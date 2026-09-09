"""Column-aware, dtype-preserving preprocessing executor.

Replaces the old ColumnTransformer-to-float-matrix path. Each column carries an
ORDERED list of operations; each op is applied to the DataFrame directly (not a
numpy matrix), fit on TRAIN only, applied to train + test. So:

  age -> [impute(median)]                 stays Int64, NOT scaled, NOT floated
  bmi -> [impute(median), scale(robust)]  float only because scale was chosen
  sex -> [encode(onehot)]                 -> int 0/1 columns

Operations are a fixed, validated allowlist -- the LLM SELECTS from it, it never
supplies code. Tabular only. Cleaning (dedupe / trim / retype / merge) lives in
clean.py and runs on the whole frame BEFORE this; this module only does the
train-fitted transforms.

plan columns = { "age": [{"op":"impute","strategy":"median"}], ... }
  impute  : strategy = mean | median | most_frequent | constant  (+ fill_value)
  encode  : method   = onehot | ordinal
  scale   : method   = standard | robust | minmax
  outliers: method   = clip_iqr | remove_rows          (remove_rows = TRAIN only)
  drop_column        : drop the column
  drop_rows_missing  : drop rows where this column is missing (before split)
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from profiler import is_numeric
import advanced

_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
_IMPUTE = {"mean", "median", "most_frequent", "constant"}
_ENCODE = {"onehot", "ordinal"}
_OUTLIER = {"clip_iqr", "remove_rows", "zscore", "winsorize"}
_OPS = {"impute", "encode", "scale", "outliers", "drop_column", "drop_rows_missing"}


def _validate(op):
    name = op.get("op")
    if name not in _OPS:
        raise ValueError(f"Unknown operation '{name}'. Allowed: {sorted(_OPS)}")
    if name == "impute" and op.get("strategy", "median") not in _IMPUTE:
        raise ValueError(f"impute.strategy must be one of {sorted(_IMPUTE)}")
    if name == "encode" and op.get("method", "onehot") not in _ENCODE:
        raise ValueError(f"encode.method must be one of {sorted(_ENCODE)}")
    if name == "scale" and op.get("method", "standard") not in _SCALERS:
        raise ValueError(f"scale.method must be one of {sorted(_SCALERS)}")
    if name == "outliers" and op.get("method", "clip_iqr") not in _OUTLIER:
        raise ValueError(f"outliers.method must be one of {sorted(_OUTLIER)}")


def _is_intlike(series):
    """A column is integer-like if every non-null value is a whole number, even
    when pandas has promoted it to float64 because of NaNs. We use this to keep
    age an integer after a median fill instead of floating the whole column."""
    s = series.dropna()
    return len(s) > 0 and is_numeric(s) and bool((s % 1 == 0).all())


def _iqr_bounds(series, k=1.5):
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return -np.inf, np.inf  # zero spread -> nothing to clip
    return q1 - k * iqr, q3 + k * iqr


def _as_int(tr, te, col):
    tr[col] = tr[col].round().astype("Int64")
    te[col] = te[col].round().astype("Int64")


def _impute(op, col, tr, te, intlike):
    strat = op.get("strategy", "median")
    # blank/whitespace strings count as missing too (same rule as the profiler)
    if not is_numeric(tr[col]):
        tr[col] = tr[col].replace(r"^\s*$", np.nan, regex=True)
        te[col] = te[col].replace(r"^\s*$", np.nan, regex=True)

    if strat in ("mean", "median"):
        vals = pd.to_numeric(tr[col], errors="coerce")
        fill = vals.mean() if strat == "mean" else vals.median()
    elif strat == "most_frequent":
        mode = tr[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else 0
    else:  # constant
        fill = op.get("fill_value", 0 if intlike else "Missing")

    tr[col] = tr[col].fillna(fill)
    te[col] = te[col].fillna(fill)
    # median / most_frequent / constant keep integer semantics; mean can be
    # fractional so it's allowed to become float.
    if intlike and strat != "mean":
        _as_int(tr, te, col)
    return tr, te


def _scale(op, col, tr, te, intlike):
    method = op.get("method", "standard")
    sc = _SCALERS[method]()
    a_tr = pd.to_numeric(tr[col], errors="coerce").to_numpy().reshape(-1, 1)
    sc.fit(a_tr)  # fit on train only
    tr[col] = sc.transform(a_tr).ravel()
    te[col] = sc.transform(pd.to_numeric(te[col], errors="coerce").to_numpy().reshape(-1, 1)).ravel()
    return tr, te  # scaling intentionally yields float


def _outliers(op, col, tr, te, intlike):
    # remove_rows is handled before this loop (it drops train rows); the rest clip
    # to train-fitted bounds so the column keeps its dtype and shape.
    method = op.get("method", "clip_iqr")
    if method == "remove_rows":
        return tr, te
    s = pd.to_numeric(tr[col], errors="coerce")
    if method == "zscore":
        m, sd = s.mean(), s.std()
        lo, hi = (m - 3 * sd, m + 3 * sd) if sd else (-np.inf, np.inf)
    elif method == "winsorize":
        lo, hi = s.quantile(0.05), s.quantile(0.95)
    else:  # clip_iqr
        lo, hi = _iqr_bounds(tr[col])
    tr[col] = pd.to_numeric(tr[col], errors="coerce").clip(lo, hi)
    te[col] = pd.to_numeric(te[col], errors="coerce").clip(lo, hi)
    if intlike:
        _as_int(tr, te, col)
    return tr, te


def _encode(op, col, tr, te, intlike):
    method = op.get("method", "onehot")
    s_tr, s_te = tr[col].astype("string"), te[col].astype("string")
    if method == "ordinal":
        cats = list(pd.Series(s_tr.dropna().unique()))
        mp = {v: i for i, v in enumerate(cats)}
        tr[col] = s_tr.map(mp).fillna(-1).astype("int64")  # unseen -> -1
        te[col] = s_te.map(mp).fillna(-1).astype("int64")
        return tr, te
    # onehot: categories fixed on TRAIN; unseen test categories -> all-zero row.
    d_tr = pd.get_dummies(s_tr, prefix=col).astype("int64")
    d_te = pd.get_dummies(s_te, prefix=col).reindex(columns=d_tr.columns, fill_value=0).astype("int64")
    return d_tr, d_te


_DISPATCH = {"impute": _impute, "scale": _scale, "outliers": _outliers, "encode": _encode}


def apply_plan(df, target, columns, task="classification", test_size=0.2,
               random_state=42, pipeline=None):
    """Run a validated per-column op plan, fit on train only. Returns preview,
    cleaned CSV, and a per-column change summary. `pipeline` = optional
    dataset-level stage (advanced impute / outlier removal / feature selection /
    imbalance / reduction), all fit on train -- see advanced.py."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not a column in this dataset.")
    for _c, ops in columns.items():
        for op in ops:
            _validate(op)
    pipeline = pipeline or {}
    advanced.validate_pipeline(pipeline)

    drop_cols = [c for c, ops in columns.items() if any(o["op"] == "drop_column" for o in ops)]
    row_cols = [c for c, ops in columns.items()
                if any(o["op"] == "drop_rows_missing" for o in ops) and c in df.columns]
    if row_cols:
        df = df.dropna(subset=row_cols)

    feats = [c for c in df.columns if c != target and c not in drop_cols]
    if not feats:
        raise ValueError("No feature columns left after drops.")
    if len(df) < 5:
        raise ValueError("Too few rows left to split (need at least 5).")

    X, y = df[feats], df[target]
    stratify = y if (task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # ---- dataset-level: advanced imputation (numeric block, fit on train) ----
    pipe_notes = []
    _imp = advanced._m("imputation", pipeline)
    if _imp:
        intlike_cols = {c for c in feats if _is_intlike(df[c])}
        X_tr, X_te, _n = advanced.advanced_impute(X_tr, X_te, _imp, intlike_cols)
        pipe_notes.append(_n)

    # remove_rows outliers: drop offending TRAIN rows only (dropping test rows
    # would distort evaluation). ponytail: IQR bounds, swap for z-score if needed.
    for c in feats:
        for o in columns.get(c, []):
            if o["op"] == "outliers" and o.get("method") == "remove_rows":
                lo, hi = _iqr_bounds(X_tr[c])
                num = pd.to_numeric(X_tr[c], errors="coerce")
                keep = num.between(lo, hi) | num.isna()
                X_tr, y_tr = X_tr[keep], y_tr[keep]

    tr_parts, te_parts, summary = [], [], []
    for c in feats:
        ops = columns.get(c, [])
        tr = X_tr[[c]].copy()
        te = X_te[[c]].copy()
        intlike = _is_intlike(df[c])
        before = str(df[c].dtype)
        applied = []
        for o in ops:
            name = o["op"]
            if name in ("drop_rows_missing",):
                continue
            if name == "outliers" and o.get("method") == "remove_rows":
                applied.append("outliers(remove_rows, train)")
                continue
            tr, te = _DISPATCH[name](o, c, tr, te, intlike)
            applied.append(_label(name, o, tr.shape[1]))
        tr_parts.append(tr)
        te_parts.append(te)
        after = f"{tr.shape[1]} one-hot cols" if tr.shape[1] > 1 else str(tr[tr.columns[0]].dtype)
        summary.append({"column": c, "dtype_before": before, "dtype_after": after,
                        "ops": applied or ["kept as-is"]})

    tr_out = pd.concat(tr_parts, axis=1)
    te_out = pd.concat(te_parts, axis=1)

    # ---- dataset-level pipeline on the assembled numeric matrix (fit on train) ----
    class_weights = None
    if advanced._m("outlier_removal", pipeline):
        tr_out, y_tr, _n = advanced.remove_outliers(tr_out, y_tr, advanced._m("outlier_removal", pipeline))
        pipe_notes.append(_n)
    if advanced._m("feature_selection", pipeline):
        k = (pipeline.get("feature_selection") or {}).get("k", 10)
        tr_out, te_out, _n = advanced.select_features(
            tr_out, te_out, y_tr, advanced._m("feature_selection", pipeline), k, task)
        pipe_notes.append(_n)
    if advanced._m("reduction", pipeline):
        n = (pipeline.get("reduction") or {}).get("n", 2)
        tr_out, te_out, _n = advanced.reduce(tr_out, te_out, advanced._m("reduction", pipeline), n)
        pipe_notes.append(_n)
    if advanced._m("imbalance", pipeline):
        tr_out, y_tr, _n, class_weights = advanced.balance(
            tr_out, y_tr, advanced._m("imbalance", pipeline), task)
        pipe_notes.append(_n)

    tr_out = tr_out.reset_index(drop=True)
    te_out = te_out.reset_index(drop=True)
    tr_out[target], te_out[target] = list(y_tr), list(y_te)
    tr_out["__split__"], te_out["__split__"] = "train", "test"
    cleaned = pd.concat([tr_out, te_out], ignore_index=True)

    return {
        "rows_before": len(df),
        "train_rows": len(tr_out),
        "test_rows": len(te_out),
        "features_in": len(feats),
        "features_out": tr_out.shape[1] - 2,  # minus target + __split__
        "dropped_columns": drop_cols,
        "stratified": stratify is not None,
        "summary": summary,
        "pipeline": pipe_notes,
        "class_weights": class_weights,
        "preview": json.loads(cleaned.head(20).to_json(orient="records")),
        "csv": cleaned.to_csv(index=False),
    }


def _label(name, op, ncols):
    if name == "impute":
        return f"impute({op.get('strategy', 'median')})"
    if name == "scale":
        return f"scale({op.get('method', 'standard')})"
    if name == "encode":
        m = op.get("method", "onehot")
        return f"encode({m})" + (f" -> {ncols} cols" if m == "onehot" else "")
    if name == "outliers":
        return f"outliers({op.get('method', 'clip_iqr')})"
    return name


if __name__ == "__main__":
    df = pd.DataFrame({
        "age": [20, 30, None, 40, 50, 25, 35, 45, 55, 60],   # integer with a gap
        "bmi": [22.1, 30.4, 18.0, 27.7, 41.2, 19.9, 25.5, 33.1, 28.0, 24.2],
        "sex": ["M", "F", "M", "F", "F", "M", "F", "M", "F", "M"],
        "label": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
    })
    plan = {
        "age": [{"op": "impute", "strategy": "median"}],          # fill ONLY
        "bmi": [{"op": "impute", "strategy": "median"}, {"op": "scale", "method": "robust"}],
        "sex": [{"op": "encode", "method": "onehot"}],
    }
    out = apply_plan(df, "label", plan, task="classification")

    cleaned = pd.read_csv(pd.io.common.StringIO(out["csv"]))
    # THE fix: age was median-filled and stayed INTEGER, never scaled to 0.5/0.6.
    assert set(cleaned["age"].dropna().unique()) <= set(range(0, 200)), cleaned["age"].tolist()
    assert (cleaned["age"] == cleaned["age"].round()).all(), "age must stay whole numbers"
    # bmi WAS scaled (float, small range) because we asked for it
    assert cleaned["bmi"].abs().max() < 10
    # sex expanded to one-hot int columns
    assert any(col.startswith("sex_") for col in cleaned.columns)
    assert out["train_rows"] + out["test_rows"] == 10
    print("execute self-check passed:", [s["column"] + ":" + str(s["ops"]) for s in out["summary"]])
