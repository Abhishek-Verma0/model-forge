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
  encode  : method   = onehot | ordinal | text
            text keeps the raw column; its word + letter TF-IDF is fitted on
            train and lives in the saved pipeline (to_matrix), not in the CSV.
  scale   : method   = standard | robust | minmax
  outliers: method   = clip_iqr | remove_rows          (remove_rows = TRAIN only)
  drop_column        : drop the column
  drop_rows_missing  : drop rows where this column is missing (before split)
"""

import json

import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from preprocessing import advanced, clean
from preprocessing.profiler import is_numeric, missing_mask

# Ordered: the Preprocessing tab lists them in this order (options() below).
_SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
_IMPUTE = ("median", "mean", "most_frequent", "constant")
_ENCODE = ("onehot", "ordinal", "text")
_OUTLIER = ("clip_iqr", "zscore", "winsorize", "remove_rows")
# Defaults for op parameters the user can override per column (the UI sends them).
IQR_K = 1.5              # published default: Tukey's fences, Q1/Q3 -/+ 1.5 x IQR
ZSCORE_THRESHOLD = 3.0   # published default: the common 3-standard-deviation rule
WINSOR_LOWER = 0.05      # our starting point: cap the lowest 5%
WINSOR_UPPER = 0.95      # our starting point: cap the highest 5%
# Split and dataset-level defaults (the request can override every one).
TEST_SIZE = 0.2          # our starting point: 80:20 split
TEST_SIZE_MIN, TEST_SIZE_MAX = 0.05, 0.5  # our starting point: accepted range
RANDOM_STATE = 42        # our starting point: any fixed seed reproduces the split
STRATIFY = True          # our starting point: keep class shares equal in train and test
FS_K = 10                # our starting point: features kept by feature selection
PCA_N = 2                # our starting point: PCA components

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


def _iqr_bounds(series, k=None):
    k = IQR_K if k is None else float(k)
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return -np.inf, np.inf  # zero spread -> nothing to clip
    return q1 - k * iqr, q3 + k * iqr


def _float(s):
    """Column as plain float64. Advanced imputation hands back Int64 columns, and
    writing a fractional bound/fill into an Int64 raises -- so every per-column op
    works in float and rounds back to Int64 at the end."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


# Every per-column op below is fit-or-replay: with st=None it learns its values
# from `fr` (the TRAIN frame) and returns them as st; given st it only applies
# them. apply_plan fits on train and replays on test; transform() replays the
# saved st on new rows -- one code path, so the saved pipeline can't drift.

def _impute(op, col, fr, st, intlike):
    strat = op.get("strategy", "median")
    s = fr[col]
    # blank/whitespace strings count as missing too (same rule as the profiler)
    if not is_numeric(s):
        s = s.replace(r"^\s*$", np.nan, regex=True)
    if is_numeric(s):
        s = _float(s)
    if st is None:
        if strat in ("mean", "median"):
            vals = pd.to_numeric(s, errors="coerce")
            fill = vals.mean() if strat == "mean" else vals.median()
        elif strat == "most_frequent":
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else 0
        else:  # constant
            fill = op.get("fill_value", 0 if intlike else "Missing")
        st = {"fill": fill}
    s = s.fillna(st["fill"])
    # median / most_frequent / constant keep integer semantics; mean can be
    # fractional so it's allowed to become float.
    fr[col] = s.round().astype("Int64") if intlike and strat != "mean" else s
    return fr, st


def _scale(op, col, fr, st, intlike):
    a = pd.to_numeric(fr[col], errors="coerce").to_numpy().reshape(-1, 1)
    if st is None:
        st = _SCALERS[op.get("method", "standard")]().fit(a)
    fr[col] = st.transform(a).ravel()
    return fr, st  # scaling intentionally yields float


def _outliers(op, col, fr, st, intlike):
    # remove_rows drops TRAIN rows before this loop; the rest clip to train-fitted
    # bounds so the column keeps its dtype and shape.
    method = op.get("method", "clip_iqr")
    if method == "remove_rows":
        return fr, None
    if st is None:
        s = _float(fr[col])
        if method == "zscore":
            z = float(op.get("threshold", ZSCORE_THRESHOLD))
            m, sd = s.mean(), s.std()
            st = (m - z * sd, m + z * sd) if sd else (-np.inf, np.inf)
        elif method == "winsorize":
            st = (s.quantile(float(op.get("lower", WINSOR_LOWER))),
                  s.quantile(float(op.get("upper", WINSOR_UPPER))))
        else:  # clip_iqr
            st = _iqr_bounds(fr[col], op.get("k"))
    s = _float(fr[col]).clip(*st)
    fr[col] = s.round().astype("Int64") if intlike else s
    return fr, st


def _text(col, fr, st):
    """Keep the raw text; fit word 1-2 + letter 3-5 TF-IDF on TRAIN. Best of the
    options measured on SMS spam (10-fold CV, docs/backlog.md) -- the squashed
    SVD version lost on every fold. The vectorizers are applied by to_matrix at
    training time; a CSV can't carry ~100k sparse columns."""
    s = fr[col].fillna("").astype(str)
    if st is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            st = [TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(s),
                  TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(s)]
        except ValueError:
            raise ValueError(f"'{col}' has no text to encode (all blank).") from None
    fr[col] = s
    return fr, st


def _encode(op, col, fr, st, intlike):
    method = op.get("method", "onehot")
    if method == "text":
        return _text(col, fr, st)
    s = fr[col].astype("string")
    if method == "ordinal":
        if st is None:
            st = {v: i for i, v in enumerate(s.dropna().unique())}
        fr[col] = s.map(st).fillna(-1).astype("int64")  # unseen -> -1
        return fr, st
    # onehot: categories fixed on TRAIN; unseen categories -> all-zero row.
    d = pd.get_dummies(s, prefix=col)
    if st is None:
        st = list(d.columns)
    return d.reindex(columns=st, fill_value=0).astype("int64"), st


_DISPATCH = {"impute": _impute, "scale": _scale, "outliers": _outliers, "encode": _encode}


def guard_target(df, target, task):
    """Remove rows the target can't be learned from, and check the target fits
    the task -- before any split, so preprocessing and training see the same
    rows. Returns (df, notes). Raises ValueError with a message for the user."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not a column in this dataset.")
    notes = []
    missing = missing_mask(df[target])  # null OR blank string -- the profiler's rule
    if missing.any():
        df = df[~missing]
        notes.append(f"{int(missing.sum())} row(s) had no {target} value and were removed")
    if task == "regression" and not is_numeric(df[target]):
        raise ValueError(f"Regression needs a numeric target column; '{target}' is {df[target].dtype}. "
                         "Convert it on the Cleaning tab, or choose classification.")
    if task == "classification":
        counts = df[target].value_counts()
        if len(counts) < 2:
            raise ValueError(f"Classification needs at least 2 classes in '{target}' (found {len(counts)}).")
        single = [str(v) for v, n in counts.items() if n < 2]
        if single:
            notes.append(f"class(es) with a single row: {', '.join(single[:10])} -- the split "
                         "can't be stratified; merge or remove them")
    return df, notes


def prepare_rows(df, target, columns, task):
    """Target guard + whole-table row/column steps, shared by preprocessing and
    training so both work on the same rows. Returns (df, feats, drop_cols, notes)."""
    df, notes = guard_target(df, target, task)
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
    return df, feats, drop_cols, notes


def split(X, y, task, test_size, random_state, stratify):
    """The locked train/test split -- one function, so training reproduces the
    preprocessing run's split exactly. Returns (X_tr, X_te, y_tr, y_te, stratified)."""
    strat = y if (stratify and task == "classification" and y.nunique() > 1
                  and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat)
    return X_tr, X_te, y_tr, y_te, strat is not None


def fit_prep(X_tr, y_tr, columns, pipeline, task, intlike_cols):
    """Learn every preprocessing step from the training rows only -- once per
    preprocessing run, and once per CV fold in training. Returns (fitted, tr_out,
    summary, notes, class_weights); replay with transform(fitted, X) + to_matrix.

    Steps that add or drop TRAIN rows (outlier row removal, resampling) are not
    applied here: CV folds must apply them to fold-training rows only, or copies of
    scored rows leak into training (train.py adds them as samplers).
    class_weights changes no rows, so it is computed."""
    feats = list(X_tr.columns)
    before = {c: str(X_tr[c].dtype) for c in feats}
    notes = []
    fitted = {"sklearn": sklearn.__version__, "features": feats, "clean_ops": [], "impute": None,
              "columns": [], "text": {}, "keep": None, "pca": None}

    _imp = advanced._m("imputation", pipeline)
    if _imp:
        fitted["impute"], _n = advanced.fit_impute(
            X_tr, _imp, intlike_cols, (pipeline.get("imputation") or {}).get("n_neighbors"))
        if fitted["impute"]:
            X_tr = advanced.impute_apply(fitted["impute"], X_tr)
        notes.append(_n)

    parts, summary = [], []
    for c in feats:
        tr = X_tr[[c]].copy()
        intlike = c in intlike_cols
        applied, replay = [], []
        for o in columns.get(c, []):
            name = o["op"]
            if name in ("drop_rows_missing",):
                continue
            if name == "outliers" and o.get("method") == "remove_rows":
                applied.append("outliers(remove_rows) - applied during training, inside each fold")
                continue
            tr, st = _DISPATCH[name](o, c, tr, None, intlike)
            replay.append((o, st, intlike))
            if name == "encode" and o.get("method") == "text":
                fitted["text"][c] = st
            applied.append(_label(name, o, tr.shape[1], st))
        fitted["columns"].append((c, replay))
        parts.append(tr)
        after = f"{tr.shape[1]} cols" if tr.shape[1] > 1 else str(tr[tr.columns[0]].dtype)
        summary.append({"column": c, "dtype_before": before[c], "dtype_after": after,
                        "ops": applied or ["kept as-is"]})
    tr_out = pd.concat(parts, axis=1)

    class_weights = None
    if advanced._m("outlier_removal", pipeline):
        notes.append(f"{advanced._m('outlier_removal', pipeline)} - applied during training, "
                     "inside each fold (not written to the export)")
    if advanced._m("feature_selection", pipeline):
        k = (pipeline.get("feature_selection") or {}).get("k", FS_K)
        tr_out, _, _n, fitted["keep"] = advanced.select_features(
            tr_out, tr_out.iloc[:0], y_tr, advanced._m("feature_selection", pipeline), k, task)
        notes.append(_n)
    if advanced._m("reduction", pipeline):
        fitted["pca"], _n = advanced.fit_reduce(tr_out, (pipeline.get("reduction") or {}).get("n", PCA_N))
        if fitted["pca"]:
            tr_out = advanced.reduce_apply(fitted["pca"], tr_out)
        notes.append(_n)
    _imb = advanced._m("imbalance", pipeline)
    if _imb == "class_weights":
        tr_out, y_tr, _n, class_weights = advanced.balance(tr_out, y_tr, _imb, task)
        notes.append(_n)
    elif _imb:
        notes.append(f"{_imb} - applied during training, inside each fold (not written to the export)")
    return fitted, tr_out, summary, notes, class_weights


def options():
    """Choices and defaults for the Preprocessing tab -- the single source; the UI
    renders these instead of keeping its own copies."""
    from preprocessing.detector import MAX_CATEGORICAL_UNIQUE
    return {
        "split": {"test_size": TEST_SIZE, "test_size_min": TEST_SIZE_MIN, "test_size_max": TEST_SIZE_MAX,
                  "random_state": RANDOM_STATE, "stratify": STRATIFY},
        "missing": ["none", *_IMPUTE, "drop_rows", "drop_column"],
        "scale": ["none", *_SCALERS],
        "outliers": ["none", *_OUTLIER],
        "encode": ["none", *_ENCODE, "drop"],
        "outlier_defaults": {"clip_iqr": {"k": IQR_K}, "remove_rows": {"k": IQR_K},
                             "zscore": {"threshold": ZSCORE_THRESHOLD},
                             "winsorize": {"lower": WINSOR_LOWER, "upper": WINSOR_UPPER}},
        "pipeline": {"imputation": ["none", *advanced._IMPUTE], "outlier_removal": ["none", *advanced._OUTREM],
                     "feature_selection": ["none", *advanced._FS], "imbalance": ["none", *advanced._IMB],
                     "reduction": ["none", *advanced._RED]},
        "pipeline_defaults": {"knn_n": advanced.KNN_NEIGHBORS, "contamination": advanced.CONTAMINATION,
                              "fs_k": FS_K, "red_n": PCA_N},
        "max_onehot": MAX_CATEGORICAL_UNIQUE,  # more distinct values than this: too many to one-hot
    }


def apply_plan(df, target, columns, task="classification", test_size=None,
               random_state=None, pipeline=None, stratify=STRATIFY):
    """Run a validated per-column op plan, fit on train only. Returns preview,
    cleaned CSV, a per-column change summary and the fitted pipeline. `pipeline` =
    optional dataset-level stage (advanced impute / outlier removal / feature
    selection / imbalance / reduction) -- see fit_prep and advanced.py.

    test_size / random_state default to TEST_SIZE / RANDOM_STATE so the same settings
    reproduce the same split; `stratify=False` forces a plain random split."""
    test_size = TEST_SIZE if test_size is None else float(test_size)
    random_state = RANDOM_STATE if random_state is None else int(random_state)
    if not TEST_SIZE_MIN <= test_size <= TEST_SIZE_MAX:
        raise ValueError(f"test_size must be between {TEST_SIZE_MIN} and {TEST_SIZE_MAX} (got {test_size}).")
    for _c, ops in columns.items():
        for op in ops:
            _validate(op)
    pipeline = pipeline or {}
    advanced.validate_pipeline(pipeline)

    df, feats, drop_cols, target_notes = prepare_rows(df, target, columns, task)
    X_tr, X_te, y_tr, y_te, stratified = split(df[feats], df[target], task, test_size, random_state, stratify)
    intlike_cols = {c for c in feats if _is_intlike(df[c])}
    fitted, tr_out, summary, pipe_notes, class_weights = fit_prep(
        X_tr, y_tr, columns, pipeline, task, intlike_cols)
    fitted["target"] = target
    fitted["settings"] = {"columns": columns, "pipeline": pipeline, "task": task, "test_size": test_size,
                          "random_state": random_state, "stratify": stratify}
    te_out = transform(fitted, X_te)

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
        "stratified": stratified,
        "test_size": test_size,
        "random_state": random_state,
        "summary": summary,
        "pipeline": pipe_notes,
        "class_weights": class_weights,
        "notes": target_notes,
        "preview": json.loads(cleaned.head(20).to_json(orient="records")),
        "csv": cleaned.to_csv(index=False),
        "fitted": fitted,
    }


def transform(fitted, df):
    """Replay a fitted run on NEW raw rows: the recorded clean ops, then every
    step with its train-fitted values. Training-only row steps (dedupe, drop rows,
    outlier row removal, resampling) are skipped -- you can't drop a row you
    were asked to predict."""
    ops = [o for o in fitted["clean_ops"] if o["op"] not in clean.ROW_OPS]  # new rows are never dropped
    if ops:
        df, _ = clean.apply_clean(df, ops)
    X = df[fitted["features"]].copy()
    if fitted["impute"]:
        X = advanced.impute_apply(fitted["impute"], X)
    parts = []
    for c, replay in fitted["columns"]:
        fr = X[[c]].copy()
        for o, st, intlike in replay:
            fr, _ = _DISPATCH[o["op"]](o, c, fr, st, intlike)
        parts.append(fr)
    out = pd.concat(parts, axis=1)
    if fitted["keep"] is not None:
        out = out[fitted["keep"]]
    if fitted["pca"]:
        out = advanced.reduce_apply(fitted["pca"], out)
    return out


def to_matrix(fitted, frame):
    """Model input: the numeric columns plus each raw text column's saved TF-IDF,
    as one sparse matrix. `frame` = transform() output, or the exported CSV's
    feature columns."""
    from scipy.sparse import csr_matrix, hstack
    text = fitted["text"]
    rest = frame.drop(columns=list(text))
    bad = [str(c) for c in rest.columns if not pd.api.types.is_numeric_dtype(rest[c])]
    if bad:
        raise ValueError(f"Column(s) {', '.join(bad[:5])} are not numbers yet -- choose an encoding "
                         "(onehot, ordinal or text) for them on the Preprocessing tab.")
    blocks = [csr_matrix(rest.astype("float64").to_numpy())]
    for c, vecs in text.items():
        blocks += [v.transform(frame[c].fillna("").astype(str)) for v in vecs]
    return hstack(blocks).tocsr()


def _label(name, op, ncols, st=None):
    if name == "impute":
        return f"impute({op.get('strategy', 'median')})"
    if name == "scale":
        return f"scale({op.get('method', 'standard')})"
    if name == "encode":
        m = op.get("method", "onehot")
        if m == "text":
            terms = sum(len(v.vocabulary_) for v in st)
            return f"text kept; word+letter tf-idf fit on train ({terms:,} terms, applied at training)"
        return f"encode({m})" + (f" -> {ncols} cols" if m == "onehot" else "")
    if name == "outliers":
        m = op.get("method", "clip_iqr")
        if m == "zscore":
            return f"outliers(zscore, {float(op.get('threshold', ZSCORE_THRESHOLD)):g} sd)"
        if m == "winsorize":
            return (f"outliers(winsorize, "
                    f"{float(op.get('lower', WINSOR_LOWER)):g}-{float(op.get('upper', WINSOR_UPPER)):g})")
        if m in ("clip_iqr", "remove_rows"):
            return f"outliers({m}, k={float(op.get('k', IQR_K)):g})"
        return f"outliers({m})"
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

    # split settings are honoured and recorded (plan §10 / MVP reproducibility)
    assert out["test_size"] == 0.2 and out["random_state"] == 42
    wide = apply_plan(df, "label", plan, test_size=0.3, random_state=7)
    assert wide["test_rows"] == 3 and wide["random_state"] == 7, wide["test_rows"]
    # same seed -> same split; different seed -> (almost surely) a different one
    a = apply_plan(df, "label", plan, random_state=1)["preview"]
    assert a == apply_plan(df, "label", plan, random_state=1)["preview"], "seed must reproduce the split"
    assert not apply_plan(df, "label", plan, stratify=False)["stratified"]
    for bad in (0.9, 0.01):
        try:
            apply_plan(df, "label", plan, test_size=bad)
            raise AssertionError(f"test_size={bad} should have been rejected")
        except ValueError:
            pass
    # per-op outlier parameters override the config defaults
    tight = apply_plan(df, "label", {"bmi": [{"op": "outliers", "method": "zscore", "threshold": 0.5}]})
    loose = apply_plan(df, "label", {"bmi": [{"op": "outliers", "method": "zscore", "threshold": 3}]})
    assert tight["preview"] != loose["preview"], "zscore threshold must change the clipping"
    # the chosen parameter must reach the summary, or the report can't cite it (plan §15)
    assert "0.5 sd" in str(tight["summary"]), tight["summary"]
    kk = apply_plan(df, "label", {"bmi": [{"op": "outliers", "method": "clip_iqr", "k": 2.5}]})
    assert "k=2.5" in str(kk["summary"]), kk["summary"]

    # regression: advanced imputation returns Int64 columns, and every outlier
    # method writes fractional bounds -- zscore/winsorize used to raise TypeError.
    whole = pd.DataFrame({"a": [float(i) for i in range(39)] + [900.0],
                          "b": [i % 7 for i in range(40)], "y": ["p", "q"] * 20})
    for m in ("clip_iqr", "zscore", "winsorize", "remove_rows"):
        r = apply_plan(whole, "y", {"a": [{"op": "outliers", "method": m}]},
                       pipeline={"imputation": {"method": "knn"}})
        assert r["train_rows"] > 0, m
    print("int64-after-advanced-impute regression passed")

    # saved pipeline: replaying it on the raw TEST rows must reproduce the run's
    # own test output exactly -- for numbers, categories, text, and dataset steps.
    import io, joblib
    from sklearn.linear_model import LogisticRegression
    words = ["great", "awful", "fast", "broken", "love", "refund", "cheap", "solid"]
    raw = pd.DataFrame({
        "Review ": [f" {words[i % 8]} item {words[(i * 3) % 8]} delivery order {i}" for i in range(80)],
        "age": [None if i % 9 == 0 else 20 + i % 50 for i in range(80)],
        "bmi": [18 + (i * 7) % 25 + 0.5 for i in range(80)],
        "sex": ["M", "F", "f"] * 26 + ["M", "F"],
        "y": ["p", "q"] * 40,
    })
    raw.loc[3, "Review "] = None                                         # blank text is fine
    clean_ops = [{"op": "rename_column", "column": "Review ", "to": "review"},
                 {"op": "merge_categories", "column": "sex"}]
    df_c, _ = clean.apply_clean(raw, clean_ops)
    plan = {"review": [{"op": "encode", "method": "text"}],
            "age": [{"op": "impute", "strategy": "median"}, {"op": "outliers", "method": "zscore"}],
            "bmi": [{"op": "scale", "method": "robust"}],
            "sex": [{"op": "encode", "method": "onehot"}]}
    r = apply_plan(df_c, "y", plan, pipeline={"imputation": {"method": "knn"},
                                              "feature_selection": {"method": "anova", "k": 2}})
    fitted = r["fitted"]
    fitted["clean_ops"] = clean_ops
    buf = io.BytesIO(); joblib.dump(fitted, buf); buf.seek(0)
    fitted = joblib.load(buf)                                            # survives a save/load

    out = pd.read_csv(io.StringIO(r["csv"]))
    assert out["review"].str.contains("delivery").any()                 # raw text kept in the CSV
    test_out = out[out["__split__"] == "test"].drop(columns=["y", "__split__"]).reset_index(drop=True)
    _, raw_te = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw["y"])  # same split
    replayed = transform(fitted, raw_te.drop(columns=["y"]))
    replayed = pd.read_csv(io.StringIO(replayed.to_csv(index=False)))    # same text round trip as the CSV
    pd.testing.assert_frame_equal(replayed, test_out, check_dtype=False)

    # row ops recorded at cleaning time (dedupe, drop rows with missing values) must NOT
    # run again on new rows -- predicting returns one row per row given
    with_row_ops = dict(fitted, clean_ops=clean_ops + [{"op": "drop_duplicates"}, {"op": "drop_missing_rows"}])
    new_rows = raw.drop(columns=["y"]).head(10).copy()
    new_rows.iloc[0, new_rows.columns.get_loc("age")] = None      # a row that cleaning would have dropped
    new_rows.iloc[1] = new_rows.iloc[2]                           # and a duplicate row
    assert len(transform(with_row_ops, new_rows)) == len(new_rows), "predicting must never drop rows"

    train_out = out[out["__split__"] == "train"]
    X_tr = to_matrix(fitted, train_out.drop(columns=["y", "__split__"]))
    n_terms = sum(len(v.vocabulary_) for v in fitted["text"]["review"])
    assert X_tr.shape[1] == test_out.shape[1] - 1 + n_terms, X_tr.shape  # numbers + every text term
    LogisticRegression(max_iter=500).fit(X_tr, train_out["y"]).predict(to_matrix(fitted, replayed))
    print("saved-pipeline replay self-check passed:", X_tr.shape, [s["ops"] for s in r["summary"]])
    # target guard: missing targets removed and reported; impossible targets rejected clearly
    g = pd.DataFrame({"x": range(12), "y": [1.5, None] + [float(i) for i in range(10)]})
    r = apply_plan(g, "y", {}, task="regression")
    assert r["train_rows"] + r["test_rows"] == 11, (r["train_rows"], r["test_rows"])
    assert "1 row(s) had no y value" in r["notes"][0], r["notes"]
    blank = pd.DataFrame({"x": range(10), "y": ["a", "b", "", "a", "b", "a", "b", "a", "b", "a"]})
    assert apply_plan(blank, "y", {})["rows_before"] == 9          # blank string counts as missing
    for bad_df, task, msg in (
            (pd.DataFrame({"x": range(10), "y": list("abcdefghij")}), "regression", "numeric target"),
            (pd.DataFrame({"x": range(10), "y": ["a"] * 10}), "classification", "at least 2 classes")):
        try:
            apply_plan(bad_df, "y", {}, task=task)
            raise AssertionError(f"{task} target should have been rejected")
        except ValueError as e:
            assert msg in str(e), e
    one = apply_plan(pd.DataFrame({"x": range(10), "y": ["a"] * 9 + ["b"]}), "y", {})
    assert any("single row" in n for n in one["notes"]) and not one["stratified"], one["notes"]
    print("target guard self-check passed")

    # row-changing steps are NOT written into the export; training applies them per fold
    imb = pd.DataFrame({"x": list(range(60)), "z": [i % 7 for i in range(60)], "y": ["a"] * 50 + ["b"] * 10})
    imb.loc[5, "x"] = 10_000                                           # an obvious outlier
    r = apply_plan(imb, "y", {"x": [{"op": "outliers", "method": "remove_rows"}]},
                   pipeline={"imbalance": {"method": "oversample"},
                             "outlier_removal": {"method": "isolation_forest"}})
    assert (r["train_rows"], r["test_rows"]) == (48, 12), (r["train_rows"], r["test_rows"])
    assert sum("applied during training" in n for n in r["pipeline"]) == 2, r["pipeline"]
    assert "applied during training" in r["summary"][0]["ops"][0], r["summary"][0]
    assert r["fitted"]["settings"]["pipeline"]["imbalance"]["method"] == "oversample"
    cw = apply_plan(imb, "y", {}, pipeline={"imbalance": {"method": "class_weights"}})
    assert cw["class_weights"] and cw["train_rows"] == 48, cw["class_weights"]
    print("row-changing steps self-check passed")

    # fit_prep is the single fitting path; to_matrix names columns that still need an encoding
    fx = pd.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0, 6.0], "s": list("xyxyxy")})
    fitted_fx, tr_fx, summ_fx, notes_fx, _ = fit_prep(
        fx, pd.Series([0, 1] * 3), {"a": [{"op": "impute", "strategy": "median"}]}, {}, "classification", set())
    assert tr_fx["a"].tolist() == [1.0, 4.0, 3.0, 4.0, 5.0, 6.0], tr_fx["a"].tolist()
    assert summ_fx[0]["ops"] == ["impute(median)"] and fitted_fx["features"] == ["a", "s"]
    try:
        to_matrix(fitted_fx, tr_fx)
        raise AssertionError("an unencoded text column must be refused")
    except ValueError as e:
        assert "s" in str(e) and "encoding" in str(e), e
    assert apply_plan(fx.assign(y=[0, 1] * 3), "y", {})["fitted"]["settings"]["stratify"] is True
    print("fit_prep self-check passed")

    # options() is what the Preprocessing tab renders: every choice must pass validation
    o = options()
    ui_only = {"none", "drop_rows", "drop_column", "drop"}
    for m in set(o["missing"]) - ui_only:
        _validate({"op": "impute", "strategy": m})
    for m in set(o["scale"]) - ui_only:
        _validate({"op": "scale", "method": m})
    for m in set(o["encode"]) - ui_only:
        _validate({"op": "encode", "method": m})
    for m, params in o["outlier_defaults"].items():
        assert m in o["outliers"], m
        _validate({"op": "outliers", "method": m, **params})
    for section, methods in o["pipeline"].items():
        for m in methods:
            advanced.validate_pipeline({section: {"method": m}})
    assert o["split"]["test_size_min"] <= o["split"]["test_size"] <= o["split"]["test_size_max"]
    json.dumps(o)  # sent inside the analyze response
    print("options self-check passed")
    print("split + parameter self-check passed")
