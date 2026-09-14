"""Deterministic, per-cell data cleaning -- runs on the WHOLE frame before the
train/test split. Nothing here is "fitted" from the data distribution (no means,
no scaler stats), so applying it before the split is leakage-safe. Fitted
transforms (impute / scale / encode / outliers) live in execute.py and run after
the split.

Operations are a fixed, validated allowlist -- the LLM SELECTS from it, it never
supplies code. ops is an ORDERED list; each is column-scoped except
drop_duplicates which is dataset-level:

  [{"op":"trim_whitespace","column":"name"},
   {"op":"merge_categories","column":"sex","mapping":{"male":"Male"}},  # or omit mapping = trim+lower
   {"op":"retype","column":"amount","to":"number"},   # to = number | datetime | category
   {"op":"map_values","column":"grade","mapping":{"A":"1"}},
   {"op":"drop_column","column":"notes"},
   {"op":"drop_duplicates"}]
"""

import pandas as pd

from preprocessing.profiler import is_numeric

_CLEAN_OPS = {"trim_whitespace", "merge_categories", "retype", "map_values", "nullify",
              "rename_column", "drop_column", "drop_duplicates"}
_RETYPE = {"number", "datetime", "category"}
# Textual placeholders that mean "missing" but read as real values. Numeric
# sentinels (999, -1) are dataset-specific -- NOT auto-nullified (would corrupt data).
_NULL_TOKENS = ["", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "unknown"]


def _intlike(series):
    s = series.dropna()
    return len(s) > 0 and is_numeric(s) and bool((s % 1 == 0).all())


def _validate(op):
    name = op.get("op")
    if name not in _CLEAN_OPS:
        raise ValueError(f"Unknown clean op '{name}'. Allowed: {sorted(_CLEAN_OPS)}")
    if name != "drop_duplicates" and not op.get("column"):
        raise ValueError(f"clean op '{name}' needs a column.")
    if name == "retype" and op.get("to", "number") not in _RETYPE:
        raise ValueError(f"retype.to must be one of {sorted(_RETYPE)}")
    if name == "rename_column" and not str(op.get("to", "")).strip():
        raise ValueError("rename_column needs a non-empty 'to'.")


def apply_clean(df, ops):
    """Apply an ordered list of validated clean ops to the whole frame.
    Returns (cleaned_df, summary[]) where each summary entry says what changed."""
    df = df.copy()
    summary = []
    for op in ops:
        _validate(op)
        name = op["op"]

        if name == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            summary.append({"op": "drop_duplicates", "detail": f"{before - len(df)} duplicate rows removed"})
            continue

        col = op["column"]
        if col not in df.columns:
            continue  # already dropped by an earlier op, or a stale plan entry
        before_dtype = str(df[col].dtype)

        if name == "trim_whitespace":
            df[col] = df[col].astype("string").str.strip()

        elif name == "merge_categories":
            s = df[col].astype("string")
            mapping = op.get("mapping")
            # explicit mapping wins; otherwise normalize spellings (trim + lower)
            df[col] = s.map(lambda v: mapping.get(v, v)) if mapping else s.str.strip().str.lower()

        elif name == "retype":
            to = op.get("to", "number")
            if to == "number":
                s = df[col].astype("string").str.strip().str.replace(",", "", regex=False)
                num = pd.to_numeric(s, errors="coerce")
                df[col] = num.astype("Int64") if _intlike(num) else num
            elif to == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:  # category
                df[col] = df[col].astype("category")

        elif name == "map_values":
            df[col] = df[col].replace(op.get("mapping", {}))

        elif name == "nullify":
            tokens = [str(t).lower() for t in (op.get("tokens") or _NULL_TOKENS)]
            mask = df[col].astype("string").str.strip().str.lower().isin(tokens)
            n = int(mask.sum())
            df[col] = df[col].mask(mask)
            summary.append({"op": name, "column": col, "detail": f"{n} placeholder value(s) -> missing"})
            continue

        elif name == "rename_column":
            to = str(op["to"]).strip()
            if to != col:
                if to in df.columns:
                    raise ValueError(f"Cannot rename '{col}' to '{to}': that column already exists.")
                df = df.rename(columns={col: to})
                summary.append({"op": name, "column": col, "detail": f"renamed to {to}"})
            continue

        elif name == "drop_column":
            df = df.drop(columns=[col])
            summary.append({"op": name, "column": col, "detail": "column dropped"})
            continue

        summary.append({"op": name, "column": col,
                        "detail": f"{before_dtype} -> {df[col].dtype}"})
    return df, summary


if __name__ == "__main__":
    df = pd.DataFrame({
        "sex": ["Male", "male", " Male", "female", "Female "],
        "amount": ["1,200", "3,400", "900", "  50", "1,000"],
        "const": [1, 1, 1, 1, 1],
        "keep": [10, 20, 30, 40, 50],
        "notes": ["ok", "N/A", "-", "?", "fine"],
    })
    ops = [
        {"op": "trim_whitespace", "column": "sex"},
        {"op": "merge_categories", "column": "sex"},          # -> all lower, trimmed
        {"op": "retype", "column": "amount", "to": "number"},  # "1,200" -> 1200 (Int64)
        {"op": "nullify", "column": "notes"},                  # N/A, -, ? -> missing
        {"op": "drop_column", "column": "const"},
        {"op": "rename_column", "column": "keep", "to": "score"},  # renames last
    ]
    out, summ = apply_clean(df, ops)

    assert out["sex"].nunique() == 2, out["sex"].tolist()      # male / female, merged
    assert is_numeric(out["amount"]) and int(out["amount"].iloc[0]) == 1200
    assert str(out["amount"].dtype) == "Int64"                 # kept integer, not float
    assert "const" not in out.columns
    assert "score" in out.columns and "keep" not in out.columns  # renamed
    assert out["notes"].isna().sum() == 3, out["notes"].tolist()  # N/A, -, ? -> missing
    try:
        apply_clean(df, [{"op": "rename_column", "column": "sex", "to": "amount"}])
        raise AssertionError("collision rename should have raised")
    except ValueError:
        pass
    print("clean self-check passed:", [s["detail"] for s in summ])
