"""Deterministic, per-cell data cleaning -- runs on the WHOLE frame before the
train/test split. Nothing here is "fitted" from the data distribution (no means,
no scaler stats), so applying it before the split is leakage-safe. Fitted
transforms (impute / scale / encode / outliers) live in execute.py and run after
the split.

Operations are a fixed, validated allowlist -- the LLM SELECTS from it, it never
supplies code. ops is an ORDERED list; each is column-scoped except the row ops
(ROW_OPS), which act on whole rows of the dataset:

  [{"op":"trim_whitespace","column":"name"},
   {"op":"merge_categories","column":"sex","mapping":{"male":"Male"}},  # or omit mapping = trim+lower
   {"op":"retype","column":"amount","to":"number"},   # to = number | datetime | category
   {"op":"map_values","column":"grade","mapping":{"A":"1"}},
   {"op":"drop_column","column":"notes"},
   {"op":"drop_duplicates"},
   {"op":"drop_missing_rows"},                       # any column; or "columns": ["age"] for some
   {"op":"reverse_score","column":"q4","low":1,"high":5},          # questionnaire: low + high - x
   {"op":"composite","columns":["q1","q2","q4"],"to":"wellbeing","method":"mean"},  # or "sum"
  ]

Row ops drop rows, so they are applied to the dataset being prepared and SKIPPED when
the saved pipeline is replayed on new rows (execute.transform) -- predicting must return
one row per row given, never fewer.
"""

import warnings

import pandas as pd

from preprocessing.profiler import empty_string_mask, is_numeric, number_text

ROW_OPS = {"drop_duplicates", "drop_missing_rows"}   # drop whole rows; never replayed on new rows
_CLEAN_OPS = {"trim_whitespace", "merge_categories", "retype", "map_values", "nullify",
              "rename_column", "drop_column", "reverse_score", "composite"} | ROW_OPS
_COLUMNLESS = ROW_OPS | {"composite"}   # ops that name no single column
_RETYPE = {"number", "datetime", "category"}
# Textual placeholders that mean "missing" but read as real values. Numeric
# sentinels (999, -1) are dataset-specific -- NOT auto-nullified (would corrupt data).
_NULL_TOKENS = ["", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "unknown"]


_DMY = r"^(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}"   # 03/04/2020, 25.12.20, 1-2-2021 (not ISO)


def _intlike(series):
    s = series.dropna()
    return len(s) > 0 and is_numeric(s) and bool((s % 1 == 0).all())


def _parse_dates(s, dayfirst):
    # utc=True, then the zone dropped: a timestamp with an offset ("2020-01-01T10:00:00+05:30")
    # is read as its UTC time, a plain one is unchanged -- without it, writing offset times
    # into the result column raised TypeError (HTTP 500 on cleaning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, utc=True)
        if out.isna().sum() > s.isna().sum():   # several formats in one column: parse each value
            mixed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed", utc=True)
            if mixed.notna().sum() > out.notna().sum():
                out = mixed
    return out.dt.tz_localize(None)


def to_datetime(series, dayfirst=None):
    """Text -> dates, deciding day/month order from the column's OWN values:
    a first number above 12 (25/12/2020) proves day-first, a second above 12 proves
    month-first. pandas alone reads every nn/nn/yyyy month-first and turns the
    day-first ones into missing. Only nn/nn/yy(yy) values get the order -- pandas'
    dayfirst also flips ISO 2020-03-04, so those rows are parsed on their own.
    Pass `dayfirst` to replay an earlier decision. Returns (dates, dayfirst, note)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, bool(dayfirst), None
    s = series.astype("string").str.strip()
    parts = s.str.extract(_DMY)
    first, second = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
    dmy = first.notna()
    note = None
    if dayfirst is None:
        day_proof, month_proof = bool((first > 12).any()), bool((second > 12).any())
        dayfirst = day_proof and not month_proof
        if not dmy.any():
            note = None
        elif dayfirst:
            note = "read as day/month/year (a first number above 12 proves it)"
        elif month_proof and not day_proof:
            note = "read as month/day/year (a second number above 12 proves it)"
        elif day_proof and month_proof:
            note = "day/month order differs between rows - check this column"
        else:
            note = "day/month order can't be told from the values - read as month/day/year"
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if dmy.any():
        out[dmy] = _parse_dates(s[dmy], dayfirst)
    if (~dmy).any():
        out[~dmy] = _parse_dates(s[~dmy], False)
    return out, dayfirst, note


def _validate(op):
    name = op.get("op")
    if name not in _CLEAN_OPS:
        raise ValueError(f"Unknown clean op '{name}'. Allowed: {sorted(_CLEAN_OPS)}")
    if name not in _COLUMNLESS and not op.get("column"):
        raise ValueError(f"clean op '{name}' needs a column.")
    if name == "reverse_score":
        try:
            if float(op["low"]) >= float(op["high"]):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("reverse_score needs numeric low < high (the scale's end points).") from None
    if name == "composite":
        if len(op.get("columns") or []) < 2 or not str(op.get("to", "")).strip():
            raise ValueError("composite needs at least 2 columns and a name ('to').")
        if op.get("method", "mean") not in ("mean", "sum"):
            raise ValueError("composite.method must be mean or sum.")
    if name == "drop_missing_rows":
        missing = [c for c in (op.get("columns") or []) if not str(c).strip()]
        if missing:
            raise ValueError("drop_missing_rows.columns must be non-empty column names.")
    if name == "retype" and op.get("to", "number") not in _RETYPE:
        raise ValueError(f"retype.to must be one of {sorted(_RETYPE)}")
    if name == "rename_column" and not str(op.get("to", "")).strip():
        raise ValueError("rename_column needs a non-empty 'to'.")


def apply_clean(df, ops, replay=False):
    """Apply an ordered list of validated clean ops to the whole frame.
    Returns (cleaned_df, summary[]) where each summary entry says what changed.
    replay=True: re-running saved ops on new rows (execute.transform) -- a composite
    column that is already there is recomputed instead of refused."""
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

        if name == "drop_missing_rows":
            # "missing" means the same here as in the audit: nulls AND blank strings
            cols = [c for c in (op.get("columns") or df.columns) if c in df.columns]
            before = len(df)
            missing = pd.Series(False, index=df.index)
            for c in cols:
                missing |= df[c].isnull() | empty_string_mask(df[c])
            df = df[~missing]
            where = "any column" if not op.get("columns") else ", ".join(cols)
            summary.append({"op": "drop_missing_rows",
                            "detail": f"{before - len(df)} rows with missing values removed ({where})"})
            continue

        if name == "composite":
            cols = [c for c in op["columns"] if c in df.columns]
            to = str(op["to"]).strip()
            if len(cols) != len(op["columns"]):
                raise ValueError(f"composite '{to}': missing column(s) "
                                 f"{', '.join(c for c in op['columns'] if c not in df.columns)}.")
            if to in df.columns and not replay:
                raise ValueError(f"Cannot create '{to}': that column already exists.")
            items = df[cols].apply(pd.to_numeric, errors="coerce")
            # mean of the items answered; sum only when every item is answered (a partial sum is not comparable)
            df[to] = items.mean(axis=1) if op.get("method", "mean") == "mean" else items.sum(axis=1, min_count=len(cols))
            summary.append({"op": name, "column": to,
                            "detail": f"{op.get('method', 'mean')} of {len(cols)} items, "
                                      f"{int(df[to].isna().sum())} row(s) without a score"})
            continue

        col = op["column"]
        if col not in df.columns:
            continue  # already dropped by an earlier op, or a stale plan entry
        before_dtype = str(df[col].dtype)
        before_values = int(df[col].nunique(dropna=True))
        before_missing = int(df[col].isna().sum())
        reading = None

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
                if is_numeric(df[col]):
                    num = df[col]
                else:   # the decision is written into the op, so new rows are read the same way
                    num, op["commas"] = number_text(df[col], op.get("commas"))
                    reading = f"commas read as {op['commas']}" if op["commas"] else None
                df[col] = num.astype("Int64") if _intlike(num) else num
            elif to == "datetime":
                df[col], op["dayfirst"], reading = to_datetime(df[col], op.get("dayfirst"))
            else:  # category
                df[col] = df[col].astype("category")

        elif name == "map_values":
            df[col] = df[col].replace(op.get("mapping", {}))

        elif name == "reverse_score":
            num = pd.to_numeric(df[col], errors="coerce")
            df[col] = float(op["low"]) + float(op["high"]) - num
            if _intlike(num):
                df[col] = df[col].astype("Int64")
            summary.append({"op": name, "column": col,
                            "detail": f"reversed on the {op['low']:g}-{op['high']:g} scale ({op['low']:g} <-> {op['high']:g})"})
            continue

        elif name == "nullify":
            if op.get("values") is not None:   # e.g. [0]: zeros that stand for "not measured"
                mask = df[col].isin(op["values"])
                what = "value(s) " + ", ".join(map(str, op["values"]))
            else:
                tokens = [str(t).lower() for t in (op.get("tokens") or _NULL_TOKENS)]
                mask = df[col].astype("string").str.strip().str.lower().isin(tokens)
                what = "placeholder value(s)"
            n = int(mask.sum())
            df[col] = df[col].mask(mask)
            summary.append({"op": name, "column": col, "detail": f"{n} {what} -> missing"})
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

        # say what actually changed: merged spellings show as fewer distinct values,
        # a retype shows as a new dtype, a failed retype as new missing cells
        after_values, after_dtype = int(df[col].nunique(dropna=True)), str(df[col].dtype)
        bits = []
        if after_values != before_values:
            bits.append(f"{before_values} distinct values -> {after_values}")
        if after_dtype != before_dtype:
            bits.append(f"{before_dtype} -> {after_dtype}")
        new_missing = int(df[col].isna().sum()) - before_missing
        if new_missing > 0:   # e.g. a "convert to number" that could not parse some values
            bits.append(f"{new_missing} value(s) could not be converted and are now missing")
        if reading:
            bits.append(reading)
        summary.append({"op": name, "column": col, "detail": ", ".join(bits) or "no change"})
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

    # row ops: duplicates and rows with missing values (blank strings count as missing,
    # and "missing" here must mean what the audit reported)
    rows = pd.DataFrame({"a": [1, 1, 2, None, 4], "b": ["x", "x", "", "y", "z"]})
    from preprocessing.detector import detect_missing_values
    reported = detect_missing_values(rows)["summary"]["rows_with_missing"]
    dropped, summ_rows = apply_clean(rows, [{"op": "drop_missing_rows"}])
    assert len(dropped) == len(rows) - reported == 3, (len(dropped), reported)
    assert apply_clean(rows, [{"op": "drop_missing_rows", "columns": ["a"]}])[0].shape[0] == 4  # only column a
    assert len(apply_clean(rows, [{"op": "drop_duplicates"}])[0]) == 4
    assert "2 rows with missing values removed (any column)" in summ_rows[0]["detail"], summ_rows
    assert set(ROW_OPS) <= _CLEAN_OPS and "drop_missing_rows" in ROW_OPS

    # the summary must say what changed, not just the dtype
    spellings = pd.DataFrame({"sex": ["Male", "male", " Male", "female", "Female "],
                              "amt": ["1,200", "oops", "900", "50", "1,000"]})
    _, s2 = apply_clean(spellings, [{"op": "merge_categories", "column": "sex"},
                                    {"op": "retype", "column": "amt", "to": "number"}])
    assert "5 distinct values -> 2" in s2[0]["detail"], s2[0]
    assert "could not be converted" in s2[1]["detail"] and "-> Int64" in s2[1]["detail"], s2[1]

    # dates: 25/12 proves day-first, so 03/04 is 3 April -- and ISO rows are not flipped
    dates = pd.DataFrame({"d": ["03/04/2020", "25/12/2020", "2021-02-05", None]})
    op = {"op": "retype", "column": "d", "to": "datetime"}
    out, s3 = apply_clean(dates, [op])
    assert out["d"].tolist()[:3] == [pd.Timestamp("2020-04-03"), pd.Timestamp("2020-12-25"),
                                     pd.Timestamp("2021-02-05")], out["d"].tolist()
    assert op["dayfirst"] is True and "day/month/year" in s3[0]["detail"], (op, s3)
    # the decision rides on the op: one new ambiguous row is read the way training was
    assert apply_clean(pd.DataFrame({"d": ["03/04/2020"]}), [op])[0]["d"].iloc[0] == pd.Timestamp("2020-04-03")
    us = apply_clean(pd.DataFrame({"d": ["12/31/2020", "03/04/2020"]}), [{"op": "retype", "column": "d", "to": "datetime"}])
    assert us[0]["d"].tolist() == [pd.Timestamp("2020-12-31"), pd.Timestamp("2020-03-04")], us
    # timestamps with a time zone are read as UTC (they raised TypeError); plain ones are unchanged
    tz = apply_clean(pd.DataFrame({"t": ["2020-01-01T10:00:00Z", "2020-01-01T10:00:00+05:30", "2020-01-02 08:00"]}),
                     [{"op": "retype", "column": "t", "to": "datetime"}])[0]["t"]
    assert tz.tolist() == [pd.Timestamp("2020-01-01 10:00"), pd.Timestamp("2020-01-01 04:30"),
                           pd.Timestamp("2020-01-02 08:00")], tz.tolist()

    # numbers: "3,5" is three and a half (was 35), "1,200" is twelve hundred
    nop = {"op": "retype", "column": "x", "to": "number"}
    eu, s4 = apply_clean(pd.DataFrame({"x": ["3,5", "1.234,5", "7"]}), [nop])
    assert eu["x"].tolist() == [3.5, 1234.5, 7.0] and "decimal comma" in s4[0]["detail"], (eu, s4)
    assert apply_clean(pd.DataFrame({"x": ["2,25"]}), [nop])[0]["x"].iloc[0] == 2.25

    # zeros that mean "not measured" (Pima-style): numeric values -> missing, others untouched
    z, s5 = apply_clean(pd.DataFrame({"glucose": [0, 85, 0, 120]}),
                        [{"op": "nullify", "column": "glucose", "values": [0]}])
    assert z["glucose"].isna().sum() == 2 and z["glucose"].dropna().tolist() == [85, 120], z
    assert "2 value(s) 0 -> missing" in s5[0]["detail"], s5
    # questionnaire: reverse a negatively worded item, then a composite score
    qs = pd.DataFrame({"q1": [4, 5, 2], "q2": [4, None, 1], "q3": [2, 1, 4]})
    out, s6 = apply_clean(qs, [{"op": "reverse_score", "column": "q3", "low": 1, "high": 5},
                               {"op": "composite", "columns": ["q1", "q2", "q3"], "to": "score", "method": "mean"},
                               {"op": "composite", "columns": ["q1", "q2", "q3"], "to": "total", "method": "sum"}])
    assert out["q3"].tolist() == [4, 5, 2] and out["score"].tolist() == [4.0, 5.0, 5 / 3], out
    assert out["total"].isna().tolist() == [False, True, False] and out["total"].iloc[0] == 12, out
    try:
        apply_clean(qs, [{"op": "reverse_score", "column": "q3", "low": 5, "high": 1}])
        raise AssertionError("low >= high must be refused")
    except ValueError:
        pass
    print("clean self-check passed:", [s["detail"] for s in summ])
