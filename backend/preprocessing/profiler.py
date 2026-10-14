import codecs
import csv
import re

import numpy as np
import pandas as pd

SNIFF_BYTES = 64 * 1024   # our starting point: enough lines for csv.Sniffer to see the pattern
_DELIMITERS = ",;\t|"
# "22,5" / "-3,75" / "1.234,5": a comma decimal (with optional dot thousands). A comma
# followed by exactly 3 digits ("1,200") is a thousands separator -- not matched here.
_DECIMAL_COMMA = re.compile(r"^[+-]?(\d{1,3}(\.\d{3})+|\d+),\d+$")
_THOUSANDS_COMMA = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def is_numeric(series):
    """True for real numbers only. pandas counts bool as numeric, but you
    can't take a mean or quantile of True/False -- it crashes IQR and skews
    stats -- so bool is excluded here. One source of truth for 'is this a
    number I can do math on', used by both profiler and detector."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def empty_string_mask(series):
    """Whitespace-only / empty strings count as missing in text columns.
    One definition of 'blank', shared by profiler and detector so their missing
    counts can't drift."""
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return (series.astype("string").str.strip() == "").fillna(False)
    return pd.Series(False, index=series.index)


def missing_mask(series):
    """One definition of 'missing': null OR blank string. Used by both the
    profiler's per-column stats and the detector's missing-value check."""
    return series.isnull() | empty_string_mask(series)


DECIMAL_COMMA = "decimal comma"


def number_text(series, reading=None):
    """Numbers written as text -> (floats, how the commas were read). Decides per
    column, from the values: "1,200" (every comma before exactly 3 digits) is a
    thousands separator; "3,5" / "1.234,5" is a decimal comma. Mixed columns keep the
    old reading (commas removed) and say so. Pass `reading` to replay a decision
    made earlier (new rows must be read the way the training rows were)."""
    s = series.astype("string").str.strip()
    if reading is None:
        has = s[s.str.contains(",", regex=False, na=False)]
        if has.empty:
            reading = None
        elif (has.str.fullmatch(_DECIMAL_COMMA.pattern).all()
              and not has.str.fullmatch(_THOUSANDS_COMMA.pattern).all()):
            reading = DECIMAL_COMMA
        elif has.str.fullmatch(_THOUSANDS_COMMA.pattern).all():
            reading = "thousands separator"
        else:
            reading = "commas removed (mixed formats - check this column)"
    if reading == DECIMAL_COMMA:
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce"), reading


def _decode(path):
    """(encoding, sample text). UTF-8 first; Excel's "CSV" export uses the Windows
    code page (cp1252 in Western locales); latin-1 decodes any byte, the last resort."""
    with open(path, "rb") as f:
        head = f.read(SNIFF_BYTES)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            # incremental, final=False: a sample cut inside a multi-byte character is fine
            return enc, codecs.getincrementaldecoder(enc)().decode(head, final=False)
        except UnicodeDecodeError:
            continue
    return "latin-1", head.decode("latin-1")


def _read_text(path, sep, notes):
    enc, sample = _decode(path)
    if enc != "utf-8-sig":
        notes.append(f"file is not UTF-8 - read as {enc}")
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
        except csv.Error:   # one column, or no consistent pattern
            sep = ","
        if sep != ",":
            notes.append(f"columns separated by {'tab' if sep == chr(9) else repr(sep)}")
    try:
        df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
    except UnicodeDecodeError:   # the bad byte was past the sample
        df = pd.read_csv(path, sep=sep, encoding="latin-1", low_memory=False)
        notes.append("file is not UTF-8 - read as latin-1")
    if sep != ",":
        # ; and tab files come from locales where the comma is the decimal mark
        for c in df.columns:
            s = df[c]
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                vals = s.dropna().astype(str).str.strip()
                if len(vals) and vals.str.fullmatch(_DECIMAL_COMMA.pattern).all():
                    df[c] = number_text(s)[0].astype("float64")   # what read_csv gives a number column
                    notes.append(f"{c}: decimal commas read as numbers")
    return df


def load_dataset(file_path, notes=None):
    """Load a CSV / TSV / JSON / Excel dataset. `notes` (a list) receives what the
    reader had to decide: encoding, delimiter, decimal commas, infinite values."""
    notes = [] if notes is None else notes
    ext = str(file_path).lower().rsplit(".", 1)[-1]
    if ext in ("csv", "tsv"):
        df = _read_text(file_path, "\t" if ext == "tsv" else None, notes)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(file_path)
    elif ext == "json":
        df = pd.read_json(file_path)   # an array of row objects, or {column: [values]}
    else:
        raise ValueError("Only CSV, TSV, JSON and Excel files are supported.")
    num = df.select_dtypes("number")
    inf = np.isinf(num.to_numpy(dtype="float64", na_value=np.nan)) if num.shape[1] else np.zeros((0, 0), bool)
    if inf.any():
        cols = [c for c, hit in zip(num.columns, inf.any(axis=0)) if hit]
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
        notes.append(f"{int(inf.sum())} infinite value(s) treated as missing ({', '.join(map(str, cols[:5]))})")
    return df


def _stat(fn, *args):
    """A statistic as a plain float, or None when it can't be computed: an empty
    column, NaN (std of one value), or pd.NA (the same on nullable Int64 / Float64)."""
    try:
        v = fn(*args)
    except (TypeError, ValueError):
        return None
    return None if v is None or pd.isna(v) else float(v)


def profile_dataset(df):
    """Create a general-purpose profile of an already-loaded DataFrame.

    Takes a DataFrame (not a path) so the caller loads the file once and the
    profiler + detector run on the exact same frame."""

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),

        "duplicate_rows": int(df.duplicated().sum()),

        "columns_info": {}
    }

    for column in df.columns:

        series = df[column]

        # Basic information. missing_mask = null OR blank string, the same rule
        # the detector uses -- so the two never report different missing counts.
        missing = missing_mask(series)
        column_info = {
            "data_type": str(series.dtype),
            "missing_values": int(missing.sum()),
            "missing_percent": round(float(missing.mean() * 100), 2),
            "unique_values": int(series.nunique(dropna=True)),
            "is_constant": bool(series.nunique(dropna=True) <= 1)
        }

        # Numeric columns
        if is_numeric(series):

            column_info["type"] = "numeric"

            column_info["statistics"] = {
                "min": _stat(series.min), "max": _stat(series.max), "mean": _stat(series.mean),
                "median": _stat(series.median), "std": _stat(series.std),
                "q1": _stat(series.quantile, 0.25), "q3": _stat(series.quantile, 0.75),
                # bias-corrected sample skewness / excess kurtosis (Excel SKEW / KURT);
                # not computable (null) below 3 / 4 values
                "skew": _stat(series.skew), "kurtosis": _stat(series.kurt),
            }

        # Non-numeric columns
        else:

            column_info["type"] = "categorical_or_text"

            value_counts = series.value_counts(dropna=True)

            column_info["top_values"] = [
                {
                    "value": str(value),
                    "count": int(count),
                    "percent": round(
                        float(count / len(series) * 100), 2
                    )
                }
                for value, count in value_counts.head(10).items()
            ]

        profile["columns_info"][column] = column_info

    return profile


if __name__ == "__main__":
    # self-check: bool must NOT count as numeric (that was the IQR crash)
    assert is_numeric(pd.Series([1, 2, 3]))
    assert is_numeric(pd.Series([1.0, 2.5]))
    assert not is_numeric(pd.Series([True, False]))
    assert not is_numeric(pd.Series(["a", "b"]))
    # missing = null OR blank string ("", "  ", None all count; "a" does not)
    assert int(missing_mask(pd.Series(["a", "", "  ", None])).sum()) == 3

    import os
    import tempfile
    d = tempfile.mkdtemp()

    def load(name, text, enc="utf-8"):
        p = os.path.join(d, name)
        with open(p, "w", encoding=enc, newline="") as f:
            f.write(text)
        notes = []
        return load_dataset(p, notes), notes

    # semicolon CSV with decimal commas (European Excel): columns split, "22,5" -> 22.5,
    # a dot-decimal column in the same file stays as it was
    df, notes = load("semi.csv", "age;bmi;label;ratio\n30;22,5;yes;0.5\n40;31,0;no;1.25\n50;27,3;yes;2\n")
    assert list(df.columns) == ["age", "bmi", "label", "ratio"], df.columns
    assert df["bmi"].tolist() == [22.5, 31.0, 27.3] and df["ratio"].tolist() == [0.5, 1.25, 2.0], df
    assert any("decimal commas" in n for n in notes) and any("';'" in n for n in notes), notes
    # an ordinary comma CSV with quoted text containing commas and semicolons is untouched
    df, notes = load("c.csv", 'name,score\n"Smith, J.; Doe",3\n"Lee, K.",4\n')
    assert df.shape == (2, 2) and notes == [], (df, notes)
    df, _ = load("t.tsv", "a\tb\n1\t2\n")
    assert df.shape == (1, 2)
    df, _ = load("j.json", '[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]')
    assert df["a"].tolist() == [1, 2]
    df, notes = load("w.csv", "name,city\nJosé,Málaga\n", enc="cp1252")
    assert df["city"].iloc[0] == "Málaga" and "cp1252" in notes[0], (df, notes)
    df, notes = load("inf.csv", "a,b\n1,x\ninf,y\n-inf,z\n")
    assert df["a"].isna().sum() == 2 and "2 infinite" in notes[0], (df, notes)
    df, _ = load("one.csv", "value\n1\n2\n")   # one column: nothing to sniff
    assert list(df.columns) == ["value"]
    # numbers written as text: the column's own values decide what a comma means
    assert number_text(pd.Series(["1,200", "3,400", "900"]))[0].tolist() == [1200, 3400, 900]
    assert number_text(pd.Series(["3,5", "1.234,5", "7"]))[0].tolist() == [3.5, 1234.5, 7]
    assert number_text(pd.Series(["3,5", "1,200"]))[1] == "decimal comma"   # 3,5 proves it
    # statistics that can't be computed are None, on every numeric dtype (was a TypeError on pd.NA)
    few = pd.DataFrame({"i": pd.array([5, None, None], dtype="Int64"), "f": pd.array([1.5, 2.5], dtype="Float64")
                        .tolist() + [None]})
    st = profile_dataset(few)["columns_info"]
    assert st["i"]["statistics"]["std"] is None and st["i"]["statistics"]["kurtosis"] is None, st["i"]
    assert profile_dataset(df)["columns_info"]["value"]["statistics"]["skew"] is None   # 2 values: no skew
    print("profiler self-check passed")