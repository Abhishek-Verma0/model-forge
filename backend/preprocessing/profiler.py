import pandas as pd


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


def load_dataset(file_path):
    """Load a CSV or Excel dataset."""
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV and Excel files are supported.")

    return df


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
                "min": float(series.min()) if not series.dropna().empty else None,
                "max": float(series.max()) if not series.dropna().empty else None,
                "mean": float(series.mean()) if not series.dropna().empty else None,
                "median": float(series.median()) if not series.dropna().empty else None,
                "std": float(series.std()) if not series.dropna().empty else None,
                "q1": float(series.quantile(0.25)) if not series.dropna().empty else None,
                "q3": float(series.quantile(0.75)) if not series.dropna().empty else None
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
    print("profiler self-check passed")