import warnings

import pandas as pd

from profiler import is_numeric, empty_string_mask

# Detection thresholds. Deliberately NOT user-tunable: a researcher overrides the
# VERDICT (untick the suggested drop on the Cleaning tab), not the dial -- "is
# patient_code an ID?" is a judgement call, 0.95 is not a number they can reason about.
IQR_MULTIPLIER = 1.5
CATEGORICAL_THRESHOLD = 0.5
MAX_CATEGORICAL_UNIQUE = 50
ID_RATIO_THRESHOLD = 0.95
CARDINALITY_THRESHOLD = 0.5
CARDINALITY_MIN_UNIQUE = 20
# Free text = most values distinct AND a typical cell holds several words
# (AutoGluon's rule). Other many-valued strings -- cities, CPU models, codes -- are
# "string": short fragments that repeat across rows.
FREETEXT_MIN_UNIQUE = 0.5
FREETEXT_MIN_WORDS = 3  # ponytail: median words per cell; 2-word names stay ID-like


def detect_missing_values(df, detect_empty_strings=True):
    """
    Detect missing values in every column of a dataset.
    General-purpose. Also catches empty/whitespace strings as missing.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """

    total_rows = len(df)
    columns_with_missing = []

    for column in df.columns:
        series = df[column]
        null_mask = series.isnull()
        # blank-string rule lives in profiler.empty_string_mask -- one source of
        # truth so this count can't drift from the profiler's missing_percent.
        esm = empty_string_mask(series) if detect_empty_strings else pd.Series(False, index=series.index)
        missing = null_mask | esm
        missing_count = int(missing.sum())

        if missing_count > 0:
            missing_percent = round(
                float(missing_count / total_rows * 100), 2
            ) if total_rows > 0 else 0.0

            columns_with_missing.append({
                "column": column,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "null_count": int(null_mask.sum()),
                "empty_string_count": int(esm.sum()),
            })

    total_missing_cells = sum(c["missing_count"] for c in columns_with_missing)

    return {
        "check": "missing_values",
        "has_missing": len(columns_with_missing) > 0,
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns_with_missing": len(columns_with_missing),
            "total_missing_cells": total_missing_cells,
        },
        "columns": columns_with_missing,
    }


def detect_duplicates(df, subset=None):
    """
    Detect duplicate rows. By default checks all columns; pass `subset`
    to check specific columns only.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """

    total_rows = len(df)

    duplicate_mask = df.duplicated(subset=subset, keep="first")
    duplicate_count = int(duplicate_mask.sum())

    all_duplicated_mask = df.duplicated(subset=subset, keep=False)
    rows_involved = int(all_duplicated_mask.sum())

    duplicate_percent = round(
        float(duplicate_count / total_rows * 100), 2
    ) if total_rows > 0 else 0.0

    duplicate_row_indices = df.index[duplicate_mask].tolist()

    return {
        "check": "duplicate_rows",
        "has_duplicates": duplicate_count > 0,
        "subset": subset if subset is not None else "all_columns",
        "summary": {
            "total_rows": total_rows,
            "duplicate_rows": duplicate_count,
            "duplicate_percent": duplicate_percent,
            "rows_involved_in_duplication": rows_involved,
            "unique_rows": total_rows - duplicate_count,
        },
        "duplicate_row_indices": duplicate_row_indices,
    }


def detect_outliers(df, iqr_multiplier=None):
    """
    Detect outliers in numeric columns using the IQR method.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """
    iqr_multiplier = IQR_MULTIPLIER if iqr_multiplier is None else float(iqr_multiplier)

    total_rows = len(df)
    columns_with_outliers = []

    for column in df.columns:
        series = df[column]

        if not is_numeric(series):
            continue

        clean = series.dropna()
        if clean.empty:
            continue

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outlier_mask = (clean < lower_bound) | (clean > upper_bound)
        outlier_count = int(outlier_mask.sum())

        if outlier_count > 0:
            outlier_percent = round(
                float(outlier_count / total_rows * 100), 2
            ) if total_rows > 0 else 0.0

            columns_with_outliers.append({
                "column": column,
                "outlier_count": outlier_count,
                "outlier_percent": outlier_percent,
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "min_value": float(clean.min()),
                "max_value": float(clean.max()),
            })

    return {
        "check": "outliers",
        "method": "IQR",
        "iqr_multiplier": iqr_multiplier,
        "has_outliers": len(columns_with_outliers) > 0,
        "summary": {
            "total_rows": total_rows,
            "numeric_columns_checked": sum(
                1 for c in df.columns if is_numeric(df[c])
            ),
            "columns_with_outliers": len(columns_with_outliers),
        },
        "columns": columns_with_outliers,
    }


def detect_constant_columns(df):
    """
    Detect columns where every row has the same value (zero variance).
    All-NaN columns are reported separately as `all_missing`.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """

    total_rows = len(df)
    constant_columns = []

    for column in df.columns:
        series = df[column]
        distinct = series.nunique(dropna=True)

        if distinct == 0:
            constant_columns.append({
                "column": column,
                "constant_value": None,
                "all_missing": True,
            })
        elif distinct == 1:
            the_value = series.dropna().iloc[0]
            constant_columns.append({
                "column": column,
                "constant_value": str(the_value),
                "all_missing": False,
            })

    return {
        "check": "constant_columns",
        "has_constant_columns": len(constant_columns) > 0,
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "constant_columns": len(constant_columns),
        },
        "columns": constant_columns,
    }


# Common textual true/false pairs used to spot boolean-like columns.
_BOOLEAN_VALUE_SETS = [
    {"yes", "no"},
    {"true", "false"},
    {"t", "f"},
    {"y", "n"},
    {"0", "1"},
]

# How many values to sample when guessing whether a column is dates. Type is
# uniform down a column, so a random sample decides it -- and a full-column
# pd.to_datetime(format="mixed") parses one value at a time in Python (~10s on
# a large text column, which is what hung the whole report).
DATETIME_SAMPLE = 1000


def text_kind(series):
    """'free_text' (reviews, notes) or 'string' (many-valued short labels).
    Reads the values only -- never the column name, never the target."""
    s = series.dropna().astype(str)
    if s.empty:
        return "string"
    probe = s if len(s) <= DATETIME_SAMPLE else s.sample(DATETIME_SAMPLE, random_state=0)
    words = probe.str.split().str.len().median()
    unique = s.nunique() / len(s)
    return "free_text" if unique >= FREETEXT_MIN_UNIQUE and words >= FREETEXT_MIN_WORDS else "string"


def detect_column_types(df, categorical_threshold=None, max_categorical_unique=None):
    """
    Classify each column into a semantic type beyond the raw pandas dtype:
    boolean, datetime, numeric, categorical, or text.

    General-purpose: uses cardinality ratios, not hard-coded column names.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """
    categorical_threshold = (CATEGORICAL_THRESHOLD if categorical_threshold is None
                             else float(categorical_threshold))
    max_categorical_unique = (MAX_CATEGORICAL_UNIQUE if max_categorical_unique is None
                              else int(max_categorical_unique))

    total_rows = len(df)
    columns = []
    type_counts = {}

    for column in df.columns:
        series = df[column]
        clean = series.dropna()

        distinct = int(clean.nunique())
        unique_ratio = round(distinct / total_rows, 4) if total_rows > 0 else 0.0

        is_boolean = False
        if distinct == 2:
            values = {str(v).strip().lower() for v in clean.unique()}
            if any(values == pair for pair in _BOOLEAN_VALUE_SETS):
                is_boolean = True

        if is_boolean:
            semantic_type = "boolean"

        elif is_numeric(series):
            semantic_type = "numeric"

        else:
            is_datetime = False
            if not clean.empty:
                # Sample, don't scan: parse a random 1,000 values, not all of them.
                # random_state fixes the sample so the type is stable across runs.
                probe = clean if len(clean) <= DATETIME_SAMPLE else clean.sample(
                    DATETIME_SAMPLE, random_state=0
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(probe, errors="coerce", format="mixed")
                if parsed.notna().mean() >= 0.8:
                    is_datetime = True

            if is_datetime:
                semantic_type = "datetime"
            elif distinct <= max_categorical_unique and unique_ratio <= categorical_threshold:
                semantic_type = "categorical"
            else:
                semantic_type = "text"

        type_counts[semantic_type] = type_counts.get(semantic_type, 0) + 1

        columns.append({
            "column": column,
            "pandas_dtype": str(series.dtype),
            "detected_type": semantic_type,
            "distinct_values": distinct,
            "unique_ratio": unique_ratio,
            "text_kind": text_kind(series) if semantic_type == "text" else None,
        })

    return {
        "check": "column_types",
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "type_counts": type_counts,
        },
        "columns": columns,
    }


def detect_id_columns(df, id_ratio_threshold=None):
    """
    Detect candidate identifier (ID) columns.

    An ID column uniquely (or near-uniquely) labels each row: an index,
    record number, order ID, etc. It carries no predictive value and can
    leak information, so it should be excluded from modeling.

    General-purpose heuristic: a column is a candidate ID when its ratio
    of distinct non-null values to rows is at or above `id_ratio_threshold`
    (i.e. almost every row has its own value). Fully-null columns are not
    IDs. Detection is based on cardinality, not column names, so it works
    on any dataset regardless of what the ID column is called.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """
    id_ratio_threshold = (ID_RATIO_THRESHOLD if id_ratio_threshold is None
                          else float(id_ratio_threshold))

    total_rows = len(df)
    id_columns = []

    for column in df.columns:
        series = df[column]
        clean = series.dropna()

        if clean.empty or total_rows == 0:
            continue

        distinct = int(clean.nunique())
        unique_ratio = distinct / total_rows

        # every review is unique too, but sentences aren't identifiers
        if (unique_ratio >= id_ratio_threshold
                and (is_numeric(series) or text_kind(series) != "free_text")):
            id_columns.append({
                "column": column,
                "distinct_values": distinct,
                "unique_ratio": round(unique_ratio, 4),
                "is_fully_unique": distinct == total_rows,
            })

    return {
        "check": "id_columns",
        "id_ratio_threshold": id_ratio_threshold,
        "has_id_columns": len(id_columns) > 0,
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "id_columns": len(id_columns),
        },
        "columns": id_columns,
    }


def detect_high_cardinality(df, cardinality_threshold=None, min_unique=None):
    """
    Detect high-cardinality categorical/text columns.

    A high-cardinality column is a non-numeric column with a very large
    number of distinct values relative to the row count (e.g. free-text
    notes, names, addresses). Such columns are problematic for one-hot
    encoding and usually need special handling: dropping, hashing, or
    target/frequency encoding.

    This differs from ID detection: an ID is near-unique per row (ratio
    ~1.0), while high cardinality is "many but not necessarily unique".

    General-purpose: only non-numeric columns are considered, and the
    decision is driven by a cardinality ratio, not column names.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """
    cardinality_threshold = (CARDINALITY_THRESHOLD if cardinality_threshold is None
                             else float(cardinality_threshold))
    min_unique = CARDINALITY_MIN_UNIQUE if min_unique is None else int(min_unique)

    total_rows = len(df)
    high_cardinality_columns = []

    for column in df.columns:
        series = df[column]

        # Only non-numeric columns are candidates for this problem.
        if is_numeric(series):
            continue

        clean = series.dropna()
        if clean.empty or total_rows == 0:
            continue

        distinct = int(clean.nunique())
        unique_ratio = distinct / total_rows

        if distinct >= min_unique and unique_ratio >= cardinality_threshold:
            high_cardinality_columns.append({
                "column": column,
                "distinct_values": distinct,
                "unique_ratio": round(unique_ratio, 4),
            })

    return {
        "check": "high_cardinality",
        "cardinality_threshold": cardinality_threshold,
        "min_unique": min_unique,
        "has_high_cardinality": len(high_cardinality_columns) > 0,
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "high_cardinality_columns": len(high_cardinality_columns),
        },
        "columns": high_cardinality_columns,
    }


def detect_value_issues(df):
    """
    Validate cell values and flag suspicious ones per column.

    Catches common data-entry / quality problems that other checks miss:
      - numeric-looking text stored as strings ("1,200", "42 ")
      - negative values in a numeric column
      - zero values in a numeric column (sometimes a stand-in for missing)
      - whitespace issues: text values with leading/trailing spaces.

    General-purpose: driven by value patterns, not column names. Nothing
    is modified; issues are reported for the researcher to confirm.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """

    total_rows = len(df)
    columns_with_issues = []

    for column in df.columns:
        series = df[column]
        clean = series.dropna()
        if clean.empty:
            continue

        # A True/False column has no negatives, zeros-as-missing, numbers-as-
        # text, or whitespace to find. Skipping it also avoids the wasted
        # per-string scan on every bool column of a wide dataset.
        if pd.api.types.is_bool_dtype(series):
            continue

        issues = {}

        if is_numeric(series):
            negative_count = int((clean < 0).sum())
            zero_count = int((clean == 0).sum())
            if negative_count > 0:
                issues["negative_values"] = negative_count
            if zero_count > 0:
                issues["zero_values"] = zero_count

        else:
            str_values = clean.astype(str)

            # Numeric-looking text: strip commas/spaces, try to convert.
            stripped = str_values.str.strip().str.replace(",", "", regex=False)
            as_numeric = pd.to_numeric(stripped, errors="coerce")
            numeric_like_ratio = float(as_numeric.notna().mean())
            if numeric_like_ratio >= 0.8:
                issues["numeric_stored_as_text"] = round(numeric_like_ratio, 4)

            # Leading/trailing whitespace in the raw strings.
            whitespace_count = int((str_values != str_values.str.strip()).sum())
            if whitespace_count > 0:
                issues["whitespace_values"] = whitespace_count

        if issues:
            columns_with_issues.append({
                "column": column,
                "issues": issues,
            })

    return {
        "check": "value_issues",
        "has_value_issues": len(columns_with_issues) > 0,
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns_with_issues": len(columns_with_issues),
        },
        "columns": columns_with_issues,
    }


def detect_consistency_issues(df):
    """
    Detect inconsistent category representations within text columns.

    The same real-world category is often written several ways: "Male",
    "male", "MALE", " Male". These are distinct strings to pandas, so they
    inflate cardinality and break encoding. This check groups a column's
    values by their normalized form (trimmed + lowercased) and reports any
    group that has more than one raw spelling.

    General-purpose: only text columns are checked, and grouping is purely
    by value normalization, so it works on any dataset.

    Returns:
        dict: Structured report for the AI reasoning layer.
    """

    total_rows = len(df)
    columns_with_inconsistencies = []

    for column in df.columns:
        series = df[column]

        # Numbers have no spellings; bool "True"/"False" has no variants either.
        if is_numeric(series) or pd.api.types.is_bool_dtype(series):
            continue

        clean = series.dropna()
        if clean.empty:
            continue

        raw_values = clean.astype(str)
        normalized = raw_values.str.strip().str.lower()

        # Map each normalized form to the set of raw spellings seen.
        groups = {}
        for raw, norm in zip(raw_values, normalized):
            groups.setdefault(norm, set()).add(raw)

        inconsistent = {
            norm: sorted(spellings)
            for norm, spellings in groups.items()
            if len(spellings) > 1
        }

        if inconsistent:
            columns_with_inconsistencies.append({
                "column": column,
                "inconsistent_groups": len(inconsistent),
                "examples": dict(list(inconsistent.items())[:5]),
            })

    return {
        "check": "consistency_issues",
        "has_consistency_issues": len(columns_with_inconsistencies) > 0,
        "summary": {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "columns_with_inconsistencies": len(columns_with_inconsistencies),
        },
        "columns": columns_with_inconsistencies,
    }


def quality_score(checks, df):
    """0-100 dataset health, from checks already computed -- no second pass.

    Only counts things that are genuinely WRONG with the data. Outliers, ID and
    high-cardinality columns are deliberately excluded: they are modelling notes,
    not defects, and penalising them makes a clean dataset look dirty.

    Every deduction is returned with the score, because an unexplained number is
    worse than no number -- the user must see why it is 72. ponytail: the weights
    are a judgement call, not a standard; tune them if they mis-rank real data.
    """
    rows, cols = max(len(df), 1), max(len(df.columns), 1)
    pct = lambda n, d: 100.0 * n / max(d, 1)

    # only the two value issues that are real defects; negatives/zeros are data
    broken = sum(1 for c in checks["value_issues"]["columns"]
                 if {"numeric_stored_as_text", "whitespace_values"} & set(c["issues"]))

    items = [  # (label, points off) -- each capped so one check can't sink the score
        ("missing values", min(30.0, pct(checks["missing_values"]["summary"]["total_missing_cells"],
                                         rows * cols) * 1.5)),
        ("duplicate rows", min(20.0, checks["duplicate_rows"]["summary"]["duplicate_percent"] * 1.5)),
        ("constant columns", min(15.0, pct(len(checks["constant_columns"]["columns"]), cols))),
        ("inconsistent categories", min(20.0, pct(len(checks["consistency_issues"]["columns"]), cols))),
        ("malformed values", min(15.0, pct(broken, cols))),
    ]
    deductions = [{"reason": label, "points": round(pts, 1)} for label, pts in items if pts > 0]
    score = round(max(0.0, 100.0 - sum(p for _, p in items)), 1)
    grade = ("good" if score >= 85 else "fair" if score >= 70 else
             "poor" if score >= 50 else "very poor")
    return {"score": score, "grade": grade, "deductions": deductions}


def run_quality_report(df):
    """
    Run every detector and return one unified data-quality report.

    This is the single hand-off point to the AI reasoning / recommendation
    layer: it calls all detectors on the given DataFrame and collects their
    structured results into one dict, plus a top-level summary of which
    problems were found.

    General-purpose: works on any DataFrame from any CSV/XLSX.

    Returns:
        dict: One report containing every check plus an overall summary.
    """

    checks = {
        "missing_values": detect_missing_values(df),
        "duplicate_rows": detect_duplicates(df),
        "outliers": detect_outliers(df),
        "constant_columns": detect_constant_columns(df),
        "column_types": detect_column_types(df),
        "id_columns": detect_id_columns(df),
        "high_cardinality": detect_high_cardinality(df),
        "value_issues": detect_value_issues(df),
        "consistency_issues": detect_consistency_issues(df),
    }

    issues_found = {
        "missing_values": checks["missing_values"]["has_missing"],
        "duplicate_rows": checks["duplicate_rows"]["has_duplicates"],
        "outliers": checks["outliers"]["has_outliers"],
        "constant_columns": checks["constant_columns"]["has_constant_columns"],
        "id_columns": checks["id_columns"]["has_id_columns"],
        "high_cardinality": checks["high_cardinality"]["has_high_cardinality"],
        "value_issues": checks["value_issues"]["has_value_issues"],
        "consistency_issues": checks["consistency_issues"]["has_consistency_issues"],
    }

    return {
        "report": "data_quality",
        "dataset_summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
        },
        "quality_score": quality_score(checks, df),
        "issues_found": issues_found,
        "total_issue_types_found": sum(1 for v in issues_found.values() if v),
        "checks": checks,
    }

if __name__ == "__main__":
    import pandas as pd
    clean = pd.DataFrame({"a": range(40), "b": [i * 1.5 for i in range(40)],
                          "c": ["x", "y"] * 20})
    dirty = pd.concat([clean.assign(c=None, d=1)] * 2, ignore_index=True)
    cs = quality_score(run_quality_report(clean)["checks"], clean)
    ds = quality_score(run_quality_report(dirty)["checks"], dirty)
    assert cs["score"] == 100 and not cs["deductions"], cs   # clean data loses nothing
    assert cs["score"] > ds["score"], (cs, ds)          # dirtier data must score lower
    assert 0 <= ds["score"] <= 100 and cs["score"] <= 100
    assert ds["deductions"], "a bad dataset must say WHY it lost points"
    # the score is only ever the deductions subtracted from 100
    assert abs(100 - sum(d["points"] for d in ds["deductions"]) - ds["score"]) < 0.2
    # free text is kept (not an ID); many-valued short labels are "string"
    t = pd.DataFrame({"review": [f"great product number {i} works well" for i in range(60)],
                      "cpu": [f"Intel Core i{3 + i % 3} {i % 5}U" for i in range(60)],  # 15 labels
                      "code": [f"ORD{i}" for i in range(60)]})
    kinds = {c["column"]: c["text_kind"] for c in detect_column_types(t)["columns"]}
    assert kinds == {"review": "free_text", "cpu": None, "code": "string"}, kinds  # cpu: few labels -> categorical
    ids = [c["column"] for c in detect_id_columns(t)["columns"]]
    assert "review" not in ids and "code" in ids, ids
    print(f"quality score self-check passed: clean={cs['score']} ({cs['grade']}) "
          f"dirty={ds['score']} ({ds['grade']}) because {[d['reason'] for d in ds['deductions']]}")
