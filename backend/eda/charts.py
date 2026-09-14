"""Correlation matrix for the frontend's heatmap.

The only chart datum the profile/report don't already carry -- categorical bars
come from the profile's top_values, and the histograms / missing map were
dropped. pandas does .corr(); no new dependency.
"""

import pandas as pd

from preprocessing.profiler import is_numeric


def chart_data(df):
    """Return {correlation} -- a small, JSON-safe correlation matrix of numeric cols."""
    numeric = [c for c in df.columns if is_numeric(df[c])]

    correlation = None
    if len(numeric) >= 2:  # correlation needs at least two numeric columns
        corr = df[numeric].corr(numeric_only=True).round(3)
        correlation = {
            "columns": list(corr.columns),
            "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in corr.values],  # NaN -> null
        }

    return {"correlation": correlation}


if __name__ == "__main__":
    d = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1], "c": list("xxyyz")})
    out = chart_data(d)
    assert out["correlation"]["columns"] == ["a", "b"]           # numeric only, text excluded
    assert abs(out["correlation"]["matrix"][0][1] + 1.0) < 1e-9  # a,b perfectly anti-correlated
    print("charts self-check passed")