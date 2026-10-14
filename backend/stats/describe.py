"""Descriptive statistics, frequency tables and normality (plan §13.1, §13.2, §13.4).

Numeric summaries use the sample (n - 1) variance; skewness / kurtosis are the
bias-corrected sample estimates (pandas = Excel SKEW / KURT; kurtosis is EXCESS,
0 for a normal distribution). The 95% confidence interval of the mean uses the t
distribution. Shapiro-Wilk is scipy.stats.shapiro (matches R's shapiro.test --
verified in stats/tests.py); scipy documents that its p-value may be inaccurate
above 5,000 values, so larger samples are tested on a seeded random 5,000 and
the result says so.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from preprocessing.profiler import is_numeric

SHAPIRO_MAX = 5000   # scipy.stats.shapiro: "for N > 5000 the p-value may not be accurate"
CONFIDENCE = 0.95    # convention; every interval says its level
QQ_POINTS = 200      # our starting point: points drawn in a Q-Q plot


def _f(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) or pd.isna(v) else float(v)


def numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna().astype(float)


def summary(series):
    """One numeric column -> every §13.1 statistic."""
    v = numeric(series)
    n = len(v)
    out = {"n": n, "missing": int(len(series) - n)}
    if not n:
        return out
    modes = v.mode()
    sd = v.std(ddof=1) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    ci = stats.t.interval(CONFIDENCE, n - 1, loc=v.mean(), scale=se) if se else (None, None)
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    out.update(mean=_f(v.mean()), median=_f(v.median()), mode=_f(modes.iloc[0]) if len(modes) else None,
               modes=int(len(modes)), min=_f(v.min()), max=_f(v.max()), range=_f(v.max() - v.min()),
               variance=_f(v.var(ddof=1)) if n > 1 else None, sd=_f(sd), se=_f(se),
               q1=_f(q1), q3=_f(q3), iqr=_f(q3 - q1), skewness=_f(v.skew()), kurtosis=_f(v.kurt()),
               ci_low=_f(ci[0]), ci_high=_f(ci[1]), confidence=CONFIDENCE)
    return out


def describe(df, columns):
    """§13.1 table: numeric columns get the full summary, others n / missing / distinct / mode."""
    rows = []
    for c in columns:
        s = df[c]
        if is_numeric(s):
            rows.append({"column": c, "kind": "numeric", **summary(s)})
        else:
            vc = s.dropna().astype(str).value_counts()
            rows.append({"column": c, "kind": "categorical", "n": int(s.notna().sum()),
                         "missing": int(s.isna().sum()), "distinct": int(len(vc)),
                         "mode": vc.index[0] if len(vc) else None,
                         "mode_percent": round(100 * vc.iloc[0] / vc.sum(), 2) if len(vc) else None})
    return {"analysis": "describe", "title": "Descriptive statistics", "rows": rows}


def frequencies(df, columns, max_values=50):
    """§13.1 / §13.4 demographic tables: count, % of all rows, valid % (missing
    excluded), cumulative valid %. The most frequent max_values values are listed."""
    tables = []
    for c in columns:
        s = df[c]
        vc = s.dropna().astype(str).value_counts()
        total, valid = len(s), int(vc.sum())
        cum, rows = 0, []
        for value, count in vc.head(max_values).items():
            cum += count
            rows.append({"value": value, "count": int(count), "percent": round(100 * count / total, 2),
                         "valid_percent": round(100 * count / valid, 2), "cumulative": round(100 * cum / valid, 2)})
        tables.append({"column": c, "rows": rows, "missing": int(total - valid), "total": int(total),
                       "distinct": int(len(vc)), "shown": int(min(len(vc), max_values))})
    return {"analysis": "frequencies", "title": "Frequency tables", "tables": tables}


def qq(values):
    """Normal Q-Q plot points: sample quantiles against theoretical ones + the fitted line."""
    (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm")
    idx = np.unique(np.linspace(0, len(osm) - 1, min(len(osm), QQ_POINTS)).round().astype(int))
    return {"theoretical": np.round(osm[idx], 5).tolist(), "sample": np.round(osr[idx], 5).tolist(),
            "slope": float(slope), "intercept": float(intercept), "r": float(r)}


def shapiro(values, seed=0):
    """Shapiro-Wilk on a 1-D array -> {W, p, n, note}; None below 3 values."""
    v = np.asarray(values, float)
    if len(v) < 3:
        return None
    note = None
    if len(v) > SHAPIRO_MAX:
        v = np.random.default_rng(seed).choice(v, SHAPIRO_MAX, replace=False)
        note = f"tested on a random {SHAPIRO_MAX:,} of the values (scipy's p-value is unreliable above that)"
    if np.ptp(v) == 0:
        return {"W": None, "p": None, "n": int(len(v)), "note": "all values are equal - normality is undefined"}
    r = stats.shapiro(v)
    return {"W": float(r.statistic), "p": float(r.pvalue), "n": int(len(v)), "note": note}


def normality(df, columns, by=None, alpha=0.05):
    """§13.2: Shapiro-Wilk + skewness / kurtosis + Q-Q plot, per column (and per group of `by`)."""
    rows = []
    for c in columns:
        groups = [(None, df[c])] if not by else [(g, sub[c]) for g, sub in df.groupby(by, dropna=True)]
        for g, s in groups:
            v = numeric(s)
            sw = shapiro(v.to_numpy())
            rows.append({"column": c, "group": None if g is None else str(g), "n": int(len(v)),
                         "skewness": _f(v.skew()), "kurtosis": _f(v.kurt()), "shapiro": sw,
                         "normal": None if not sw or sw["p"] is None else bool(sw["p"] > alpha),
                         "qq": qq(v.to_numpy()) if len(v) >= 3 else None})
    return {"analysis": "normality", "title": "Normality (Shapiro-Wilk)", "alpha": alpha, "rows": rows,
            "note": "p > alpha: no evidence against normality (it does not prove normality); with large samples "
                    "even tiny departures give small p - read the Q-Q plot too."}
