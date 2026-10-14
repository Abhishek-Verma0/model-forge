"""Inferential tests (plan §13.3), assumption checks (§13.2) and the test
recommendation (§13.4 note): every result carries its assumptions, effect size,
confidence interval and an automated decision that the researcher must confirm.

Recommendation rules (test="auto"), each stated in the result:
  numeric outcome, 2 independent groups: every group normal (Shapiro-Wilk p > alpha)
      -> Welch's t-test (no equal-variance assumption; Delacre, Lakens & Leys 2017
      recommend it by default), otherwise Mann-Whitney U
  numeric outcome, 3+ groups: all normal -> one-way ANOVA when Levene p > alpha,
      Welch's ANOVA otherwise; not normal -> Kruskal-Wallis
  paired measurements: differences normal -> paired t-test, otherwise Wilcoxon signed-rank
  two categorical variables: chi-square test of independence (Yates' correction for
      2x2, like R); Fisher's exact test reported too for 2x2
  two numeric variables: both normal -> Pearson, otherwise Spearman

Rank tests follow R's defaults: an exact p-value when there are fewer than 50
values and no ties (and, for Wilcoxon, no zero differences), otherwise the normal
approximation with continuity correction -- verified below against R.

Effect-size labels use Cohen (1988): d 0.2 / 0.5 / 0.8; r 0.1 / 0.3 / 0.5;
eta² 0.01 / 0.06 / 0.14; Cramér's V via w = V * sqrt(min(rows, cols) - 1) with
w 0.1 / 0.3 / 0.5. They are conventions, not laws.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from preprocessing.profiler import is_numeric
from stats.describe import CONFIDENCE, numeric, shapiro

ALPHA = 0.05            # convention (Fisher); every request may set its own
EXACT_MAX = 50          # R wilcox.test: exact p-value below 50 values without ties
EXPECTED_MIN = 5        # Cochran (1954): expected counts below 5 ...
EXPECTED_SHARE = 0.2    # ... in more than 20% of cells make the chi-square approximation unreliable
REVIEW = "Automated interpretation - confirm the assumptions and read the result in context before reporting it."
THRESHOLDS = {"d": (0.2, 0.5, 0.8), "r": (0.1, 0.3, 0.5), "eta2": (0.01, 0.06, 0.14), "w": (0.1, 0.3, 0.5)}


def magnitude(kind, value):
    if value is None or not math.isfinite(value):
        return None
    lo, mid, hi = THRESHOLDS[kind]
    a = abs(value)
    return "negligible" if a < lo else "small" if a < mid else "medium" if a < hi else "large"


def _effect(name, value, kind, scale=1.0):
    v = None if value is None or not math.isfinite(value) else float(value)
    return {"name": name, "value": v, "magnitude": magnitude(kind, v * scale if v is not None else None)}


def _decision(p, alpha):
    if p is None:
        return "not computable"
    return f"reject H0 (p {'<' if p < alpha else '>='} {alpha:g})" if p < alpha else \
        f"do not reject H0 (p >= {alpha:g})"


def _result(analysis, title, test, h0, stat_name, stat, p, alpha, df=None, effect=None, ci=None, extra=None):
    return {"analysis": analysis, "title": title, "test": test, "h0": h0, "statistic_name": stat_name,
            "statistic": None if stat is None else float(stat), "df": df, "p": None if p is None else float(p),
            "alpha": alpha, "decision": _decision(p, alpha), "significant": None if p is None else bool(p < alpha),
            "effect": effect, "ci": ci, "review": REVIEW, **(extra or {})}


def _ties(values):
    v = np.asarray(values)
    return len(v) != len(np.unique(v))


def _groups(df, outcome, group):
    if outcome not in df.columns or group not in df.columns:
        raise ValueError("Choose columns that exist in the data.")
    if not is_numeric(df[outcome]):
        raise ValueError(f"'{outcome}' must be numeric to compare means or ranks.")
    data = pd.DataFrame({"y": pd.to_numeric(df[outcome], errors="coerce"), "g": df[group]}).dropna()
    groups = [(str(k), v["y"].to_numpy(float)) for k, v in data.groupby("g", sort=True)]
    groups = [(k, v) for k, v in groups if len(v)]
    if len(groups) < 2:
        raise ValueError(f"'{group}' needs at least 2 groups with values.")
    if any(len(v) < 2 for _, v in groups):
        raise ValueError("Every group needs at least 2 values.")
    return groups, int(len(data))


def _group_table(groups):
    return [{"group": k, "n": int(len(v)), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
             "median": float(np.median(v))} for k, v in groups]


def compare_groups(df, outcome, group, test="auto", alpha=ALPHA):
    """Numeric outcome across the groups of a categorical column (independent samples)."""
    groups, n = _groups(df, outcome, group)
    k = len(groups)
    norm = [{"group": g, **(shapiro(v) or {"W": None, "p": None, "n": len(v), "note": "fewer than 3 values"})}
            for g, v in groups]
    normal = all(r["p"] is not None and r["p"] > alpha for r in norm)
    lev = stats.levene(*[v for _, v in groups])          # center="median": Brown-Forsythe, as R car::leveneTest
    equal_var = bool(lev.pvalue > alpha)
    assumptions = {"normality": norm, "all_normal": normal,
                   "levene": {"F": float(lev.statistic), "p": float(lev.pvalue), "equal_variances": equal_var}}
    two = k == 2
    choices = ({"auto", "welch_t", "student_t", "mann_whitney"} if two
               else {"auto", "anova", "welch_anova", "kruskal"})
    if test not in choices:
        raise ValueError(f"Test must be one of {', '.join(sorted(choices))} for {k} groups.")
    reason = None
    if test == "auto":
        if two:
            test = "welch_t" if normal else "mann_whitney"
            reason = ("both groups look normal -> Welch's t-test (does not assume equal variances)" if normal
                      else "a group is not normal (Shapiro-Wilk) -> Mann-Whitney U (rank-based)")
        else:
            test = ("anova" if equal_var else "welch_anova") if normal else "kruskal"
            reason = (("all groups normal and variances equal (Levene) -> one-way ANOVA" if equal_var else
                       "all groups normal but variances differ (Levene) -> Welch's ANOVA") if normal
                      else "a group is not normal (Shapiro-Wilk) -> Kruskal-Wallis (rank-based)")
    h0 = f"{outcome} has the same {'mean' if test in ('welch_t', 'student_t', 'anova', 'welch_anova') else 'distribution'} " \
         f"in every {group} group"
    title = f"{outcome} by {group}"
    extra = {"groups": _group_table(groups), "assumptions": assumptions, "reason": reason, "n": n}
    vals = [v for _, v in groups]
    if test in ("welch_t", "student_t"):
        a, b = vals
        r = stats.ttest_ind(a, b, equal_var=test == "student_t")
        ci = r.confidence_interval(CONFIDENCE)
        sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp else None
        return _result("compare", title, "Welch's t-test" if test == "welch_t" else "Student's t-test", h0, "t",
                       r.statistic, r.pvalue, alpha, float(r.df), _effect("Cohen's d", d, "d"),
                       {"of": f"mean difference ({groups[0][0]} - {groups[1][0]})", "level": CONFIDENCE,
                        "low": float(ci.low), "high": float(ci.high)}, extra)
    if test == "mann_whitney":
        a, b = vals
        exact = len(a) < EXACT_MAX and len(b) < EXACT_MAX and not _ties(np.r_[a, b])
        r = stats.mannwhitneyu(a, b, method="exact" if exact else "asymptotic", use_continuity=True)
        rb = 1 - 2 * r.statistic / (len(a) * len(b))      # rank-biserial correlation (Kerby 2014)
        extra["method"] = "exact" if exact else "normal approximation with continuity correction"
        return _result("compare", title, "Mann-Whitney U", h0, "U", r.statistic, r.pvalue, alpha, None,
                       _effect("rank-biserial r", -rb, "r"), None, extra)
    if test in ("anova", "welch_anova"):
        r = stats.f_oneway(*vals, equal_var=test == "anova")
        grand = np.concatenate(vals)
        ss_b = sum(len(v) * (v.mean() - grand.mean()) ** 2 for v in vals)
        ss_t = ((grand - grand.mean()) ** 2).sum()
        extra["post_hoc"] = _tukey(groups) if r.pvalue < alpha and test == "anova" else None
        return _result("compare", title, "One-way ANOVA" if test == "anova" else "Welch's ANOVA", h0, "F",
                       r.statistic, r.pvalue, alpha, [k - 1, int(len(grand) - k)] if test == "anova" else None,
                       _effect("eta squared", ss_b / ss_t if ss_t else None, "eta2"), None, extra)
    r = stats.kruskal(*vals)
    eps2 = r.statistic / (n - 1)                          # epsilon squared (Tomczak & Tomczak 2014)
    return _result("compare", title, "Kruskal-Wallis", h0, "H", r.statistic, r.pvalue, alpha, k - 1,
                   _effect("epsilon squared", eps2, "eta2"), None, extra)


def _tukey(groups):
    r = stats.tukey_hsd(*[v for _, v in groups])
    ci = r.confidence_interval(CONFIDENCE)
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            out.append({"a": groups[i][0], "b": groups[j][0], "difference": float(r.statistic[i, j]),
                        "p": float(r.pvalue[i, j]), "low": float(ci.low[i, j]), "high": float(ci.high[i, j])})
    return {"test": "Tukey HSD", "pairs": out}


def paired(df, before, after, test="auto", alpha=ALPHA):
    """Two measurements of the same rows (before / after, left / right)."""
    for c in (before, after):
        if c not in df.columns or not is_numeric(df[c]):
            raise ValueError(f"'{c}' must be a numeric column.")
    data = pd.DataFrame({"a": pd.to_numeric(df[before], errors="coerce"),
                         "b": pd.to_numeric(df[after], errors="coerce")}).dropna()
    if len(data) < 3:
        raise ValueError("Need at least 3 rows with both values.")
    a, b = data["a"].to_numpy(float), data["b"].to_numpy(float)
    d = a - b
    sw = shapiro(d)
    normal = bool(sw and sw["p"] is not None and sw["p"] > alpha)
    if test not in ("auto", "paired_t", "wilcoxon"):
        raise ValueError("Test must be auto, paired_t or wilcoxon.")
    reason = None
    if test == "auto":
        test = "paired_t" if normal else "wilcoxon"
        reason = ("the differences look normal -> paired t-test" if normal
                  else "the differences are not normal (Shapiro-Wilk) -> Wilcoxon signed-rank")
    extra = {"assumptions": {"normality_of_differences": sw, "normal": normal}, "reason": reason,
             "n": int(len(d)), "means": {before: float(a.mean()), after: float(b.mean())},
             "mean_difference": float(d.mean())}
    title = f"{before} vs {after} (paired)"
    if test == "paired_t":
        r = stats.ttest_rel(a, b)
        ci = r.confidence_interval(CONFIDENCE)
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) else None
        return _result("paired", title, "Paired t-test", f"the mean difference {before} - {after} is 0", "t",
                       r.statistic, r.pvalue, alpha, int(r.df), _effect("Cohen's dz", dz, "d"),
                       {"of": f"mean difference ({before} - {after})", "level": CONFIDENCE,
                        "low": float(ci.low), "high": float(ci.high)}, extra)
    nz = d[d != 0]
    exact = len(nz) < EXACT_MAX and not _ties(np.abs(nz)) and len(nz) == len(d)
    r = stats.wilcoxon(a, b, method="exact" if exact else "approx", correction=not exact)
    ranks = stats.rankdata(np.abs(nz))
    w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    rrb = (w_pos - w_neg) / (w_pos + w_neg) if (w_pos + w_neg) else None   # matched-pairs rank-biserial (Kerby 2014)
    extra["method"] = "exact" if exact else "normal approximation with continuity correction"
    extra["zero_differences_dropped"] = int((d == 0).sum())
    return _result("paired", title, "Wilcoxon signed-rank", f"the differences {before} - {after} are centred on 0",
                   "V", w_pos, r.pvalue, alpha, None, _effect("rank-biserial r", rrb, "r"), None, extra)


def correlation(df, columns, method="auto", alpha=ALPHA):
    """Every pair of numeric columns: coefficient, p-value, 95% CI (Pearson), method chosen per pair."""
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2 or any(not is_numeric(df[c]) for c in cols):
        raise ValueError("Choose at least 2 numeric columns.")
    if method not in ("auto", "pearson", "spearman"):
        raise ValueError("Method must be auto, pearson or spearman.")
    normal = {c: (lambda s: bool(s and s["p"] is not None and s["p"] > alpha))(shapiro(numeric(df[c]).to_numpy()))
              for c in cols}
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            data = pd.DataFrame({"a": pd.to_numeric(df[a], errors="coerce"),
                                 "b": pd.to_numeric(df[b], errors="coerce")}).dropna()
            if len(data) < 3:
                pairs.append({"a": a, "b": b, "n": int(len(data)), "error": "fewer than 3 complete rows"})
                continue
            m = method if method != "auto" else ("pearson" if normal[a] and normal[b] else "spearman")
            if m == "pearson":
                r = stats.pearsonr(data["a"], data["b"])
                ci = r.confidence_interval(CONFIDENCE)
                ci = {"level": CONFIDENCE, "low": float(ci.low), "high": float(ci.high)}
            else:
                r = stats.spearmanr(data["a"], data["b"])
                ci = None
            pairs.append({"a": a, "b": b, "n": int(len(data)), "method": m, "r": float(r.statistic),
                          "p": float(r.pvalue), "ci": ci, "magnitude": magnitude("r", float(r.statistic)),
                          "decision": _decision(float(r.pvalue), alpha)})
    return {"analysis": "correlation", "title": f"Correlation: {', '.join(cols)}", "alpha": alpha,
            "h0": "no correlation (coefficient = 0)", "pairs": pairs, "normal": normal, "review": REVIEW,
            "reason": None if method != "auto" else "Pearson where both columns look normal, Spearman otherwise"}


def association(df, row, col, alpha=ALPHA):
    """Chi-square test of independence between two categorical columns."""
    for c in (row, col):
        if c not in df.columns:
            raise ValueError(f"'{c}' is not a column.")
    table = pd.crosstab(df[row].astype("string"), df[col].astype("string"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both columns need at least 2 categories.")
    chi2, p, dof, expected = stats.chi2_contingency(table.to_numpy())
    chi2_raw = stats.chi2_contingency(table.to_numpy(), correction=False)[0]
    n = int(table.to_numpy().sum())
    k = min(table.shape) - 1
    v = math.sqrt(chi2_raw / (n * k)) if n and k else None
    low_share = float((expected < EXPECTED_MIN).mean())
    warnings = []
    if low_share > EXPECTED_SHARE:
        warnings.append(f"{low_share:.0%} of cells have an expected count below {EXPECTED_MIN} "
                        f"(Cochran's rule: at most {EXPECTED_SHARE:.0%}) - the chi-square p-value is unreliable"
                        + ("; use Fisher's exact test below" if table.shape == (2, 2) else
                           "; merge sparse categories"))
    extra = {"table": {"rows": [str(x) for x in table.index], "cols": [str(x) for x in table.columns],
                       "counts": table.to_numpy().tolist(), "expected": np.round(expected, 2).tolist()},
             "warnings": warnings, "n": n,
             "yates": table.shape == (2, 2)}
    if table.shape == (2, 2):
        f = stats.fisher_exact(table.to_numpy())
        extra["fisher"] = {"odds_ratio": float(f.statistic), "p": float(f.pvalue), "decision": _decision(float(f.pvalue), alpha)}
    return _result("association", f"{row} x {col}", "Chi-square test of independence",
                   f"{row} and {col} are independent", "chi2", chi2, p, alpha, int(dof),
                   _effect("Cramér's V", v, "w", scale=math.sqrt(k) if k else 1.0), None, extra)


def decisions(saved):
    """§13.4 hypothesis decision table from saved results: one row per test."""
    rows = []
    for r in saved:
        if r.get("analysis") == "correlation":
            for pr in r.get("pairs", []):
                if "p" in pr:
                    rows.append({"hypothesis": f"H0: no correlation between {pr['a']} and {pr['b']}",
                                 "test": pr["method"].title(), "statistic": pr["r"], "p": pr["p"],
                                 "effect": f"r = {pr['r']:.3f} ({pr['magnitude']})", "decision": pr["decision"]})
        elif r.get("h0") and "p" in r and r.get("decision"):
            eff = r.get("effect") or {}
            effect = (f"{eff['name']} = {eff['value']:.3f} ({eff['magnitude']})"
                      if eff.get("value") is not None else
                      f"R² = {r['r2']:.3f}" if r.get("r2") is not None else
                      f"pseudo R² = {r['pseudo_r2']:.3f}" if r.get("pseudo_r2") is not None else "")
            rows.append({"hypothesis": f"H0: {r['h0']}", "test": r.get("test"), "statistic": r.get("statistic"),
                         "p": r.get("p"), "effect": effect, "decision": r["decision"]})
    return rows


if __name__ == "__main__":
    # Every expected value below is R's published output for its built-in dataset.
    sleep1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
    sleep2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
    sleep = pd.DataFrame({"extra": sleep1 + sleep2, "group": ["1"] * 10 + ["2"] * 10,
                          "d1": sleep1 + [None] * 10, "d2": sleep2 + [None] * 10})
    w = compare_groups(sleep, "extra", "group", "welch_t")                 # R t.test(extra ~ group, sleep)
    assert round(w["statistic"], 4) == -1.8608 and round(w["df"], 3) == 17.776 and round(w["p"], 5) == 0.07939, w
    mw = compare_groups(sleep, "extra", "group", "mann_whitney")           # R wilcox.test(extra ~ group, sleep)
    assert mw["statistic"] == 25.5 and round(mw["p"], 5) == 0.06933, mw
    pt = paired(sleep, "d1", "d2", "paired_t")                             # R t.test(..., paired = TRUE)
    assert round(pt["statistic"], 4) == -4.0621 and pt["df"] == 9 and round(pt["p"], 6) == 0.002833, pt
    wx = paired(sleep, "d1", "d2", "wilcoxon")                             # R wilcox.test(..., paired = TRUE)
    assert wx["statistic"] == 0 and round(wx["p"], 6) == 0.009091 and wx["zero_differences_dropped"] == 1, wx
    assert round(pt["effect"]["value"], 2) == -1.28 and pt["effect"]["magnitude"] == "large"

    plant = pd.DataFrame({"weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
                                     4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
                                     6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26],
                          "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10})
    av = compare_groups(plant, "weight", "group", "anova")                 # R summary(aov(weight ~ group))
    assert round(av["statistic"], 3) == 4.846 and av["df"] == [2, 27] and round(av["p"], 5) == 0.01591, av
    assert round(av["effect"]["value"], 3) == round(3.766 / (3.766 + 10.492), 3)   # R: SS group 3.766, resid 10.492
    lev = av["assumptions"]["levene"]                                      # R car::leveneTest(weight ~ group)
    assert round(lev["F"], 4) == 1.1192 and round(lev["p"], 4) == 0.3412, lev
    kw = compare_groups(plant, "weight", "group", "kruskal")               # R kruskal.test(weight ~ group)
    assert round(kw["statistic"], 4) == 7.9882 and round(kw["p"], 5) == 0.01842, kw
    assert av["post_hoc"] and {(x["a"], x["b"]) for x in av["post_hoc"]["pairs"]} == {
        ("ctrl", "trt1"), ("ctrl", "trt2"), ("trt1", "trt2")}
    assert compare_groups(plant, "weight", "group")["test"] == "One-way ANOVA"   # auto: normal + equal variances

    party = pd.DataFrame({"gender": ["F"] * (762 + 327 + 468) + ["M"] * (484 + 239 + 477),
                          "party": ["Democrat"] * 762 + ["Independent"] * 327 + ["Republican"] * 468 +
                                   ["Democrat"] * 484 + ["Independent"] * 239 + ["Republican"] * 477})
    ch = association(party, "gender", "party")                             # R chisq.test (Agresti 2007 table)
    assert round(ch["statistic"], 2) == 30.07 and ch["df"] == 2 and abs(ch["p"] - 2.954e-07) < 1e-9, ch
    assert round(ch["effect"]["value"], 4) == round(math.sqrt(30.0701 / 2757), 4)

    speed = [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15,
             16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25]
    dist = [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54,
            32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]
    cars = pd.DataFrame({"speed": speed, "dist": dist})
    pe = correlation(cars, ["speed", "dist"], "pearson")["pairs"][0]       # R cor.test(cars$speed, cars$dist)
    assert round(pe["r"], 7) == 0.8068949 and abs(pe["p"] - 1.49e-12) < 1e-14, pe
    sp = correlation(cars, ["speed", "dist"], "spearman")["pairs"][0]
    assert round(sp["r"], 7) == 0.8303568, sp
    assert correlation(cars, ["speed", "dist"])["pairs"][0]["method"] == "spearman"  # dist is not normal (W 0.951, p .039)

    # small 2x2 with sparse cells: the warning appears and Fisher's exact test is given
    tea = pd.DataFrame({"poured": ["milk"] * 4 + ["tea"] * 4, "guess": ["milk"] * 3 + ["tea"] + ["milk"] + ["tea"] * 3})
    ft = association(tea, "poured", "guess")
    assert ft["warnings"] and round(ft["fisher"]["p"], 4) == 0.4857, ft["fisher"]   # R fisher.test(TeaTasting)
    print("statistical tests self-check passed (R reference values reproduced)")
