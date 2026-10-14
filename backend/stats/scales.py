"""Questionnaire support (plan §13.2, §13.4): Likert-item detection and
reliability. Reverse scoring and composite scores are cleaning ops
(preprocessing/clean.py: reverse_score, composite) so they are saved with the
data and replayed on new rows like every other cleaning step.

  Cronbach's alpha = k/(k-1) * (1 - sum of item variances / variance of the total)
  (Cronbach 1951); standardised alpha = k*r/(1 + (k-1)*r), r = mean inter-item
  correlation. Corrected item-total correlation = item vs the total of the OTHER
  items; "alpha if deleted" = alpha of the remaining items.
  Labels (George & Mallery 2003): >= .9 excellent, .8 good, .7 acceptable,
  .6 questionable, .5 poor, below unacceptable -- a convention, not a test.

Likert detection is a heuristic, shown as "possible": a whole-number column with
3 to 11 different values, all between 0 and 10 (5- and 7-point scales are the
usual; 0-10 covers 11-point ratings).
"""

import numpy as np
import pandas as pd

from preprocessing.profiler import is_numeric
from stats.tests import REVIEW

LIKERT_MIN_LEVELS, LIKERT_MAX_LEVELS = 3, 11   # our starting point (see module docstring)
LIKERT_RANGE = (0, 10)
ALPHA_LABELS = [(0.9, "excellent"), (0.8, "good"), (0.7, "acceptable"), (0.6, "questionable"), (0.5, "poor")]


def likert_candidates(df):
    """{'items': [{column, min, max, levels}], 'scales': [{range, columns}]} -- possible Likert items,
    grouped by their value range (items of one scale share it)."""
    items = []
    for c in df.columns:
        s = df[c]
        if not is_numeric(s):
            continue
        v = pd.to_numeric(s, errors="coerce").dropna()
        if v.empty or not bool((v % 1 == 0).all()):
            continue
        levels = int(v.nunique())
        lo, hi = float(v.min()), float(v.max())
        if LIKERT_MIN_LEVELS <= levels <= LIKERT_MAX_LEVELS and LIKERT_RANGE[0] <= lo and hi <= LIKERT_RANGE[1]:
            items.append({"column": str(c), "min": lo, "max": hi, "levels": levels})
    scales = {}
    for it in items:
        scales.setdefault(f"{it['min']:g}-{it['max']:g}", []).append(it["column"])
    return {"analysis": "likert", "title": "Possible Likert items", "items": items,
            "scales": [{"range": r, "columns": cols} for r, cols in scales.items() if len(cols) >= 2],
            "note": "A heuristic: whole numbers, 3-11 levels, within 0-10. Confirm these are rating items."}


def _alpha(M):
    k = M.shape[1]
    total_var = M.sum(axis=1).var(ddof=1)
    if k < 2 or total_var == 0:
        return None
    return float(k / (k - 1) * (1 - M.var(axis=0, ddof=1).sum() / total_var))


def label(alpha):
    if alpha is None:
        return None
    return next((name for cut, name in ALPHA_LABELS if alpha >= cut), "unacceptable")


def reliability(df, items, reverse=None, low=None, high=None):
    """Cronbach's alpha for a set of items; `reverse` items are scored low+high-x
    first (low / high default to the lowest / highest value seen in the items)."""
    items = list(dict.fromkeys(items or []))
    if len(items) < 2:
        raise ValueError("Choose at least 2 items.")
    for c in items:
        if c not in df.columns or not is_numeric(df[c]):
            raise ValueError(f"'{c}' must be a numeric item.")
    data = df[items].apply(pd.to_numeric, errors="coerce")
    complete = data.dropna()
    if len(complete) < 3:
        raise ValueError("Need at least 3 respondents who answered every item.")
    lo = float(complete.min().min()) if low is None else float(low)
    hi = float(complete.max().max()) if high is None else float(high)
    rev = [c for c in (reverse or []) if c in items]
    M = complete.copy()
    for c in rev:
        M[c] = lo + hi - M[c]
    X = M.to_numpy(float)
    a = _alpha(X)
    corr = np.corrcoef(X, rowvar=False)
    r_mean = float(corr[np.triu_indices(len(items), 1)].mean())
    k = len(items)
    rows = []
    for j, c in enumerate(items):
        rest = np.delete(X, j, axis=1)
        other_total = rest.sum(axis=1)
        r = float(np.corrcoef(X[:, j], other_total)[0, 1]) if X[:, j].std() and other_total.std() else None
        rows.append({"item": c, "reversed": c in rev, "mean": float(X[:, j].mean()), "sd": float(X[:, j].std(ddof=1)),
                     "item_total_r": r, "alpha_if_deleted": _alpha(rest)})
    weak = [r["item"] for r in rows if r["item_total_r"] is not None and r["item_total_r"] < 0]
    return {"analysis": "reliability", "title": f"Reliability: {', '.join(items)}", "test": "Cronbach's alpha",
            "alpha": a, "standardized_alpha": k * r_mean / (1 + (k - 1) * r_mean) if r_mean > -1 / (k - 1) else None,
            "label": label(a), "items": rows, "k": k, "n": int(len(complete)),
            "dropped_rows": int(len(data) - len(complete)), "mean_inter_item_r": r_mean,
            "scale_range": [lo, hi], "reversed": rev,
            "warnings": [f"{', '.join(weak)} correlate(s) negatively with the rest - reverse-scored item not "
                         "marked as reversed?"] if weak else [],
            "review": REVIEW}


if __name__ == "__main__":
    # a 4-item, 6-person example worked by hand with the formula
    d = pd.DataFrame({"q1": [4, 5, 3, 2, 4, 5], "q2": [4, 4, 3, 2, 5, 5], "q3": [3, 5, 2, 1, 4, 4],
                      "q4": [2, 1, 3, 4, 2, 1]})       # q4 is worded the other way round
    item_vars = d[["q1", "q2", "q3"]].var(ddof=1)
    total_var = d[["q1", "q2", "q3"]].sum(axis=1).var(ddof=1)
    by_hand = 3 / 2 * (1 - item_vars.sum() / total_var)
    r3 = reliability(d, ["q1", "q2", "q3"])
    assert abs(r3["alpha"] - by_hand) < 1e-12 and r3["label"] in ("excellent", "good"), r3
    # without reversing q4 the scale looks bad and q4 is flagged; reversed (1..5) it fits
    raw = reliability(d, ["q1", "q2", "q3", "q4"])
    fixed = reliability(d, ["q1", "q2", "q3", "q4"], reverse=["q4"], low=1, high=5)
    assert raw["warnings"] and "q4" in raw["warnings"][0] and fixed["alpha"] > raw["alpha"] and not fixed["warnings"]
    # identical items: alpha = 1; standardised alpha follows Spearman-Brown on r = 1
    same = reliability(pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]}), ["a", "b"])
    assert abs(same["alpha"] - 1) < 1e-12 and abs(same["standardized_alpha"] - 1) < 1e-12
    # alpha if deleted = alpha computed on the remaining items
    assert abs(fixed["items"][3]["alpha_if_deleted"] - r3["alpha"]) < 1e-12
    lk = likert_candidates(d.assign(age=[23, 35, 41, 52, 29, 60], score=[1.5, 2, 3, 4, 5, 2]))
    assert {i["column"] for i in lk["items"]} == {"q1", "q2", "q3", "q4"} and lk["scales"][0]["columns"], lk
    print("reliability self-check passed")
