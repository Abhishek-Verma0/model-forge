"""Regression for explanation (plan §13.3 "linear, multiple or logistic
regression"): coefficient tables with standard errors, test statistics, p-values
and 95% confidence intervals -- the statistical view, unlike the ML Train tab
which scores predictions. Validated against R (lm / glm on mtcars) below.

  linear    ordinary least squares; t tests on coefficients; R², adjusted R², F test;
            variance inflation factors (VIF) for multicollinearity
  logistic  maximum likelihood by Newton-Raphson (IRLS); Wald z tests; odds ratios
            with 95% CI; McFadden pseudo R²; likelihood-ratio test

Categorical predictors become indicator columns against their first category (R's
default treatment coding). Rows with any missing value in the used columns are
left out, and the result says how many.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from preprocessing.profiler import is_numeric
from stats.describe import CONFIDENCE, shapiro
from stats.tests import ALPHA, REVIEW

MAX_ITER = 100       # Newton-Raphson iterations (glm's default maxit is 25; more never hurts convergence)
TOL = 1e-10          # change in log-likelihood that counts as converged
MAX_LEVELS = 50      # our starting point: categories a predictor may have (indicator columns)


def _design(df, outcome, predictors):
    cols = [outcome] + list(predictors)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Not columns: {', '.join(missing)}.")
    if not predictors:
        raise ValueError("Choose at least one predictor.")
    data = df[cols].dropna()
    dropped = int(len(df) - len(data))
    parts, names, source = [], ["(Intercept)"], {}
    for c in predictors:
        s = data[c]
        if is_numeric(s):
            parts.append(pd.to_numeric(s).astype(float).to_frame(c))
            names.append(str(c))
            source[str(c)] = c
        else:
            levels = sorted(s.astype(str).unique())
            if len(levels) > MAX_LEVELS:
                raise ValueError(f"'{c}' has {len(levels)} categories (max {MAX_LEVELS}).")
            for lv in levels[1:]:     # first category = reference (R treatment coding)
                name = f"{c}[{lv}]"
                parts.append((s.astype(str) == lv).astype(float).rename(name).to_frame())
                names.append(name)
                source[name] = c
    X = np.c_[np.ones(len(data)), pd.concat(parts, axis=1).to_numpy(float)] if parts else np.ones((len(data), 1))
    if len(data) <= X.shape[1]:
        raise ValueError(f"Only {len(data)} complete rows for {X.shape[1]} coefficients - not enough data.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("Predictors are perfectly collinear (one is a combination of others) - remove one.")
    return data, X, names, dropped, source


def _coef_table(names, b, se, stat, p, crit, stat_name):
    return [{"term": t, "estimate": float(bi), "se": float(si), stat_name: float(zi), "p": float(pi),
             "low": float(bi - crit * si), "high": float(bi + crit * si)}
            for t, bi, si, zi, pi in zip(names, b, se, stat, p)]


def linear(df, outcome, predictors, alpha=ALPHA):
    data, X, names, dropped, source = _design(df, outcome, predictors)
    if not is_numeric(data[outcome]):
        raise ValueError(f"Linear regression needs a numeric outcome; '{outcome}' is not.")
    y = pd.to_numeric(data[outcome]).to_numpy(float)
    n, p = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    rss, tss = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    dfr = n - p
    sigma2 = rss / dfr
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = b / se
    pv = 2 * stats.t.sf(np.abs(t), dfr)
    crit = stats.t.ppf(0.5 + CONFIDENCE / 2, dfr)
    r2 = 1 - rss / tss if tss else None
    adj = 1 - (1 - r2) * (n - 1) / dfr if r2 is not None else None
    f = ((tss - rss) / (p - 1)) / sigma2 if p > 1 else None
    fp = float(stats.f.sf(f, p - 1, dfr)) if f is not None else None
    vif = {}
    if p > 2:   # VIF_j = 1 / (1 - R²_j), R²_j from regressing predictor j on the others
        for j in range(1, p):
            others = np.delete(X, j, axis=1)
            fit, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
            res = X[:, j] - others @ fit
            r2j = 1 - (res @ res) / (((X[:, j] - X[:, j].mean()) ** 2).sum() or 1)
            vif[names[j]] = float(1 / (1 - r2j)) if r2j < 1 else None
    return {"analysis": "regression", "kind": "linear", "title": f"Linear regression: {outcome} ~ {' + '.join(predictors)}",
            "test": "Ordinary least squares", "outcome": outcome, "n": n, "dropped_rows": dropped,
            "coefficients": _coef_table(names, b, se, t, pv, crit, "t"), "df_residual": dfr,
            "r2": r2, "adj_r2": adj, "f": f, "f_df": [p - 1, dfr], "p": fp, "alpha": alpha,
            "sigma": math.sqrt(sigma2), "vif": vif,
            "residual_normality": shapiro(resid),
            "h0": "no predictor explains the outcome (all slopes are 0)",
            "decision": None if fp is None else ("reject H0" if fp < alpha else "do not reject H0") + f" (F test, alpha {alpha:g})",
            "significant": None if fp is None else bool(fp < alpha),
            "notes": ["VIF above 10 (a common rule of thumb) signals strong multicollinearity"] if any(
                v and v > 10 for v in vif.values()) else [],
            "review": REVIEW}


def logistic(df, outcome, predictors, event=None, alpha=ALPHA):
    data, X, names, dropped, _ = _design(df, outcome, predictors)
    yv = data[outcome].astype(str)
    levels = sorted(yv.unique())
    if len(levels) != 2:
        raise ValueError(f"Logistic regression needs an outcome with exactly 2 values; '{outcome}' has {len(levels)}.")
    event = str(event) if event is not None else levels[1]
    if event not in levels:
        raise ValueError(f"'{event}' is not a value of {outcome}.")
    y = (yv == event).to_numpy(float)
    n, p = X.shape
    beta = np.zeros(p)

    def loglik(bv):
        eta = X @ bv
        return float(np.sum(y * eta - np.logaddexp(0, eta)))

    ll, converged, notes = loglik(beta), False, []
    for _ in range(MAX_ITER):
        prob = 1 / (1 + np.exp(-(X @ beta)))
        W = prob * (1 - prob)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - prob))
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        new = loglik(beta)
        if abs(new - ll) < TOL:
            ll, converged = new, True
            break
        ll = new
    prob = 1 / (1 + np.exp(-(X @ beta)))
    H = X.T @ (X * (prob * (1 - prob))[:, None])
    if not converged or np.any(np.abs(beta) > 30) or np.linalg.cond(H) > 1e12:
        notes.append("the fit did not settle (possible perfect separation: a predictor splits the outcome "
                     "exactly) - estimates and p-values are not trustworthy")
    se = np.sqrt(np.diag(np.linalg.pinv(H)))
    z = beta / se
    pv = 2 * stats.norm.sf(np.abs(z))
    crit = stats.norm.ppf(0.5 + CONFIDENCE / 2)
    p0 = y.mean()
    ll0 = float(len(y) * (p0 * math.log(p0) + (1 - p0) * math.log(1 - p0))) if 0 < p0 < 1 else 0.0
    lr = 2 * (ll - ll0)
    lr_p = float(stats.chi2.sf(lr, p - 1)) if p > 1 else None
    coefs = _coef_table(names, beta, se, z, pv, crit, "z")
    for c in coefs:
        c.update(odds_ratio=math.exp(c["estimate"]) if c["estimate"] < 700 else None,
                 or_low=math.exp(c["low"]) if c["low"] < 700 else None,
                 or_high=math.exp(c["high"]) if c["high"] < 700 else None)
    return {"analysis": "regression", "kind": "logistic",
            "title": f"Logistic regression: P({outcome} = {event}) ~ {' + '.join(predictors)}",
            "test": "Logistic regression (maximum likelihood)", "outcome": outcome, "event": event, "n": n,
            "dropped_rows": dropped, "coefficients": coefs, "log_likelihood": ll, "null_log_likelihood": ll0,
            "pseudo_r2": 1 - ll / ll0 if ll0 else None, "lr_chi2": lr, "lr_df": p - 1, "p": lr_p,
            "converged": converged, "alpha": alpha, "notes": notes,
            "h0": "no predictor changes the odds of the outcome (all slopes are 0)",
            "decision": None if lr_p is None else ("reject H0" if lr_p < alpha else "do not reject H0") + f" (likelihood-ratio test, alpha {alpha:g})",
            "significant": None if lr_p is None else bool(lr_p < alpha),
            "review": REVIEW}


if __name__ == "__main__":
    mtcars = pd.DataFrame({
        "mpg": [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4,
                14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
        "wt": [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440, 4.070, 3.730, 3.780,
               5.250, 5.424, 5.345, 2.200, 1.615, 1.835, 2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513,
               3.170, 2.770, 3.570, 2.780],
        "hp": [110, 110, 93, 110, 175, 105, 245, 62, 95, 123, 123, 180, 180, 180, 205, 215, 230, 66, 52, 65, 97,
               150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109],
        "am": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]})
    lm = linear(mtcars, "mpg", ["wt", "hp"])           # R: summary(lm(mpg ~ wt + hp, data = mtcars))
    est = {c["term"]: c for c in lm["coefficients"]}
    assert round(est["(Intercept)"]["estimate"], 5) == 37.22727 and round(est["wt"]["estimate"], 5) == -3.87783
    assert round(est["hp"]["estimate"], 5) == -0.03177 and round(lm["r2"], 4) == 0.8268
    assert round(lm["adj_r2"], 4) == 0.8148 and round(lm["f"], 2) == 69.21 and lm["f_df"] == [2, 29], lm
    simple = linear(pd.DataFrame({"x": [1, 2, 3, 4, 5.0], "y": [2, 4.1, 5.9, 8.2, 9.9]}), "y", ["x"])
    ref = stats.linregress([1, 2, 3, 4, 5], [2, 4.1, 5.9, 8.2, 9.9])
    assert abs(simple["coefficients"][1]["estimate"] - ref.slope) < 1e-12
    assert abs(simple["coefficients"][1]["se"] - ref.stderr) < 1e-12 and abs(simple["coefficients"][1]["p"] - ref.pvalue) < 1e-12

    gl = logistic(mtcars, "am", ["wt"])                # R: summary(glm(am ~ wt, binomial, data = mtcars))
    g = {c["term"]: c for c in gl["coefficients"]}
    assert round(g["(Intercept)"]["estimate"], 3) == 12.040 and round(g["wt"]["estimate"], 3) == -4.024, g
    assert round(g["(Intercept)"]["z"], 3) == 2.670 and round(g["wt"]["z"], 3) == -2.801
    assert round(g["(Intercept)"]["p"], 5) == 0.00759 and round(g["wt"]["p"], 5) == 0.00509 and gl["converged"]
    sep = logistic(pd.DataFrame({"x": [1, 2, 3, 4, 5, 6.0], "y": ["a", "a", "a", "b", "b", "b"]}), "y", ["x"])
    assert sep["notes"] and "separation" in sep["notes"][0]
    cat = linear(mtcars.assign(gear=["4", "4", "4", "3", "3", "3", "3", "4", "4", "4", "4", "3", "3", "3", "3", "3",
                                     "3", "4", "4", "4", "3", "3", "3", "3", "3", "4", "5", "5", "5", "5", "5", "4"]),
                 "mpg", ["wt", "gear"])
    assert {c["term"] for c in cat["coefficients"]} == {"(Intercept)", "wt", "gear[4]", "gear[5]"}
    print("regression self-check passed (R lm / glm reproduced)")
