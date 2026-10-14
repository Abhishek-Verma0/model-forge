"""Result charts from a run's saved test-set predictions -- no model is re-run.

predictions/<key>.json holds y_true, y_pred (classification: encoded 0..k-1) and,
when the model has predict_proba, proba (columns in class order). results.json
holds "classes" (the label strings, same order) and "task". This turns those into
small arrays the frontend draws (and the PDF report re-draws):
  classification: confusion matrix, ROC, precision-recall and calibration curves
  regression:     actual vs predicted, residuals vs predicted, error histogram, Q-Q plot
"""

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_curve)

MAX_POINTS = 200     # our starting point: cap curve points so the JSON and SVG stay small
MAX_SCATTER = 2000   # our starting point: points in the regression scatter plots
CALIBRATION_BINS = 10  # our starting point (scikit-learn's calibration examples use 10; its default is 5)


def _thin(xs, ys, cap=MAX_POINTS):
    """Down-sample a curve to at most `cap` evenly-spaced points (a test set of
    thousands of rows gives thousands of thresholds; the plot needs far fewer)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) > cap:
        idx = np.unique(np.linspace(0, len(xs) - 1, cap).round().astype(int))
        xs, ys = xs[idx], ys[idx]
    return [round(float(v), 5) for v in xs], [round(float(v), 5) for v in ys]


def _classification(results, preds):
    classes = results.get("classes") or []
    labels = list(range(len(classes)))
    y_true = np.asarray(preds.get("y_true", []), dtype=int)
    y_pred = np.asarray(preds.get("y_pred", []), dtype=int)
    out = {"task": "classification", "classes": classes,
           "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
           "roc": [], "pr": [], "calibration": []}

    proba = preds.get("proba")
    if proba is None or not len(y_true):
        return out
    P = np.asarray(proba, dtype=float)
    # binary: one curve for the positive class; multiclass: one-vs-rest per class
    if len(classes) == 2:
        pos = results["positive_class"] if results.get("positive_class") in classes else classes[1]
        draw = [classes.index(pos)]
    else:
        draw = labels
    for i in draw:
        yb = (y_true == i).astype(int)
        if yb.min() == yb.max():
            continue    # class absent (or every row) in the test set -> no curve
        fpr, tpr, _ = roc_curve(yb, P[:, i])
        prec, rec, _ = precision_recall_curve(yb, P[:, i])
        fx, fy = _thin(fpr, tpr)
        rx, ry = _thin(rec, prec)
        out["roc"].append({"label": classes[i], "fpr": fx, "tpr": fy, "auc": round(float(auc(fpr, tpr)), 4)})
        out["pr"].append({"label": classes[i], "recall": rx, "precision": ry,
                          "ap": round(float(average_precision_score(yb, P[:, i])), 4)})
        # calibration: within each probability bin, how often the class really occurred
        true_frac, mean_pred = calibration_curve(yb, P[:, i], n_bins=CALIBRATION_BINS)
        out["calibration"].append({"label": classes[i], "predicted": [round(float(v), 5) for v in mean_pred],
                                   "observed": [round(float(v), 5) for v in true_frac],
                                   "brier": round(float(brier_score_loss(yb, P[:, i])), 4)})
    return out


def _regression(preds):
    y = np.asarray(preds.get("y_true", []), dtype=float)
    p = np.asarray(preds.get("y_pred", []), dtype=float)
    out = {"task": "regression", "n": int(len(y))}
    if len(y) < 2:
        return out
    resid = y - p
    idx = np.unique(np.linspace(0, len(y) - 1, min(len(y), MAX_SCATTER)).round().astype(int))
    r5 = lambda a: [round(float(v), 5) for v in a]  # noqa: E731
    counts, edges = np.histogram(resid, bins="auto")   # numpy's rule: max of Sturges and Freedman-Diaconis
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    qx, qy = _thin(osm, osr)
    out.update(
        actual=r5(y[idx]), predicted=r5(p[idx]), residual=r5(resid[idx]), shown=int(len(idx)),
        histogram={"counts": counts.tolist(), "edges": r5(edges)},
        # Q-Q: residual quantiles against normal quantiles; on the line = normally distributed errors
        qq={"theoretical": qx, "sample": qy, "slope": round(float(slope), 5),
            "intercept": round(float(intercept), 5), "r": round(float(r), 5)},
        residual_mean=round(float(resid.mean()), 5), residual_sd=round(float(resid.std(ddof=1)), 5))
    return out


def curves(results, preds):
    """Chart data for one model of a run (see module docstring)."""
    task = results.get("task")
    if task == "classification":
        return _classification(results, preds)
    if task == "regression":
        return _regression(preds)
    return {"task": task}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 2-class: a decent classifier -> AUC well above 0.5, confusion counts sum to n
    n = 200
    y = rng.integers(0, 2, n)
    p1 = np.clip(y * 0.6 + rng.normal(0, 0.25, n) + 0.2, 0, 1)
    preds = {"y_true": y.tolist(), "y_pred": (p1 > 0.5).astype(int).tolist(),
             "proba": np.c_[1 - p1, p1].tolist()}
    res = {"task": "classification", "classes": ["no", "yes"], "positive_class": "yes"}
    c = curves(res, preds)
    assert np.sum(c["confusion"]) == n and np.array(c["confusion"]).shape == (2, 2)
    assert len(c["roc"]) == 1 and c["roc"][0]["label"] == "yes"
    assert 0.5 < c["roc"][0]["auc"] <= 1.0, c["roc"][0]["auc"]
    assert all(0 <= v <= 1 for v in c["roc"][0]["fpr"] + c["roc"][0]["tpr"])
    assert len(c["pr"][0]["recall"]) == len(c["pr"][0]["precision"]) <= MAX_POINTS
    cal = c["calibration"][0]
    assert abs(cal["brier"] - brier_score_loss(y, p1)) < 1e-4 and len(cal["predicted"]) <= CALIBRATION_BINS
    # a perfectly calibrated model: observed frequency tracks the predicted probability
    pp = rng.random(20000)
    yy = (rng.random(20000) < pp).astype(int)
    good = curves(res, {"y_true": yy.tolist(), "y_pred": (pp > .5).astype(int).tolist(),
                        "proba": np.c_[1 - pp, pp].tolist()})["calibration"][0]
    assert max(abs(a - b) for a, b in zip(good["predicted"], good["observed"])) < 0.03, good

    # 3-class OvR: one curve per class, matrix 3x3; a class missing from the test set is skipped
    y3 = np.array([0, 1, 2] * 40)
    P3 = rng.dirichlet(np.ones(3), len(y3))
    P3[np.arange(len(y3)), y3] += 1.0
    P3 /= P3.sum(1, keepdims=True)
    c3 = curves({"task": "classification", "classes": ["a", "b", "c"]},
                {"y_true": y3.tolist(), "y_pred": P3.argmax(1).tolist(), "proba": P3.tolist()})
    assert np.array(c3["confusion"]).shape == (3, 3) and len(c3["roc"]) == 3

    # no proba (e.g. SVM without probability): matrix only, no curves
    c0 = curves(res, {"y_true": y.tolist(), "y_pred": (p1 > 0.5).astype(int).tolist()})
    assert c0["roc"] == [] and c0["pr"] == [] and np.sum(c0["confusion"]) == n

    # down-sampling caps a huge test set
    big = 5000
    yb = rng.integers(0, 2, big); pb = rng.random(big)
    cb = curves(res, {"y_true": yb.tolist(), "y_pred": (pb > .5).astype(int).tolist(), "proba": np.c_[1 - pb, pb].tolist()})
    assert len(cb["roc"][0]["fpr"]) <= MAX_POINTS

    # regression: residuals = actual - predicted; normal errors sit on the Q-Q line
    ya = rng.normal(50, 10, 3000)
    pr = ya + rng.normal(0, 2, 3000)
    rg = curves({"task": "regression"}, {"y_true": ya.tolist(), "y_pred": pr.tolist()})
    assert rg["shown"] == MAX_SCATTER and len(rg["actual"]) == len(rg["residual"]) == MAX_SCATTER
    assert sum(rg["histogram"]["counts"]) == 3000 and rg["qq"]["r"] > 0.99, rg["qq"]
    assert abs(rg["residual_mean"] - float(np.mean(ya - pr))) < 1e-4
    assert abs(rg["qq"]["slope"] - float(np.std(ya - pr, ddof=1))) < 0.1   # Q-Q slope ~ residual sd
    assert curves({"task": "regression"}, {"y_true": [1.0], "y_pred": [1.0]}) == {"task": "regression", "n": 1}
    print("curves self-check passed")
