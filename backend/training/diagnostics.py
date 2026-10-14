"""Learning and validation curves (plan §11.1) for one model of a finished run,
on that run's own folds and with its parameters -- scikit-learn's learning_curve /
validation_curve on the whole leak-free pipeline. Both retrain the model many
times, so they run as background jobs (core/jobs) and land in
runs/<run>/diagnostics/.

  learning curve   -- score on the rows trained on vs on the held-out fold, as the
                      training share grows: a gap that stays wide = overfitting
                      (more data may help); both low = underfitting.
  validation curve -- the same two scores across values of ONE parameter: where the
                      held-out score peaks is the sweet spot for that setting.

Models are fitted in one call here (no live epochs / early stopping).
"""

import numpy as np
from sklearn.model_selection import learning_curve, validation_curve

from core import store
from training import ensembles, models, train, tuning
from training import params as P
from training.metrics import METRICS, scorers
from training.pipeline import build_pipeline

TRAIN_SIZES = np.linspace(0.1, 1.0, 5)   # published default (scikit-learn learning_curve train_sizes)
CURVE_POINTS = 7                         # our starting point: parameter values in a validation curve


def _setup(ds_id, run_id, key):
    run = store.run_dir(ds_id, run_id)
    res = store.read_json(run / "results.json") or {}
    settings = store.read_json(run / "settings.json") or {}
    rec = (res.get("models") or {}).get(key)
    if not rec or "error" in rec:
        raise LookupError(f"No trained model '{key}' in this run.")
    ctx = train.load_context(ds_id)
    if train.fingerprint(ctx, res["folds"]) != res["fingerprint"]:
        raise ValueError("The data or preprocessing changed since this run - train again first.")
    rows = dict(models.table(ctx["task"]))
    for e in settings.get("ensembles") or []:
        kw = {m: P.resolve(rows[m], (settings.get("params") or {}).get(m))[0] for m in e["members"]}
        rows[e["key"]] = ensembles.row(ctx["task"], e, rows, kw)
    row = rows[key]
    user = {k: v for k, v in (rec.get("params") or {}).items() if k != "learning_rate_used"}
    kwargs, _fit, _ = P.resolve(row, user)
    f = train.facts(ctx)
    pipe = build_pipeline(ctx, key, row, f, res["folds"], kwargs)
    primary = res["primary_metric"]
    pos = ctx["classes"].index(res["positive_class"]) if res.get("positive_class") else None
    sc = scorers(ctx["task"], len(ctx["classes"] or []), pos, hasattr(pipe, "predict_proba"),
                 f.get("target_has_zero", False), hasattr(pipe, "decision_function"))
    metric = primary if primary in sc else ("r2" if ctx["task"] == "regression" else "accuracy")
    return run, ctx, row, pipe, sc[metric], metric, train.cv_splits(ctx, res["folds"])


def _sign(metric):
    return -1.0 if metric in ("mae", "mse", "rmse", "mape", "log_loss") else 1.0   # scorers negate errors


def _summ(scores, metric):
    s = np.asarray(scores, float) * _sign(metric)
    return ([None if np.isnan(v) else round(float(v), 6) for v in np.nanmean(s, axis=1)],
            [None if np.isnan(v) else round(float(v), 6) for v in np.nanstd(s, axis=1, ddof=1)])


def learning(job, ds_id, run_id, key):
    run, ctx, row, pipe, scorer, metric, splits = _setup(ds_id, run_id, key)
    job.progress(step="learning curve", model=row["label"])
    sizes, tr, va = learning_curve(pipe, ctx["X_tr"], ctx["y_tr"], cv=splits, scoring=scorer,
                                   train_sizes=TRAIN_SIZES)
    (tm, ts), (vm, vs) = _summ(tr, metric), _summ(va, metric)
    out = {"kind": "learning", "model": key, "label": row["label"], "metric": metric,
           "metric_label": METRICS[metric][0], "higher_is_better": METRICS[metric][1],
           "train_rows": [int(x) for x in sizes], "train_mean": tm, "train_sd": ts, "val_mean": vm, "val_sd": vs}
    _save(run, f"{key}_learning", out, job)


def values_for(key, row, param):
    """The values a validation curve tries: the tuning search range, spread evenly
    (log-spaced for log-scale parameters); every choice for a choice parameter."""
    sp = tuning.space(key, row).get(param)
    if sp is None:
        raise ValueError(f"'{param}' can't be swept for {row['label']} (no search range).")
    if sp["kind"] == "choice":
        return sp["choices"]
    lo, hi = sp["low"], sp["high"]
    vals = np.geomspace(lo, hi, CURVE_POINTS) if sp["log"] else np.linspace(lo, hi, CURVE_POINTS)
    return sorted({int(round(v)) for v in vals}) if sp["kind"] == "int" else [float(f"{v:.6g}") for v in vals]


def validation(job, ds_id, run_id, key, param):
    run, ctx, row, pipe, scorer, metric, splits = _setup(ds_id, run_id, key)
    vals = values_for(key, row, param)
    job.progress(step="validation curve", model=row["label"], param=param)
    tr, va = validation_curve(pipe, ctx["X_tr"], ctx["y_tr"], param_name=f"model__{param}", param_range=vals,
                              cv=splits, scoring=scorer)
    (tm, ts), (vm, vs) = _summ(tr, metric), _summ(va, metric)
    out = {"kind": "validation", "model": key, "label": row["label"], "param": param, "values": vals,
           "metric": metric, "metric_label": METRICS[metric][0], "higher_is_better": METRICS[metric][1],
           "train_mean": tm, "train_sd": ts, "val_mean": vm, "val_sd": vs}
    _save(run, f"{key}_validation_{param}", out, job)


def _save(run, name, out, job):
    folder = run / "diagnostics"
    folder.mkdir(exist_ok=True)
    store.write_json(folder / f"{name}.json", out)
    store.write_json(job.dir / "results.json", {"kind": out["kind"] + "_curve", "file": name})


def cached(ds_id, run_id, name):
    try:
        return store.read_json(store.run_dir(ds_id, run_id) / "diagnostics" / f"{name}.json")
    except KeyError:
        return None


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path

    import pandas as pd
    from core import jobs
    from preprocessing import execute
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = np.where(df["a"] + rng.normal(0, .5, n) > 0, "pos", "neg")
    ds = store.create(df, "t.csv")
    r = execute.apply_plan(df, "y", {})
    r["fitted"]["clean_ops"] = []
    store.save_prep(ds, r["csv"], r["fitted"])
    s = train.validate(ds, {"models": ["decision_tree", "logistic_regression"], "folds": 3,
                            "primary_metric": "accuracy", "time_limit_min": 5})
    run = jobs.start(ds, lambda job: train.run(job, ds, s), s, 300)
    jobs.wait(run)

    j = jobs.start(ds, lambda job: learning(job, ds, run, "decision_tree"), {"kind": "learning_curve"}, 300)
    jobs.wait(j)
    lc = cached(ds, run, "decision_tree_learning")
    assert lc["train_rows"][0] < lc["train_rows"][-1] and len(lc["val_mean"]) == 5, lc
    assert all(t == 1.0 for t in lc["train_mean"]), lc["train_mean"]       # a full tree memorises its rows
    assert lc["val_mean"][-1] < 1.0                                         # ...but not the held-out fold
    # the last point trains on all fold-training rows = the run's own CV
    res = store.read_json(store.run_dir(ds, run) / "results.json")
    assert abs(lc["val_mean"][-1] - res["models"]["decision_tree"]["cv"]["accuracy"]["mean"]) < 1e-9

    j = jobs.start(ds, lambda job: validation(job, ds, run, "logistic_regression", "C"), {"kind": "v"}, 300)
    jobs.wait(j)
    vc = cached(ds, run, "logistic_regression_validation_C")
    assert vc["values"][0] == 1e-3 and vc["values"][-1] == 1e3 and len(vc["val_mean"]) == CURVE_POINTS, vc
    try:
        values_for("logistic_regression", models.table("classification")["logistic_regression"], "max_iter")
        raise AssertionError("max_iter has no search range")
    except ValueError:
        pass
    print("diagnostics self-check passed")
