"""Epoch-by-epoch training with live reporting, one adapter per model family.

fit_model(row, model, X, y, X_val, y_val, fit, on_epoch, check, classes)
  X, X_val    already preprocessed (fold rows / held-back rows); X_val may be None
  fit         {"early_stopping": bool, "patience": int} (empty for one-call models)
  on_epoch    on_epoch(epoch, total, train_loss, val_loss, metric_name)
  check       raises jobs.Stopped on cancel / time limit; polled every epoch
returns (fitted model, info) -- info: epochs_trained, best_epoch, total, metric

Hooks verified on the installed versions (spec 2026-09-14): MLP partial_fit loop,
XGBoost / LightGBM / CatBoost callbacks, GradientBoosting monitor (training loss
only -- it cannot score other rows mid-fit), HistGradientBoosting X_val/y_val
(early stopping on our rows, curve only after the fit).
"""

import copy

import numpy as np
from sklearn.metrics import log_loss


class _Stop:
    """Remembers a Stopped raised inside a library callback, so it can be re-raised
    after the library has shut training down cleanly."""

    def __init__(self, check):
        self.check, self.error = check, None

    def poll(self):
        try:
            self.check()
        except Exception as exc:  # noqa: BLE001 -- jobs.Stopped (or anything the check raises)
            self.error = exc
        return self.error is not None

    def reraise(self):
        if self.error is not None:
            raise self.error


def _required(model, name):
    """Round count must be explicit: XGBoost / CatBoost hide their defaults, and
    params.resolve passes the value read from the library (models.py)."""
    value = model.get_params().get(name)
    if value is None:
        raise ValueError(f"{type(model).__name__}.{name} is not set -- build the model through params.resolve.")
    return int(value)


def _early(fit, X_val):
    return bool(fit.get("early_stopping")) and X_val is not None


def fit_model(row, model, X, y, X_val, y_val, fit, on_epoch, check, classes=None):
    family = row["epochs"]
    if family is None:
        check()
        return model.fit(X, y), {}
    return _ADAPTERS[family](model, X, y, X_val, y_val, fit, on_epoch, check, classes)


def _mlp(model, X, y, X_val, y_val, fit, on_epoch, check, classes):
    clf = hasattr(model, "predict_proba")
    total = int(model.max_iter)          # partial_fit trains one epoch per call; max_iter = epochs
    metric = "log loss" if clf else "half squared error"
    tol, patience = model.tol, int(fit.get("patience") or 0)
    best, best_loss, best_epoch, wait, epoch = None, np.inf, None, 0, 0
    for epoch in range(1, total + 1):
        check()
        if clf:
            model.partial_fit(X, y, classes=classes)
        else:
            model.partial_fit(X, y)
        val = None
        if X_val is not None:
            if clf:
                val = float(log_loss(y_val, model.predict_proba(X_val), labels=classes))
            else:
                val = float(0.5 * np.mean((np.asarray(y_val) - model.predict(X_val)) ** 2))
        on_epoch(epoch, total, float(model.loss_), val, metric)
        if _early(fit, X_val):
            if val < best_loss - tol:     # MLP's own tol (published default 1e-4)
                best, best_loss, best_epoch, wait = copy.deepcopy(model), val, epoch, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    info = {"total": total, "metric": metric, "epochs_trained": epoch, "best_epoch": best_epoch}
    return (best if best is not None else model), info


def _xgboost(model, X, y, X_val, y_val, fit, on_epoch, check, classes):
    import xgboost
    total = _required(model, "n_estimators")
    stop = _Stop(check)
    seen = {}

    class Report(xgboost.callback.TrainingCallback):
        def after_iteration(self, bst, epoch, evals_log):
            names = list(evals_log)
            metric = list(evals_log[names[0]])[-1]
            train = evals_log[names[0]][metric][-1]
            val = evals_log[names[1]][metric][-1] if len(names) > 1 else None
            seen["metric"], seen["epoch"] = metric, epoch + 1
            on_epoch(epoch + 1, total, float(train), None if val is None else float(val), metric)
            return stop.poll()

    evals = [(X, y)] + ([(X_val, y_val)] if X_val is not None else [])
    model.set_params(callbacks=[Report()], early_stopping_rounds=int(fit["patience"]) if _early(fit, X_val) else None)
    model.fit(X, y, eval_set=evals, verbose=False)
    stop.reraise()
    model.set_params(callbacks=None)      # the fitted model must pickle without our closure
    best = getattr(model, "best_iteration", None) if _early(fit, X_val) else None
    return model, {"total": total, "metric": seen.get("metric"), "epochs_trained": seen.get("epoch"),
                   "best_epoch": None if best is None else int(best) + 1}


def _lightgbm(model, X, y, X_val, y_val, fit, on_epoch, check, classes):
    import lightgbm
    total = int(model.get_params()["n_estimators"])
    stop = _Stop(check)
    seen = {}

    def report(env):
        res = {name: value for name, _metric, value, _hib in env.evaluation_result_list}
        metric = env.evaluation_result_list[0][1]
        seen["metric"], seen["epoch"] = metric, env.iteration + 1
        on_epoch(env.iteration + 1, total, float(res["training"]), float(res["valid"]) if "valid" in res else None,
                 metric)
        if stop.poll():
            raise lightgbm.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    # "training" is LightGBM's own name for the training set, which its early-stopping
    # callback skips -- so early stopping watches only the held-back rows.
    evals, names = [(X, y)], ["training"]
    if X_val is not None:
        evals.append((X_val, y_val))
        names.append("valid")
    callbacks = [report]
    if _early(fit, X_val):
        callbacks.append(lightgbm.early_stopping(int(fit["patience"]), first_metric_only=True, verbose=False))
    model.fit(X, y, eval_set=evals, eval_names=names, callbacks=callbacks)
    stop.reraise()
    best = model.best_iteration_ if _early(fit, X_val) else None
    return model, {"total": total, "metric": seen.get("metric"), "epochs_trained": seen.get("epoch"),
                   "best_epoch": int(best) if best else None}


def _catboost(model, X, y, X_val, y_val, fit, on_epoch, check, classes):
    total = _required(model, "iterations")
    stop = _Stop(check)
    seen = {}
    gpu = model.get_params().get("task_type") == "GPU"

    class Report:
        def after_iteration(self, info):
            learn = info.metrics.get("learn", {})
            metric = next(iter(learn)) if learn else None
            val = info.metrics.get("validation", {}).get(metric)
            seen["metric"], seen["epoch"] = metric, info.iteration
            on_epoch(info.iteration, total, float(learn[metric][-1]) if metric else None,
                     float(val[-1]) if val else None, metric)
            return not stop.poll()

    early = _early(fit, X_val)
    # with an eval set CatBoost keeps its best iteration by default (use_best_model) --
    # that IS early-stopping selection, so it is only allowed when the user turned it on
    model.set_params(use_best_model=early, **({"early_stopping_rounds": int(fit["patience"])} if early else {}))
    eval_set = (X_val, y_val) if X_val is not None else None
    if gpu:  # "User defined callbacks are not supported for GPU" (verified): replay the curve after the fit
        check()
        model.fit(X, y, eval_set=eval_set)
        log = model.get_evals_result()
        learn = log.get("learn", {})
        metric = next(iter(learn), None)
        val = log.get("validation", {}).get(metric) or []
        for e, train in enumerate(learn.get(metric, []), 1):
            on_epoch(e, total, float(train), float(val[e - 1]) if e <= len(val) else None, metric)
        seen["metric"], seen["epoch"] = metric, len(learn.get(metric, [])) or None
    else:
        model.fit(X, y, eval_set=eval_set, callbacks=[Report()])
        stop.reraise()
    info = {"total": total, "metric": seen.get("metric"), "epochs_trained": seen.get("epoch"),
            "best_epoch": int(model.get_best_iteration()) + 1 if early and model.get_best_iteration() is not None else None,
            "learning_rate_used": float(model.get_all_params()["learning_rate"]),
            "note": "GPU: curve shown after the fit" if gpu else None}
    return model, info


def _gradient_boosting(model, X, y, X_val, y_val, fit, on_epoch, check, classes):
    total = int(model.n_estimators)
    stop = _Stop(check)
    seen = {}

    def monitor(i, est, _locals):
        seen["epoch"] = i + 1
        on_epoch(i + 1, total, float(est.train_score_[i]), None, "training loss")
        return stop.poll()

    model.fit(X, y, monitor=monitor)
    stop.reraise()
    return model, {"total": total, "metric": "training loss", "epochs_trained": seen.get("epoch"),
                   "best_epoch": None}


def _hist_gradient_boosting(model, X, y, X_val, y_val, fit, on_epoch, check, classes):
    total = int(model.max_iter)
    check()
    early = _early(fit, X_val)
    model.set_params(early_stopping=early, **({"n_iter_no_change": int(fit["patience"])} if early else {}))
    if early:
        model.fit(X, y, X_val=X_val, y_val=y_val)
        # scores are negated losses ("loss" scoring): flip them so the chart shows loss;
        # entry 0 is before the first round
        for e in range(1, len(model.train_score_)):
            on_epoch(e, total, float(-model.train_score_[e]), float(-model.validation_score_[e]), "loss")
        best = int(np.argmax(model.validation_score_))
    else:
        model.fit(X, y)
        best = None
    return model, {"total": total, "metric": "loss" if early else None, "epochs_trained": int(model.n_iter_),
                   "best_epoch": best or None, "note": None if early else "curve only with early stopping on"}


_ADAPTERS = {"mlp": _mlp, "xgboost": _xgboost, "lightgbm": _lightgbm, "catboost": _catboost,
             "gradient_boosting": _gradient_boosting, "hist_gradient_boosting": _hist_gradient_boosting}


if __name__ == "__main__":
    import pickle
    import warnings
    from training import models, params
    warnings.filterwarnings("ignore")

    class Stopped(Exception):
        pass

    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 6))
    y_c = (X[:, 0] + 0.8 * rng.normal(size=n) > 0).astype(int)
    y_r = X[:, 0] * 3 + rng.normal(size=n)
    tr, va = slice(0, 700), slice(700, n)

    for task, y in (("classification", y_c), ("regression", y_r)):
        rows = models.table(task)
        for key in ("mlp", "xgboost", "lightgbm", "catboost", "gradient_boosting", "hist_gradient_boosting"):
            row = rows[key]
            big = {"mlp": {"max_iter": 400}, "xgboost": {"n_estimators": 400}, "lightgbm": {"n_estimators": 400},
                   "catboost": {"iterations": 400}, "gradient_boosting": {"n_estimators": 60},
                   "hist_gradient_boosting": {"max_iter": 400}}[key]
            kwargs, fit, _ = params.resolve(row, {**big, "early_stopping": True, "patience": 5}
                                            if row["epochs"] != "gradient_boosting" else big)
            classes = np.array([0, 1]) if task == "classification" else None

            # 1. every epoch is reported; early stopping on held-back rows stops before the end
            log = []
            model, info = fit_model(row, row["make"](0).set_params(**kwargs), X[tr], y[tr], X[va], y[va], fit,
                                    lambda e, t, a, b, m: log.append((e, a, b)), lambda: None, classes)
            assert log and all(e == i + 1 for i, (e, *_rest) in enumerate(log)), (task, key, log[:3])
            if key != "gradient_boosting":
                assert all(b is not None for _e, _a, b in log), (task, key, "missing validation loss")
                assert info["epochs_trained"] < info["total"], (task, key, info)  # stopped early
                assert info["best_epoch"] is not None and info["best_epoch"] <= info["epochs_trained"], (task, key, info)
            model.predict(X[va])
            pickle.dumps(model)  # a fitted model must be savable
            # 2. a cancel inside training stops it within a few epochs and surfaces as Stopped
            # HistGB and CatBoost on GPU have no per-round hook: checked before fitting only
            if key != "hist_gradient_boosting" and not (key == "catboost" and row["device"] == "gpu"):
                calls = {"n": 0}

                def check():
                    calls["n"] += 1
                    if calls["n"] > 3:
                        raise Stopped()

                kwargs_nf, fit_nf, _ = params.resolve(row, big)
                seen = []
                try:
                    fit_model(row, row["make"](0).set_params(**kwargs_nf), X[tr], y[tr], None, None, fit_nf,
                              lambda e, t, a, b, m: seen.append(e), check, classes)
                    raise AssertionError((task, key, "cancel ignored"))
                except Stopped:
                    assert len(seen) <= 5, (task, key, len(seen))

            # 3. no held-back rows -> training loss only, no early stopping
            if key not in ("hist_gradient_boosting",):
                log0 = []
                kwargs_s, fit_s, _ = params.resolve(row, {k: min(v, 20) for k, v in big.items()})
                fit_model(row, row["make"](0).set_params(**kwargs_s), X[tr], y[tr], None, None, fit_s,
                          lambda e, t, a, b, m: log0.append(b), lambda: None, classes)
                assert len(log0) == 20 and all(b is None for b in log0), (task, key, len(log0))
        print(f"epochs self-check passed for {task}")
