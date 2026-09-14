"""Training runs (spec 2026-09-13 §5, 2026-09-14): every chosen model is
cross-validated with the preprocessing refitted inside each fold, refit on all
training rows and scored once on the locked test split. Each model's parameters
are editable; epoch-based models stream a live curve. Results, epoch curves and
every fitted model land in runs/<run_id>/ as models finish.

"Best" comes only from CV results on the primary metric; suggestions are the
models.suggest rules. The fold-safe pipeline lives in training/pipeline.py,
metrics in training/metrics.py, epoch adapters in training/epochs.py.
"""

import hashlib
import json
import math
import time
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from core import jobs, store
from preprocessing import advanced, execute
from training import models
from training import params as P
from training.metrics import METRICS, TASK_METRICS, adjusted_r2, score, scorers, summary
from training.pipeline import Prep, build_pipeline, fit_fold, input_schema

DEFAULT_FOLDS = 5          # published default (scikit-learn cross_validate cv=5)
MAX_FOLDS = 20             # our starting point: upper bound accepted from the UI
IMBALANCE_CUT = 0.20       # our starting point: smallest class share below this suggests F1 / balanced accuracy
VALIDATION_SHARE = 0.1     # published default (scikit-learn MLP validation_fraction=0.1)
MAX_VALIDATION_SHARE = 0.5  # our starting point
EPOCH_FLUSH_S = 1.0        # our starting point: live epoch file written at most once per second
ESTIMATE_ROWS = 250        # our starting point: Estimate fits each model on 250, then 500 training rows
ESTIMATE_EXP = (1.0, 3.0)  # our starting point: clip range for the measured time-growth exponent
LIBRARIES = ("scikit-learn", "imbalanced-learn", "xgboost", "lightgbm", "catboost", "pandas", "numpy", "scipy")


def versions():
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            pass
    return out


def load_context(ds_id):
    """Everything a run needs, rebuilt from disk as the preprocessing run saw it:
    same rows, same locked split, encoded labels. LookupError = nothing to train
    on yet; ValueError = can't train as it stands."""
    df, prep = store.load_frame(ds_id), store.load_prep(ds_id)
    if df is None:
        raise LookupError("Dataset not found. Re-upload.")
    if prep is None:
        raise LookupError("Run preprocessing first.")
    fitted = prep["fitted"]
    if fitted["clean_ops"] != store.clean_ops(ds_id):
        raise ValueError("The data was cleaned again after preprocessing - run preprocessing again.")
    s, target = fitted["settings"], fitted["target"]
    task = s["task"]
    df, feats, _, notes = execute.prepare_rows(df, target, s["columns"], task)
    X_tr, X_te, y_tr, y_te, _ = execute.split(df[feats], df[target], task, s["test_size"],
                                              s["random_state"], s.get("stratify", True))
    ctx = {"target": target, "task": task, "prep": s, "seed": s["random_state"], "notes": notes,
           "X_tr": X_tr, "X_te": X_te, "classes": None, "clean_ops": fitted["clean_ops"], "features": feats}
    if task == "classification":
        enc = LabelEncoder().fit(df[target].astype(str))
        ctx["classes"] = [str(c) for c in enc.classes_]
        ctx["y_tr"], ctx["y_te"] = enc.transform(y_tr.astype(str)), enc.transform(y_te.astype(str))
    else:
        ctx["y_tr"], ctx["y_te"] = y_tr.to_numpy(dtype=float), y_te.to_numpy(dtype=float)
    return ctx


def fingerprint(ctx, folds):
    """Runs are comparable only when this matches: same cleaning, preprocessing,
    target, locked split and folds."""
    s = ctx["prep"]
    key = {"clean_ops": ctx["clean_ops"], "target": ctx["target"], "folds": folds,  # folds=None: data only
           "prep": {k: s.get(k) for k in ("columns", "pipeline", "task", "test_size", "random_state", "stratify")}}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def facts(ctx):
    """Measured on the preprocessed training rows; decides compatibility and defaults."""
    t0 = time.time()
    m = Prep(ctx["prep"]["columns"], ctx["prep"]["pipeline"], ctx["task"]).fit_transform(ctx["X_tr"], ctx["y_tr"])
    values = m.data if issparse(m) else np.asarray(m)
    f = {"rows": int(m.shape[0]), "features": int(m.shape[1]), "sparse": bool(issparse(m)),
         "nan": bool(np.isnan(values).any()), "negative": bool((values < 0).any()),
         "prep_seconds": round(time.time() - t0, 3)}
    if ctx["task"] == "classification":
        counts = np.bincount(ctx["y_tr"], minlength=len(ctx["classes"]))
        f["class_counts"] = {c: int(n) for c, n in zip(ctx["classes"], counts)}
        f["minority"] = ctx["classes"][int(np.argmin(counts))]
        f["imbalanced"] = bool(counts.min() / counts.sum() < IMBALANCE_CUT)
    else:
        f["target_has_zero"] = bool((ctx["y_tr"] == 0).any() or (ctx["y_te"] == 0).any())
    return f


def options(ds_id):
    ctx = load_context(ds_id)
    f = facts(ctx)
    task, rows = ctx["task"], models.table(ctx["task"])
    picks = models.suggest(task, f, rows)
    binary = task == "classification" and len(ctx["classes"]) == 2
    if task == "regression":
        primary, folds_max = "rmse", min(MAX_FOLDS, f["rows"])
    else:
        primary = ("f1" if binary else "balanced_accuracy") if f["imbalanced"] else "accuracy"
        folds_max = min(MAX_FOLDS, min(f["class_counts"].values()))
    listed = []
    for key, row in rows.items():
        reason = models.incompatible(row, f)
        defaults = P.defaults(row)
        listed.append({"key": key, "label": row["label"], "family": row["family"], "compatible": reason is None,
                       "reason": reason, "suggested": key in picks, "why": picks.get(key), "note": row["note"],
                       "baseline": key == "baseline", "epochs": row["epochs"] is not None, "about": row["about"],
                       "device": row["device"],
                       "params": [{**p, "default": defaults[p["name"]]} for p in row["params"]]})
    return {"task": task, "target": ctx["target"], "classes": ctx["classes"], "notes": ctx["notes"], "facts": f,
            "fingerprint": fingerprint(ctx, None),
            "models": listed,
            "metrics": [{"key": m, "label": METRICS[m][0], "higher_is_better": METRICS[m][1]}
                        for m in TASK_METRICS[task]],
            "defaults": {"primary_metric": primary, "folds": DEFAULT_FOLDS,
                         "time_limit_min": jobs.DEFAULT_TIME_LIMIT_S / 60,
                         "positive_class": f["minority"] if binary else None, "seed": ctx["seed"],
                         "validation_share": VALIDATION_SHARE},
            "limits": {"folds_max": folds_max, "validation_share_max": MAX_VALIDATION_SHARE}}


def validate(ds_id, payload, kind="train"):
    """Check a Train / Estimate request against this dataset's measured facts and
    each model's parameter specs. Returns the settings recorded with the run;
    ValueError messages are for the user."""
    opt = options(ds_id)
    by_key = {m["key"]: m for m in opt["models"]}
    rows = models.table(opt["task"])
    chosen = [k for k in dict.fromkeys(payload.get("models") or []) if k != "baseline"]
    if not chosen:
        raise ValueError("Pick at least one model to train.")
    for k in chosen:
        if k not in by_key:
            raise ValueError(f"Unknown model '{k}'.")
        if not by_key[k]["compatible"]:
            raise ValueError(f"{by_key[k]['label']} can't train on this data: {by_key[k]['reason']}.")
    fmax = opt["limits"]["folds_max"]
    if fmax < 2:
        raise ValueError("Not enough rows per class for cross-validation (every class needs at least 2).")
    folds = int(payload.get("folds") or DEFAULT_FOLDS)
    if not 2 <= folds <= fmax:
        raise ValueError(f"Folds must be between 2 and {fmax} for this data.")
    primary = payload.get("primary_metric") or opt["defaults"]["primary_metric"]
    if primary not in TASK_METRICS[opt["task"]]:
        raise ValueError(f"Unknown metric '{primary}' for {opt['task']}.")
    minutes = float(payload.get("time_limit_min") or opt["defaults"]["time_limit_min"])
    if minutes <= 0:
        raise ValueError("Time limit must be more than 0 minutes.")
    share = payload.get("validation_share")
    share = VALIDATION_SHARE if share is None else float(share)
    if not 0 <= share <= MAX_VALIDATION_SHARE:
        raise ValueError(f"Validation share must be between 0 and {MAX_VALIDATION_SHARE}.")
    positive = None
    if opt["task"] == "classification" and len(opt["classes"]) == 2:
        positive = payload.get("positive_class") or opt["defaults"]["positive_class"]
        if positive not in opt["classes"]:
            raise ValueError(f"Positive class '{positive}' is not a class of {opt['target']}.")
    user_params = payload.get("params") or {}
    chosen_params = {}
    for k in chosen:
        _kwargs, fit, record = P.resolve(rows[k], user_params.get(k))
        if fit.get("early_stopping") and share == 0:
            raise ValueError(f"{rows[k]['label']}: early stopping needs a validation share above 0.")
        chosen_params[k] = record
    return {"kind": kind, "target": opt["target"], "task": opt["task"], "models": chosen, "folds": folds,
            "primary_metric": primary, "positive_class": positive, "time_limit_s": minutes * 60,
            "validation_share": share, "params": chosen_params}


def _rank(rec, metric, higher):
    """Sort key for 'best' (spec §6): primary CV mean, then smaller sd, shorter fit, name."""
    m = rec.get("cv", {}).get(metric)
    if not m or m["mean"] is None:
        return None
    return (-m["mean"] if higher else m["mean"], m["sd"] if m["sd"] is not None else math.inf,
            rec["fit_seconds"], rec["label"])


def _eta(t0, done, total):
    return None if done == 0 else round((time.time() - t0) / done * (total - done), 1)


def _predictions(est, X, y):
    out = {"y_true": np.asarray(y).tolist(), "y_pred": np.asarray(est.predict(X)).tolist()}
    if hasattr(est, "predict_proba"):
        out["proba"] = np.round(est.predict_proba(X), 6).tolist()
    return out


class _EpochLog:
    """Collects one model's epoch curves (one line per fit) and writes them for the
    live chart at most every EPOCH_FLUSH_S, plus at the end of each fit."""

    def __init__(self, job, key):
        self.job, self.path = job, job.dir / "epochs" / f"{key}.json"
        self.data, self.fit, self.last, self.progress = {"metric": None, "total": None, "fits": {}}, None, 0.0, {}

    def start(self, name, progress):
        self.fit, self.progress = name, progress
        self.data["fits"][name] = []

    def __call__(self, epoch, total, train, val, metric):
        self.data["fits"][self.fit].append([epoch, train, val])
        self.data["metric"], self.data["total"] = metric, total
        if time.time() - self.last >= EPOCH_FLUSH_S:
            self.flush(epoch)

    def flush(self, epoch=None):
        self.last = time.time()
        store.write_json(self.path, self.data)
        if epoch is not None:
            self.job.progress(**self.progress, epoch=epoch, epochs=self.data["total"])


def _train_score(primary, sc, est, X, y):
    """Primary metric on the rows the model was fitted on -- the overfitting check.
    One metric only: every extra scorer is another prediction pass over the training
    rows (slow for KNN / SVM). Prediction never drops or resamples rows."""
    if primary == "adj_r2":
        return adjusted_r2(score("r2", sc["r2"], est, X, y), len(y), est.named_steps["prep"].n_features_out_)
    return score(primary, sc[primary], est, X, y) if primary in sc else None


def run(job, ds_id, settings):
    """A training job for jobs.start: baseline + chosen models, CV then locked test.
    results.json is rewritten after every model; every final model is saved to
    models/<key>.joblib as soon as it is fitted, so a cancel keeps finished work."""
    ctx = load_context(ds_id)
    f = facts(ctx)
    task, folds, seed = ctx["task"], settings["folds"], ctx["seed"]
    share = settings.get("validation_share", VALIDATION_SHARE)
    rows = models.table(task)
    X_tr, y_tr, X_te, y_te = ctx["X_tr"], ctx["y_tr"], ctx["X_te"], ctx["y_te"]
    n_classes = len(ctx["classes"] or [])
    classes = np.arange(n_classes) if task == "classification" else None
    pos = ctx["classes"].index(settings["positive_class"]) if settings.get("positive_class") else None
    primary = settings["primary_metric"]
    higher = METRICS[primary][1]
    cv = (StratifiedKFold if task == "classification" else KFold)(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(X_tr, y_tr))
    keys = ["baseline"] + settings["models"]
    imb = advanced._m("imbalance", ctx["prep"]["pipeline"] or {})
    total, done, t0 = len(keys) * (folds + 1), 0, time.time()
    results = {"task": task, "target": ctx["target"], "classes": ctx["classes"],
               "positive_class": settings.get("positive_class"), "primary_metric": primary, "folds": folds,
               "seed": seed, "validation_share": share, "fingerprint": fingerprint(ctx, folds),
               "versions": versions(), "clean_ops": ctx["clean_ops"], "features": ctx["features"],
               "facts": f, "notes": ctx["notes"], "models": {}, "best": None}
    best_rank = None
    for sub in ("predictions", "epochs", "models"):
        (job.dir / sub).mkdir(exist_ok=True)
    for i, key in enumerate(keys):
        row = rows[key]
        rec = {"label": row["label"], "family": row["family"], "baseline": key == "baseline",
               "device": row["device"]}
        fits_before = done
        try:
            kwargs, fit, record = P.resolve(row, settings.get("params", {}).get(key))
            rec["params"] = record
            pipe = build_pipeline(ctx, key, row, f, folds, kwargs)
            if key != "baseline" and imb == "class_weights" and not row["balanced"]:
                rec["note"] = "class weights not supported by this model - trained without them"
            sc = scorers(task, n_classes, pos, hasattr(pipe, "predict_proba"), f.get("target_has_zero", False))
            log = _EpochLog(job, key) if row["epochs"] else None
            fold_scores = {name: [] for name in sc}
            fit_times, n_val, n_feat, train_fold = [], [], [], []
            for k, (tr, va) in enumerate(splits + [(None, None)]):
                job.check()
                progress = {"key": key, "model": row["label"], "model_index": i + 1, "models": len(keys),
                            "fold": k + 1 if va is not None else "final", "folds": folds,
                            "eta_seconds": _eta(t0, done, total)}
                job.progress(**progress)
                if log:
                    log.start(f"fold {k + 1}" if tr is not None else "final", progress)
                rows_X, rows_y = (X_tr.iloc[tr], y_tr[tr]) if tr is not None else (X_tr, y_tr)
                t = time.time()
                fitted, info = fit_fold(pipe, row, fit, rows_X, rows_y, share, seed, task,
                                        log or (lambda *a: None), job.check, classes)
                fit_times.append(time.time() - t)
                if log:
                    log.flush()
                done += 1
                if tr is None:
                    break
                train_fold.append(_train_score(primary, sc, fitted, rows_X, rows_y))
                for name, scorer in sc.items():
                    fold_scores[name].append(score(name, scorer, fitted, X_tr.iloc[va], y_tr[va]))
                n_val.append(len(va))
                n_feat.append(fitted.named_steps["prep"].n_features_out_)
            final = fitted
            if task == "regression":
                fold_scores["adj_r2"] = [adjusted_r2(r, n, p) for r, n, p in zip(fold_scores["r2"], n_val, n_feat)]
            test = {name: score(name, scorer, final, X_te, y_te) for name, scorer in sc.items()}
            if task == "regression":
                test["adj_r2"] = adjusted_r2(test["r2"], len(y_te), final.named_steps["prep"].n_features_out_)
            model_path = job.dir / "models" / f"{key}.joblib"
            joblib.dump(final, model_path)
            rec.update(cv={name: summary(v) for name, v in fold_scores.items()}, test=test,
                       # the primary metric on the rows each model was fitted on: train_cv pairs
                       # with cv (same fold models), train with test (the final model)
                       train_cv={primary: summary(train_fold)},
                       train={primary: _train_score(primary, sc, final, X_tr, y_tr)},
                       fit_seconds=round(float(np.mean(fit_times)), 3), file_mb=store.size_mb(model_path),
                       schema=input_schema(final, X_tr))
            if info:  # epoch-based: what the final fit actually did
                rec["epochs"] = {k: info.get(k) for k in ("total", "epochs_trained", "best_epoch", "metric", "note")}
                if "learning_rate_used" in info:
                    rec["params"] = {**record, "learning_rate_used": info["learning_rate_used"]}
            store.write_json(job.dir / "predictions" / f"{key}.json", _predictions(final, X_te, y_te))
            rank = _rank(rec, primary, higher)
            if key != "baseline" and rank is not None and (best_rank is None or rank < best_rank):
                best_rank, results["best"] = rank, key
        except jobs.Stopped:
            results["models"][key] = {**rec, "error": "stopped before this model finished"}
            store.write_json(job.dir / "results.json", results)
            raise
        except Exception as exc:  # noqa: BLE001 -- one model failing must not end the run
            rec["error"] = f"{type(exc).__name__}: {exc}"
            done = fits_before + folds + 1
        results["models"][key] = rec
        store.write_json(job.dir / "results.json", results)


def estimate(job, ds_id, settings):
    """Optional time estimate (spec §4, 2026-09-14): fit each model -- with the
    chosen parameters -- on two small samples of the preprocessed training rows and
    project to full folds."""
    ctx = load_context(ds_id)
    t = time.time()
    m = Prep(ctx["prep"]["columns"], ctx["prep"]["pipeline"], ctx["task"]).fit_transform(ctx["X_tr"], ctx["y_tr"])
    prep_s = time.time() - t
    n, folds = m.shape[0], settings["folds"]
    fold_rows = n * (folds - 1) / folds
    s2 = min(2 * ESTIMATE_ROWS, n)
    s1 = max(2, s2 // 2)
    order = np.random.default_rng(ctx["seed"]).permutation(n)[:s2]
    rows = models.table(ctx["task"])
    keys = ["baseline"] + settings["models"]
    out = {"kind": "estimate", "prep_seconds": round(prep_s, 3), "sample_rows": [s1, s2], "estimates": {}}
    for i, key in enumerate(keys):
        job.check()
        job.progress(model=rows[key]["label"], model_index=i + 1, models=len(keys))
        try:
            kwargs, _fit, _record = P.resolve(rows[key], settings.get("params", {}).get(key))
            times = []
            for size in (s1, s2):
                idx = order[:size]
                t = time.time()
                rows[key]["make"](ctx["seed"]).set_params(**kwargs).fit(m[idx], ctx["y_tr"][idx]).predict(m[idx])
                times.append(time.time() - t)
            lo, hi = ESTIMATE_EXP
            exp = 1.0 if s2 <= s1 else math.log(max(times[1], 1e-4) / max(times[0], 1e-4)) / math.log(s2 / s1)
            exp = min(max(exp, lo), hi)
            per_fit = times[1] * (fold_rows / s2) ** exp + prep_s * fold_rows / n
            out["estimates"][key] = {"seconds_small": round(times[0], 3), "seconds_large": round(times[1], 3),
                                     "exponent": round(exp, 2), "projected_seconds": per_fit * (folds + 1)}
        except jobs.Stopped:
            raise
        except Exception as exc:  # noqa: BLE001
            out["estimates"][key] = {"error": f"{type(exc).__name__}: {exc}"}
        store.write_json(job.dir / "results.json", out)


def runs(ds_id):
    """Run history for comparison: status + what each run trained, newest first."""
    out = []
    for status in jobs.list_runs(ds_id):
        path = store.run_dir(ds_id, status["run_id"])
        settings = store.read_json(path / "settings.json") or {}
        res = store.read_json(path / "results.json") or {}
        primary = res.get("primary_metric")
        out.append({"run_id": status["run_id"], "state": status.get("state"), "created": status.get("created"),
                    "kind": settings.get("kind"), "primary_metric": primary, "fingerprint": res.get("fingerprint"),
                    "best": res.get("best"), "size_mb": store.size_mb(path),
                    "models": [{"key": k, "label": r["label"], "baseline": r["baseline"], "error": r.get("error"),
                                "cv": (r.get("cv") or {}).get(primary, {}).get("mean"),
                                "test": (r.get("test") or {}).get(primary)}
                               for k, r in (res.get("models") or {}).items()]})
    return out


def discard(ds_id, run_id):
    status = jobs.status(ds_id, run_id)
    if status is None:
        raise LookupError("Run not found.")
    if status.get("state") in ("queued", "running"):
        raise ValueError("Cancel the run before discarding it.")
    store.delete_run(ds_id, run_id)


if __name__ == "__main__":
    import io
    import tempfile
    import warnings
    from pathlib import Path
    import pandas as pd
    from sklearn.base import BaseEstimator, clone
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import cross_validate
    from training.metrics import _NEGATED
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 240
    words = ["great", "awful", "fast", "broken", "love", "refund", "cheap", "solid"]
    label = rng.choice(["ham", "spam", "promo"], n, p=[.65, .25, .10])
    df = pd.DataFrame({
        "msg": [f"{words[rng.integers(8)]} {'win free prize' if c == 'spam' else 'big sale' if c == 'promo' else 'see you'} {i}"
                for i, c in enumerate(label)],
        "age": np.where(rng.random(n) < .1, np.nan, rng.integers(18, 80, n)),
        "amount": rng.normal(100, 30, n), "label": label})
    df.loc[7, "amount"] = 10_000

    def preprocess(frame, target, task, columns, pipeline=None):
        ds_id = store.create(frame, "t.csv")
        r = execute.apply_plan(frame, target, columns, task=task, pipeline=pipeline)
        fitted = r["fitted"]
        fitted["clean_ops"] = store.clean_ops(ds_id)
        store.save_prep(ds_id, r["csv"], fitted)
        return ds_id

    def results_of(ds_id, fn, settings, state="done"):
        r = jobs.start(ds_id, fn, settings, settings["time_limit_s"])
        jobs.wait(r)
        assert jobs.status(ds_id, r)["state"] == state, jobs.status(ds_id, r)
        return r, store.read_json(store.run_dir(ds_id, r) / "results.json")

    cols = {"msg": [{"op": "encode", "method": "text"}], "age": [{"op": "impute", "strategy": "median"}],
            "amount": [{"op": "outliers", "method": "remove_rows", "k": 1.5}, {"op": "scale", "method": "robust"}]}
    ds = preprocess(df, "label", "classification", cols, {"imbalance": {"method": "oversample"}})

    # 1. options: facts measured; compatibility, suggestions and parameter defaults follow from them
    o = options(ds)
    m = {x["key"]: x for x in o["models"]}
    assert o["facts"]["sparse"] and o["facts"]["negative"] and not o["facts"]["nan"], o["facts"]
    assert not m["gaussian_nb"]["compatible"] and "dense" in m["gaussian_nb"]["reason"]
    assert not m["multinomial_nb"]["compatible"] and "non-negative" in m["multinomial_nb"]["reason"]
    assert m["logistic_regression"]["suggested"] and m["random_forest"]["suggested"] and not m["svm"]["suggested"]
    assert o["defaults"]["primary_metric"] == "balanced_accuracy" and o["defaults"]["positive_class"] is None, o["defaults"]
    mlp_defaults = {p["name"]: p["default"] for p in m["mlp"]["params"]}
    assert m["mlp"]["epochs"] and mlp_defaults["max_iter"] == 200 and mlp_defaults["early_stopping"] is False
    json.dumps(o)

    # 2. manual fold fitting: prep never sees scored rows, resampling only in fit, == sklearn cross_validate
    ctx = load_context(ds)
    f = facts(ctx)
    lr_row = models.table("classification")["logistic_regression"]
    pipe = build_pipeline(ctx, "logistic_regression", lr_row, f, 5)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=ctx["seed"]).split(ctx["X_tr"], ctx["y_tr"]))
    sc = scorers("classification", 3, None, True, False)
    ours = {k: [] for k in sc}
    for tr, va in splits:
        p, _ = fit_fold(pipe, lr_row, {}, ctx["X_tr"].iloc[tr], ctx["y_tr"][tr], 0.1, ctx["seed"],
                        "classification", lambda *a: None, lambda: None)
        assert not set(p.named_steps["prep"].seen_index_) & set(ctx["X_tr"].index[va]), "prep saw scored rows"
        for k, s in sc.items():
            ours[k].append(score(k, s, p, ctx["X_tr"].iloc[va], ctx["y_tr"][va]))
    ref = cross_validate(pipe, ctx["X_tr"], ctx["y_tr"], cv=splits, scoring=sc)
    for k in sc:
        expect = -ref[f"test_{k}"] if k in _NEGATED else ref[f"test_{k}"]
        assert np.allclose(ours[k], expect), (k, ours[k], expect)

    class Count(BaseEstimator):  # records how many rows reach the model
        def fit(self, X, y):
            self.n_, self.classes_ = X.shape[0], np.unique(y)
            return self

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    tr0 = splits[0][0]
    counted = clone(pipe).set_params(model=Count()).fit(ctx["X_tr"].iloc[tr0], ctx["y_tr"][tr0])
    assert counted.named_steps["model"].n_ > len(tr0), "oversampling must add fold-training rows"
    assert len(counted.predict(ctx["X_te"])) == len(ctx["X_te"]), "no rows dropped when predicting"

    # epoch models: held-back rows are raw fold-training rows, never seen by prep or resampling
    mlp_row = models.table("classification")["mlp"]
    kw, fit_opts, _ = P.resolve(mlp_row, {"max_iter": 30, "hidden_layer_sizes": "16", "early_stopping": True})
    mpipe = build_pipeline(ctx, "mlp", mlp_row, f, 5, kw)
    p, info = fit_fold(mpipe, mlp_row, fit_opts, ctx["X_tr"].iloc[tr0], ctx["y_tr"][tr0], 0.2, ctx["seed"],
                       "classification", lambda *a: None, lambda: None, np.arange(3))
    seen = len(p.named_steps["prep"].seen_index_)
    # persistence: a fitted pipeline written to disk and read back predicts identically
    buf = io.BytesIO()
    joblib.dump(p, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert np.array_equal(restored.predict_proba(ctx["X_te"]), p.predict_proba(ctx["X_te"])), "reloaded model differs"
    held_back = math.ceil(len(tr0) * 0.2)  # train_test_split rounds the held-back count up
    assert seen <= len(tr0) - held_back, (seen, len(tr0), held_back)  # IQR row removal may drop a few more
    assert info["epochs_trained"] <= 30 and info["metric"] == "log loss", info

    # 3. a real run: baseline + LR + RF + MLP with early stopping; results, epoch curves, every model on disk
    settings = validate(ds, {"models": ["logistic_regression", "random_forest", "mlp"], "folds": 3,
                             "primary_metric": "balanced_accuracy", "time_limit_min": 5, "validation_share": 0.2,
                             "params": {"random_forest": {"n_estimators": 50},
                                        "mlp": {"max_iter": 60, "hidden_layer_sizes": "32", "early_stopping": True,
                                                "patience": 5}}})
    r1, res = results_of(ds, lambda job: run(job, ds, settings), settings)
    assert list(res["models"]) == ["baseline", "logistic_regression", "random_forest", "mlp"], list(res["models"])
    assert res["best"] in ("logistic_regression", "random_forest", "mlp"), res["best"]
    lr = res["models"]["logistic_regression"]
    assert len(lr["cv"]["balanced_accuracy"]["folds"]) == 3
    assert lr["test"]["balanced_accuracy"] > res["models"]["baseline"]["test"]["balanced_accuracy"]
    assert res["models"]["random_forest"]["params"]["n_estimators"] == 50
    mlp = res["models"]["mlp"]
    assert mlp["params"]["hidden_layer_sizes"] == [32] and mlp["epochs"]["epochs_trained"] <= 60, mlp
    curves = store.read_json(store.run_dir(ds, r1) / "epochs" / "mlp.json")
    assert list(curves["fits"]) == ["fold 1", "fold 2", "fold 3", "final"], list(curves["fits"])
    assert all(len(v) >= 1 and v[0][2] is not None for v in curves["fits"].values()), "validation loss missing"
    assert res["fingerprint"] == fingerprint(ctx, 3) and res["versions"]["scikit-learn"]
    prim = res["primary_metric"]
    assert prim == "balanced_accuracy", prim  # the manual check below computes this metric
    from sklearn.metrics import balanced_accuracy_score
    for key in res["models"]:
        model = joblib.load(store.run_dir(ds, r1) / "models" / f"{key}.joblib")
        assert len(model.predict(ctx["X_te"])) == len(ctx["X_te"]), key
        # training score = the saved final model scored on every training row, nothing dropped
        m = res["models"][key]
        assert np.isclose(m["train"][prim], balanced_accuracy_score(ctx["y_tr"], model.predict(ctx["X_tr"]))), key
        assert len(m["train_cv"][prim]["folds"]) == 3, key
    # on pure-noise labels a fully grown tree memorises its rows: train score 1.0, CV near chance
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    noise_X, noise_y = np.random.default_rng(0).normal(size=(300, 5)), np.random.default_rng(1).integers(0, 2, 300)
    memoriser = DecisionTreeClassifier(random_state=0).fit(noise_X, noise_y)
    assert _train_score("accuracy", {"accuracy": "accuracy"}, memoriser, noise_X, noise_y) == 1.0
    assert cross_val_score(DecisionTreeClassifier(random_state=0), noise_X, noise_y, cv=3).mean() < 0.65
    listed = runs(ds)
    assert listed[0]["run_id"] == r1 and listed[0]["size_mb"] > 0 and len(listed[0]["models"]) == 4, listed[0]

    # 4. same seed + settings -> identical CV table and fingerprint
    r2, res2 = results_of(ds, lambda job: run(job, ds, settings), settings)
    assert {k: v["cv"] for k, v in res["models"].items()} == {k: v["cv"] for k, v in res2["models"].items()}
    assert res2["fingerprint"] == res["fingerprint"]
    assert {k: v.get("test") for k, v in res["models"].items()} == {k: v.get("test") for k, v in res2["models"].items()}
    p1 = joblib.load(store.run_dir(ds, r1) / "models" / "mlp.joblib").predict_proba(ctx["X_te"])
    p2 = joblib.load(store.run_dir(ds, r2) / "models" / "mlp.joblib").predict_proba(ctx["X_te"])
    assert np.array_equal(p1, p2), "same seed + settings must give identical fitted models"
    discard(ds, r2)
    assert r2 not in [x["run_id"] for x in runs(ds)]

    # 5. clear refusals
    for bad, msg in (({"models": []}, "at least one"), ({"models": ["gaussian_nb"]}, "dense"),
                     ({"models": ["logistic_regression"], "folds": 999}, "Folds"),
                     ({"models": ["mlp"], "params": {"mlp": {"max_iter": 0}}}, "between 1 and 10000"),
                     ({"models": ["mlp"], "validation_share": 0, "params": {"mlp": {"early_stopping": True}}},
                      "validation share above 0"),
                     ({"models": ["mlp"], "validation_share": 0.9}, "Validation share")):
        try:
            validate(ds, bad)
            raise AssertionError(bad)
        except ValueError as e:
            assert msg in str(e), (bad, e)

    # 6. a cancel during a long epoch loop stops within seconds and keeps finished models
    slow = validate(ds, {"models": ["mlp"], "folds": 2, "time_limit_min": 5,
                         "params": {"mlp": {"max_iter": 10000, "hidden_layer_sizes": "8"}}})
    rs = jobs.start(ds, lambda job: run(job, ds, slow), slow, slow["time_limit_s"])
    t_wait = time.time()
    while (jobs.status(ds, rs).get("progress") or {}).get("epoch") is None and time.time() - t_wait < 60:
        time.sleep(0.05)
    t_cancel = time.time()
    assert jobs.cancel(ds, rs)
    jobs.wait(rs)
    assert jobs.status(ds, rs)["state"] == "cancelled" and time.time() - t_cancel < 5, jobs.status(ds, rs)
    stopped = store.read_json(store.run_dir(ds, rs) / "results.json")
    assert "cv" in stopped["models"]["baseline"] and "error" in stopped["models"]["mlp"], stopped["models"]

    # 7. binary: positive class = minority, PR-AUC is for that class (index 0 here), SVM gets AUC without proba
    bin_df = df.assign(label=np.where(label == "spam", "spam", "zz_other"))
    ds_b = preprocess(bin_df, "label", "classification", cols)
    ob = options(ds_b)
    assert ob["defaults"]["positive_class"] == "spam" and ob["defaults"]["primary_metric"] == "accuracy", ob["defaults"]
    ctx_b = load_context(ds_b)
    pos = ctx_b["classes"].index("spam")
    assert pos == 0
    svm_pipe = build_pipeline(ctx_b, "svm", models.table("classification")["svm"], facts(ctx_b), 5)
    svm_pipe.fit(ctx_b["X_tr"], ctx_b["y_tr"])
    sc_b = scorers("classification", 2, pos, False, False)
    assert "log_loss" not in sc_b and "roc_auc" in sc_b
    got = score("pr_auc", sc_b["pr_auc"], svm_pipe, ctx_b["X_te"], ctx_b["y_te"])
    d = svm_pipe.decision_function(ctx_b["X_te"])
    want = average_precision_score(ctx_b["y_te"] == pos, -d)
    assert abs(got - want) < 1e-12, (got, want)
    assert "roc_auc" not in scorers("classification", 3, None, False, False)

    # 8. regression: error metrics positive, MAPE n/a when the target has a zero; LightGBM epochs
    reg = df.drop(columns="label").assign(y=np.r_[0.0, rng.normal(50, 10, n - 1)])
    ds_r = preprocess(reg, "y", "regression", {"msg": [{"op": "encode", "method": "text"}],
                                               "age": [{"op": "impute", "strategy": "median"}]})
    s_r = validate(ds_r, {"models": ["ridge", "lightgbm"], "folds": 3, "primary_metric": "rmse", "time_limit_min": 5,
                          "params": {"lightgbm": {"n_estimators": 200, "early_stopping": True, "patience": 5}}})
    r_r, rres = results_of(ds_r, lambda job: run(job, ds_r, s_r), s_r)
    ridge = rres["models"]["ridge"]
    assert ridge["cv"]["rmse"]["mean"] > 0 and "mape" not in ridge["cv"] and "adj_r2" in ridge["cv"], ridge["cv"].keys()
    lg = rres["models"]["lightgbm"]
    assert lg["epochs"]["epochs_trained"] < 200 and lg["epochs"]["metric"] == "l2", lg["epochs"]
    assert rres["best"] in ("ridge", "lightgbm")

    # 9. estimate projects a time per model, with the chosen parameters
    s_e = validate(ds, {"models": ["logistic_regression"], "folds": 3, "time_limit_min": 5}, kind="estimate")
    _, est = results_of(ds, lambda job: estimate(job, ds, s_e), s_e)
    assert est["estimates"]["logistic_regression"]["projected_seconds"] > 0 and "baseline" in est["estimates"], est

    # 10. data cleaned again after preprocessing -> refuse instead of training on a mismatch
    store.add_clean_ops(ds, [{"op": "trim_whitespace", "column": "msg"}])
    try:
        load_context(ds)
        raise AssertionError("stale preprocessing accepted")
    except ValueError as e:
        assert "preprocessing again" in str(e), e
    print("train self-check passed")
