"""Hyperparameter search (plan §10; spec 2026-09-13 §6): random or grid search,
each candidate cross-validated on the run's own folds, then the best settings are
trained like any run (all metrics, refit on all training rows, scored once on the
locked test set). The test set never chooses anything.

Search ranges are our starting point -- shown on the page and editable there --
and narrower than the parameter ranges, which only say what the library accepts.
Epoch counts and layer sizes are not searched (early stopping decides epochs).

Known optimism (spec §6): the CV score of the tuned model is slightly optimistic,
because the same folds chose the settings and score them; the locked-test score
is the honest one. Nested CV (folds x the cost) is not built.
"""

import math
import time

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterGrid, ParameterSampler

from core import jobs, store
from training import models, train
from training import params as P
from training.metrics import METRICS, score, scorers
from training.pipeline import build_pipeline, fit_fold

N_TRIALS = 20            # our starting point (spec §8), range below
MAX_TRIALS = 500         # our starting point
MAX_GRID = 500           # our starting point: refuse grids larger than this many combinations
GRID_POINTS = 3          # our starting point: default grid = low, middle, high of each range
LEADERS_K = 3            # our starting point (spec §8): leaders pre-ticked for tuning after a run

# (low, high, log scale) per parameter name; (model, name) overrides; None = not searched.
SEARCH = {
    "C": (1e-3, 1e3, True), "alpha": (1e-4, 1e2, True), "l1_ratio": (0.0, 1.0, False),
    "n_estimators": (50, 500, False), "max_depth": (2, 20, False), "min_samples_leaf": (1, 20, False),
    "learning_rate": (0.01, 0.3, True), "n_neighbors": (1, 50, False), "num_leaves": (8, 128, False),
    "depth": (3, 10, False), "iterations": (100, 1000, False), "var_smoothing": (1e-12, 1e-6, True),
    "learning_rate_init": (1e-4, 1e-1, True),
    # deep-learning models (training/deep.py)
    "hidden": (16, 256, True), "layers": (1, 4, False), "dropout": (0.0, 0.5, False),
}
SEARCH_BY_MODEL = {
    ("multinomial_nb", "alpha"): (0.01, 10.0, True), ("complement_nb", "alpha"): (0.01, 10.0, True),
    ("mlp", "alpha"): (1e-6, 1e-1, True), ("xgboost", "max_depth"): (2, 12, False),
    ("gradient_boosting", "max_depth"): (2, 8, False), ("hist_gradient_boosting", "max_iter"): (50, 500, False),
    ("adaboost", "learning_rate"): (0.01, 2.0, True),
    # Adam step sizes: the boosters' 0.01-0.3 is too large for these networks
    ("lstm", "learning_rate"): (1e-4, 1e-2, True), ("gru", "learning_rate"): (1e-4, 1e-2, True),
    ("cnn1d", "learning_rate"): (1e-4, 1e-2, True), ("tab_transformer", "learning_rate"): (1e-5, 1e-3, True),
    ("tabnet", "learning_rate"): (1e-3, 5e-2, True), ("tabnet", "hidden"): (8, 64, True),
    # convergence / training-length settings, not model choices
    ("logistic_regression", "max_iter"): None, ("lasso", "max_iter"): None, ("elastic_net", "max_iter"): None,
}


def space(key, row):
    """{param: {kind, low, high, log, choices, none}} -- what a search varies by default."""
    out = {}
    key = row.get("base", key)     # a tested pipeline searches its model's ranges (fs_xgboost -> xgboost)
    for p in row["params"]:
        if p["fit"] or p["kind"] in ("bool", "layers"):
            continue
        if p["kind"] == "choice":
            out[p["name"]] = {"kind": "choice", "choices": list(p["choices"])}
            continue
        rng = SEARCH_BY_MODEL.get((key, p["name"]), SEARCH.get(p["name"])) \
            if (key, p["name"]) in SEARCH_BY_MODEL else SEARCH.get(p["name"])
        if rng is None:
            continue
        lo, hi, log = rng
        out[p["name"]] = {"kind": p["kind"], "low": lo, "high": hi, "log": log, "none": p["none"]}
    return out


def _check_space(label, row, user):
    """User edits to the ranges, validated against the parameter specs."""
    specs = {p["name"]: p for p in row["params"]}
    out = {}
    for name, s in (user or {}).items():
        if name not in specs:
            raise ValueError(f"{label}: unknown parameter '{name}'.")
        p = specs[name]
        if p["kind"] == "choice":
            choices = [c for c in s.get("choices", []) if c in p["choices"]]
            if not choices:
                raise ValueError(f"{label} · {p['label']}: choose at least one of {p['choices']}.")
            out[name] = {"kind": "choice", "choices": choices}
            continue
        lo, hi = P._check(label, p, s.get("low")), P._check(label, p, s.get("high"))
        if lo is None or hi is None or lo > hi:
            raise ValueError(f"{label} · {p['label']}: the search range needs low <= high.")
        log = bool(s.get("log", False)) and lo > 0
        out[name] = {"kind": p["kind"], "low": lo, "high": hi, "log": log, "none": p["none"] and s.get("none", False)}
    return out


def _distribution(s):
    if s["kind"] == "choice":
        return s["choices"]
    lo, hi = s["low"], s["high"]
    if s["kind"] == "int":
        if s.get("log") and lo > 0:
            vals = sorted({int(round(v)) for v in np.geomspace(lo, hi, 50)})
            return vals + ([None] if s.get("none") else [])
        return (list(range(int(lo), int(hi) + 1)) + [None]) if s.get("none") else randint(int(lo), int(hi) + 1)
    return loguniform(lo, hi) if s.get("log") else uniform(lo, hi - lo)


def _grid_values(s, points=GRID_POINTS):
    if s["kind"] == "choice":
        return s["choices"]
    lo, hi = s["low"], s["high"]
    vals = np.geomspace(lo, hi, points) if s.get("log") and lo > 0 else np.linspace(lo, hi, points)
    if s["kind"] == "int":
        return sorted({int(round(v)) for v in vals})
    return [float(f"{v:.6g}") for v in vals]


def candidates(sp, method, n_trials, seed, grid=None):
    """Parameter settings to try. grid: optional {param: [values]} replacing the
    default low / middle / high points."""
    if not sp:
        return [{}]
    if method == "grid":
        g = {k: (grid or {}).get(k) or _grid_values(s) for k, s in sp.items()}
        return list(ParameterGrid(g))
    dist = {k: _distribution(s) for k, s in sp.items()}
    total = math.prod(len(d) for d in dist.values()) if all(isinstance(d, list) for d in dist.values()) else None
    n = min(n_trials, total) if total else n_trials   # a small discrete space can't give more distinct trials
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in c.items()}
            for c in ParameterSampler(dist, n_iter=n, random_state=seed)]


def grid_size(sp, grid=None):
    return len(candidates(sp, "grid", 0, 0, grid))


def validate(ds_id, payload):
    """A tune request: the normal run settings plus method, trials and ranges."""
    settings = train.validate(ds_id, payload, "tune")
    method = payload.get("method", "random")
    if method not in ("random", "grid"):
        raise ValueError("Search method must be random or grid.")
    n_trials = int(payload.get("n_trials") or N_TRIALS)
    if not 1 <= n_trials <= MAX_TRIALS:
        raise ValueError(f"Trials must be between 1 and {MAX_TRIALS}.")
    rows = models.table(settings["task"])
    spaces, grids = {}, payload.get("grid") or {}
    for k in settings["models"]:
        sp = {**space(k, rows[k]), **_check_space(rows[k]["label"], rows[k], (payload.get("space") or {}).get(k))}
        spaces[k] = sp
        if method == "grid" and grid_size(sp, grids.get(k)) > MAX_GRID:
            raise ValueError(f"{rows[k]['label']}: the grid has {grid_size(sp, grids.get(k)):,} combinations "
                             f"(max {MAX_GRID}) - narrow it or use random search.")
    settings.update(kind="tune", method=method, n_trials=n_trials, spaces=spaces, grids=grids)
    return settings


def leaders(results, k=LEADERS_K):
    """The exact leader rule (spec §6): rank by CV mean of the primary metric
    (direction-aware), only models that beat the baseline, ties -> smaller sd ->
    shorter fit -> name. The test set is never used."""
    primary = results["primary_metric"]
    higher = METRICS[primary][1]
    base = (results["models"].get("baseline", {}).get("cv") or {}).get(primary, {}).get("mean")
    ranked = []
    for key, rec in results["models"].items():
        r = train._rank(rec, primary, higher)
        m = (rec.get("cv") or {}).get(primary, {}).get("mean")
        beats = base is None or (m is not None and (m > base if higher else m < base))
        if key != "baseline" and "error" not in rec and r is not None and beats:
            ranked.append((r, key))
    return [key for _, key in sorted(ranked)[:k]]


def _cv_trial(ctx, key, row, f, folds, splits, kwargs, fit, sc_primary, primary, share, check):
    pipe = build_pipeline(ctx, key, row, f, folds, kwargs)
    X, y, groups = ctx["X_tr"], ctx["y_tr"], ctx.get("groups_tr")
    vals = []
    for tr, va in splits:
        check()
        fitted, _ = fit_fold(pipe, row, fit, X.iloc[tr], y[tr], share, ctx["seed"], ctx["task"],
                             lambda *a: None, check, np.arange(len(ctx["classes"] or [])) if ctx["classes"] else None,
                             None if groups is None else groups[tr])
        vals.append(score(primary, sc_primary, fitted, X.iloc[va], y[va]))
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)) if vals else None), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def tune(job, ds_id, settings):
    """Search each chosen model, then train the winners as a normal run. Trials are
    written to tuning.json as they finish (the page shows them live) and merged
    into results.json at the end."""
    ctx = train.load_context(ds_id)
    f = train.facts(ctx)
    rows = models.table(ctx["task"])
    folds, primary = settings["folds"], settings["primary_metric"]
    higher = METRICS[primary][1]
    splits = train.cv_splits(ctx, folds)
    pos = ctx["classes"].index(settings["positive_class"]) if settings.get("positive_class") else None
    log = {"method": settings["method"], "primary_metric": primary, "models": {}}
    best_params = {}
    for i, key in enumerate(settings["models"]):
        row, sp = rows[key], settings["spaces"][key]
        base_user = dict((settings.get("params") or {}).get(key) or {})
        cands = candidates(sp, settings["method"], settings["n_trials"], ctx["seed"], settings["grids"].get(key))
        trials, best = [], None
        log["models"][key] = {"label": row["label"], "space": sp, "trials": trials, "best": None}
        for t, cand in enumerate(cands):
            job.check()
            job.progress(step="search", model=row["label"], model_index=i + 1, models=len(settings["models"]),
                         trial=t + 1, trials=len(cands))
            user = {**base_user, **cand}
            t0 = time.time()
            try:
                kwargs, fit, record = P.resolve(row, user)
                probe = build_pipeline(ctx, key, row, f, folds, kwargs)
                sc = scorers(ctx["task"], len(ctx["classes"] or []), pos, hasattr(probe, "predict_proba"),
                             f.get("target_has_zero", False), hasattr(probe, "decision_function"))
                metric = primary if primary in sc else ("r2" if primary == "adj_r2" else None)
                if metric is None:
                    raise ValueError(f"{METRICS[primary][0]} can't be computed for this model")
                mean, sd = _cv_trial(ctx, key, row, f, folds, splits, kwargs, fit, sc[metric], metric,
                                     settings.get("validation_share", train.VALIDATION_SHARE), job.check)
                trial = {"params": {k: record[k] for k in cand}, "cv_mean": mean, "cv_sd": sd,
                         "seconds": round(time.time() - t0, 3)}
            except jobs.Stopped:
                raise
            except Exception as exc:  # noqa: BLE001 -- one bad combination must not end the search
                trial = {"params": cand, "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 3)}
            trials.append(trial)
            m = trial.get("cv_mean")
            if m is not None and (best is None or (m > best["cv_mean"] if higher else m < best["cv_mean"])):
                best = trial
            log["models"][key]["best"] = best
            store.write_json(job.dir / "tuning.json", log)
        best_params[key] = {**base_user, **(best["params"] if best else {})}
    final = {**settings, "params": best_params}
    train.run(job, ds_id, final)
    res = store.read_json(job.dir / "results.json") or {}
    res["tuning"] = log
    store.write_json(job.dir / "results.json", res)


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path

    import pandas as pd
    from preprocessing import execute
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y"], n)})
    df["y"] = np.where(df["a"] * 2 + df["b"] + rng.normal(0, .5, n) > 0, "pos", "neg")
    ds = store.create(df, "t.csv")
    r = execute.apply_plan(df, "y", {"c": [{"op": "encode", "method": "onehot"}]})
    r["fitted"]["clean_ops"] = []
    store.save_prep(ds, r["csv"], r["fitted"])

    rows = models.table("classification")
    sp = space("random_forest", rows["random_forest"])
    assert set(sp) == {"n_estimators", "max_depth", "min_samples_leaf"} and sp["max_depth"]["none"], sp
    assert "max_iter" not in space("logistic_regression", rows["logistic_regression"])
    assert grid_size(sp) == 27                                         # 3 x 3 x 3 default grid
    c1 = candidates(sp, "random", 20, 7)
    assert len(c1) == 20 and c1 == candidates(sp, "random", 20, 7)     # seeded: the same trials every time
    assert all(50 <= c["n_estimators"] <= 500 for c in c1)
    lr_sp = space("logistic_regression", rows["logistic_regression"])
    assert all(1e-3 <= c["C"] <= 1e3 for c in candidates(lr_sp, "random", 50, 0))
    assert len(candidates({"kernel": {"kind": "choice", "choices": ["linear", "rbf"]}}, "random", 20, 0)) == 2
    try:
        validate(ds, {"models": ["random_forest"], "method": "grid", "grid": {"random_forest": {
            "n_estimators": list(range(50, 100)), "max_depth": list(range(2, 20))}}})
        raise AssertionError("an oversized grid must be refused")
    except ValueError as e:
        assert "combinations" in str(e), e

    # a real search: every trial cross-validated, the winner trained like a normal run
    s = validate(ds, {"models": ["logistic_regression", "knn"], "folds": 3, "method": "random", "n_trials": 6,
                      "primary_metric": "accuracy", "time_limit_min": 5,
                      "space": {"knn": {"n_neighbors": {"low": 1, "high": 30}}}})
    run = jobs.start(ds, lambda job: tune(job, ds, s), s, 300)
    jobs.wait(run)
    assert jobs.status(ds, run)["state"] == "done", jobs.status(ds, run)
    res = store.read_json(store.run_dir(ds, run) / "results.json")
    t = res["tuning"]["models"]["knn"]
    assert len(t["trials"]) == 6 and all(1 <= x["params"]["n_neighbors"] <= 30 for x in t["trials"]), t["trials"]
    best = max(t["trials"], key=lambda x: x["cv_mean"])
    assert t["best"] == best and res["models"]["knn"]["params"]["n_neighbors"] == best["params"]["n_neighbors"]
    # the final run's CV on the same folds reproduces the winning trial's CV
    assert abs(res["models"]["knn"]["cv"]["accuracy"]["mean"] - best["cv_mean"]) < 1e-9
    assert "test" in res["models"]["knn"] and res["best"] in ("knn", "logistic_regression")
    assert leaders(res, 1) == [res["best"]]
    print("tuning self-check passed")
