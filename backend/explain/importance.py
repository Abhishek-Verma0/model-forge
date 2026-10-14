"""Explainable AI (plan §12): why a trained model predicts what it does, measured
on the run's locked test rows. Three views, each labelled for what it is:

  permutation  -- how much the run's primary metric drops when one INPUT column is
                  shuffled (scikit-learn permutation_importance on the whole
                  pipeline, so it speaks in the user's original columns). Works
                  for every model.
  native       -- the model's own global importance (tree impurity importance, or
                  |coefficient| for linear models), summed per input column.
  shap         -- SHAP values: each column's contribution to each prediction, in the
                  model's raw output (log-odds for classification, target units for
                  regression). Exact for XGBoost / LightGBM / CatBoost (their built-in
                  TreeSHAP; additivity verified on the installed versions) and for
                  linear models (phi_j = w_j * (x_j - mean_j), background = training
                  rows). Other model types need the `shap` package, which is not
                  installed -- they get permutation importance only, and say so.

Contributions of the features one input column turns into (one-hot categories,
TF-IDF terms) are summed into that column: SHAP values are additive, so a group's
contribution is the sum of its members'. Explanations support interpretation; they
do not prove causation (plan §12).
"""

import time

import joblib
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.inspection import permutation_importance

from core import store
from preprocessing.profiler import is_numeric
from training import train
from training.metrics import METRICS, scorers
from training.pipeline import transform_steps

MAX_ROWS = 2000          # our starting point: test rows used (a sample when the test set is larger)
N_REPEATS = 5            # published default: scikit-learn permutation_importance n_repeats=5
SHAP_CELLS = 20_000_000  # our starting point: rows x features held at once while computing SHAP (~160 MB)
TOP_DOTS = 10            # our starting point: columns in the SHAP dot plot
DOT_ROWS = 300           # our starting point: rows per column in the dot plot
LINEAR = ("logistic_regression", "linear_regression", "ridge", "lasso", "elastic_net")
TREE_SHAP = ("xgboost", "lightgbm", "catboost")
PCA_GROUP = "reduced components (all numeric columns)"


def feature_groups(prep):
    """Input column of every model feature, in the exact order execute.to_matrix
    builds them: numeric / encoded columns (after feature selection and PCA), then
    each text column's word and letter TF-IDF terms."""
    fitted = prep.fitted_
    origin, outputs = {}, []
    for col, replay in fitted["columns"]:
        if col in fitted["text"]:
            continue
        onehot = next((st for o, st, _ in replay if o["op"] == "encode" and o.get("method") == "onehot"), None)
        for out in (onehot or [col]):
            origin[out] = str(col)
            outputs.append(out)
    if fitted["keep"] is not None:
        outputs = [c for c in fitted["keep"] if c in origin]
    if fitted["pca"]:
        n = fitted["pca"].get("n") or fitted["pca"]["pca"].n_components_
        rest = [c for c in outputs if c not in fitted["pca"]["num"]]
        groups = [PCA_GROUP] * n + [origin[c] for c in rest]
    else:
        groups = [origin[c] for c in outputs]
    for col, vecs in fitted["text"].items():
        groups += [str(col)] * sum(len(v.vocabulary_) for v in vecs)
    if len(groups) != prep.n_features_out_:
        raise ValueError(f"feature map has {len(groups)} entries, the model {prep.n_features_out_}")
    return groups


def _indicator(groups):
    """Sparse features x groups 0/1 matrix: contributions @ G = per-group sums."""
    names = list(dict.fromkeys(groups))
    pos = {g: i for i, g in enumerate(names)}
    G = csr_matrix((np.ones(len(groups)), (np.arange(len(groups)), [pos[g] for g in groups])),
                   shape=(len(groups), len(names)))
    return names, G


def _rows(n, seed):
    if n <= MAX_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS, replace=False))


def _native(model, groups):
    """(kind, {input column: share}) from the model's own importance, or (None, {})."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, float)
        kind = ("impurity / gain importance (the model's own) - favours columns with many values or many "
                "derived features, e.g. text (scikit-learn docs); permutation importance does not")
    elif hasattr(model, "coef_"):
        imp = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
        kind = "|coefficient| (comparable only when the features share a scale)"
    else:
        return None, {}
    if len(imp) != len(groups):
        return None, {}
    total = {}
    for g, v in zip(groups, imp):
        total[g] = total.get(g, 0.0) + float(v)
    s = sum(total.values()) or 1.0
    return kind, {g: v / s for g, v in sorted(total.items(), key=lambda kv: -kv[1])}


def _tree_contribs(key, model, X):
    """Per-row SHAP contributions (rows x classes x features+1, bias last) from the
    library's own TreeSHAP; classes = 1 for binary and regression."""
    if key == "xgboost":
        import xgboost
        c = model.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
    elif key == "lightgbm":
        c = model.predict(X, pred_contrib=True)
        c = np.asarray(c.todense()) if issparse(c) else np.asarray(c)
        k = c.shape[1] // (X.shape[1] + 1)
        c = c.reshape(len(c), k, X.shape[1] + 1) if k > 1 else c
    else:  # catboost
        import catboost
        c = model.get_feature_importance(catboost.Pool(X), type="ShapValues")
    c = np.asarray(c, dtype=float)
    return c[:, None, :] if c.ndim == 2 else c


def _linear_contribs(model, X, mean):
    """Exact linear SHAP: w_j * (x_j - mean_j); rows x classes x features (no bias)."""
    W = np.atleast_2d(model.coef_)
    if issparse(X):
        # row-wise w*x stays sparse; subtracting w*mean happens on the group sums
        return [X.multiply(w).tocsr() for w in W], [w * mean for w in W]
    return np.stack([(X - mean) * w for w in W], axis=1), None


def _shap(key, model, X, Xtr_mean, groups, check):
    """Grouped SHAP: rows x classes x input columns, in chunks so wide text data
    never materialises rows x 100k features at once."""
    names, G = _indicator(groups)
    n, F = X.shape
    step = max(1, SHAP_CELLS // max(F, 1))
    parts = []
    for start in range(0, n, step):
        check()
        Xc = X[start:start + step]
        if key in TREE_SHAP:
            c = _tree_contribs(key, model, Xc)[:, :, :F]          # drop the bias column
            parts.append(np.stack([np.asarray(c[:, k, :] @ G) for k in range(c.shape[1])], axis=1))
        else:
            blocks, shift = _linear_contribs(model, Xc, Xtr_mean)
            if shift is None:
                parts.append(np.stack([blocks[:, k, :] @ G for k in range(blocks.shape[1])], axis=1))
            else:
                parts.append(np.stack([(b @ G).toarray() - np.asarray(G.T @ s).ravel()
                                       for b, s in zip(blocks, shift)], axis=1))
    return names, np.concatenate(parts, axis=0)


def explain(job, ds_id, run_id, key):
    """Background job (core/jobs): writes runs/<run_id>/explain/<key>.json."""
    res = store.read_json(store.run_dir(ds_id, run_id) / "results.json") or {}
    rec = (res.get("models") or {}).get(key)
    if not rec or "error" in rec:
        raise LookupError(f"No trained model '{key}' in this run.")
    ctx = train.load_context(ds_id)
    if train.fingerprint(ctx, res["folds"]) != res["fingerprint"]:
        raise ValueError("The data or preprocessing changed since this run - train again to explain it.")
    pipe = joblib.load(store.run_dir(ds_id, run_id) / "models" / f"{key}.joblib")
    seed, task, primary = ctx["seed"], ctx["task"], res["primary_metric"]
    idx = _rows(len(ctx["X_te"]), seed)
    X, y = ctx["X_te"].iloc[idx], ctx["y_te"][idx]
    out = {"model": key, "label": rec["label"], "task": task, "rows": int(len(idx)),
           "test_rows": int(len(ctx["X_te"])), "created": time.time()}

    # 1. permutation importance on the run's primary metric (adjusted R² has no scorer -> R²)
    metric = "r2" if primary == "adj_r2" else primary
    pos = ctx["classes"].index(res["positive_class"]) if res.get("positive_class") else None
    sc = scorers(task, len(ctx["classes"] or []), pos, hasattr(pipe, "predict_proba"),
                 bool((ctx["y_tr"] == 0).any()) if task == "regression" else False)
    if metric not in sc:
        metric = "accuracy" if task == "classification" else "r2"
    job.progress(step="permutation importance", columns=len(X.columns))
    pi = permutation_importance(pipe, X, y, scoring=sc[metric], n_repeats=N_REPEATS, random_state=seed)
    order = np.argsort(-pi.importances_mean)
    out["permutation"] = {"metric": metric, "metric_label": METRICS[metric][0], "repeats": N_REPEATS,
                          "columns": [{"column": str(X.columns[i]), "mean": float(pi.importances_mean[i]),
                                       "sd": float(pi.importances_std[i])} for i in order]}

    # 2. + 3. model-native importance and SHAP, on the model's own input features
    job.check()
    job.progress(step="model importance and SHAP")
    steps = list(pipe.steps[:-1])
    model = pipe.steps[-1][1]
    groups = feature_groups(pipe.named_steps["prep"])
    kind, shares = _native(model, groups)
    out["native"] = {"kind": kind, "columns": [{"column": g, "share": v} for g, v in shares.items()]} if kind else None
    if key in TREE_SHAP or key in LINEAR:
        Xt = transform_steps(steps, X)
        Xtr_mean = None
        if key in LINEAR:
            Xtr = transform_steps(steps, ctx["X_tr"])
            Xtr_mean = np.asarray(Xtr.mean(axis=0)).ravel()
        names, phi = _shap(key, model, Xt, Xtr_mean, groups, job.check)   # rows x classes x columns
        classes = ctx["classes"]
        if phi.shape[1] == 1 and task == "classification" and pos == 0:
            phi = -phi            # one output = log-odds of class index 1; flip to the positive class
        bar = np.abs(phi).mean(axis=(0, 1))     # mean |SHAP| (averaged over classes for multiclass)
        top = list(np.argsort(-bar))
        if phi.shape[1] > 1:   # multiclass: each row explained for the class the model predicted
            pred = np.asarray(pipe.predict(X)).astype(int)
            row_phi = phi[np.arange(len(phi)), pred, :]
            explained = "the class each row was predicted as"
        else:
            row_phi = phi[:, 0, :]
            explained = (f"class {res.get('positive_class') or (classes[1] if classes else '')}"
                         if task == "classification" else "the predicted value")
        dots = []
        pick = np.random.default_rng(seed).permutation(len(row_phi))[:DOT_ROWS]
        for j in top[:TOP_DOTS]:
            col = names[j]
            vals = X[col].iloc[pick] if col in X.columns and is_numeric(X[col]) else None
            if vals is not None:   # colour = the row's value, as a rank 0..1 (high = red, like the shap library)
                vals = vals.rank(pct=True).fillna(-1).round(3).tolist()
            dots.append({"column": col, "shap": np.round(row_phi[pick, j], 5).tolist(), "value_rank": vals})
        out["shap"] = {"method": "TreeSHAP (built into the library)" if key in TREE_SHAP else
                       "exact linear SHAP (features independent, background = training rows)",
                       "units": "log-odds" if task == "classification" else "target units",
                       "explained": explained,
                       "columns": [{"column": names[j], "mean_abs": float(bar[j])} for j in top],
                       "dots": dots}
    else:
        out["shap"] = None
        out["shap_note"] = ("SHAP for this model type needs the shap package (pip install shap), which is not "
                            "installed - permutation importance above works for every model.")
    folder = store.run_dir(ds_id, run_id) / "explain"
    folder.mkdir(exist_ok=True)
    store.write_json(folder / f"{key}.json", out)
    store.write_json(job.dir / "results.json", {"kind": "explain", "run": run_id, "model": key})


def cached(ds_id, run_id, key):
    """A stored explanation, or None. `key` becomes part of a path, so only model
    keys (letters, digits, underscores) are accepted -- never "../"."""
    if not isinstance(key, str) or not key.replace("_", "").isalnum():
        return None
    try:
        return store.read_json(store.run_dir(ds_id, run_id) / "explain" / f"{key}.json")
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
    n = 400
    df = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n),
                       "color": rng.choice(["red", "blue", "green"], n),
                       "note": [f"{'urgent' if i % 4 == 0 else 'fine'} item {i}" for i in range(n)]})
    df["y"] = np.where(df["signal"] + 0.3 * (df["color"] == "red") + rng.normal(0, .3, n) > 0, "yes", "no")

    def prepared(frame, target, task, cols):
        ds = store.create(frame, "t.csv")
        r = execute.apply_plan(frame, target, cols, task=task)
        r["fitted"]["clean_ops"] = []
        store.save_prep(ds, r["csv"], r["fitted"])
        return ds

    def run_and_explain(ds, models, primary):
        s = train.validate(ds, {"models": models, "folds": 3, "primary_metric": primary, "time_limit_min": 5,
                                "params": {"xgboost": {"n_estimators": 60}, "lightgbm": {"n_estimators": 60},
                                           "catboost": {"iterations": 60}}})
        r = jobs.start(ds, lambda job: train.run(job, ds, s), s, 300)
        jobs.wait(r)
        out = {}
        for k in models:
            e = jobs.start(ds, lambda job, k=k: explain(job, ds, r, k), {"kind": "explain"}, 300)
            jobs.wait(e)
            assert jobs.status(ds, e)["state"] == "done", jobs.status(ds, e)
            out[k] = cached(ds, r, k)
        return r, out

    cols = {"color": [{"op": "encode", "method": "onehot"}], "note": [{"op": "encode", "method": "text"}]}
    ds = prepared(df, "y", "classification", cols)
    run, ex = run_and_explain(ds, ["xgboost", "lightgbm", "catboost", "logistic_regression", "random_forest"],
                              "accuracy")
    for k, e in ex.items():
        perm = [c["column"] for c in e["permutation"]["columns"]]
        assert perm[0] == "signal" and set(perm) == {"signal", "noise", "color", "note"}, (k, perm)
        if k == "random_forest":
            assert e["shap"] is None and "shap package" in e["shap_note"]
            # impurity importance summed over thousands of TF-IDF terms ranks the near-noise text
            # column first; permutation (above) ranks the real signal first -- why both are shown
            share = {c["column"]: c["share"] for c in e["native"]["columns"]}
            assert share["note"] > share["signal"] and "favours" in e["native"]["kind"], share
            continue
        assert e["shap"]["columns"][0]["column"] == "signal", (k, e["shap"]["columns"][:3])
        assert {c["column"] for c in e["shap"]["columns"]} == {"signal", "noise", "color", "note"}, k

    # additivity through the grouping: per-row SHAP sums equal the raw model output minus its base value
    ctx = train.load_context(ds)
    pipe = joblib.load(store.run_dir(ds, run) / "models" / "xgboost.joblib")
    Xt = transform_steps(list(pipe.steps[:-1]), ctx["X_te"])
    groups = feature_groups(pipe.named_steps["prep"])
    names, phi = _shap("xgboost", pipe.steps[-1][1], Xt, None, groups, lambda: None)
    import xgboost
    full = pipe.steps[-1][1].get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True)
    margin = pipe.steps[-1][1].get_booster().predict(xgboost.DMatrix(Xt), output_margin=True)
    assert np.allclose(phi[:, 0, :].sum(1) + full[:, -1], margin, atol=1e-3)
    lr = joblib.load(store.run_dir(ds, run) / "models" / "logistic_regression.joblib")
    Xtr = transform_steps(list(lr.steps[:-1]), ctx["X_tr"])
    mean = np.asarray(Xtr.mean(axis=0)).ravel()
    _, lphi = _shap("logistic_regression", lr.steps[-1][1], transform_steps(list(lr.steps[:-1]), ctx["X_te"]),
                    mean, feature_groups(lr.named_steps["prep"]), lambda: None)
    m = lr.steps[-1][1]
    dec = m.decision_function(transform_steps(list(lr.steps[:-1]), ctx["X_te"]))
    assert np.allclose(lphi[:, 0, :].sum(1), dec - (mean @ m.coef_[0] + m.intercept_[0]), atol=1e-6)

    # regression + multiclass run through the same path
    reg = df.assign(y=df["signal"] * 3 + rng.normal(0, .2, n))
    dsr = prepared(reg, "y", "regression", cols)
    _, exr = run_and_explain(dsr, ["lightgbm", "ridge"], "rmse")
    assert exr["ridge"]["shap"]["columns"][0]["column"] == "signal" and exr["ridge"]["shap"]["units"] == "target units"
    multi = df.assign(y=pd.cut(df["signal"], [-9, -.5, .5, 9], labels=["low", "mid", "high"]).astype(str))
    dsm = prepared(multi, "y", "classification", cols)
    _, exm = run_and_explain(dsm, ["catboost"], "accuracy")
    assert exm["catboost"]["shap"]["columns"][0]["column"] == "signal"
    assert exm["catboost"]["shap"]["explained"] == "the class each row was predicted as"
    assert cached(ds, run, "../../meta") is None and cached(ds, run, "xgboost") is not None
    print("explain self-check passed")
