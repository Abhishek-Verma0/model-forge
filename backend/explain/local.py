"""One prediction explained (plan §12: "individual prediction explanations", LIME) --
for one locked test row of a run model:

  shap  the exact SHAP contributions of each input column (the same methods as the
        global view: booster / scikit-learn TreeSHAP, linear SHAP) where the model
        type has one.
  lime  a local surrogate for EVERY model (Ribeiro, Singh & Guestrin 2016): many
        copies of the row with some columns replaced by a random training row's
        values, the model's output on each, and a weighted ridge regression of that
        output on "this column kept its value" (1/0). A column's weight = how much the
        output moves when it keeps this row's value instead of a typical one.
        Columns are switched on/off as whole values -- how LIME treats words of a text
        -- so number, category and text columns are handled alike. Published LIME
        defaults: 5000 samples, kernel sqrt(exp(-d^2 / w^2)) with w = 0.75 * sqrt(columns),
        Ridge(alpha=1); the surrogate's weighted R^2 says how well it fits locally.

Explanations support interpretation; they do not prove causation (plan §12)."""

import json

import joblib
import numpy as np

from core import store
from explain import importance
from training import train

LIME_SAMPLES = 5000        # published default: lime explain_instance num_samples=5000
LIME_KERNEL = 0.75         # published default: lime kernel_width = 0.75 * sqrt(number of features)
LIME_ALPHA = 1.0           # published default: lime's surrogate Ridge(alpha=1)
TOP = 12                   # our starting point: columns listed per explanation


def _output(pipe, task, classes, ci):
    """The number explained and its units: the probability of class index ci, else a
    decision score or a 0/1 vote (models without probabilities); the predicted value
    for regression."""
    if task != "classification":
        return lambda Z: np.asarray(pipe.predict(Z), float), "predicted value (target units)"
    name = classes[ci]
    if hasattr(pipe, "predict_proba"):
        return lambda Z: pipe.predict_proba(Z)[:, ci], f"probability of {name}"
    if hasattr(pipe, "decision_function") and len(classes) == 2:
        sign = 1.0 if ci == 1 else -1.0
        return lambda Z: sign * np.asarray(pipe.decision_function(Z), float), f"decision score for {name}"
    return lambda Z: (np.asarray(pipe.predict(Z)) == ci).astype(float), f"1 if predicted {name}, else 0"


def lime(f, X_train, x, seed, samples=LIME_SAMPLES):
    """Weighted ridge surrogate around row x (a 1-row frame) -> {intercept, score, columns}."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    cols, d = list(x.columns), x.shape[1]
    replaced = rng.integers(1, d + 1, samples)                      # how many columns to replace (lime's text way)
    keep = np.argsort(rng.random((samples, d)), axis=1) >= replaced[:, None]
    keep[0] = True                                                  # sample 0 = the row itself
    Z = X_train.iloc[rng.integers(0, len(X_train), samples)].reset_index(drop=True)
    for j, c in enumerate(cols):
        Z[c] = Z[c].where(~keep[:, j], x[c].iloc[0])
    y = f(Z)
    dist = np.sqrt((~keep).sum(axis=1))                             # euclidean distance to all-kept
    width = LIME_KERNEL * np.sqrt(d)
    w = np.sqrt(np.exp(-dist ** 2 / width ** 2))
    model = Ridge(alpha=LIME_ALPHA).fit(keep.astype(float), y, sample_weight=w)
    order = np.argsort(-np.abs(model.coef_))
    return {"samples": samples, "intercept": float(model.intercept_),
            "local_prediction": float(model.intercept_ + model.coef_.sum()), "model_output": float(y[0]),
            "score": float(model.score(keep.astype(float), y, sample_weight=w)),
            "columns": [{"column": str(cols[i]), "weight": float(model.coef_[i])} for i in order]}


def explain_row(ds_id, run_id, key, row):
    """Explain the model's prediction for locked test row `row` (0-based)."""
    if not isinstance(key, str) or not key.replace("_", "").isalnum():   # checked first: it becomes a path
        raise LookupError(f"No trained model '{key}' in this run.")
    res = store.read_json(store.run_dir(ds_id, run_id) / "results.json") or {}
    rec = (res.get("models") or {}).get(key)
    if not rec or "error" in rec:
        raise LookupError(f"No trained model '{key}' in this run.")
    ctx = train.load_context(ds_id)
    if train.fingerprint(ctx, res["folds"]) != res["fingerprint"]:
        raise ValueError("The data or preprocessing changed since this run - train again to explain it.")
    X_te = ctx["X_te"]
    if not isinstance(row, int) or not 0 <= row < len(X_te):
        raise ValueError(f"Row must be between 1 and {len(X_te)}.")
    pipe = joblib.load(store.run_dir(ds_id, run_id) / "models" / f"{key}.joblib")
    task, classes, seed = ctx["task"], ctx["classes"] or [], ctx["seed"]
    x = X_te.iloc[[row]]
    pred = pipe.predict(x)[0]
    out = {"model": key, "label": rec["label"], "task": task, "row": row, "rows": int(len(X_te)),
           "values": json.loads(x.to_json(orient="records", date_format="iso", double_precision=15))[0]}
    if task == "classification":
        pos = classes.index(res["positive_class"]) if res.get("positive_class") else None
        ci = pos if len(classes) == 2 and pos is not None else int(pred)
        out.update(actual=classes[int(ctx["y_te"][row])], predicted=classes[int(pred)], explained=classes[ci])
        if hasattr(pipe, "predict_proba"):
            out["probabilities"] = {classes[i]: float(p) for i, p in enumerate(pipe.predict_proba(x)[0])}
    else:
        ci = 0
        out.update(actual=float(ctx["y_te"][row]), predicted=float(pred), explained="the predicted value")
    f, units = _output(pipe, task, classes, ci)

    model = pipe.steps[-1][1]
    kind, base = importance.shap_kind(key, task, model)
    out["shap"] = None
    if kind:
        names, phi, _, shap_units = importance.shap_rows(kind, base, pipe, x, ctx)
        k = ci if phi.shape[1] > 1 else 0
        vals = phi[0, k]
        if phi.shape[1] == 1 and task == "classification" and ci == 0:
            vals = -vals                       # one output = class index 1; flip for class index 0
        order = np.argsort(-np.abs(vals))
        out["shap"] = {"method": importance.SHAP_METHOD[kind], "units": shap_units,
                       "columns": [{"column": names[i], "value": float(vals[i])} for i in order[:TOP]]}
    lm = lime(f, ctx["X_tr"], x, seed)
    lm["columns"] = lm["columns"][:TOP]
    out["lime"] = {**lm, "units": units}
    return out


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
    df = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n),
                       "color": rng.choice(["red", "blue"], n),
                       "note": [f"{'urgent' if i % 4 == 0 else 'fine'} item {i}" for i in range(n)]})
    df["y"] = np.where(df["signal"] + rng.normal(0, .3, n) > 0, "yes", "no")
    ds = store.create(df, "t.csv")
    r = execute.apply_plan(df, "y", {"color": [{"op": "encode", "method": "onehot"}],
                                     "note": [{"op": "encode", "method": "text"}]}, task="classification")
    r["fitted"]["clean_ops"] = []
    store.save_prep(ds, r["csv"], r["fitted"])
    s = train.validate(ds, {"models": ["random_forest", "svm", "logistic_regression"], "folds": 3,
                            "primary_metric": "accuracy", "time_limit_min": 5})
    run = jobs.start(ds, lambda job: train.run(job, ds, s), s, 300)
    jobs.wait(run)
    ctx = train.load_context(ds)
    big = int(np.argmax(np.abs(ctx["X_te"]["signal"].to_numpy())))     # a row the signal decides
    for key in ("random_forest", "svm", "logistic_regression"):
        e = explain_row(ds, run, key, big)
        lime_top = [c["column"] for c in e["lime"]["columns"][:2]]
        assert e["predicted"] in ("yes", "no") and np.isclose(e["values"]["signal"], ctx["X_te"]["signal"].iloc[big])
        if key == "svm":     # no SHAP here: LIME alone, and it finds the signal
            assert e["shap"] is None and "signal" in lime_top, (key, e["lime"]["columns"][:3])
            continue
        # LIME (a sampled surrogate) and exact SHAP agree on what drives this row. The forest leans on
        # the memorised text for it (both say so: the per-row "item <i>" note), the linear model on the signal.
        assert e["shap"]["columns"][0]["column"] == e["lime"]["columns"][0]["column"], (key, e["shap"], e["lime"])
        assert np.sign(e["shap"]["columns"][0]["value"]) == np.sign(e["lime"]["columns"][0]["weight"]), key
    e = explain_row(ds, run, "logistic_regression", big)
    assert e["shap"]["columns"][0]["column"] == "signal" and e["lime"]["score"] > 0.2, e["lime"]
    assert abs(e["lime"]["local_prediction"] - e["lime"]["model_output"]) < 0.1, e["lime"]
    for bad in (-1, 10 ** 6):
        try:
            explain_row(ds, run, "random_forest", bad)
            raise AssertionError("bad row accepted")
        except ValueError:
            pass
    for bad_key in ("../x", ["random_forest"], None):
        try:
            explain_row(ds, run, bad_key, 0)
            raise AssertionError(f"bad key accepted: {bad_key!r}")
        except LookupError:
            pass
    print("local explanation self-check passed")
