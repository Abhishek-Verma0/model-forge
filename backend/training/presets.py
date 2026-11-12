"""Controlled hybrid pipelines (plan §9.6): one fixed preprocessing step + one model,
offered as named rows of the model table -- instead of letting any step meet any
model ("tested pipelines instead of arbitrary combinations").

A preset sets ONE section of the run's dataset-level preprocessing for its own row
(training/pipeline.build_pipeline merges it over the run's settings, replacing that
section only); cleaning, column steps, split, folds and metrics stay the run's. Like
every step, it is refitted inside each CV fold, so a held-out fold never shapes it.
The model and its editable parameters are the base model's row, unchanged.

Why these pairs (each is the plan's §9.6 list):
  PCA + SVM          SVM distances are dominated by whichever columns have the largest
                     numbers -- columns are standardised, then PCA keeps the components
                     that explain most of the variance (decorrelated, fewer inputs).
  SMOTE + forest     synthetic minority rows (training folds only) for an imbalanced
                     target, with a model that is not distance-based itself.
  Feature selection  mutual information (catches non-linear links, as boosted trees
  + XGBoost          use them) keeps the better half of the numeric features.
  Autoencoder +      a small network compresses the standardised columns; a linear model
  logistic reg.      on its units shows whether the learned code separates the classes.
"Optimization algorithm + classifier" is the plan's future research extension.

The self-check trains every preset through the real run code and measures that its
step actually happened (components, units, kept columns, synthetic rows).
"""

VARIANCE_KEPT = 0.95   # our starting point: PCA / autoencoder size = components explaining 95% of the variance
FS_SHARE = 0.5         # our starting point: feature selection keeps the better half of the numeric features

_PRESETS = {
    "pca_svm": {
        "tasks": ("classification", "regression"), "model": "svm", "label": "PCA + {model}",
        "step": ("reduction", {"method": "pca", "n": VARIANCE_KEPT, "scale": True}), "needs_numeric": 2,
        "what": "Standardise the number columns, keep the principal components that explain "
                f"{VARIANCE_KEPT:.0%} of their variance, then train {{model}} on them.",
        "good": "SVMs measure distances; decorrelated, equally-scaled inputs keep one large column from "
                "dominating, and fewer inputs train faster.",
        "watch": "Components mix all columns, so explanations are per component, not per original column."},
    "smote_random_forest": {
        "tasks": ("classification",), "model": "random_forest", "label": "SMOTE + {model}",
        "step": ("imbalance", {"method": "smote"}), "needs_numeric": 0,
        "what": "Add synthetic rows of the smaller classes (SMOTE, training folds only), then train {model}.",
        "good": "Helps the forest see enough of a rare class; test rows are never synthetic.",
        "watch": "Synthetic rows are interpolations of real ones - judge it on recall / F1 / balanced accuracy."},
    "fs_xgboost": {
        "tasks": ("classification", "regression"), "model": "xgboost", "label": "Feature selection + {model}",
        "step": ("feature_selection", {"method": "mutual_info", "k": FS_SHARE}), "needs_numeric": 2,
        "what": f"Keep the {FS_SHARE:.0%} of number columns with the highest mutual information with the "
                "target (chosen inside each training fold), then train {model}.",
        "good": "Drops columns that carry little information about the target - fewer inputs, less noise.",
        "watch": "Mutual information scores one column at a time: a column useful only together with "
                 "another may be dropped."},
    "autoencoder_logreg": {
        "tasks": ("classification",), "model": "logistic_regression", "label": "Autoencoder + {model}",
        "step": ("reduction", {"method": "autoencoder", "n": VARIANCE_KEPT, "scale": True}), "needs_numeric": 2,
        "what": "Standardise the number columns, compress them with a one-layer autoencoder (as many units as "
                f"principal components for {VARIANCE_KEPT:.0%} of the variance), then train {{model}} on the units.",
        "good": "Tests whether a learned, non-linear compression of the inputs separates the classes.",
        "watch": "Advanced option: the code is learned without the target, so it can discard what predicts it."},
}


def rows(task, base):
    """Preset rows for this task, built on the base model rows (skipped when the base
    model is not installed, e.g. XGBoost)."""
    out = {}
    for key, p in _PRESETS.items():
        if task not in p["tasks"] or p["model"] not in base:
            continue
        b = base[p["model"]]
        name = b["label"]
        section, settings = p["step"]
        # nan=False: none of the steps (PCA, autoencoder, SMOTE, mutual information) accepts missing values,
        # even when the base model does (forest, XGBoost) -- the table then asks for an impute step
        out[key] = {**b, "nan": False, "label": p["label"].format(model=name), "family": "Tested pipeline",
                    "pipeline": {section: dict(settings)}, "needs_numeric": p["needs_numeric"], "base": p["model"],
                    "about": {"what": p["what"].format(model=name), "how": b["about"]["how"],
                              "good": p["good"], "watch": p["watch"]}}
    return out


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    from core import jobs, store
    from preprocessing import execute
    from training import models, train
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 300
    # 8 columns driven by 3 hidden factors (+ a little noise): PCA at 95% keeps fewer than 8
    X = (rng.normal(size=(n, 3)) @ rng.normal(size=(3, 8)) + rng.normal(0, .1, (n, 8))) * [1, 1, 50, 1, 1, 1, 1, 1]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(8)])
    df["y"] = np.where(df["x0"] + df["x1"] + rng.normal(0, .5, n) > 1.4, "rare", "common")   # imbalanced

    def ran(frame, target, task, keys, primary):
        ds = store.create(frame, "t.csv")
        r = execute.apply_plan(frame, target, {}, task=task)
        r["fitted"]["clean_ops"] = []
        store.save_prep(ds, r["csv"], r["fitted"])
        opt = train.options(ds)
        listed = {m["key"]: m for m in opt["models"]}
        assert all(listed[k]["compatible"] and listed[k]["family"] == "Tested pipeline" for k in keys), listed
        s = train.validate(ds, {"models": keys, "folds": 3, "primary_metric": primary, "time_limit_min": 5,
                                "params": {"xgboost": {"n_estimators": 30}, "fs_xgboost": {"n_estimators": 30}}})
        run = jobs.start(ds, lambda job: train.run(job, ds, s), s, 300)
        jobs.wait(run)
        res = store.read_json(store.run_dir(ds, run) / "results.json")
        folder = store.run_dir(ds, run) / "models"
        return ds, res, {k: joblib.load(folder / f"{k}.joblib") for k in keys if (folder / f"{k}.joblib").exists()}

    clf = [k for k in rows("classification", models.table("classification"))]
    ds, res, fitted = ran(df, "y", "classification", clf, "balanced_accuracy")
    for k in clf:
        assert "error" not in res["models"][k], (k, res["models"][k])
        section, settings = _PRESETS[k]["step"]
        assert res["models"][k]["pipeline_step"] == {section: settings}, res["models"][k]
    ctx = train.load_context(ds)
    prep = {k: f.named_steps["prep"] for k, f in fitted.items()}
    # the step really ran: components / units sized by the variance rule, half the columns kept
    pcs = prep["pca_svm"].fitted_["pca"]
    assert pcs["prefix"] == "pc" and pcs["scaler"] is not None and 1 <= pcs["n"] < 8, pcs["n"]
    assert prep["autoencoder_logreg"].fitted_["pca"]["prefix"] == "ae"
    if "fs_xgboost" in prep:
        assert len(prep["fs_xgboost"].fitted_["keep"]) == 4, prep["fs_xgboost"].fitted_["keep"]
    smote = fitted["smote_random_forest"].named_steps["resample"]
    assert type(smote).__name__ == "SMOTE"
    # saved models predict new rows; the plain model's pipeline is untouched by its preset
    for k, f in fitted.items():
        assert len(f.predict(ctx["X_te"])) == len(ctx["X_te"]), k
    from training.pipeline import build_pipeline
    svm_pipe = build_pipeline(ctx, "svm", models.table("classification")["svm"], train.facts(ctx), 3)
    assert svm_pipe.named_steps["prep"].pipeline == (ctx["prep"]["pipeline"] or {}), "base model must not change"
    # too few number columns: the preset says why it can't run, the base model still can
    one = pd.DataFrame({"x": rng.normal(size=60), "t": list("ab") * 30})
    one["y"] = np.where(one["x"] > 0, "p", "n")
    d1 = store.create(one, "o.csv")
    r1 = execute.apply_plan(one, "y", {"t": [{"op": "drop_column"}]}, task="classification")
    r1["fitted"]["clean_ops"] = []
    store.save_prep(d1, r1["csv"], r1["fitted"])
    listed = {m["key"]: m for m in train.options(d1)["models"]}
    assert not listed["pca_svm"]["compatible"] and "numeric" in listed["pca_svm"]["reason"], listed["pca_svm"]
    assert listed["svm"]["compatible"] and listed["smote_random_forest"]["compatible"]
    # missing values left in: every preset refuses with the impute reason (their steps can't take NaN),
    # while the forest itself (which can) stays available
    gaps = df.copy()
    gaps.loc[::7, "x0"] = np.nan
    dn = store.create(gaps, "n.csv")
    rn = execute.apply_plan(gaps, "y", {}, task="classification")
    rn["fitted"]["clean_ops"] = []
    store.save_prep(dn, rn["csv"], rn["fitted"])
    listed = {m["key"]: m for m in train.options(dn)["models"]}
    for k in clf:
        assert not listed[k]["compatible"] and "impute" in listed[k]["reason"], (k, listed[k]["reason"])
    assert listed["random_forest"]["compatible"]
    # regression presets
    reg = df.drop(columns="y").assign(y=df["x0"] * 3 + rng.normal(0, .3, n))
    rkeys = list(rows("regression", models.table("regression")))
    assert set(rkeys) <= {"pca_svm", "fs_xgboost"} and "pca_svm" in rkeys
    _, rres, _ = ran(reg, "y", "regression", rkeys, "rmse")
    assert all("error" not in rres["models"][k] for k in rkeys), rres["models"]
    print(f"presets self-check passed ({len(clf)} classification, {len(rkeys)} regression)")
