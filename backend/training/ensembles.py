"""Ensembles (plan §9.5; spec 2026-09-13 §7 piece 5): voting (hard / soft),
stacking with a meta-model, and weighted averaging -- built from models of the
table, each with the parameters the user set for it. An ensemble is one more row
of the run: same preprocessing (fitted once per fold, shared by its members), same
folds, same locked test set, same metrics. Bagging and boosting are already rows
of the table (random forest / extra trees; AdaBoost, gradient boosting, XGBoost,
LightGBM, CatBoost).

Only valid combinations are offered (plan §9.6 "tested pipelines"):
  - soft voting / weighted averaging of classifiers need every member to give
    probabilities (SVM does not by default);
  - stacking's internal cross-validation can't keep groups together, so it is
    refused when the split uses a group column;
  - every member must be able to train on this data (the model table's rules).
Members train with their constructor parameters; their early stopping (an epoch
option) is not used inside an ensemble.
"""

from sklearn import ensemble as E

KINDS = {
    "classification": {"voting_hard": "Voting (hard - majority of predicted classes)",
                       "voting_soft": "Voting (soft - average of probabilities, optionally weighted)",
                       "stacking": "Stacking (logistic regression on the members' predictions)"},
    "regression": {"averaging": "Weighted averaging (mean of predictions, optionally weighted)",
                   "stacking": "Stacking (ridge regression on the members' predictions)"},
}
MIN_MEMBERS = 2


def _has_proba(row):
    return hasattr(row["make"](0), "predict_proba")


def validate(task, specs, rows, compatible, grouped):
    """User ensemble requests -> [{key, kind, members, weights}] or ValueError."""
    out, seen = [], set()
    for spec in specs or []:
        kind = spec.get("kind")
        if kind not in KINDS.get(task, {}):
            raise ValueError(f"Ensemble kind must be one of {', '.join(KINDS.get(task, {}))} for {task}.")
        if kind in seen:
            raise ValueError(f"Only one {kind.replace('_', ' ')} ensemble per run.")
        seen.add(kind)
        members = list(dict.fromkeys(spec.get("members") or []))
        if len(members) < MIN_MEMBERS:
            raise ValueError(f"{KINDS[task][kind]}: choose at least {MIN_MEMBERS} models.")
        for m in members:
            if m not in rows or m == "baseline":
                raise ValueError(f"'{m}' can't be an ensemble member.")
            if not compatible.get(m, False):
                raise ValueError(f"{rows[m]['label']} can't train on this data, so it can't join an ensemble.")
            if kind in ("voting_soft",) and not _has_proba(rows[m]):
                raise ValueError(f"Soft voting needs probabilities - {rows[m]['label']} gives none. "
                                 "Use hard voting or remove it.")
        if kind == "stacking" and grouped:
            raise ValueError("Stacking's internal cross-validation can't keep groups together - "
                             "use voting / averaging with a group split.")
        weights = spec.get("weights")
        if weights is not None:
            weights = [float(w) for w in weights]
            if len(weights) != len(members) or any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError("Weights: one non-negative number per member, not all zero.")
            if all(w == 1 for w in weights):
                weights = None                      # equal weights = plain voting / averaging
            elif kind == "stacking":
                raise ValueError("Stacking learns its own weights - leave them at 1.")
        out.append({"key": kind, "kind": kind, "members": members, "weights": weights})
    return out


def row(task, spec, rows, member_kwargs):
    """A model-table row (training/models.py shape) for one ensemble."""
    members = spec["members"]

    def make(seed):
        est = [(m, rows[m]["make"](seed).set_params(**member_kwargs.get(m, {}))) for m in members]
        kind, w = spec["kind"], spec["weights"]
        if kind == "voting_hard":
            return E.VotingClassifier(est, voting="hard", weights=w)
        if kind == "voting_soft":
            return E.VotingClassifier(est, voting="soft", weights=w)
        if kind == "averaging":
            return E.VotingRegressor(est, weights=w)
        # stacking: scikit-learn's defaults -- LogisticRegression / RidgeCV meta-model on 5-fold
        # out-of-fold member predictions (so every fit costs members x 6)
        return (E.StackingClassifier if task == "classification" else E.StackingRegressor)(est)

    member_rows = [rows[m] for m in members]
    # class weights reach each member that supports them, as nested parameters (lr__class_weight=...)
    balanced = {f"{m}__{k}": v for m in members for k, v in (rows[m]["balanced"] or {}).items()}
    names = ", ".join(rows[m]["label"] for m in members)
    weights = f" (weights {', '.join(f'{w:g}' for w in spec['weights'])})" if spec["weights"] else ""
    return {"label": f"{KINDS[task][spec['kind']].split(' (')[0]}: {names}{weights}", "family": "Ensemble",
            "make": make, "sparse": all(r["sparse"] for r in member_rows),
            "nonneg": any(r["nonneg"] for r in member_rows), "nan": all(r["nan"] for r in member_rows),
            "balanced": balanced or None, "note": "members train without early stopping inside an ensemble",
            "params": [], "epochs": None, "resolve": None,
            "device": "gpu" if any(r["device"] == "gpu" for r in member_rows) else "cpu",
            "about": {"what": KINDS[task][spec["kind"]], "how": f"Members: {names}.",
                      "good": "Combining different kinds of models often beats each one alone.",
                      "watch": "Costs the members' training time combined (stacking: x6)."}}


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from core import jobs, store
    from preprocessing import execute
    from training import train
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 240
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "subject": np.repeat(range(60), 4)})
    df["y"] = np.where(df["a"] + rng.normal(0, .4, n) > 0, "pos", "neg")

    def prepared(frame, target, task, group=None):
        ds = store.create(frame, "t.csv")
        r = execute.apply_plan(frame, target, {}, task=task, group=group)
        r["fitted"]["clean_ops"] = []
        store.save_prep(ds, r["csv"], r["fitted"])
        return ds

    def ran(ds, payload):
        s = train.validate(ds, {"folds": 3, "time_limit_min": 5, **payload})
        r = jobs.start(ds, lambda job: train.run(job, ds, s), s, 300)
        jobs.wait(r)
        assert jobs.status(ds, r)["state"] == "done", jobs.status(ds, r)
        return store.read_json(store.run_dir(ds, r) / "results.json")

    ds = prepared(df.drop(columns="subject"), "y", "classification")
    small = {"xgboost": {"n_estimators": 30}, "catboost": {"iterations": 30}}
    res = ran(ds, {"models": ["logistic_regression"], "primary_metric": "accuracy", "params": small,
                   "ensembles": [{"kind": "voting_soft", "members": ["logistic_regression", "xgboost", "catboost"],
                                  "weights": [2, 1, 1]},
                                 {"kind": "voting_hard", "members": ["logistic_regression", "svm", "knn"]},
                                 {"kind": "stacking", "members": ["logistic_regression", "random_forest"]}]})
    for key in ("voting_soft", "voting_hard", "stacking"):
        rec = res["models"][key]
        assert "error" not in rec and rec["family"] == "Ensemble" and rec["test"]["accuracy"] > 0.7, (key, rec)
    assert "roc_auc" not in res["models"]["voting_hard"]["cv"]         # no probabilities -> no AUC, no crash
    assert "roc_auc" in res["models"]["voting_soft"]["cv"] and "weights 2, 1, 1" in res["models"]["voting_soft"]["label"]

    # only valid combinations
    for bad, msg in (({"kind": "voting_soft", "members": ["logistic_regression", "svm"]}, "probabilities"),
                     ({"kind": "voting_hard", "members": ["logistic_regression"]}, "at least 2"),
                     ({"kind": "stacking", "members": ["logistic_regression", "knn"], "weights": [2, 1]}, "own weights"),
                     ({"kind": "averaging", "members": ["logistic_regression", "knn"]}, "kind must be")):
        try:
            train.validate(ds, {"models": [], "ensembles": [bad]})
            raise AssertionError(bad)
        except ValueError as e:
            assert msg in str(e), (bad, str(e))
    dg = prepared(df, "y", "classification", group="subject")
    try:
        train.validate(dg, {"models": [], "ensembles": [{"kind": "stacking", "members": ["logistic_regression", "knn"]}]})
        raise AssertionError("stacking + groups")
    except ValueError as e:
        assert "groups together" in str(e), e

    # regression: equal-weight averaging == the mean of its members' predictions
    reg = df.drop(columns="subject").assign(y=df["a"] * 3 + rng.normal(0, .3, n))
    dr = prepared(reg, "y", "regression")
    rr = ran(dr, {"models": ["ridge", "knn"], "primary_metric": "rmse",
                  "ensembles": [{"kind": "averaging", "members": ["ridge", "knn"]}]})
    assert rr["models"]["averaging"]["test"]["rmse"] > 0
    import joblib
    folder = store.run_dir(dr, store.run_ids(dr)[-1]) / "models"
    ctx = train.load_context(dr)
    avg = joblib.load(folder / "averaging.joblib").predict(ctx["X_te"])
    each = [joblib.load(folder / f"{k}.joblib").predict(ctx["X_te"]) for k in ("ridge", "knn")]
    assert np.allclose(avg, np.mean(each, axis=0)), "equal-weight averaging must be the plain mean"
    print("ensembles self-check passed")
