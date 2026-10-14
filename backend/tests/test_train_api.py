"""Train API check: options, refusals, a run, results, best model, estimate.
Run from backend/: .venv\\Scripts\\python -B -m tests.test_train_api"""
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

warnings.filterwarnings("ignore")
import app  # noqa: E402,F401  (the whole app -- routers + startup -- must import cleanly)
from api import data as D, training as T  # noqa: E402
from core import jobs, store  # noqa: E402

store.ROOT = Path(tempfile.mkdtemp())
rng = np.random.default_rng(1)
n = 150
df = pd.DataFrame({"note": [f"{'refund broken' if i % 3 == 0 else 'love it'} {i}" for i in range(n)],
                   "price": rng.normal(50, 5, n), "y": ["bad" if i % 3 == 0 else "good" for i in range(n)]})
ds = store.create(df, "t.csv")

try:
    T.train_options(ds)
    raise AssertionError("options before preprocessing should 404")
except HTTPException as e:
    assert e.status_code == 404, e.status_code

D.do_preprocess({"id": ds, "target": "y", "task": "classification",
                 "columns": {"note": [{"op": "encode", "method": "text"}]}})
opt = T.train_options(ds)
assert opt["target"] == "y" and opt["defaults"]["positive_class"] == "bad", opt["defaults"]

try:
    T.start_training({"id": ds, "models": ["gaussian_nb"]})
    raise AssertionError("incompatible model should 422")
except HTTPException as e:
    assert e.status_code == 422 and "dense" in e.detail, e.detail

run = T.start_training({"id": ds, "models": ["logistic_regression"], "folds": 3})["run"]
jobs.wait(run)
assert T.run_status(ds, run)["state"] == "done"
res = T.run_results(ds, run)
assert res["best"] == "logistic_regression" and res["models"]["logistic_regression"]["test"]["f1"] > 0.9, res
model = joblib.load(T.run_model(ds, run, "").path)
assert len(model.predict(df.drop(columns="y").head(5))) == 5   # raw feature rows in, predictions out

# result charts come straight from the saved predictions (binary "bad"/"good")
ch = T.run_curves(ds, run, "logistic_regression")
lr_test = res["models"]["logistic_regression"]["test"]
assert len(ch["confusion"]) == 2 and ch["classes"] == res["classes"]
diag = ch["confusion"][0][0] + ch["confusion"][1][1]
assert diag == round(lr_test["accuracy"] * sum(sum(r) for r in ch["confusion"]))  # matrix agrees with reported accuracy
assert abs(ch["roc"][0]["auc"] - lr_test["roc_auc"]) < 1e-6                        # ROC agrees with reported AUC


# boosters run end to end on whatever device the registry picked (GPU when one is usable)
device = {m["key"]: m["device"] for m in opt["models"]}
boost = T.start_training({"id": ds, "models": ["xgboost", "catboost"], "folds": 3,
                          "params": {"xgboost": {"n_estimators": 50, "early_stopping": True},
                                     "catboost": {"iterations": 50, "early_stopping": True}}})["run"]
jobs.wait(boost)
res = T.run_results(ds, boost)
for key in ("xgboost", "catboost"):
    rec = res["models"][key]
    assert "error" not in rec and rec["device"] == device[key], (key, rec.get("error"), rec.get("device"))
    curve = store.read_json(store.run_dir(ds, boost) / "epochs" / f"{key}.json")
    assert curve, (key, "no epoch curve")
    fitted = joblib.load(T.run_model(ds, boost, key).path)
    assert len(fitted.predict(df.drop(columns="y").head(5))) == 5, key

est = T.start_estimate({"id": ds, "models": ["logistic_regression"], "folds": 3})["run"]
jobs.wait(est)
assert T.run_results(ds, est)["estimates"]["logistic_regression"]["projected_seconds"] > 0
print("train api check passed")
