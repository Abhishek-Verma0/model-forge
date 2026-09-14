"""Train-workflow API check: parameters + epochs in a run, run history, epoch curves,
predict on an uploaded file (JSON + CSV), locked test set, save / list / download /
delete models, discard a run, bad ids refused.
Run from backend/: .venv\\Scripts\\python -B -m tests.test_inference_api"""
import asyncio
import io
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile

warnings.filterwarnings("ignore")
import app  # noqa: E402,F401  (the whole app -- routers + startup -- must import cleanly)
from api import data as D, inference as I, training as T  # noqa: E402
from core import jobs, store  # noqa: E402
from training import suggest  # noqa: E402

store.ROOT = Path(tempfile.mkdtemp())
rng = np.random.default_rng(1)
n = 180
df = pd.DataFrame({"note": [f"{'refund broken' if i % 3 == 0 else 'love it'} {i}" for i in range(n)],
                   "price": rng.normal(50, 5, n), "y": ["bad" if i % 3 == 0 else "good" for i in range(n)]})
ds = store.create(df, "t.csv")
D.do_preprocess({"id": ds, "target": "y", "task": "classification",
                 "columns": {"note": [{"op": "encode", "method": "text"}]}})


def expect(status, call):
    try:
        call()
    except HTTPException as e:
        assert e.status_code == status, (status, e.status_code, e.detail)
        return e.detail
    raise AssertionError(f"expected HTTP {status}")


opt = T.train_options(ds)
lgbm = next(m for m in opt["models"] if m["key"] == "lightgbm")
assert lgbm["epochs"] and {p["name"] for p in lgbm["params"]} >= {"n_estimators", "early_stopping", "patience"}
assert "between" in expect(422, lambda: T.start_training({"id": ds, "models": ["lightgbm"],
                                                          "params": {"lightgbm": {"n_estimators": -5}}}))

run = T.start_training({"id": ds, "models": ["logistic_regression", "lightgbm"], "folds": 3,
                        "params": {"lightgbm": {"n_estimators": 80, "early_stopping": True, "patience": 5}}})["run"]
jobs.wait(run)
assert T.run_status(ds, run)["state"] == "done"
curves = T.run_epochs(ds, run, "lightgbm")
assert set(curves["fits"]) == {"fold 1", "fold 2", "fold 3", "final"} and curves["metric"], curves.keys()
expect(404, lambda: T.run_epochs(ds, run, "../../settings"))          # not a model of this run
summary = T.runs_summary(ds)
assert summary[0]["run_id"] == run and {m["key"] for m in summary[0]["models"]} == {"baseline", "logistic_regression", "lightgbm"}
assert joblib.load(T.run_model(ds, run, "lightgbm").path).predict(df.drop(columns="y").head(3)).shape == (3,)

upload = df.drop(columns="y").head(20).assign(y=df["y"].head(20))
csv_bytes = upload.to_csv(index=False).encode()


def post_predict(fmt="json", **fields):
    file = UploadFile(file=io.BytesIO(csv_bytes), filename="new.csv")
    return asyncio.run(I.predict(file=file, id=ds, format=fmt, **{"run": "", "model": "", "saved": "", **fields}))


got = post_predict(run=run, model="logistic_regression")
assert got["rows"] == 20 and got["scored_rows"] == 20 and got["metrics"]["accuracy"] > 0.9, got["metrics"]
assert got["preview"][0]["prediction"] in ("bad", "good") and "probability bad" in got["columns"]
as_csv = pd.read_csv(io.StringIO(post_predict("csv", run=run, model="logistic_regression").body.decode()))
assert len(as_csv) == 20 and "prediction" in as_csv.columns
assert "Choose" in expect(422, lambda: post_predict())

page = I.test_set(ds, run, "lightgbm", False, 0, 5)
assert page["total"] == 36 and len(page["rows"]) == 5

saved = I.save_model({"id": ds, "run": run, "model": "", "name": "first model"})   # "" = the run's best
assert "file" not in saved and saved["key"] == T.run_results(ds, run)["best"]
assert [m["name"] for m in I.list_models(ds)] == ["first model"]
assert Path(I.download_model(ds, saved["model_id"]).path).exists()
assert post_predict(saved=saved["model_id"])["rows"] == 20
expect(404, lambda: I.download_model(ds, "../../etc"))
I.delete_model(ds, saved["model_id"])
assert I.list_models(ds) == []

# suggestions: fake AI, then cache, then fallback -- through the API
suggest._ask = lambda context: {"models": [{"key": "lightgbm", "reason": "wide text"}, {"key": "gaussian_nb"}]}
first = T.suggest_models({"id": ds})
assert first["source"] == "ai" and [m["key"] for m in first["models"]] == ["lightgbm"], first
assert T.suggest_models({"id": ds})["source"] == "ai_cached"

# training data view: readable page + summary, never the model matrix
page = T.training_data(ds, "train", 0, 5)
assert len(page["rows"]) == 5 and "note" in page["columns"] and len(page["columns"]) < 10, page["columns"]
assert page["summary"]["model_features"] > 100

# manual row through the API (exact equality with the file path is proven in inference.predict self-check)
schema = I.input_schema(ds, run, "logistic_regression", "")["schema"]
assert {f["name"]: f["kind"] for f in schema} == {"note": "text", "price": "number"}, schema
row = I.predict_row({"id": ds, "run": run, "model": "logistic_regression", "values": {"note": "refund broken 7", "price": 50}})
assert row["preview"][0]["prediction"] == "bad" and row["model"]["label"] == "Logistic regression", row
assert "trained on" not in str(row["notes"])
assert "number" in expect(422, lambda: I.predict_row({"id": ds, "run": run, "model": "logistic_regression",
                                                       "values": {"note": "x", "price": "abc"}}))

T.discard_run(ds, run)
assert T.runs_summary(ds) == []
expect(404, lambda: T.discard_run(ds, run))
print("inference api check passed")
