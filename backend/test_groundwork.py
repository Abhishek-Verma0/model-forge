"""Groundwork check: datasets, clean ops and pipelines survive a restart; unknown
ids 404; run endpoints answer. Run from backend/: .venv\\Scripts\\python -B test_groundwork.py"""
import io
import tempfile
import warnings
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

warnings.filterwarnings("ignore")
import app as A  # noqa: E402  (adds preprocessing/ to sys.path)
import jobs      # noqa: E402
import store     # noqa: E402

store.ROOT = Path(tempfile.mkdtemp())
df = pd.DataFrame({"review": [f" good item {i}" if i % 2 else f"bad item {i} " for i in range(40)],
                   "age": [20 + i % 30 for i in range(40)], "y": ["pos", "neg"] * 20})
ds = store.create(df, "t.csv")

A.do_clean({"id": ds, "ops": [{"op": "trim_whitespace", "column": "review"}]})
res = A.do_preprocess({"id": ds, "target": "y", "task": "classification",
                       "columns": {"review": [{"op": "encode", "method": "text"}]}})
assert "fitted" not in res and "csv" not in res and res["train_rows"] == 32, res.keys()

store._cache.clear()                                       # restart: memory gone, disk remains
assert A.download(ds).body.startswith(b"review")
fitted = joblib.load(io.BytesIO(A.download_pipeline(ds).body))
assert fitted["clean_ops"] == [{"op": "trim_whitespace", "column": "review"}], fitted["clean_ops"]
assert A.chart(ds, "histogram", x="age").status_code == 200

for call in (lambda: A.download("000000000000"), lambda: A.chart("../../x", "histogram", x="age"),
             lambda: A.run_status(ds, "../x"), lambda: A.run_status(ds, "000000000000")):
    try:
        call()
        raise AssertionError("unknown id should 404")
    except HTTPException as e:
        assert e.status_code == 404, e.status_code

r = jobs.start(ds, lambda job: job.progress(step=1), {"kind": "demo"}, time_limit_s=60)
jobs.wait(r)
assert A.run_status(ds, r)["state"] == "done" and A.list_runs(ds)[0]["run_id"] == r
assert A.cancel_run({"id": ds, "run": r}) == {"cancelled": False}   # already finished
print("groundwork check passed")
