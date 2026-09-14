"""Groundwork check: datasets, clean ops and pipelines survive a restart; unknown
ids 404; run endpoints answer. Run from backend/: .venv\\Scripts\\python -B -m tests.test_groundwork"""
import io
import tempfile
import warnings
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

warnings.filterwarnings("ignore")
import app  # noqa: E402,F401  (the whole app -- routers + startup -- must import cleanly)
from api import data as D, training as T  # noqa: E402
from core import jobs, store  # noqa: E402

store.ROOT = Path(tempfile.mkdtemp())
df = pd.DataFrame({"review": [f" good item {i}" if i % 2 else f"bad item {i} " for i in range(40)],
                   "age": [20 + i % 30 for i in range(40)], "y": ["pos", "neg"] * 20})
ds = store.create(df, "t.csv")

cleaned = D.do_clean({"id": ds, "ops": [{"op": "trim_whitespace", "column": "review"}]})
# the page renders the backend's choices + defaults, never its own copies
from core import config  # noqa: E402
from preprocessing import execute  # noqa: E402
assert cleaned["preprocess_options"] == execute.options()
assert D.upload_limits() == {"max_mb": config.MAX_UPLOAD_MB, "allowed": [".csv", ".xls", ".xlsx"]}
res = D.do_preprocess({"id": ds, "target": "y", "task": "classification",
                       "columns": {"review": [{"op": "encode", "method": "text"}]}})
assert "fitted" not in res and "csv" not in res and res["train_rows"] == 32, res.keys()

store._cache.clear()                                       # restart: memory gone, disk remains
assert D.download(ds).body.startswith(b"review")
fitted = joblib.load(io.BytesIO(D.download_pipeline(ds).body))
assert fitted["clean_ops"] == [{"op": "trim_whitespace", "column": "review"}], fitted["clean_ops"]
assert D.chart(ds, "histogram", x="age").status_code == 200

for call in (lambda: D.download("000000000000"), lambda: D.chart("../../x", "histogram", x="age"),
             lambda: T.run_status(ds, "../x"), lambda: T.run_status(ds, "000000000000")):
    try:
        call()
        raise AssertionError("unknown id should 404")
    except HTTPException as e:
        assert e.status_code == 404, e.status_code

r = jobs.start(ds, lambda job: job.progress(step=1), {"kind": "demo"}, time_limit_s=60)
jobs.wait(r)
assert T.run_status(ds, r)["state"] == "done" and T.list_runs(ds)[0]["run_id"] == r
assert T.cancel_run({"id": ds, "run": r}) == {"cancelled": False}   # already finished
print("groundwork check passed")
