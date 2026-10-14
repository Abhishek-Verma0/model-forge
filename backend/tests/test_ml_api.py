"""ML features over real HTTP, the way the page uses them: result charts
(classification + regression), hyperparameter search, ensembles, learning and
validation curves, explanations, clustering -- each as a background job.
Run from backend/: .venv\Scripts\python -B -m tests.test_ml_api"""
import io
import time

import numpy as np
import pandas as pd

from tests.server import account, serve  # first: isolates DATA_DIR + database

import app as app_module  # noqa: E402

base = serve(app_module.app)
me = account(base, "ml@example.com")


def upload(df, name="d.csv"):
    r = me.post("/api/analyze", files={"file": (name, df.to_csv(index=False).encode(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def wait(ds, run, field="run"):
    for _ in range(1200):
        s = me.get(f"/api/run?id={ds}&run={run}").json()
        if s["state"] not in ("queued", "running"):
            assert s["state"] == "done", s
            return s
        time.sleep(0.25)
    raise AssertionError("job did not finish")


rng = np.random.default_rng(3)
n = 240
df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y", "z"], n)})
df["y"] = np.where(df["a"] + 0.5 * (df["c"] == "x") + rng.normal(0, .4, n) > 0, "yes", "no")
ds = upload(df)
assert me.post("/api/preprocess", json={"id": ds, "target": "y", "task": "classification",
                                        "columns": {"c": [{"op": "encode", "method": "onehot"}]}}).status_code == 200
opt = me.get(f"/api/train/options?id={ds}").json()
assert "voting_soft" in opt["ensembles"] and opt["group"] is None

# search: random over the default ranges, then the winners train as a normal run
space = me.get(f"/api/train/space?id={ds}").json()
assert "C" in space["logistic_regression"]["space"]
run = me.post("/api/train/tune", json={"id": ds, "models": ["logistic_regression", "decision_tree"], "folds": 3,
                                       "method": "random", "n_trials": 4, "primary_metric": "accuracy",
                                       "ensembles": [{"kind": "voting_soft",
                                                      "members": ["logistic_regression", "decision_tree"]}]}).json()["run"]
wait(ds, run)
res = me.get(f"/api/run/results?id={ds}&run={run}").json()
assert len(res["tuning"]["models"]["decision_tree"]["trials"]) == 4 and "voting_soft" in res["models"], res.keys()
assert me.get(f"/api/run/tuning?id={ds}&run={run}").json()["method"] == "random"
assert me.get(f"/api/run/leaders?id={ds}&run={run}&k=2").json()
summary = me.get(f"/api/runs/summary?id={ds}").json()
assert summary[0]["kind"] == "tune" and summary[0]["task"] == "classification"

# result charts incl. calibration; explanation; learning + validation curves
ch = me.get(f"/api/run/curves?id={ds}&run={run}&model=logistic_regression").json()
assert ch["calibration"] and ch["roc"] and len(ch["confusion"]) == 2
assert me.post("/api/explain/start", json={"id": ds, "run": run, "model": "logistic_regression"}).json()["job"]
for _ in range(400):
    ex = me.get(f"/api/explain?id={ds}&run={run}&model=logistic_regression")
    if ex.status_code == 200:
        break
    time.sleep(0.25)
ex = ex.json()
assert ex["permutation"]["columns"][0]["column"] == "a" and ex["shap"]["columns"][0]["column"] == "a", ex
assert me.post("/api/explain/start", json={"id": ds, "run": run, "model": "logistic_regression"}).json() == {"cached": True}
assert me.get(f"/api/explain?id={ds}&run={run}&model=../../meta").status_code == 404
job = me.post("/api/run/learning", json={"id": ds, "run": run, "model": "decision_tree"}).json()["job"]
wait(ds, job)
lc = me.get(f"/api/run/diagnostics?id={ds}&run={run}&name=decision_tree_learning").json()
assert len(lc["val_mean"]) == 5
job = me.post("/api/run/validation", json={"id": ds, "run": run, "model": "logistic_regression", "param": "C"}).json()["job"]
wait(ds, job)
assert me.get(f"/api/run/diagnostics?id={ds}&run={run}&name=logistic_regression_validation_C").json()["values"]
assert me.get(f"/api/run/diagnostics?id={ds}&run={run}&name=..%2F..%2Fmeta").status_code == 404

# regression plots
reg = df.assign(y=df["a"] * 2 + rng.normal(0, .3, n))
dr = upload(reg, "r.csv")
me.post("/api/preprocess", json={"id": dr, "target": "y", "task": "regression",
                                 "columns": {"c": [{"op": "encode", "method": "onehot"}]}})
rr = me.post("/api/train", json={"id": dr, "models": ["ridge"], "folds": 3}).json()["run"]
wait(dr, rr)
rc = me.get(f"/api/run/curves?id={dr}&run={rr}&model=ridge").json()
assert rc["task"] == "regression" and rc["qq"]["r"] > 0.9 and len(rc["actual"]) == len(rc["predicted"])

# clustering: no target, same Train routes, CSV of assignments
blobs = pd.DataFrame(np.vstack([c + rng.normal(0, .5, (60, 2)) for c in ([0, 0], [5, 5], [0, 6])]), columns=["u", "v"])
dc = upload(blobs, "blobs.csv")
pre = me.post("/api/preprocess", json={"id": dc, "task": "clustering", "columns": {}})
assert pre.status_code == 200 and pre.json()["test_rows"] == 0, pre.text
co = me.get(f"/api/train/options?id={dc}").json()
assert co["task"] == "clustering" and co["elbow"]["best_k"] == 3
cr = me.post("/api/train", json={"id": dc, "models": ["kmeans", "gaussian_mixture"]}).json()["run"]
wait(dc, cr)
cres = me.get(f"/api/run/results?id={dc}&run={cr}").json()
assert cres["models"]["kmeans"]["clusters"] == 3
csv = pd.read_csv(io.StringIO(me.get(f"/api/run/clusters?id={dc}&run={cr}&model=kmeans").text))
assert len(csv) == 180 and csv["cluster"].nunique() == 3
assert me.get(f"/api/runs/summary?id={dc}").json()[0]["task"] == "clustering"
print("ml api check passed")
