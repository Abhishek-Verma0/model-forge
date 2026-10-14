"""Reports over real HTTP (plan §15, §22 "a complete PDF report can be generated and
downloaded"): PDF + Excel + JSON of a classification project with statistics and an
explanation, a regression and a clustering project, and a bibliometric collection.
Run from backend/: .venv\Scripts\python -B -m tests.test_report_api"""
import io
import time

import numpy as np
import openpyxl
import pandas as pd

from tests.server import account, serve  # first: isolates DATA_DIR + database

import app as app_module  # noqa: E402

base = serve(app_module.app)
me, other = account(base, "report@example.com"), account(base, "nosy@example.com")


def wait(ds, run):
    for _ in range(1200):
        s = me.get(f"/api/run?id={ds}&run={run}").json()
        if s["state"] not in ("queued", "running"):
            assert s["state"] == "done", s
            return
        time.sleep(0.25)


def pages(pdf):
    assert pdf.startswith(b"%PDF"), pdf[:20]
    return pdf.count(b"/Type /Page\n") + pdf.count(b"/Type /Page ") + pdf.count(b"/Type /Page>")


def project(df, target, task, models, columns=None):
    ds = me.post("/api/analyze", files={"file": ("d.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["id"]
    body = {"id": ds, "task": task, "columns": columns or {}}
    if target:
        body["target"] = target
    assert me.post("/api/preprocess", json=body).status_code == 200
    run = me.post("/api/train", json={"id": ds, "models": models, **({} if task == "clustering" else {"folds": 3})}).json()["run"]
    wait(ds, run)
    return ds, run


rng = np.random.default_rng(5)
n = 200
df = pd.DataFrame({"age": rng.integers(20, 70, n), "bmi": rng.normal(27, 4, n).round(1), "sex": rng.choice(["F", "M"], n)})
df["outcome"] = np.where(df["bmi"] + rng.normal(0, 2, n) > 27, "risk", "ok")

# empty project: the report still builds and says what is missing
blank = me.post("/api/analyze", files={"file": ("b.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["id"]
r = me.get(f"/api/report/pdf?id={blank}")
assert r.status_code == 200 and pages(r.content) >= 3

ds, run = project(df, "outcome", "classification", ["logistic_regression", "random_forest"],
                  {"sex": [{"op": "encode", "method": "onehot"}]})
me.post("/api/stats/run", json={"id": ds, "analysis": "compare", "params": {"outcome": "bmi", "group": "outcome"}})
me.post("/api/stats/run", json={"id": ds, "analysis": "describe", "params": {"columns": ["age", "bmi"]}})
best = me.get(f"/api/run/results?id={ds}&run={run}").json()["best"]
me.post("/api/explain/start", json={"id": ds, "run": run, "model": best})
for _ in range(400):
    if me.get(f"/api/explain?id={ds}&run={run}&model={best}").status_code == 200:
        break
    time.sleep(0.25)
pdf = me.get(f"/api/report/pdf?id={ds}")
assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
assert "attachment" in pdf.headers["content-disposition"] and pages(pdf.content) >= 6, pages(pdf.content)
xl = openpyxl.load_workbook(io.BytesIO(me.get(f"/api/report/excel?id={ds}").content))
assert {"Data dictionary", "Model metrics", "Parameters", "Hypothesis decisions", "Descriptives"} <= set(xl.sheetnames)
js = me.get(f"/api/report/json?id={ds}").json()
assert js["run"]["id"] == run and js["preprocessing"]["task"] == "classification" and len(js["statistics"]) == 2
assert other.get(f"/api/report/pdf?id={ds}").status_code == 404

reg = df.assign(outcome=df["bmi"] * 2 + rng.normal(0, 1, n))
dr, _ = project(reg, "outcome", "regression", ["ridge"], {"sex": [{"op": "encode", "method": "onehot"}]})
assert pages(me.get(f"/api/report/pdf?id={dr}").content) >= 5
dc, _ = project(df[["age", "bmi"]], None, "clustering", ["kmeans"])
assert pages(me.get(f"/api/report/pdf?id={dc}").content) >= 4
assert "Clustering" in openpyxl.load_workbook(io.BytesIO(me.get(f"/api/report/excel?id={dc}").content)).sheetnames

rows = [{"Authors": "A B.; C D.", "Title": f"T{i}", "Year": 2015 + i % 6, "Source title": "J", "Cited by": i,
         "Affiliations": "Univ X, Rome, Italy", "Author Keywords": "alpha; beta" if i % 2 else "beta; gamma",
         "References": "R1; R2", "DOI": f"10.1000/x{i}"} for i in range(30)]
bib = me.post("/api/biblio/upload", files={"file": ("s.csv", pd.DataFrame(rows).to_csv(index=False).encode(), "text/csv")}).json()["id"]
me.post("/api/biblio/run", json={"id": bib, "analysis": "network", "params": {"kind": "keyword"}})
assert pages(me.get(f"/api/report/pdf?id={bib}").content) >= 3
assert "Author impact" in openpyxl.load_workbook(io.BytesIO(me.get(f"/api/report/excel?id={bib}").content)).sheetnames
assert me.get(f"/api/report/json?id={bib}").status_code == 422
print("report api check passed")
