"""Statistics over real HTTP: every analysis type runs on an uploaded table, results
are kept (history + hypothesis decision table), questionnaire ops go through
cleaning, bad requests are refused.
Run from backend/: .venv\Scripts\python -B -m tests.test_stats_api"""
import numpy as np
import pandas as pd

from tests.server import account, serve  # first: isolates DATA_DIR + database

import app as app_module  # noqa: E402

base = serve(app_module.app)
me = account(base, "stats@example.com")
rng = np.random.default_rng(0)
n = 120
df = pd.DataFrame({"group": rng.choice(["a", "b", "c"], n), "sex": rng.choice(["F", "M"], n),
                   "pre": rng.normal(50, 10, n).round(1), "q1": rng.integers(1, 6, n), "q2": rng.integers(1, 6, n),
                   "q3": rng.integers(1, 6, n)})
df["post"] = (df["pre"] + rng.normal(3, 4, n)).round(1)
df["score"] = df["pre"] + (df["group"] == "c") * 8 + rng.normal(0, 5, n)
df["passed"] = np.where(df["score"] > 52, "yes", "no")
ds = me.post("/api/analyze", files={"file": ("survey.csv", df.to_csv(index=False).encode(), "text/csv")}).json()["id"]


def run(analysis, params, ok=200, **extra):
    r = me.post("/api/stats/run", json={"id": ds, "analysis": analysis, "params": params, **extra})
    assert r.status_code == ok, (analysis, r.status_code, r.text[:300])
    return r.json()


d = run("describe", {"columns": ["pre", "group"]})
assert d["rows"][0]["ci_low"] < d["rows"][0]["mean"] < d["rows"][0]["ci_high"] and d["rows"][1]["kind"] == "categorical"
assert run("frequencies", {"columns": ["group"]})["tables"][0]["rows"][-1]["cumulative"] == 100
assert run("normality", {"columns": ["pre"], "by": "sex"})["rows"][0]["qq"]
c = run("compare", {"outcome": "score", "group": "group"})
assert c["test"] in ("One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis") and c["reason"] and c["p"] < 0.05
assert run("compare", {"outcome": "score", "group": "sex", "test": "student_t"})["test"] == "Student's t-test"
assert run("paired", {"before": "pre", "after": "post"})["significant"] is True
assert len(run("correlation", {"columns": ["pre", "post", "score"]})["pairs"]) == 3
assert run("association", {"row": "sex", "col": "passed"})["fisher"]
assert run("regression", {"kind": "linear", "outcome": "score", "predictors": ["pre", "group"]})["r2"] > 0.3
assert run("regression", {"kind": "logistic", "outcome": "passed", "predictors": ["pre"], "event": "yes"})["converged"]
rel = run("reliability", {"items": ["q1", "q2", "q3"]})
assert "alpha" in rel and rel["k"] == 3
assert "q1" in {i["column"] for i in me.get(f"/api/stats/likert?id={ds}").json()["items"]}

# refusals
run("compare", {"outcome": "group", "group": "sex"}, ok=422)          # non-numeric outcome
run("describe", {"columns": ["nope"]}, ok=422)
run("describe", {"columns": ["pre"]}, ok=422, alpha=0.9)
run("magic", {}, ok=422)

hist = me.get(f"/api/stats/history?id={ds}").json()
assert len(hist) == 11 and all("id" in h for h in hist)
table = me.get(f"/api/stats/decisions?id={ds}").json()
assert len(table["rows"]) >= 8 and all(r["decision"] for r in table["rows"]) and "confirm" in table["review"]
assert me.delete(f"/api/stats?id={ds}&stat={hist[0]['id']}").status_code == 200
assert len(me.get(f"/api/stats/history?id={ds}").json()) == 10

# questionnaire ops are cleaning ops: saved and replayed
cl = me.post("/api/clean", json={"id": ds, "ops": [{"op": "reverse_score", "column": "q3", "low": 1, "high": 5},
                                                   {"op": "composite", "columns": ["q1", "q2", "q3"], "to": "scale"}]})
assert cl.status_code == 200 and "scale" in cl.json()["profile"]["column_names"], cl.text[:300]
assert run("describe", {"columns": ["scale"]})["rows"][0]["n"] == n
print("stats api check passed")
