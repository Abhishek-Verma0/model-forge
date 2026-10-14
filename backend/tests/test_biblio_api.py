"""Bibliometrics over real HTTP: upload a Scopus export and a Web of Science export,
summary + every analysis, history for the report, ownership, wrong files refused.
Run from backend/: .venv\Scripts\python -B -m tests.test_biblio_api"""
import random

import pandas as pd

from tests.server import account, serve  # first: isolates DATA_DIR + database

import app as app_module  # noqa: E402

base = serve(app_module.app)
me, other = account(base, "biblio@example.com"), account(base, "other@example.com")
rnd = random.Random(1)
authors = ["Aria M.", "Cuccurullo C.", "Cobo M.J.", "Herrera F.", "Small H.", "Garfield E.", "Lee K.", "Smith J."]
kws = ["bibliometrics", "science mapping", "citation analysis", "co-word", "networks", "deep learning",
       "machine learning", "altmetrics"]
places = ["University of Naples, Naples, Italy", "University of Granada, Granada, Spain",
          "Harvard University, Cambridge, United States", "Seoul National University, Seoul, South Korea"]
rows = []
for i in range(60):
    year = 2010 + i % 12
    au = rnd.sample(authors, rnd.randint(1, 3))
    rows.append({"Authors": "; ".join(au), "Title": f"Paper {i} on {rnd.choice(kws)}", "Year": year,
                 "Source title": rnd.choice(["Scientometrics", "Journal of Informetrics", "JASIST"]),
                 "Cited by": rnd.randint(0, 120), "DOI": f"10.1016/j.test.{i}",
                 "Affiliations": "; ".join(rnd.sample(places, rnd.randint(1, 2))),
                 "Author Keywords": "; ".join(rnd.sample(kws[:5] if year < 2016 else kws[3:], 3)),
                 "Index Keywords": "", "References": "; ".join(
                     [f"Ref{rnd.randint(1, 15)} et al., Journal, ({rnd.randint(1990, 2010)})" for _ in range(4)] +
                     ([f"Earlier work, DOI 10.1016/j.test.{i - 1}"] if i else [])),
                 "Document Type": "Article"})
scopus = pd.DataFrame(rows).to_csv(index=False).encode()
up = me.post("/api/biblio/upload", files={"file": ("scopus.csv", scopus, "text/csv")})
assert up.status_code == 200, up.text[:300]
ds = up.json()["id"]
s = up.json()
assert s["format"] == "scopus" and s["overview"]["documents"] == 60 and s["completeness"]["countries"] == 100.0
assert s["annual"]["years"][0] == 2010 and sum(s["annual"]["documents"]) == 60
assert me.get(f"/api/biblio/summary?id={ds}").json()["overview"]["authors"] == len(authors)
projects = {p["id"]: p for p in me.get("/api/projects").json()}
assert projects[ds]["kind"] == "bibliometric"


def run(analysis, params=None, ok=200):
    r = me.post("/api/biblio/run", json={"id": ds, "analysis": analysis, "params": params or {}})
    assert r.status_code == ok, (analysis, r.status_code, r.text[:300])
    return r.json()


pr = run("productivity")
assert pr["countries"][0]["name"] in ("Italy", "Spain", "USA", "South Korea") and pr["institutions"]
ci = run("citations")
assert ci["has_citations"] and ci["author_impact"][0]["h_index"] >= 1 and ci["local_citations_matched"] >= 50
for kind in ("coauthor", "country", "keyword", "cocitation", "coupling"):
    net = run("network", {"kind": kind, "top_n": 20})
    assert net["nodes"] and all(0 <= n["x"] <= 1 for n in net["nodes"]), kind
tm = run("thematic", {"min_freq": 3})
assert tm["themes"] and {t["quadrant"] for t in tm["themes"]}
ev = run("evolution", {"cuts": [2015], "min_freq": 3})
assert len(ev["periods"]) == 2 and ev["periods"][0]["themes"]
assert run("three_field", {"n": 5})["links"]
assert run("wordcloud", {"field": "title"})["words"]
assert run("trends", {"min_freq": 5})["terms"]
run("network", {"kind": "nope"}, ok=422)
run("wordcloud", {"field": "../x"}, ok=422)
assert len(me.get(f"/api/biblio/history?id={ds}").json()) == 12

# ownership + wrong files
assert other.get(f"/api/biblio/summary?id={ds}").status_code == 404
assert other.post("/api/biblio/run", json={"id": ds, "analysis": "citations"}).status_code == 404
bad = me.post("/api/biblio/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
assert bad.status_code == 422 and "publication export" in bad.json()["detail"]
assert me.post("/api/biblio/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
wos = ("FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\nAU Smith, J\nTI A study\nSO SCIENTOMETRICS\n"
       "DE citation analysis; co-word\nC1 [Smith, J] Harvard Univ, Cambridge, MA 02138 USA.\nTC 4\nPY 2019\nER\nEF\n")
w = me.post("/api/biblio/upload", files={"file": ("savedrecs.txt", wos.encode(), "text/plain")})
assert w.status_code == 200 and w.json()["format"] == "wos_plain" and w.json()["overview"]["documents"] == 1
# a bibliometric collection is not a table for the analysis workflow
assert me.get(f"/api/projects/{ds}").json()["kind"] == "bibliometric"
print("bibliometrics api check passed")
