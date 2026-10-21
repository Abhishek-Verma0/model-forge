"""Plan §23 security testing + §24 over real HTTP: every data route needs a session,
one user can never reach another's dataset (id in query, JSON body, form or path),
malicious / malformed uploads are refused, files that used to crash the server now
answer 200, and the owner can rename and delete their project.
Run from backend/: .venv\\Scripts\\python -B -m tests.test_security"""

from tests.server import Client, account, serve  # first: isolates DATA_DIR + database

import app as app_module  # noqa: E402

base = serve(app_module.app)
anon = Client(base)
alice, bob = account(base, "alice@example.com"), account(base, "bob@example.com")


def upload(client, name, text, mime="text/csv"):
    return client.post("/api/analyze", files={"file": (name, text.encode(), mime)})


# no session: every data route refuses, the public limits answer
assert upload(anon, "a.csv", "x,y\n1,a\n").status_code == 401
assert anon.get("/api/projects").status_code == 401
assert anon.get("/api/train/options?id=000000000000").status_code == 401
assert anon.get("/api/upload/limits").status_code == 200

# the files that used to answer HTTP 500 (NaN / inf in the JSON) now load
for name, text in (("one_row.csv", "x,y\n1.5,a\n"), ("one_value.csv", "x,y\n1.0,a\n,b\n,a\n,b\n"),
                   ("inf.csv", "x,y\n1,a\ninf,b\n3,a\n-inf,b\n")):
    r = upload(alice, name, text)
    assert r.status_code == 200, (name, r.status_code, r.text[:200])
assert any("infinite" in n for n in r.json()["load_notes"]), r.json()["load_notes"]
semi = upload(alice, "eu.csv", "age;bmi;y\n30;22,5;a\n40;31,0;b\n50;27,3;a\n")
assert semi.json()["profile"]["columns"] == 3 and semi.json()["profile"]["columns_info"]["bmi"]["type"] == "numeric"

# malicious / malformed uploads
assert upload(alice, "evil.exe", "MZ\x90\x00", "application/octet-stream").status_code == 415
assert upload(alice, "evil.csv.exe", "x\n1\n").status_code == 415
assert upload(alice, "empty.csv", "").status_code == 422
assert upload(alice, "bad.json", "{not json").status_code == 422

# alice's dataset is invisible to bob through every way of naming it
up = upload(alice, "data.csv", "x,y\n" + "".join(f"{i},{'ab'[i % 2]}\n" for i in range(40))).json()
ds = up["id"]
assert up["charts"]["builder"]["x"]["bar"] == ["x", "y"], up["charts"]["builder"]   # the builder's choices travel with the data
assert alice.get(f"/api/chart?id={ds}&kind=histogram&x=x").status_code == 200
assert alice.get(f"/api/chart?id={ds}&kind=density&x=y").status_code == 422   # a category has no density: refused, not drawn
assert bob.get(f"/api/chart?id={ds}&kind=histogram&x=x").status_code == 404                  # query
assert bob.post("/api/clean", json={"id": ds, "ops": []}).status_code == 404                  # JSON body
assert bob.post("/api/predict", data={"id": ds, "run": "x"},
                files={"file": ("n.csv", b"x\n1\n", "text/csv")}).status_code == 404          # form
assert bob.get(f"/api/projects/{ds}").status_code == 404                                       # path
assert bob.delete(f"/api/projects/{ds}").status_code == 404
assert ds not in [p["id"] for p in bob.get("/api/projects").json()]
for bad in ("../../etc", "ABCDEF123456", "0" * 13):                                            # traversal / junk
    assert alice.get(f"/api/chart?id={bad}&kind=histogram&x=x").status_code == 404

# the owner's project: listed, reopened, renamed, deleted (plan §8.1, §24)
pre = alice.post("/api/preprocess", json={"id": ds, "target": "y", "task": "classification", "columns": {}})
assert pre.status_code == 200, pre.text
listed = {p["id"]: p for p in alice.get("/api/projects").json()}
assert listed[ds]["preprocessed"] and listed[ds]["shape"] == [40, 2], listed[ds]
again = alice.get(f"/api/projects/{ds}").json()
assert again["restore"] == {"target": "y", "task": "classification"} and again["profile"]["rows"] == 40
assert alice.patch(f"/api/projects/{ds}", json={"title": "Pilot study"}).json()["title"] == "Pilot study"
assert alice.delete(f"/api/projects/{ds}").status_code == 200
assert alice.get(f"/api/projects/{ds}").status_code == 404
print("security check passed")
