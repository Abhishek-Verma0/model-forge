"""Datasets, fitted pipelines and training runs on disk, so a restart or a new
upload loses nothing. Layout under ROOT:

  <dataset_id>/frame.pkl          current table (after cleaning)
  <dataset_id>/meta.json          {"filename", "title", "owner", "kind", "created", "clean_ops": [...]}
  <dataset_id>/prep.joblib        {"csv", "fitted"} from the last preprocessing run
  <dataset_id>/runs/<run_id>/     one folder per training run (jobs.py)

pickle/joblib keep pandas dtypes exactly, but can run code when loaded -- so this
module only loads files it wrote, and every id is validated before it becomes a
path. Every dataset has an owner (a user id); only the owner reaches it (api/auth.py
guard). The owner can delete it, and DATA_RETENTION_DAYS removes stale ones (plan §24).
"""

import json
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import joblib
import pandas as pd

from core import config

ROOT = Path(config.DATA_DIR)
CACHE_CAP = config.STORE_CAP  # frames kept in memory; disk is the source of truth (STORE_CAP in backend/.env)
_ID = re.compile(r"[0-9a-f]{12}")
_cache = OrderedDict()
_lock = threading.Lock()  # meta.json read-modify-write from concurrent requests


def _new_id():
    return uuid.uuid4().hex[:12]


def _check(an_id):
    if not isinstance(an_id, str) or not _ID.fullmatch(an_id):
        raise KeyError(an_id)
    return an_id


def _dir(ds_id):
    return ROOT / _check(ds_id)


# Windows refuses to replace a file another thread has open (and to open one being
# replaced) with PermissionError; the clash lasts microseconds. Measured 2026-10-31: a
# writer every 10 ms + 2 readers every 5 ms failed 12-24 of ~430 writes, 4-7 of ~1780
# reads. Retrying is the standard answer; elsewhere the first try succeeds.
SHARE_RETRIES = 40        # our starting point: x SHARE_WAIT_S = 2 s worst case
SHARE_WAIT_S = 0.05


def _retry(fn):
    for attempt in range(SHARE_RETRIES):
        try:
            return fn()
        except PermissionError:
            if attempt == SHARE_RETRIES - 1:
                raise
            time.sleep(SHARE_WAIT_S)


def write_json(path, obj):
    path = Path(path)
    # one temp file per thread: two writers of the same file never share a temp file
    tmp = path.with_suffix(f"{path.suffix}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
    _retry(lambda: tmp.replace(path))  # atomic: a crash never leaves half a file


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(_retry(lambda: path.read_text(encoding="utf-8")))
    except FileNotFoundError:   # deleted between the check and the read
        return None


def _remember(ds_id, df):
    _cache[ds_id] = df
    _cache.move_to_end(ds_id)
    while len(_cache) > CACHE_CAP:
        _cache.popitem(last=False)


def create(df, filename, owner=None, title=None, kind="dataset"):
    """kind: "dataset" (the analysis workflow) or "bibliometric" (publication records)."""
    ds_id = _new_id()
    _dir(ds_id).mkdir(parents=True)
    write_json(_dir(ds_id) / "meta.json", {"filename": filename, "title": (title or filename), "owner": owner,
                                           "kind": kind, "created": time.time(), "clean_ops": []})
    save_frame(ds_id, df)
    df.to_pickle(_dir(ds_id) / "original.pkl")   # never overwritten: what "reset" restores
    return ds_id


def owned_by(ds_id, user_id):
    """True only for an existing dataset of this user. Datasets created before
    accounts existed have no owner and belong to nobody."""
    m = meta(ds_id)
    return bool(m) and m.get("owner") is not None and m.get("owner") == user_id


def datasets(owner):
    """This user's datasets, newest change first: id, title, kind, rows/columns, runs."""
    out = []
    for p in ROOT.iterdir() if ROOT.exists() else []:
        if not _ID.fullmatch(p.name):
            continue
        m = read_json(p / "meta.json") or {}
        if m.get("owner") != owner:
            continue
        frame = _cache.get(p.name)
        out.append({"id": p.name, "title": m.get("title") or m.get("filename"), "filename": m.get("filename"),
                    "kind": m.get("kind", "dataset"), "created": m.get("created"), "updated": _touched(p),
                    "shape": m.get("shape") or (list(frame.shape) if frame is not None else None),
                    "runs": sum(1 for r in run_ids(p.name)   # training runs, not explanation / curve jobs
                                if (read_json(p / "runs" / r / "settings.json") or {}).get("kind") in ("train", "tune", "cluster")),
                    "saved_models": len(model_ids(p.name)),
                    "preprocessed": (p / "prep.joblib").exists(), "size_mb": size_mb(p)})
    return sorted(out, key=lambda d: d["updated"], reverse=True)


def _touched(path):
    return max((f.stat().st_mtime for f in path.rglob("*") if f.is_file()), default=path.stat().st_mtime)


def update_meta(ds_id, **fields):
    with _lock:
        m = meta(ds_id)
        m.update(fields)
        write_json(_dir(ds_id) / "meta.json", m)


def rename(ds_id, title):
    title = (title or "").strip()
    if not title:
        raise ValueError("Give the project a name.")
    with _lock:
        m = meta(ds_id)
        m["title"] = title[:200]
        write_json(_dir(ds_id) / "meta.json", m)


def delete_dataset(ds_id):
    """Remove a dataset and everything made from it (runs, models, results)."""
    path = _dir(ds_id)
    _cache.pop(ds_id, None)
    shutil.rmtree(path, ignore_errors=True)


def sweep(days, now=None):
    """Delete datasets not changed for `days` days (0 = keep everything). Returns
    the ids removed. Runs at startup (app.py)."""
    if days <= 0 or not ROOT.exists():
        return []
    cutoff = (now or time.time()) - days * 86400
    gone = [p.name for p in ROOT.iterdir() if _ID.fullmatch(p.name) and _touched(p) < cutoff]
    for ds_id in gone:
        delete_dataset(ds_id)
    return gone


def has_original(ds_id):
    return (_dir(ds_id) / "original.pkl").exists()


def reset_frame(ds_id):
    """Undo every cleaning step: restore the uploaded table and forget the ops.
    Returns the restored frame, or None when the dataset predates this copy."""
    path = _dir(ds_id) / "original.pkl"
    if not path.exists():
        return None
    df = pd.read_pickle(path)
    save_frame(ds_id, df)
    with _lock:
        m = meta(ds_id) or {}
        m["clean_ops"] = []
        write_json(_dir(ds_id) / "meta.json", m)
    return df


def save_frame(ds_id, df):
    path = _dir(ds_id) / "frame.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    df.to_pickle(tmp)
    tmp.replace(path)
    _remember(ds_id, df)
    with _lock:   # the project list shows the shape without loading every table
        m = meta(ds_id)
        if m is not None:
            m["shape"] = list(df.shape)
            write_json(_dir(ds_id) / "meta.json", m)


def load_frame(ds_id):
    """The current table, or None for an unknown or invalid id. Callers must not
    modify the returned frame in place (it is shared through the cache)."""
    try:
        path = _dir(ds_id) / "frame.pkl"
    except KeyError:
        return None
    if ds_id in _cache:
        _cache.move_to_end(ds_id)
        return _cache[ds_id]
    if not path.exists():
        return None
    df = pd.read_pickle(path)
    _remember(ds_id, df)
    return df


def meta(ds_id):
    try:
        return read_json(_dir(ds_id) / "meta.json")
    except KeyError:
        return None


def add_clean_ops(ds_id, ops):
    with _lock:
        m = meta(ds_id)
        m["clean_ops"] = m["clean_ops"] + list(ops)
        write_json(_dir(ds_id) / "meta.json", m)


def clean_ops(ds_id):
    return (meta(ds_id) or {}).get("clean_ops", [])


def save_prep(ds_id, csv, fitted):
    path = _dir(ds_id) / "prep.joblib"
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump({"csv": csv, "fitted": fitted}, tmp)
    tmp.replace(path)


def load_prep(ds_id):
    try:
        path = _dir(ds_id) / "prep.joblib"
    except KeyError:
        return None
    return joblib.load(path) if path.exists() else None


def new_run(ds_id):
    run_id = _new_id()
    run_dir(ds_id, run_id).mkdir(parents=True)
    return run_id


def run_dir(ds_id, run_id):
    return _dir(ds_id) / "runs" / _check(run_id)


def run_ids(ds_id):
    runs = _dir(ds_id) / "runs"
    return sorted(p.name for p in runs.iterdir() if _ID.fullmatch(p.name)) if runs.exists() else []


def delete_run(ds_id, run_id):
    shutil.rmtree(run_dir(ds_id, run_id), ignore_errors=True)


def model_dir(ds_id, model_id):
    return _dir(ds_id) / "models" / _check(model_id)


def model_ids(ds_id):
    root = _dir(ds_id) / "models"
    return sorted(p.name for p in root.iterdir() if _ID.fullmatch(p.name)) if root.exists() else []


def new_model_id(ds_id):
    model_id = _new_id()
    model_dir(ds_id, model_id).mkdir(parents=True)
    return model_id


def delete_model(ds_id, model_id):
    shutil.rmtree(model_dir(ds_id, model_id), ignore_errors=True)


def suggestion_path(ds_id, key):
    folder = _dir(ds_id) / "suggestions"
    folder.mkdir(exist_ok=True)
    return folder / f"{_check(key)}.json"


_ANALYSES = {"stats", "biblio"}   # folders of saved analysis results inside a dataset


def save_analysis(ds_id, kind, result):
    """Keep one analysis result (statistics, bibliometrics) for the history and the report."""
    if kind not in _ANALYSES:
        raise KeyError(kind)
    folder = _dir(ds_id) / kind
    folder.mkdir(exist_ok=True)
    a_id = _new_id()
    write_json(folder / f"{a_id}.json", {**result, "id": a_id, "created": time.time()})
    return a_id


def analyses(ds_id, kind):
    if kind not in _ANALYSES:
        raise KeyError(kind)
    folder = _dir(ds_id) / kind
    out = [read_json(p) for p in folder.glob("*.json") if _ID.fullmatch(p.stem)] if folder.exists() else []
    return sorted([a for a in out if a], key=lambda a: a["created"], reverse=True)


def delete_analysis(ds_id, kind, a_id):
    if kind not in _ANALYSES:
        raise KeyError(kind)
    (_dir(ds_id) / kind / f"{_check(a_id)}.json").unlink(missing_ok=True)


def size_mb(path):
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    return round(sum(p.stat().st_size for p in files) / 1e6, 3)


def all_run_dirs():
    if not ROOT.exists():
        return []
    return [p for p in ROOT.glob("*/runs/*") if _ID.fullmatch(p.parent.parent.name) and _ID.fullmatch(p.name)]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    import pandas as pd
    ROOT = Path(tempfile.mkdtemp())
    df = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64"), "t": ["x", "y", None]})
    ds = create(df, "demo.csv")
    _cache.clear()                                               # simulate a restart
    pd.testing.assert_frame_equal(load_frame(ds), df)            # dtypes survive (Int64 stays Int64)
    assert meta(ds)["filename"] == "demo.csv"
    add_clean_ops(ds, [{"op": "trim_whitespace", "column": "t"}])
    add_clean_ops(ds, [{"op": "drop_duplicates"}])
    assert [o["op"] for o in clean_ops(ds)] == ["trim_whitespace", "drop_duplicates"]
    assert load_prep(ds) is None
    save_prep(ds, "a,t\n", {"features": ["a"]})
    assert load_prep(ds)["fitted"]["features"] == ["a"]
    for bad in ("../../etc", "ABCDEF123456", "", None, "0" * 13):
        assert load_frame(bad) is None and meta(bad) is None     # never touches a path outside ROOT
    run = new_run(ds)
    write_json(run_dir(ds, run) / "status.json", {"state": "done"})
    assert run_ids(ds) == [run] and read_json(run_dir(ds, run) / "status.json") == {"state": "done"}
    assert all_run_dirs() == [run_dir(ds, run)]
    # a status file polled while it is rewritten: no reader or writer may fail (Windows sharing clash)
    status, errors, stop = run_dir(ds, run) / "status.json", [], time.time() + 1.5

    def hammer(fn):
        while time.time() < stop:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 -- collected and asserted below
                errors.append(repr(exc))

    workers = [threading.Thread(target=hammer, args=(lambda: write_json(status, {"state": "running", "n": 1}),)),
               *[threading.Thread(target=hammer, args=(lambda: read_json(status),)) for _ in range(2)]]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not errors, errors[:3]
    assert read_json(status) == {"state": "running", "n": 1}
    write_json(status, {"state": "done"})
    try:
        run_dir(ds, "../x")
        raise AssertionError("bad run id accepted")
    except KeyError:
        pass
    for _ in range(CACHE_CAP + 2):
        create(df, "x.csv")
    assert len(_cache) == CACHE_CAP

    # ownership, projects, deletion, retention
    mine = create(df, "mine.csv", owner=7, title="My study")
    assert owned_by(mine, 7) and not owned_by(mine, 8) and not owned_by(ds, 7)   # ds: no owner = nobody's
    assert not owned_by("../../etc", 7)
    listed = datasets(7)
    assert [d["id"] for d in listed] == [mine] and listed[0]["title"] == "My study" and listed[0]["shape"] == [3, 2]
    rename(mine, "  Renamed  ")
    assert datasets(7)[0]["title"] == "Renamed"
    delete_dataset(mine)
    assert datasets(7) == [] and load_frame(mine) is None
    old = create(df, "old.csv", owner=7)
    assert sweep(0) == [] and sweep(30) == []                        # fresh data is kept
    assert sweep(30, now=time.time() + 31 * 86400) and load_frame(old) is None
    ds = create(df, "a.csv", owner=7)                     # the sweep above removed the others
    a = save_analysis(ds, "stats", {"analysis": "describe"})
    assert [x["id"] for x in analyses(ds, "stats")] == [a]
    delete_analysis(ds, "stats", a)
    assert analyses(ds, "stats") == []
    for bad in (lambda: analyses(ds, "../x"), lambda: delete_analysis(ds, "stats", "../../meta")):
        try:
            bad()
            raise AssertionError("bad name accepted")
        except KeyError:
            pass
    print("store self-check passed")
