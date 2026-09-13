"""Datasets, fitted pipelines and training runs on disk, so a restart or a new
upload loses nothing. Layout under ROOT:

  <dataset_id>/frame.pkl          current table (after cleaning)
  <dataset_id>/meta.json          {"filename", "created", "clean_ops": [...]}
  <dataset_id>/prep.joblib        {"csv", "fitted"} from the last preprocessing run
  <dataset_id>/runs/<run_id>/     one folder per training run (jobs.py)

pickle/joblib keep pandas dtypes exactly, but can run code when loaded -- so this
module only loads files it wrote, and every id is validated before it becomes a
path. ponytail: no deletion/retention yet (plan §24) -- add with auth.
"""

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import joblib
import pandas as pd

import config

ROOT = Path(config.DATA_DIR)
CACHE_CAP = 5            # frames kept in memory; disk is the source of truth (our starting point)
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


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves half a file


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _remember(ds_id, df):
    _cache[ds_id] = df
    _cache.move_to_end(ds_id)
    while len(_cache) > CACHE_CAP:
        _cache.popitem(last=False)


def create(df, filename):
    ds_id = _new_id()
    _dir(ds_id).mkdir(parents=True)
    write_json(_dir(ds_id) / "meta.json", {"filename": filename, "created": time.time(), "clean_ops": []})
    save_frame(ds_id, df)
    return ds_id


def save_frame(ds_id, df):
    path = _dir(ds_id) / "frame.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    df.to_pickle(tmp)
    tmp.replace(path)
    _remember(ds_id, df)


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
    try:
        run_dir(ds, "../x")
        raise AssertionError("bad run id accepted")
    except KeyError:
        pass
    for _ in range(CACHE_CAP + 2):
        create(df, "x.csv")
    assert len(_cache) == CACHE_CAP
    print("store self-check passed")
