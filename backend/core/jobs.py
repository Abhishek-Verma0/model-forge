"""Background runs: one at a time, with progress, a time limit and cancel. Status
lives in runs/<run_id>/status.json, so a restart shows "interrupted" instead of
"running forever".

A job is fn(job): job.progress(**info) reports progress, job.check() raises
Stopped when the user cancelled or the time limit passed. Both are honoured
between steps only -- a fit already running in this thread finishes first.
ponytail: a thread, not a process; hard-killing one long fit needs a process
pool plus copying the data into it.
"""

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from core import store

DEFAULT_TIME_LIMIT_S = 3600   # spec §8: 60 min per run, our starting point
_ACTIVE = {"queued", "running"}
_pool = ThreadPoolExecutor(max_workers=1)  # our starting point -- one run at a time: a run already uses the cores and the GPU
_cancel = {}                  # run_id -> Event, only for runs owned by this process
_futures = {}
_lock = threading.Lock()


class Stopped(Exception):
    """Raised by Job.check(); its message becomes the run state."""


def _update(run_path, **fields):
    with _lock:
        s = store.read_json(run_path / "status.json") or {}
        s.update(fields)
        store.write_json(run_path / "status.json", s)


class Job:
    def __init__(self, ds_id, run_id, time_limit_s):
        self.ds_id, self.run_id, self.time_limit_s = ds_id, run_id, time_limit_s
        self.dir = store.run_dir(ds_id, run_id)
        self.started = None

    def check(self):
        if _cancel[self.run_id].is_set():
            raise Stopped("cancelled")
        if self.time_limit_s and time.time() - self.started > self.time_limit_s:
            raise Stopped("timed_out")

    def progress(self, **info):
        _update(self.dir, progress=info)


def start(ds_id, fn, settings, time_limit_s):
    run_id = store.new_run(ds_id)
    job = Job(ds_id, run_id, time_limit_s)
    store.write_json(job.dir / "settings.json", settings)
    _update(job.dir, state="queued", created=time.time(), time_limit_s=time_limit_s, progress={})
    _cancel[run_id] = threading.Event()
    _futures[run_id] = _pool.submit(_run, job, fn)
    return run_id


def _run(job, fn):
    state, error = "done", None
    if _cancel[job.run_id].is_set():
        state = "cancelled"
    else:
        job.started = time.time()
        _update(job.dir, state="running", started=job.started)
        try:
            fn(job)
        except Stopped as stop:
            state = str(stop)
        except Exception as exc:  # noqa: BLE001 -- any failure must end as a visible state
            traceback.print_exc()
            state, error = "failed", f"{type(exc).__name__}: {exc}"
    _update(job.dir, state=state, error=error, finished=time.time())
    _cancel.pop(job.run_id, None)


def wait(run_id, timeout=None):
    _futures[run_id].result(timeout)


def cancel(ds_id, run_id):
    store.run_dir(ds_id, run_id)  # validates both ids; KeyError on bad input
    event = _cancel.get(run_id)
    if event is None:             # finished, or owned by a previous process
        return False
    event.set()
    return True


def status(ds_id, run_id):
    try:
        return store.read_json(store.run_dir(ds_id, run_id) / "status.json")
    except KeyError:
        return None


def list_runs(ds_id):
    try:
        ids = store.run_ids(ds_id)
    except KeyError:
        return []
    runs = [{"run_id": r, **(status(ds_id, r) or {})} for r in ids]
    return sorted(runs, key=lambda s: s.get("created", 0), reverse=True)


def mark_interrupted():
    n = 0
    for run_path in store.all_run_dirs():
        s = store.read_json(run_path / "status.json") or {}
        if s.get("state") in _ACTIVE and run_path.name not in _cancel:
            _update(run_path, state="interrupted", finished=time.time())
            n += 1
    return n


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    import pandas as pd
    store.ROOT = Path(tempfile.mkdtemp())
    ds = store.create(pd.DataFrame({"a": [1]}), "d.csv")

    def work(job):
        for i in range(3):
            job.check()
            job.progress(step=i + 1, total=3)
    r = start(ds, work, {"kind": "demo"}, time_limit_s=60)
    wait(r)
    s = status(ds, r)
    assert s["state"] == "done" and s["progress"] == {"step": 3, "total": 3}, s
    assert store.read_json(store.run_dir(ds, r) / "settings.json") == {"kind": "demo"}

    gate = threading.Event()
    def slow(job):
        gate.wait(5)
        job.check()
    r = start(ds, slow, {}, time_limit_s=60)
    assert cancel(ds, r)
    gate.set()
    wait(r)
    assert status(ds, r)["state"] == "cancelled", status(ds, r)
    assert not cancel(ds, r)                                  # finished runs can't be cancelled

    def late(job):
        time.sleep(0.3)
        job.check()
    r = start(ds, late, {}, time_limit_s=0.1)
    wait(r)
    assert status(ds, r)["state"] == "timed_out", status(ds, r)

    def boom(job):
        raise RuntimeError("bad data")
    r = start(ds, boom, {}, time_limit_s=60)
    wait(r)
    assert status(ds, r)["state"] == "failed" and "bad data" in status(ds, r)["error"], status(ds, r)

    ghost = store.new_run(ds)                                 # left "running" by a dead process
    store.write_json(store.run_dir(ds, ghost) / "status.json", {"state": "running", "created": 0})
    assert mark_interrupted() == 1
    assert status(ds, ghost)["state"] == "interrupted"
    runs = list_runs(ds)
    assert len(runs) == 5 and runs[0]["run_id"] == r and runs[-1]["run_id"] == ghost, [x["run_id"] for x in runs]
    print("jobs self-check passed")
