"""Backend console logging: where a request is, how long each step took, and what
is stuck.

Every module logs through a child of the "modelforge" logger (get("clean") ->
modelforge.clean); one console handler prints them in the uvicorn window:

  14:02:11 INFO    #7   modelforge.http: > POST /api/clean
  14:02:11 INFO    #7   modelforge.payload: EDA images ...
  14:02:26 WARNING #7   modelforge.http: POST /api/clean still running after 15s - now: EDA images
  14:03:40 INFO    #7   modelforge.payload: EDA images done in 89.10s
  14:03:40 WARNING #7   modelforge.http: < 200 POST /api/clean in 89.42s

#7 is the request's number, so one request can be followed when several overlap; a
step that logged "..." without its "done" line is where a request is stuck.
LOG_LEVEL in backend/.env: INFO (default), DEBUG adds the polling requests,
WARNING keeps only slow and failing ones.
"""

import contextvars
import itertools
import logging
import sys
import time
from contextlib import contextmanager

from core import config

request_id = contextvars.ContextVar("request_id", default="-")   # set by api.common.RequestLog
_ids = itertools.count(1)
_doing = {}   # request number -> the step it is in now (the "still running" line shows it)


def next_request_id():
    return str(next(_ids))


def doing(rid):
    return _doing.get(rid)


def done(rid):
    _doing.pop(rid, None)


def get(name):
    return logging.getLogger(f"modelforge.{name}")


class _Console(logging.StreamHandler):
    """stderr, like uvicorn. A character the console can't print (a column named
    "Café" in a cp1252 window) is escaped instead of breaking the line."""

    def format(self, record):
        record.rid = request_id.get()
        enc = getattr(self.stream, "encoding", None) or "utf-8"
        return super().format(record).encode(enc, "backslashreplace").decode(enc)


_root = logging.getLogger("modelforge")
if not _root.handlers:   # once per process
    _handler = _Console(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s #%(rid)-4s %(name)s: %(message)s",
                                            "%H:%M:%S"))
    _root.addHandler(_handler)
    _root.propagate = False   # uvicorn's root logger would print each line a second time
_root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))   # an unknown name means INFO


@contextmanager
def step(log, what):
    """`what ...` when a step starts, `what done in N s` when it ends (or `failed`)."""
    rid = request_id.get()
    before = _doing.get(rid)
    if rid != "-":
        _doing[rid] = what
    log.info("%s ...", what)
    t = time.perf_counter()
    try:
        yield
    except Exception as exc:
        log.warning("%s failed after %.2fs: %s", what, time.perf_counter() - t, exc)
        raise
    finally:
        if rid != "-":
            if before is None:
                _doing.pop(rid, None)
            else:
                _doing[rid] = before
    log.info("%s done in %.2fs", what, time.perf_counter() - t)


if __name__ == "__main__":
    import io
    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    _root.handlers[0].setStream(buf)
    log = get("selfcheck")
    token = request_id.set("42")
    with step(log, "outer"):
        with step(log, "inner Café ☃"):   # ☃ is not in cp1252 -> escaped, not an error
            assert doing("42") == "inner Café ☃"
        assert doing("42") == "outer"
    assert doing("42") is None
    try:
        with step(log, "boom"):
            raise ValueError("bad")
    except ValueError:
        pass
    request_id.reset(token)
    buf.flush()
    out = buf.buffer.getvalue().decode("cp1252")
    assert "#42" in out and "outer done in" in out and "inner Café \\u2603 done in" in out, out
    assert "boom failed after" in out and "bad" in out, out
    print("logs self-check passed")
