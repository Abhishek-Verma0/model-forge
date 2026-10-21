"""Helpers shared by more than one router."""

import asyncio
import json
import logging
import math
import tempfile
import time
from importlib.util import find_spec
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core import config, logs, store
from eda import plot
from eda.charts import chart_data
from eda.eda import render_charts
from preprocessing import execute
from preprocessing.detector import run_quality_report
from preprocessing.profiler import load_dataset, profile_dataset

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
# .xls needs the optional xlrd reader -- only offered when it is installed.
ALLOWED = {".csv", ".tsv", ".json", ".xlsx"} | ({".xls"} if find_spec("xlrd") else set())

http_log, payload_log, upload_log = logs.get("http"), logs.get("payload"), logs.get("upload")
STILL_RUNNING_S = 15   # our starting point: how often a request that has not answered yet is reported
# polled every 1.5 s while a run is going -- logged only when slow or failing (or at LOG_LEVEL=DEBUG)
_QUIET = {"/api/run", "/api/run/results", "/api/run/epochs", "/api/run/tuning", "/health"}


class RequestLog:
    """Console lines for every request: when it arrives, when it answers (status +
    duration), and every STILL_RUNNING_S while it runs, with the step it is in. Pure
    ASGI middleware: it never reads the body, so uploads and the streamed chat pass
    through untouched."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] == "OPTIONS":   # CORS preflights: noise
            await self.app(scope, receive, send)
            return
        rid = logs.next_request_id()
        token = logs.request_id.set(rid)
        method, path = scope["method"], scope["path"]
        query = scope.get("query_string", b"").decode("latin-1")
        quiet = path in _QUIET
        http_log.log(logging.DEBUG if quiet else logging.INFO, "> %s %s%s", method, path, f"?{query}" if query else "")
        status = {}

        async def send_logged(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        async def watch():
            waited = 0
            while True:
                await asyncio.sleep(STILL_RUNNING_S)
                waited += STILL_RUNNING_S
                now = logs.doing(rid)
                http_log.warning("%s %s still running after %ds%s", method, path, waited, f" - now: {now}" if now else "")

        watcher = asyncio.create_task(watch())
        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_logged)
        finally:
            watcher.cancel()
            took, code = time.perf_counter() - start, status.get("code")
            level = (logging.WARNING if took >= STILL_RUNNING_S or code is None or code >= 500
                     else logging.DEBUG if quiet else logging.INFO)
            http_log.log(level, "< %s %s %s in %.2fs", code or "no answer", method, path, took)
            logs.done(rid)
            logs.request_id.reset(token)


def _finite(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


class SafeJSONResponse(JSONResponse):
    """JSON where NaN / inf become null. Statistics of a 1-row file or a one-value
    column are NaN (pandas std of one value), and the standard encoder refuses
    them -- every route answered HTTP 500 instead of "not computable"."""

    def render(self, content):
        return json.dumps(_finite(content), ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")


def _payload(df, filename, ds_id):
    """The full analyze response for a frame: profile + quality report + charts +
    a small row sample (for the Cleaning tab's preview grid). Shared by
    /api/analyze and /api/clean so cleaned data re-renders identically."""
    m = store.meta(ds_id) or {}
    payload_log.info("page data for %s: %s rows x %d columns", ds_id, f"{df.shape[0]:,}", df.shape[1])
    with logs.step(payload_log, "profile"):
        profile = profile_dataset(df)
    with logs.step(payload_log, "quality audit"):
        report = run_quality_report(df)
    with logs.step(payload_log, "chart data"):
        # builder: the columns each chart can draw -- an ID or many-valued text is never offered as one bar per value
        charts = {**chart_data(df), "builder": plot.builder_options(report["checks"])}
    with logs.step(payload_log, "EDA images"):
        eda = render_charts(df)
    return {"id": ds_id, "filename": filename, "title": m.get("title") or filename,
            "profile": profile,
            "report": report,
            "charts": charts,
            "eda": eda,
            "sample": json.loads(df.head(50).to_json(orient="records")),
            "preprocess_options": execute.options(),
            "can_reset": store.has_original(ds_id), "ethics": m.get("ethics")}


def frame_or_404(ds_id):
    df = store.load_frame(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found. Re-upload.")
    return df


async def read_upload(file: UploadFile, allowed=None):
    """An uploaded file's bytes, after the type and size checks every upload gets."""
    suffix = Path(file.filename or "").suffix.lower()
    allowed = ALLOWED if allowed is None else allowed
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Use {', '.join(sorted(allowed))}.")
    raw = await file.read()
    if not raw:
        raise HTTPException(422, "File is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File too large ({len(raw) // 1024 // 1024} MB). Max {config.MAX_UPLOAD_MB} MB.")
    return raw, suffix


async def read_table(file: UploadFile, notes=None):
    """An uploaded CSV / TSV / JSON / Excel file as a DataFrame, with the same checks
    everywhere. `notes` collects what reading changed (encoding, inf, decimal commas)."""
    raw, suffix = await read_upload(file)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        with logs.step(upload_log, f"read {file.filename} ({len(raw) / 1e6:.1f} MB)"):
            return load_dataset(tmp_path, notes)
    except Exception as exc:
        raise HTTPException(422, f"Could not read file: {exc}") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
