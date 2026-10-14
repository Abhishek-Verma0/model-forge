"""Helpers shared by more than one router."""

import json
import math
import tempfile
from importlib.util import find_spec
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core import config, store
from eda.charts import chart_data
from eda.eda import render_charts
from preprocessing import execute
from preprocessing.detector import run_quality_report
from preprocessing.profiler import load_dataset, profile_dataset

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
# .xls needs the optional xlrd reader -- only offered when it is installed.
ALLOWED = {".csv", ".tsv", ".json", ".xlsx"} | ({".xls"} if find_spec("xlrd") else set())


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
    return {"id": ds_id, "filename": filename, "title": m.get("title") or filename,
            "profile": profile_dataset(df),
            "report": run_quality_report(df),
            "charts": chart_data(df),
            "eda": render_charts(df),
            "sample": json.loads(df.head(50).to_json(orient="records")),
            "preprocess_options": execute.options(),
            "can_reset": store.has_original(ds_id)}


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
        return load_dataset(tmp_path, notes)
    except Exception as exc:
        raise HTTPException(422, f"Could not read file: {exc}") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
