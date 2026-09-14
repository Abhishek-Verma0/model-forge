"""Helpers shared by more than one router."""

import tempfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from core import config, store
from preprocessing.profiler import load_dataset

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
ALLOWED = {".csv", ".xlsx", ".xls"}


def frame_or_404(ds_id):
    df = store.load_frame(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found. Re-upload.")
    return df


async def read_table(file: UploadFile):
    """An uploaded CSV / Excel file as a DataFrame, with the same checks everywhere."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Use CSV or Excel.")
    raw = await file.read()
    if not raw:
        raise HTTPException(422, "File is empty.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File too large ({len(raw) // 1024 // 1024} MB). Max {config.MAX_UPLOAD_MB} MB.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return load_dataset(tmp_path)
    except Exception as exc:
        raise HTTPException(422, f"Could not read file: {exc}") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
