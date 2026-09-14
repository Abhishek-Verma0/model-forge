"""Data-phase routes: upload + audit, charts, cleaning, preprocessing, downloads.
Routes only -- the work happens in preprocessing/ and eda/."""

import io
import json

import joblib
from fastapi import APIRouter, Body, HTTPException, Response, UploadFile

from api.common import ALLOWED, frame_or_404, read_table
from core import config, store
from eda import plot
from eda.charts import chart_data
from eda.eda import render_charts
from preprocessing import clean, execute
from preprocessing.detector import run_quality_report
from preprocessing.profiler import profile_dataset

router = APIRouter()


def _payload(df, filename, ds_id):
    """The full analyze response for a frame: profile + quality report + charts +
    a small row sample (for the Cleaning tab's preview grid). Shared by
    /api/analyze and /api/clean so cleaned data re-renders identically."""
    return {"id": ds_id, "filename": filename,
            "profile": profile_dataset(df),
            "report": run_quality_report(df),
            "charts": chart_data(df),
            "eda": render_charts(df),
            "sample": json.loads(df.head(50).to_json(orient="records")),
            "preprocess_options": execute.options()}


@router.get("/api/upload/limits")
def upload_limits():
    """What the upload page may accept -- read from the backend so the page never
    keeps its own copy (MAX_UPLOAD_MB lives in backend/.env)."""
    return {"max_mb": config.MAX_UPLOAD_MB, "allowed": sorted(ALLOWED)}


@router.post("/api/analyze")
async def analyze(file: UploadFile):
    df = await read_table(file)
    ds_id = store.create(df, file.filename)
    return _payload(df, file.filename, ds_id)


@router.get("/api/chart")
def chart(id: str, kind: str, x: str = "", y: str = "", hue: str = "", fmt: str = "png"):
    df = frame_or_404(id)
    try:
        img, media = plot.render(df, kind, x=x or None, y=y or None, hue=hue or None, fmt=fmt)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return Response(content=img, media_type=media)


@router.post("/api/clean")
def do_clean(payload: dict = Body(...)):
    """Apply validated per-cell clean ops to the WHOLE frame (before split),
    save the cleaned frame, and return the re-profiled payload + a summary.
    Body: {id, ops}."""
    df = frame_or_404(payload.get("id"))
    try:
        cleaned, summary = clean.apply_clean(df, payload.get("ops", []))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    store.save_frame(payload["id"], cleaned)  # downstream tabs now see the cleaned frame
    store.add_clean_ops(payload["id"], payload.get("ops", []))
    out = _payload(cleaned, payload.get("filename", "cleaned"), payload["id"])
    out["clean_summary"] = summary
    return out


@router.post("/api/preprocess")
def do_preprocess(payload: dict = Body(...)):
    """Apply a validated per-column op plan (fit on train only), return a preview
    + change summary. Body: {id, target, task, columns, test_size?, random_state?,
    stratify?}. Split settings fall back to execute.TEST_SIZE / RANDOM_STATE. The cleaned CSV and
    fitted pipeline are saved for /api/download and /api/pipeline."""
    df = frame_or_404(payload.get("id"))
    if not payload.get("target"):
        raise HTTPException(422, "No target column given.")
    try:
        result = execute.apply_plan(
            df, payload["target"], payload.get("columns", {}),
            task=payload.get("task", "classification"),
            pipeline=payload.get("pipeline"),
            test_size=payload.get("test_size"),
            random_state=payload.get("random_state"),
            stratify=payload.get("stratify", execute.STRATIFY),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    fitted = result.pop("fitted")
    fitted["clean_ops"] = store.clean_ops(payload["id"])
    store.save_prep(payload["id"], result.pop("csv"), fitted)  # big CSV stays out of the JSON response
    return result


def _prep(ds_id):
    prep = store.load_prep(ds_id)
    if prep is None:
        raise HTTPException(404, "No preprocessing run yet. Run preprocessing first.")
    return prep


@router.get("/api/download")
def download(id: str):
    return Response(
        content=_prep(id)["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned.csv"},
    )


@router.get("/api/pipeline")
def download_pipeline(id: str):
    """The fitted pipeline (joblib): clean ops + every train-fitted step + text
    TF-IDF. Load with joblib and use preprocessing.execute.transform / to_matrix,
    same scikit-learn version. Only ever load a pipeline file you produced
    yourself -- joblib files can run code."""
    buf = io.BytesIO()
    joblib.dump(_prep(id)["fitted"], buf)
    return Response(content=buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=pipeline.joblib"})
