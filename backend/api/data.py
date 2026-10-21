"""Data-phase routes: upload + audit, charts, cleaning, preprocessing, downloads,
sample datasets. Routes only -- the work happens in preprocessing/ and eda/.
Every route is behind api.auth.owner_guard (signed in + owns the dataset), except
`public` (upload limits, shown before sign-in)."""

import io
import json

import joblib
from fastapi import APIRouter, Body, Depends, HTTPException, Response, UploadFile

from api.auth import owner_guard
from api.common import ALLOWED, _payload, frame_or_404, read_table
from core import config, logs, store
from eda import plot
from models.user import User
from preprocessing import clean, execute

router = APIRouter(dependencies=[Depends(owner_guard)])
public = APIRouter()
log = logs.get("data")

# Bundled with scikit-learn (read from the installed package, no download):
# key -> (loader, kwargs, target column name, title, task)
SAMPLES = {
    "breast_cancer": ("load_breast_cancer", {}, "diagnosis", "Breast cancer diagnosis (Wisconsin)", "classification"),
    "diabetes": ("load_diabetes", {"scaled": False}, "progression", "Diabetes progression", "regression"),
    "iris": ("load_iris", {}, "species", "Iris flowers", "classification"),
    "wine": ("load_wine", {}, "cultivar", "Wine cultivars", "classification"),
}


@public.get("/api/upload/limits")
def upload_limits():
    """What the upload page may accept -- read from the backend so the page never
    keeps its own copy (MAX_UPLOAD_MB lives in backend/.env)."""
    return {"max_mb": config.MAX_UPLOAD_MB, "allowed": sorted(ALLOWED), "samples":
            [{"key": k, "title": v[3], "task": v[4], "target": v[2]} for k, v in SAMPLES.items()]}


@router.post("/api/analyze")
async def analyze(file: UploadFile, user: User = Depends(owner_guard)):
    notes = []
    df = await read_table(file, notes)
    ds_id = store.create(df, file.filename, owner=user.id)
    return {**_payload(df, file.filename, ds_id), "load_notes": notes}


@router.post("/api/samples/{name}")
def load_sample(name: str, user: User = Depends(owner_guard)):
    """Create a dataset from a sample bundled with scikit-learn."""
    if name not in SAMPLES:
        raise HTTPException(404, "Unknown sample dataset.")
    from sklearn import datasets
    loader, kwargs, target, title, task = SAMPLES[name]
    bunch = getattr(datasets, loader)(as_frame=True, **kwargs)
    df = bunch.frame.rename(columns={"target": target})
    names = getattr(bunch, "target_names", None)
    if task == "classification" and names is not None:
        df[target] = [str(names[i]) for i in df[target]]
    filename = f"{name}.csv"
    ds_id = store.create(df, filename, owner=user.id, title=title)
    return {**_payload(df, filename, ds_id), "load_notes": [f"sample dataset bundled with scikit-learn ({loader})"],
            "suggested": {"target": target, "task": task}}


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
    ops = payload.get("ops", [])
    listed = "; ".join(f"{o.get('op')} {o.get('column') or ''}".strip() for o in ops)
    log.info("clean %s (%s rows x %d columns): %d operation(s)%s", payload.get("id"), f"{df.shape[0]:,}", df.shape[1],
             len(ops), f": {listed[:300]}{' ...' if len(listed) > 300 else ''}" if ops else "")
    try:
        with logs.step(log, "apply cleaning"):
            cleaned, summary = clean.apply_clean(df, ops)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    with logs.step(log, "save cleaned table"):
        store.save_frame(payload["id"], cleaned)  # downstream tabs now see the cleaned frame
        # ops carry the decisions apply_clean made (date order, comma meaning) -> replayed identically
        store.add_clean_ops(payload["id"], ops)
    out = _payload(cleaned, payload.get("filename", "cleaned"), payload["id"])
    out["clean_summary"] = summary
    return out


@router.post("/api/clean/reset")
def reset_clean(payload: dict = Body(...)):
    """Undo all cleaning: restore the uploaded table. Body: {id}."""
    ds_id = payload.get("id")
    frame_or_404(ds_id)
    df = store.reset_frame(ds_id)
    if df is None:
        raise HTTPException(404, "No original copy for this dataset (uploaded before reset existed).")
    return _payload(df, payload.get("filename", "original"), ds_id)


@router.post("/api/preprocess")
def do_preprocess(payload: dict = Body(...)):
    """Apply a validated per-column op plan (fit on train only), return a preview
    + change summary. Body: {id, target, task, columns, test_size?, random_state?,
    stratify?, group?}. Clustering needs no target. Split settings fall back to
    execute.TEST_SIZE / RANDOM_STATE. The cleaned CSV and fitted pipeline are saved
    for /api/download and /api/pipeline."""
    df = frame_or_404(payload.get("id"))
    task = payload.get("task", "classification")
    if not payload.get("target") and task != "clustering":
        raise HTTPException(422, "No target column given.")
    try:
        with logs.step(log, f"preprocess {payload.get('id')} (target {payload.get('target') or 'none'}, {task})"):
            result = execute.apply_plan(
                df, payload.get("target"), payload.get("columns", {}),
                task=task,
                pipeline=payload.get("pipeline"),
                test_size=payload.get("test_size"),
                random_state=payload.get("random_state"),
                stratify=payload.get("stratify", execute.STRATIFY),
                group=payload.get("group") or None,
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
