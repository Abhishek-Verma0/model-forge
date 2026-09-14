"""Test and keep trained models: predict on an uploaded table or browse the locked
test set; save any run model under a name; list / download / delete saved models.
Routes only -- the work happens in inference/."""

import io

from fastapi import APIRouter, Body, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from api.common import read_table
from inference import predict as P
from inference import registry

router = APIRouter()


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


def _meta(ds_id, run, model, saved):
    if saved:
        return _call(registry.saved_meta, ds_id, saved)
    if not run:
        raise HTTPException(422, "Choose a run model or a saved model.")
    return _call(registry.run_meta, ds_id, run, model or None)


@router.post("/api/predict")
async def predict(file: UploadFile = File(...), id: str = Form(...), run: str = Form(""), model: str = Form(""),
                  saved: str = Form(""), format: str = Form("json")):
    """Multipart: file + id + (run [+ model] | saved) + format json|csv."""
    meta = _meta(id, run, model, saved)
    pipe = _call(registry.load, meta)
    raw = await read_table(file)
    out = _call(P.predict, pipe, meta, raw)
    if format == "csv":
        buf = io.StringIO()
        out["table"].to_csv(buf, index=False)
        return Response(content=buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    out.pop("table")
    return {**out, "model": {k: meta.get(k) for k in ("label", "name", "task", "target", "classes", "source")}}


@router.get("/api/predict/schema")
def input_schema(id: str, run: str = "", model: str = "", saved: str = ""):
    """The fields a manual prediction needs, from the model's saved input schema."""
    meta = _meta(id, run, model, saved)
    return {k: meta.get(k) for k in ("label", "name", "task", "target", "classes", "schema")}


@router.post("/api/predict/row")
def predict_row(payload: dict = Body(...)):
    """Body: {id, run?, model?, saved?, values: {field: value}} -- validated against the
    model's saved input schema, then the same prediction path as an uploaded file."""
    meta = _meta(payload.get("id"), payload.get("run"), payload.get("model"), payload.get("saved"))
    pipe = _call(registry.load, meta)
    frame, notes = _call(P.row_frame, meta, payload.get("values") or {})
    out = _call(P.predict, pipe, meta, frame, True)
    out.pop("table")
    return {**out, "notes": notes + out["notes"],
            "model": {k: meta.get(k) for k in ("label", "name", "task", "target", "classes", "source")}}


@router.get("/api/run/testset")
def test_set(id: str, run: str, model: str = "", mistakes: bool = False, offset: int = 0, limit: int = 50):
    return _call(P.test_set, id, run, model or None, mistakes, offset, limit)


@router.get("/api/models")
def list_models(id: str):
    return [{k: v for k, v in m.items() if k != "file"} for m in registry.saved(id)]


@router.post("/api/models/save")
def save_model(payload: dict = Body(...)):
    """Body: {id, run, model?, name}; model defaults to the run's best."""
    meta = _call(registry.save, payload.get("id"), payload.get("run"), payload.get("model"), payload.get("name"))
    return {k: v for k, v in meta.items() if k != "file"}


@router.delete("/api/models")
def delete_model(id: str, model: str):
    _call(registry.delete, id, model)
    return {"deleted": model}


@router.get("/api/models/download")
def download_model(id: str, model: str):
    meta = _call(registry.saved_meta, id, model)
    return FileResponse(meta["file"], media_type="application/octet-stream", filename=f"{meta['name']}.joblib")
