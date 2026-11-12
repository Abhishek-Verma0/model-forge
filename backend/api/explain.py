"""Explainable-AI routes (plan §12): start an explanation of one run model as a
background job, read it back. Routes only -- the work happens in explain/."""

from fastapi import APIRouter, Body, Depends, HTTPException

from api.auth import owner_guard
from core import jobs, store
from explain import importance

router = APIRouter(dependencies=[Depends(owner_guard)])  # signed in + owns the dataset


@router.post("/api/explain/start")
def start(payload: dict = Body(...)):
    """Body: {id, run, model} -> {job} (or {cached: true} when it was explained already)."""
    ds_id, run, key = payload.get("id"), payload.get("run"), payload.get("model")
    try:
        if importance.cached(ds_id, run, key) is not None and not payload.get("again"):
            return {"cached": True}
        store.run_dir(ds_id, run)
    except KeyError:
        raise HTTPException(404, "Run not found.") from None
    return {"job": jobs.start(ds_id, lambda job: importance.explain(job, ds_id, run, key),
                              {"kind": "explain", "run": run, "model": key}, jobs.DEFAULT_TIME_LIMIT_S)}


@router.post("/api/explain/row")
def row(payload: dict = Body(...)):
    """Body: {id, run, model, row} -> one locked test row's prediction explained (SHAP where
    the model type has it, LIME always). row is 0-based."""
    from explain import local
    try:
        return local.explain_row(payload.get("id"), payload.get("run"), payload.get("model"), payload.get("row"))
    except KeyError:                       # a malformed id / run never becomes a path
        raise HTTPException(404, "Run not found.") from None
    except LookupError as exc:             # (KeyError is a LookupError: caught above first)
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.get("/api/explain")
def read(id: str, run: str, model: str):
    try:
        out = importance.cached(id, run, model)
    except KeyError:
        out = None
    if out is None:
        raise HTTPException(404, "Not explained yet.")
    return out
