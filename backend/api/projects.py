"""Projects (plan §7 step 1, §8.1, §24): each uploaded dataset is a private
workspace -- list, reopen, rename, delete. Routes only; the files live in core/store."""

from fastapi import APIRouter, Body, Depends, HTTPException

from api.auth import owner_guard
from api.common import _payload, frame_or_404
from core import store
from models.user import User

router = APIRouter(dependencies=[Depends(owner_guard)])
ETHICS_MAX = 2000  # our starting point: characters kept of the ethics / consent note


@router.get("/api/projects")
def list_projects(user: User = Depends(owner_guard)):
    """This user's datasets with their history (runs, saved models), newest first."""
    return store.datasets(user.id)


@router.get("/api/projects/{id}")
def reopen(id: str):
    """Everything the studio needs to continue where the user left off: the current
    (cleaned) table's profile, and the target / task of the last preprocessing run."""
    m = store.meta(id) or {}
    if m.get("kind") == "bibliometric":
        return {"id": id, "kind": "bibliometric", "title": m.get("title"), "filename": m.get("filename")}
    df = frame_or_404(id)
    prep = store.load_prep(id)
    restore = None
    if prep:
        s = prep["fitted"]["settings"]
        restore = {"target": prep["fitted"].get("target"), "task": s.get("task")}
    return {**_payload(df, m.get("filename"), id), "restore": restore, "kind": "dataset"}


@router.patch("/api/projects/{id}")
def rename(id: str, title: str | None = Body(None), ethics: str | None = Body(None)):
    """Rename and/or record the ethics approval / consent note (plan §24) shown in the report."""
    try:
        if title is not None:
            store.rename(id, title)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if ethics is not None:
        store.update_meta(id, ethics=ethics.strip()[:ETHICS_MAX] or None)
    m = store.meta(id)
    return {"id": id, "title": m["title"], "ethics": m.get("ethics")}


@router.delete("/api/projects/{id}")
def delete(id: str):
    """Delete the dataset and everything made from it (cleaned data, runs, models).
    Refused while a training run is still going."""
    from core import jobs
    if any(r.get("state") in ("queued", "running") for r in jobs.list_runs(id)):
        raise HTTPException(409, "A run is still going - cancel it first.")
    store.delete_dataset(id)
    return {"deleted": id}
