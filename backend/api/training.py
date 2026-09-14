"""Training routes: options, start a run or an estimate, run status / results /
cancel / best model. Routes only -- the work happens in training/ and core/jobs.py."""

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse

from core import jobs, store
from training import dataview, suggest, train

router = APIRouter()


def _train_call(fn, *args):
    try:
        return fn(*args)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@router.get("/api/train/options")
def train_options(id: str):
    """Measured facts, every model with compatibility + suggestion, metrics, defaults."""
    return _train_call(train.options, id)


@router.post("/api/train/suggest")
def suggest_models(payload: dict = Body(...)):
    """Body: {id}. AI-ranked models from the allowed list (cached), or the rule
    suggestions with the reason when the AI can't answer."""
    return _train_call(suggest.suggest, payload.get("id"))


@router.get("/api/train/data")
def training_data(id: str, split: str = "all", offset: int = 0, limit: int = 50):
    """One page of the preprocessed training data + a summary of the model input."""
    return _train_call(dataview.data_page, id, split, offset, limit)


@router.post("/api/train")
def start_training(payload: dict = Body(...)):
    """Body: {id, models, folds?, primary_metric?, time_limit_min?, positive_class?} -> {run}."""
    ds_id = payload.get("id")
    settings = _train_call(train.validate, ds_id, payload, "train")
    return {"run": jobs.start(ds_id, lambda job: train.run(job, ds_id, settings), settings, settings["time_limit_s"])}


@router.post("/api/train/estimate")
def start_estimate(payload: dict = Body(...)):
    """Same body as /api/train; projects a time per model from small samples -> {run}."""
    ds_id = payload.get("id")
    settings = _train_call(train.validate, ds_id, payload, "estimate")
    return {"run": jobs.start(ds_id, lambda job: train.estimate(job, ds_id, settings), settings,
                              settings["time_limit_s"])}


@router.get("/api/runs")
def list_runs(id: str):
    return jobs.list_runs(id)


@router.get("/api/run")
def run_status(id: str, run: str):
    s = jobs.status(id, run)
    if s is None:
        raise HTTPException(404, "Run not found.")
    return s


@router.post("/api/run/cancel")
def cancel_run(payload: dict = Body(...)):
    """Stops a queued/running run between steps. Body: {id, run}."""
    try:
        return {"cancelled": jobs.cancel(payload.get("id"), payload.get("run"))}
    except KeyError:
        raise HTTPException(404, "Run not found.") from None


def _run_file(ds_id, run_id, name):
    try:
        return store.run_dir(ds_id, run_id) / name
    except KeyError:
        raise HTTPException(404, "Run not found.") from None


@router.get("/api/run/results")
def run_results(id: str, run: str):
    return store.read_json(_run_file(id, run, "results.json")) or {}


@router.get("/api/run/model")
def run_model(id: str, run: str, model: str = ""):
    """A run's fitted pipeline (joblib): `model` = a model key, default the best. It
    expects the cleaned feature columns and the same library versions; only load
    files you produced yourself."""
    res = store.read_json(_run_file(id, run, "results.json")) or {}
    key = model or res.get("best")
    if key not in (res.get("models") or {}):
        raise HTTPException(404, "No such model in this run.")
    path = _run_file(id, run, "models") / f"{key}.joblib"
    if not path.exists():
        raise HTTPException(404, "No model saved for this run.")
    return FileResponse(path, media_type="application/octet-stream", filename=f"{key}.joblib")


@router.get("/api/run/epochs")
def run_epochs(id: str, run: str, model: str):
    """Live / finished epoch curves of one model: {metric, total, fits: {name: [[epoch, train, val]]}}.
    `model` must be one of the run's models -- it becomes part of a file path."""
    settings = store.read_json(_run_file(id, run, "settings.json")) or {}
    if model not in (settings.get("models") or []):
        raise HTTPException(404, "No such model in this run.")
    return store.read_json(_run_file(id, run, "epochs") / f"{model}.json") or {}


@router.get("/api/runs/summary")
def runs_summary(id: str):
    return _train_call(train.runs, id)


@router.delete("/api/run")
def discard_run(id: str, run: str):
    _train_call(train.discard, id, run)
    return {"discarded": run}
