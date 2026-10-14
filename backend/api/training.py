"""Training routes: options, start a run / an estimate / a search / clustering,
run status / results / cancel / models, result charts, learning and validation
curves. Routes only -- the work happens in training/ and core/jobs.py.
A dataset preprocessed for clustering gets the clustering models (same routes)."""

import io

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from fastapi.responses import FileResponse

from api.auth import owner_guard
from core import jobs, store
from training import clustering, curves, dataview, diagnostics, models, suggest, train, tuning


router = APIRouter(dependencies=[Depends(owner_guard)])  # signed in + owns the dataset


def _train_call(fn, *args):
    try:
        return fn(*args)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


def _clustering(ds_id):
    prep = store.load_prep(ds_id)
    return bool(prep) and prep["fitted"]["settings"].get("task") == "clustering"


@router.get("/api/train/options")
def train_options(id: str):
    """Measured facts, every model with compatibility + suggestion, metrics, defaults."""
    return _train_call(clustering.options if _clustering(id) else train.options, id)


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
    """Body: {id, models, folds?, primary_metric?, time_limit_min?, positive_class?, params?,
    ensembles?} -> {run}. Clustering data: {id, models, params?, primary_metric?}."""
    ds_id = payload.get("id")
    if _clustering(ds_id):
        settings = _train_call(clustering.validate, ds_id, payload)
        return {"run": jobs.start(ds_id, lambda job: clustering.run(job, ds_id, settings), settings,
                                  settings["time_limit_s"])}
    settings = _train_call(train.validate, ds_id, payload, "train")
    return {"run": jobs.start(ds_id, lambda job: train.run(job, ds_id, settings), settings, settings["time_limit_s"])}


@router.post("/api/train/tune")
def start_tuning(payload: dict = Body(...)):
    """The /api/train body + {method: random|grid, n_trials?, space?: {model: {param: {low, high, log}
    | {choices}}}, grid?: {model: {param: [values]}}} -> {run}. Search, then train the winners."""
    ds_id = payload.get("id")
    settings = _train_call(tuning.validate, ds_id, payload)
    return {"run": jobs.start(ds_id, lambda job: tuning.tune(job, ds_id, settings), settings, settings["time_limit_s"])}


@router.get("/api/train/space")
def search_space(id: str):
    """Default search ranges per model (our starting point; editable on the page)."""
    prep = store.load_prep(id)
    if prep is None:
        raise HTTPException(404, "Run preprocessing first.")
    task = prep["fitted"]["settings"]["task"]
    if task == "clustering":
        return {}
    rows = models.table(task)
    return {k: {"space": tuning.space(k, r), "grid_size": tuning.grid_size(tuning.space(k, r))}
            for k, r in rows.items() if k != "baseline"}


@router.get("/api/run/leaders")
def run_leaders(id: str, run: str, k: int = tuning.LEADERS_K):
    """The run's top-k models by CV mean that beat the baseline (spec §6 leader rule)."""
    res = store.read_json(_run_file(id, run, "results.json")) or {}
    if "models" not in res or res.get("task") == "clustering":
        return []
    return tuning.leaders(res, max(1, k))


@router.get("/api/run/tuning")
def run_tuning(id: str, run: str):
    """Live search trials of a tuning run."""
    return store.read_json(_run_file(id, run, "tuning.json")) or {}


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


@router.get("/api/run/curves")
def run_curves(id: str, run: str, model: str):
    """Confusion matrix + ROC + PR of one model, from its saved test-set predictions
    (classification only). Available as soon as that model finishes."""
    res = store.read_json(_run_file(id, run, "results.json")) or {}
    if model not in (res.get("models") or {}):
        raise HTTPException(404, "No such model in this run.")
    preds = store.read_json(_run_file(id, run, "predictions") / f"{model}.json") or {}
    return curves.curves(res, preds)


@router.post("/api/run/learning")
def start_learning(payload: dict = Body(...)):
    """Body: {id, run, model} -> {job}: a learning curve on the run's folds (background)."""
    ds_id, run, key = payload.get("id"), payload.get("run"), payload.get("model")
    _run_file(ds_id, run, "results.json")
    return {"job": jobs.start(ds_id, lambda job: diagnostics.learning(job, ds_id, run, key),
                              {"kind": "learning_curve", "run": run, "model": key}, jobs.DEFAULT_TIME_LIMIT_S)}


@router.post("/api/run/validation")
def start_validation(payload: dict = Body(...)):
    """Body: {id, run, model, param} -> {job}: score across values of one parameter."""
    ds_id, run, key, param = payload.get("id"), payload.get("run"), payload.get("model"), payload.get("param")
    _run_file(ds_id, run, "results.json")
    return {"job": jobs.start(ds_id, lambda job: diagnostics.validation(job, ds_id, run, key, param),
                              {"kind": "validation_curve", "run": run, "model": key, "param": param},
                              jobs.DEFAULT_TIME_LIMIT_S)}


@router.get("/api/run/diagnostics")
def run_diagnostics(id: str, run: str, name: str):
    """A finished learning / validation curve: name = <model>_learning or <model>_validation_<param>."""
    if not name.replace("_", "").isalnum():
        raise HTTPException(404, "Not found.")
    out = diagnostics.cached(id, run, name)
    if out is None:
        raise HTTPException(404, "Not computed yet.")
    return out


@router.get("/api/run/clusters")
def run_clusters(id: str, run: str, model: str):
    """CSV of the clustered rows with their cluster (-1 = noise)."""
    res = store.read_json(_run_file(id, run, "results.json")) or {}
    if model not in (res.get("models") or {}):
        raise HTTPException(404, "No such method in this run.")
    df = _train_call(clustering.assignments, id, run, model)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=clusters_{model}.csv"})


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
