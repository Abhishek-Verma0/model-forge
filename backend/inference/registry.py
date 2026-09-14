"""Trained models you can test and keep: a model from a run, or a saved copy.

Saving copies the fitted pipeline plus everything needed to use it later without
the dataset's current state: clean ops, features, classes, target, task, positive
class, parameters, test metrics, fingerprint and library versions. Loading refuses
when installed library versions differ from the recorded ones -- joblib files are
not reliable across versions, and they can run code, so only files this app wrote
are ever loaded.
"""

import shutil
import time

import joblib

from core import store
from training import train


def _run_results(ds_id, run_id):
    try:
        res = store.read_json(store.run_dir(ds_id, run_id) / "results.json")
    except KeyError:
        raise LookupError("Run not found.") from None
    if not res or "models" not in res:
        raise LookupError("This run has no trained models.")
    return res


def run_meta(ds_id, run_id, key):
    res = _run_results(ds_id, run_id)
    key = key or res.get("best")
    rec = res["models"].get(key) if key else None
    if not rec or "error" in rec:
        raise LookupError(f"No trained model '{key}' in this run.")
    return {"source": "run", "run_id": run_id, "key": key, "label": rec["label"], "task": res["task"],
            "target": res["target"], "classes": res["classes"], "positive_class": res["positive_class"],
            "primary_metric": res["primary_metric"], "features": res["features"], "clean_ops": res["clean_ops"],
            "fingerprint": res["fingerprint"], "versions": res["versions"], "params": rec.get("params"),
            "test": rec.get("test"), "cv": {m: v["mean"] for m, v in (rec.get("cv") or {}).items()},
            "schema": rec.get("schema"),
            "file": str(store.run_dir(ds_id, run_id) / "models" / f"{key}.joblib")}


def _check_versions(meta):
    now = train.versions()
    diff = [f"{lib} {v} (now {now.get(lib, 'not installed')})"
            for lib, v in (meta.get("versions") or {}).items() if now.get(lib) != v]
    if diff:
        raise ValueError("This model was trained with different library versions: " + ", ".join(diff)
                         + ". Install the pinned versions from requirements.txt to use it.")


def load(meta):
    _check_versions(meta)
    return joblib.load(meta["file"])


def save(ds_id, run_id, key, name):
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the model a name.")
    meta = run_meta(ds_id, run_id, key)
    model_id = store.new_model_id(ds_id)
    folder = store.model_dir(ds_id, model_id)
    shutil.copyfile(meta["file"], folder / "model.joblib")
    meta.update(source="saved", model_id=model_id, name=name, created=time.time(), file=str(folder / "model.joblib"),
                size_mb=store.size_mb(folder / "model.joblib"))
    store.write_json(folder / "meta.json", meta)
    return meta


def saved(ds_id):
    try:
        ids = store.model_ids(ds_id)
    except KeyError:
        return []
    metas = [store.read_json(store.model_dir(ds_id, m) / "meta.json") for m in ids]
    return sorted([m for m in metas if m], key=lambda m: m["created"], reverse=True)


def saved_meta(ds_id, model_id):
    try:
        meta = store.read_json(store.model_dir(ds_id, model_id) / "meta.json")
    except KeyError:
        meta = None
    if not meta:
        raise LookupError("Saved model not found.")
    meta["file"] = str(store.model_dir(ds_id, model_id) / "model.joblib")  # path is always rebuilt, never trusted
    return meta


def delete(ds_id, model_id):
    saved_meta(ds_id, model_id)
    store.delete_model(ds_id, model_id)
