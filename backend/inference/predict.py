"""Use a trained model on data (spec 2026-09-14): an uploaded table of new rows, or
the run's locked test set. Works for any dataset: it replays the model's own
recorded clean ops, needs the model's feature columns, maps predictions back to
class names, and scores only rows that carry a usable target value.
"""

import numpy as np
import pandas as pd

from inference import registry
from preprocessing import clean
from preprocessing.profiler import missing_mask
from core import store
from training import train
from training.metrics import score, scorers

PREVIEW_ROWS = 200   # our starting point: rows returned as JSON; the CSV download has all rows
TEST_PAGE_MAX = 500  # our starting point: largest test-set page


def _encode_target(meta, y):
    """(usable row mask, encoded y) for scoring; unusable = missing or unknown label."""
    usable = ~missing_mask(y)
    if meta["task"] == "classification":
        known = y.astype(str).isin(meta["classes"])
        usable &= known
        codes = y[usable].astype(str).map({c: i for i, c in enumerate(meta["classes"])}).to_numpy(dtype=int)
        return usable, codes, int((~known & ~missing_mask(y)).sum())
    num = pd.to_numeric(y, errors="coerce")
    bad = int((num.isna() & ~missing_mask(y)).sum())
    usable &= num.notna()
    return usable, num[usable].to_numpy(dtype=float), bad


def _metrics(pipe, meta, X, y_codes):
    task = meta["task"]
    n_classes = len(meta["classes"] or [])
    pos = meta["classes"].index(meta["positive_class"]) if meta.get("positive_class") else None
    has_zero = task == "regression" and bool((y_codes == 0).any())
    sc = scorers(task, n_classes, pos, hasattr(pipe, "predict_proba"), has_zero)
    out = {}
    for name, scorer in sc.items():
        try:
            out[name] = score(name, scorer, pipe, X, y_codes)
        except ValueError:  # e.g. only one class present in the scored rows
            out[name] = None
    return out


def row_frame(meta, values):
    """One manual row, validated ONLY against the model's saved input schema (the
    categories and columns its fitted pipeline accepts). Values are already in the
    cleaned form the schema lists, so predict(..., cleaned=True) skips clean ops."""
    schema = meta.get("schema")
    if not schema:
        raise ValueError("This model was trained before input forms existed - upload a file to test it.")
    names = {f["name"] for f in schema}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}.")
    row, notes = {}, []
    for f in schema:
        v = values.get(f["name"])
        blank = v is None or (isinstance(v, str) and not v.strip())
        if blank:
            if not f["missing_ok"]:
                raise ValueError(f"{f['name']} is required - the model has no rule for a missing value here.")
            row[f["name"]] = "" if f["kind"] == "text" else np.nan
            continue
        if f["kind"] == "number":
            try:
                num = float(v)
                if not np.isfinite(num):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"{f['name']} must be a number (got {v!r}).") from None
            if f["min"] is not None and not f["min"] <= num <= f["max"]:
                notes.append(f"{f['name']} = {num:g} is outside the training range {f['min']:g} to {f['max']:g}")
            row[f["name"]] = num
        elif f["kind"] == "category":
            if str(v) not in f["choices"]:
                raise ValueError(f"{f['name']} must be one of the values the model was trained on: "
                                 f"{', '.join(f['choices'][:20])}{' ...' if len(f['choices']) > 20 else ''}.")
            row[f["name"]] = str(v)
        elif f["kind"] == "boolean":
            if not isinstance(v, bool):
                raise ValueError(f"{f['name']} must be true or false.")
            row[f["name"]] = v
        else:
            row[f["name"]] = str(v)
    return pd.DataFrame([row]), notes


def predict(pipe, meta, raw, cleaned=False):
    """Predictions for every row of `raw`: an uploaded table in the ORIGINAL columns
    (clean ops replayed), or schema-validated manual rows (cleaned=True). Everything
    after that step is the same path."""
    ops = [] if cleaned else [o for o in meta["clean_ops"] if o["op"] != "drop_duplicates"]
    df = clean.apply_clean(raw, ops)[0] if ops else raw.copy()
    df = df.reset_index(drop=True)
    missing = [c for c in meta["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(map(str, missing[:10]))}"
                         f"{' …' if len(missing) > 10 else ''}. The file needs the columns the model was trained on.")
    X = df[meta["features"]]
    pred = np.asarray(pipe.predict(X))
    out = pd.DataFrame(index=df.index)
    if meta["task"] == "classification":
        out["prediction"] = [meta["classes"][int(i)] for i in pred]
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            for i, c in enumerate(meta["classes"]):
                out[f"probability {c}"] = np.round(proba[:, i], 6)
    else:
        out["prediction"] = pred
    notes, metrics, scored = [], None, 0
    target = meta["target"]
    if target in df.columns:
        usable, y_codes, unusable = _encode_target(meta, df[target])
        scored = int(usable.sum())
        if unusable:
            notes.append(f"{unusable} row(s) have a {target} value the model never saw - not scored")
        if scored:
            metrics = _metrics(pipe, meta, X[usable.to_numpy()], y_codes)
    table = pd.concat([raw.reset_index(drop=True), out], axis=1)
    return {"rows": len(df), "scored_rows": scored, "metrics": metrics, "notes": notes,
            "columns": list(map(str, table.columns)),
            "preview": table.head(PREVIEW_ROWS).astype(object).where(table.head(PREVIEW_ROWS).notna(), None)
                            .to_dict(orient="records"),
            "table": table}


def test_set(ds_id, run_id, key, mistakes_only=False, offset=0, limit=50):
    """The run's locked test rows with the stored predictions. Refuses when the data
    or preprocessing changed since the run (the rows could no longer be matched)."""
    meta = registry.run_meta(ds_id, run_id, key)
    ctx = train.load_context(ds_id)
    res = store.read_json(store.run_dir(ds_id, run_id) / "results.json")
    if train.fingerprint(ctx, res["folds"]) != meta["fingerprint"]:
        raise ValueError("The data or preprocessing changed since this run - its test rows can't be matched.")
    pred = store.read_json(store.run_dir(ds_id, run_id) / "predictions" / f"{meta['key']}.json")
    y_true, y_pred = np.asarray(pred["y_true"]), np.asarray(pred["y_pred"])
    if not np.array_equal(y_true, np.asarray(ctx["y_te"])):
        raise ValueError("Stored test predictions don't match the rebuilt test rows.")
    frame = ctx["X_te"].reset_index().rename(columns={"index": "row"})
    if meta["task"] == "classification":
        frame["actual"] = [meta["classes"][int(i)] for i in y_true]
        frame["prediction"] = [meta["classes"][int(i)] for i in y_pred]
        frame["correct"] = y_true == y_pred
        if pred.get("proba"):
            frame["confidence"] = np.max(np.asarray(pred["proba"]), axis=1).round(4)
    else:
        frame["actual"], frame["prediction"] = y_true, y_pred
        frame["error"] = (y_pred - y_true).round(6)
        frame["correct"] = None
    if mistakes_only and meta["task"] == "classification":
        frame = frame[~frame["correct"]]
    limit = max(1, min(int(limit), TEST_PAGE_MAX))
    page = frame.iloc[int(offset):int(offset) + limit]
    return {"total": len(frame), "offset": int(offset), "limit": limit, "columns": list(map(str, page.columns)),
            "rows": page.astype(object).where(page.notna(), None).to_dict(orient="records")}


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path
    from core import jobs
    from preprocessing import execute
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    n = 300
    raw = pd.DataFrame({"Note ": [f" {'refund broken' if i % 3 == 0 else 'love it'} {i} " for i in range(n)],
                        "price": rng.normal(50, 5, n), "y": ["bad" if i % 3 == 0 else "good" for i in range(n)]})
    ds = store.create(raw, "t.csv")
    ops = [{"op": "rename_column", "column": "Note ", "to": "note"}, {"op": "trim_whitespace", "column": "note"}]
    cleaned, _ = clean.apply_clean(raw, ops)
    store.save_frame(ds, cleaned)
    store.add_clean_ops(ds, ops)
    r = execute.apply_plan(cleaned, "y", {"note": [{"op": "encode", "method": "text"}]})
    r["fitted"]["clean_ops"] = store.clean_ops(ds)
    store.save_prep(ds, r["csv"], r["fitted"])
    s = train.validate(ds, {"models": ["logistic_regression", "lightgbm"], "folds": 3,
                            "params": {"lightgbm": {"n_estimators": 30}}})
    run_id = jobs.start(ds, lambda job: train.run(job, ds, s), s, s["time_limit_s"])
    jobs.wait(run_id)
    assert jobs.status(ds, run_id)["state"] == "done", jobs.status(ds, run_id)

    # 1. a run model predicts on RAW rows: original column names, clean ops replayed, class names back
    meta = registry.run_meta(ds, run_id, "logistic_regression")
    pipe = registry.load(meta)
    new = pd.DataFrame({"Note ": ["  refund broken 999 ", " love it 998"], "price": [49.0, 51.0], "y": ["bad", "good"]})
    out = predict(pipe, meta, new)
    assert out["table"]["prediction"].tolist() == ["bad", "good"], out["table"]
    assert out["scored_rows"] == 2 and out["metrics"]["accuracy"] == 1.0, out["metrics"]
    assert {"probability bad", "probability good"} <= set(out["columns"])
    assert "Note " in out["columns"], "the original columns stay in the output table"
    no_target = predict(pipe, meta, new.drop(columns="y"))
    assert no_target["metrics"] is None and no_target["scored_rows"] == 0
    unseen = predict(pipe, meta, new.assign(y=["bad", "meh"]))
    assert unseen["scored_rows"] == 1 and "never saw" in unseen["notes"][0], unseen["notes"]
    try:
        predict(pipe, meta, new.drop(columns="Note "))
        raise AssertionError("missing feature column accepted")
    except ValueError as e:
        assert "note" in str(e), e

    # 2. save -> list -> load -> same predictions; version mismatch refused; delete
    saved_m = registry.save(ds, run_id, None, "best spam model")   # None = the run's best model
    assert saved_m["key"] == store.read_json(store.run_dir(ds, run_id) / "results.json")["best"]
    assert [m["name"] for m in registry.saved(ds)] == ["best spam model"]
    again = registry.saved_meta(ds, saved_m["model_id"])
    assert predict(registry.load(again), again, new)["table"]["prediction"].tolist() == \
        predict(registry.load(registry.run_meta(ds, run_id, saved_m["key"])), registry.run_meta(ds, run_id, saved_m["key"]),
                new)["table"]["prediction"].tolist()
    stale = {**again, "versions": {**again["versions"], "scikit-learn": "0.0.1"}}
    try:
        registry.load(stale)
        raise AssertionError("version mismatch accepted")
    except ValueError as e:
        assert "scikit-learn 0.0.1" in str(e), e
    try:
        registry.save(ds, run_id, "logistic_regression", "  ")
        raise AssertionError("blank name accepted")
    except ValueError:
        pass
    registry.delete(ds, saved_m["model_id"])
    assert registry.saved(ds) == []

    # 3. locked test set: rows rebuilt from the recorded split, predictions matched, mistakes filter, paging
    ts = test_set(ds, run_id, "logistic_regression", limit=10)
    assert ts["total"] == 60 and len(ts["rows"]) == 10 and {"row", "note", "actual", "prediction", "correct"} <= set(ts["columns"])
    wrong = test_set(ds, run_id, "logistic_regression", mistakes_only=True)
    assert all(row["correct"] is False for row in wrong["rows"])
    store.add_clean_ops(ds, [{"op": "trim_whitespace", "column": "note"}])
    try:
        test_set(ds, run_id, "logistic_regression")
        raise AssertionError("changed data accepted")
    except ValueError as e:
        assert "preprocessing again" in str(e) or "changed" in str(e), e

    # the run model still works after the dataset changed: it carries its own clean ops
    assert predict(pipe, meta, new)["table"]["prediction"].tolist() == ["bad", "good"]

    # 4. saved input schema is authoritative for manual rows; a manual row == the same row in a file
    shop = pd.DataFrame({"review": [f"{'broken refund' if i % 3 == 0 else 'love it'} {i}" for i in range(240)],
                         "city": ["delhi", "pune", "agra"] * 80, "size": ["small", "large"] * 120,
                         "age": [None if i % 11 == 0 else 20 + i % 40 for i in range(240)],
                         "spend": rng.normal(100, 20, 240), "y": ["bad" if i % 3 == 0 else "good" for i in range(240)]})
    ds2 = store.create(shop, "shop.csv")
    plan = {"review": [{"op": "encode", "method": "text"}], "city": [{"op": "encode", "method": "onehot"}],
            "size": [{"op": "encode", "method": "ordinal"}], "age": [{"op": "impute", "strategy": "median"}]}
    r2 = execute.apply_plan(shop, "y", plan)
    r2["fitted"]["clean_ops"] = store.clean_ops(ds2)
    store.save_prep(ds2, r2["csv"], r2["fitted"])
    s2 = train.validate(ds2, {"models": ["logistic_regression"], "folds": 3})
    run2 = jobs.start(ds2, lambda job: train.run(job, ds2, s2), s2, s2["time_limit_s"])
    jobs.wait(run2)
    assert jobs.status(ds2, run2)["state"] == "done", jobs.status(ds2, run2)
    meta2 = registry.run_meta(ds2, run2, "logistic_regression")
    fields = {f["name"]: f for f in meta2["schema"]}
    assert fields["review"]["kind"] == "text"
    assert fields["city"]["kind"] == "category" and set(fields["city"]["choices"]) == {"delhi", "pune", "agra"}
    assert fields["size"]["kind"] == "category" and set(fields["size"]["choices"]) == {"small", "large"}
    assert fields["age"]["kind"] == "number" and fields["age"]["missing_ok"] and fields["age"]["integer"]
    assert fields["spend"]["kind"] == "number" and not fields["spend"]["missing_ok"]
    pipe2 = registry.load(meta2)
    values = {"review": "broken refund 999", "city": "pune", "size": "large", "age": "", "spend": 101.5}
    manual_frame, _ = row_frame(meta2, values)
    manual = predict(pipe2, meta2, manual_frame, cleaned=True)
    from_file = predict(pipe2, meta2, pd.DataFrame([{**values, "age": None}]))
    assert manual["table"].drop(columns=["age"]).equals(from_file["table"].drop(columns=["age"])), \
        (manual["table"], from_file["table"])
    for bad, msg in (({**values, "city": "mumbai"}, "trained on"), ({**values, "spend": "lots"}, "number"),
                     ({**values, "spend": ""}, "required"), ({**values, "colour": "red"}, "Unknown field")):
        try:
            row_frame(meta2, bad)
            raise AssertionError(bad)
        except ValueError as e:
            assert msg in str(e), (bad, str(e))
    far, far_notes = row_frame(meta2, {**values, "spend": 1e6})
    assert "outside the training range" in far_notes[0]
    try:
        row_frame({**meta2, "schema": None}, values)
        raise AssertionError("schema-less model accepted a manual row")
    except ValueError as e:
        assert "upload a file" in str(e)

    # 5. persistence: saved model == run model == after a restart, to the last bit
    saved2 = registry.save(ds2, run2, "logistic_regression", "shop model")
    assert saved2["schema"] == meta2["schema"], "schema must travel with the saved model"
    probe = shop.drop(columns="y").head(40)
    p_run = registry.load(meta2).predict_proba(probe)
    store._cache.clear()
    reloaded = registry.saved_meta(ds2, saved2["model_id"])
    p_saved = registry.load(reloaded).predict_proba(probe)
    assert np.array_equal(p_run, p_saved), "saved model predictions differ from the run model"
    print("inference self-check passed")
