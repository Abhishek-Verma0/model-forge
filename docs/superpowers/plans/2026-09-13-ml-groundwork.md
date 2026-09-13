# ML Groundwork (Piece 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the data phase safe to train on: pinned versions, a target guard, row-changing steps kept out of the export, datasets/pipelines/runs on disk, and a background job runner.

**Architecture:** Two new backend modules (`store.py` for disk persistence, `jobs.py` for background runs) replace the in-memory dicts in `app.py`. `execute.py` gains a shared `guard_target` and stops writing resampled / outlier-filtered rows into the export (training applies those per fold in piece 1). No training code in this piece.

**Tech Stack:** Python 3.12, FastAPI, pandas 3.0.5, scikit-learn 1.9.0, imbalanced-learn 0.14.2, joblib 1.6.0; Next.js 15 frontend.

**Spec:** `docs/superpowers/specs/2026-09-13-ml-training-design.md` (§3 is this piece).

## Global Constraints

- Do not install packages. If one is missing, stop and tell the user the exact `pip install` command.
- Versions pinned exactly as verified 2026-09-13 (Task 1 lists them). A saved pipeline/model only loads reliably with the same scikit-learn.
- No hidden constants: every new default gets a comment naming its source — "published default (<source>)", "computed", or "our starting point".
- Tests follow the repo's pattern: `assert`-based `if __name__ == "__main__":` self-checks, run with `python -B <file>`. pytest is not installed.
- Ids from requests are validated before they touch a filesystem path (`[0-9a-f]{12}`).
- Numeric/categorical preprocessing output must stay identical to commit `c3d192c` except the two documented changes (missing-target rows removed; row-changing steps no longer in the export).
- Commands below run from `backend\` in PowerShell/cmd: the interpreter is `.venv\Scripts\python`.

---

### Task 0: Branch

- [ ] **Step 1: Create the branch**

```bash
git checkout -b ml-groundwork
```

---

### Task 1: Pin requirements

**Files:**
- Modify: `backend/requirements.txt` (whole file)

**Interfaces:**
- Consumes: nothing
- Produces: an exact, reproducible dependency set for later pieces

- [ ] **Step 1: Confirm installed versions match the pins below**

Run:
```bash
.venv\Scripts\python -c "from importlib.metadata import version as v; [print(f'{d}=={v(d)}') for d in ['pandas','numpy','scipy','openpyxl','requests','fastapi','uvicorn','python-multipart','python-dotenv','matplotlib','seaborn','scikit-learn','imbalanced-learn','joblib','xgboost','lightgbm','catboost']]"
```
Expected output (verified 2026-09-13):
```
pandas==3.0.5
numpy==2.5.3
scipy==1.18.1
openpyxl==3.1.5
requests==2.34.2
fastapi==0.141.1
uvicorn==0.52.4
python-multipart==0.0.32
python-dotenv==1.2.3
matplotlib==3.11.1
seaborn==0.13.2
scikit-learn==1.9.0
imbalanced-learn==0.14.2
joblib==1.6.0
xgboost==3.4.1
lightgbm==4.7.0
catboost==1.2.10
```
If any line differs, use the installed version in Step 2 and tell the user which changed.

- [ ] **Step 2: Write `backend/requirements.txt`**

```
# Exact versions verified together on 2026-09-13. A saved pipeline or model
# only loads reliably with the same scikit-learn -- change pins deliberately.

# API
fastapi==0.141.1
uvicorn[standard]==0.52.4
python-multipart==0.0.32
python-dotenv==1.2.3
requests==2.34.2

# Data
pandas==3.0.5
numpy==2.5.3
scipy==1.18.1
openpyxl==3.1.5
joblib==1.6.0

# Charts
matplotlib==3.11.1
seaborn==0.13.2

# ML
scikit-learn==1.9.0
imbalanced-learn==0.14.2
xgboost==3.4.1
lightgbm==4.7.0
catboost==1.2.10

# Optional, not installed: pretrained fine-tuning (spec piece 6)
# torch
# torchvision
```

(`python-dotenv` was imported by `config.py` but missing from the old file; `numpy`, `scipy`, `joblib` are imported directly by our code.)

- [ ] **Step 3: Verify the file resolves against what is installed (no install)**

Run: `.venv\Scripts\python -m pip install --dry-run -r requirements.txt`
Expected: ends with no "Would install" line (every requirement already satisfied). If it lists packages, the pins and the environment disagree — stop and report.

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt
git commit -m "build: pin backend dependencies to the verified set"
```

---

### Task 2: Target guard

**Files:**
- Modify: `backend/preprocessing/execute.py` (import line, new `guard_target`, `apply_plan` start + return dict, `__main__` block)
- Modify: `frontend/app/preprocess.jsx` (results panel)

**Interfaces:**
- Consumes: `profiler.missing_mask(series)`, `profiler.is_numeric(series)` (existing)
- Produces: `execute.guard_target(df, target, task) -> (DataFrame, list[str])` — raises `ValueError` with a user-readable message; `apply_plan(...)` result gains `"notes": list[str]`. Piece 1 (`train.py`) calls `guard_target` before its own split.

- [ ] **Step 1: Write the failing self-check**

In `backend/preprocessing/execute.py`, inside `if __name__ == "__main__":`, add just before the final line `print("split + parameter self-check passed")`:

```python
    # target guard: missing targets removed and reported; impossible targets rejected clearly
    g = pd.DataFrame({"x": range(12), "y": [1.5, None] + [float(i) for i in range(10)]})
    r = apply_plan(g, "y", {}, task="regression")
    assert r["train_rows"] + r["test_rows"] == 11, (r["train_rows"], r["test_rows"])
    assert "1 row(s) had no y value" in r["notes"][0], r["notes"]
    blank = pd.DataFrame({"x": range(10), "y": ["a", "b", "", "a", "b", "a", "b", "a", "b", "a"]})
    assert apply_plan(blank, "y", {})["rows_before"] == 9          # blank string counts as missing
    for bad_df, task, msg in (
            (pd.DataFrame({"x": range(10), "y": list("abcdefghij")}), "regression", "numeric target"),
            (pd.DataFrame({"x": range(10), "y": ["a"] * 10}), "classification", "at least 2 classes")):
        try:
            apply_plan(bad_df, "y", {}, task=task)
            raise AssertionError(f"{task} target should have been rejected")
        except ValueError as e:
            assert msg in str(e), e
    one = apply_plan(pd.DataFrame({"x": range(10), "y": ["a"] * 9 + ["b"]}), "y", {})
    assert any("single row" in n for n in one["notes"]) and not one["stratified"], one["notes"]
    print("target guard self-check passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -B -W ignore preprocessing\execute.py`
Expected: FAIL — `KeyError: 'notes'` (or `ValueError: Input y contains NaN`-style error from the regression split).

- [ ] **Step 3: Implement `guard_target` and use it in `apply_plan`**

Change the import line:
```python
from profiler import is_numeric
```
to:
```python
from profiler import is_numeric, missing_mask
```

Add this function directly above `def apply_plan(`:
```python
def guard_target(df, target, task):
    """Remove rows the target can't be learned from, and check the target fits
    the task -- before any split, so preprocessing and training see the same
    rows. Returns (df, notes). Raises ValueError with a message for the user."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not a column in this dataset.")
    notes = []
    missing = missing_mask(df[target])  # null OR blank string -- the profiler's rule
    if missing.any():
        df = df[~missing]
        notes.append(f"{int(missing.sum())} row(s) had no {target} value and were removed")
    if task == "regression" and not is_numeric(df[target]):
        raise ValueError(f"Regression needs a numeric target column; '{target}' is {df[target].dtype}. "
                         "Convert it on the Cleaning tab, or choose classification.")
    if task == "classification":
        counts = df[target].value_counts()
        if len(counts) < 2:
            raise ValueError(f"Classification needs at least 2 classes in '{target}' (found {len(counts)}).")
        single = [str(v) for v, n in counts.items() if n < 2]
        if single:
            notes.append(f"class(es) with a single row: {', '.join(single[:10])} -- the split "
                         "can't be stratified; merge or remove them")
    return df, notes
```

In `apply_plan`, replace:
```python
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not a column in this dataset.")
```
with:
```python
    df, target_notes = guard_target(df, target, task)
```

In the dict returned by `apply_plan`, after the line `"class_weights": class_weights,` add:
```python
        "notes": target_notes,
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python -B -W ignore preprocessing\execute.py`
Expected: ends with `target guard self-check passed` then `split + parameter self-check passed`.

- [ ] **Step 5: Show the notes on the Preprocessing results panel**

In `frontend/app/preprocess.jsx`, find:
```jsx
          <details className="pp-summary" open>
```
and insert directly above it:
```jsx
          {result.notes?.length > 0 && (
            <div className="pp-ai" style={{ marginBottom: 10 }}>
              {result.notes.map((n, i) => <p key={i} style={{ margin: 0 }}>{n}</p>)}
            </div>
          )}

```

- [ ] **Step 6: Commit**

```bash
git add backend/preprocessing/execute.py frontend/app/preprocess.jsx
git commit -m "feat: guard the target before splitting (missing rows, task fit, single-row classes)"
```

---

### Task 3: Row-changing steps out of the export

**Files:**
- Modify: `backend/preprocessing/execute.py` (`apply_plan`: remove_rows loop, per-column label, outlier_removal and imbalance blocks; `__main__`)
- Modify: `frontend/app/preprocess.jsx` (two labels, `ruleNote`, `opCode`)

**Interfaces:**
- Consumes: `advanced.balance(X_tr, y_tr, "class_weights", task)` (existing; changes no rows)
- Note: spec §3.3 names SMOTE, over/undersampling and isolation-forest removal. Per-column `outliers.remove_rows` is included too — it also drops train rows, and a scikit-learn transformer cannot drop rows inside a CV fold, so it must become a per-fold sampler in piece 1 like the others.
- Produces: `apply_plan` no longer drops or adds train rows for `outliers.remove_rows`, `pipeline.outlier_removal`, `pipeline.imbalance` (except `class_weights`, which changes no rows). The choices stay in `result["fitted"]["settings"]["columns"]` / `["pipeline"]` for piece 1 to apply inside each CV fold. `advanced.remove_outliers` and `advanced.balance` are kept unchanged for piece 1.

- [ ] **Step 1: Write the failing self-check**

In `execute.py` `__main__`, add before `print("split + parameter self-check passed")`:

```python
    # row-changing steps are NOT written into the export; training applies them per fold
    imb = pd.DataFrame({"x": list(range(60)), "z": [i % 7 for i in range(60)], "y": ["a"] * 50 + ["b"] * 10})
    imb.loc[5, "x"] = 10_000                                           # an obvious outlier
    r = apply_plan(imb, "y", {"x": [{"op": "outliers", "method": "remove_rows"}]},
                   pipeline={"imbalance": {"method": "oversample"},
                             "outlier_removal": {"method": "isolation_forest"}})
    assert (r["train_rows"], r["test_rows"]) == (48, 12), (r["train_rows"], r["test_rows"])
    assert sum("applied during training" in n for n in r["pipeline"]) == 2, r["pipeline"]
    assert "applied during training" in r["summary"][0]["ops"][0], r["summary"][0]
    assert r["fitted"]["settings"]["pipeline"]["imbalance"]["method"] == "oversample"
    cw = apply_plan(imb, "y", {}, pipeline={"imbalance": {"method": "class_weights"}})
    assert cw["class_weights"] and cw["train_rows"] == 48, cw["class_weights"]
    print("row-changing steps self-check passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -B -W ignore preprocessing\execute.py`
Expected: FAIL at the `(48, 12)` assertion (oversampling currently inflates train rows).

- [ ] **Step 3: Implement**

In `apply_plan`, delete this whole block:
```python
    # remove_rows outliers: drop offending TRAIN rows only (dropping test rows
    # would distort evaluation). ponytail: IQR bounds, swap for z-score if needed.
    for c in feats:
        for o in columns.get(c, []):
            if o["op"] == "outliers" and o.get("method") == "remove_rows":
                lo, hi = _iqr_bounds(X_tr[c])
                num = pd.to_numeric(X_tr[c], errors="coerce")
                keep = num.between(lo, hi) | num.isna()
                X_tr, y_tr = X_tr[keep], y_tr[keep]
```

In the per-column loop, replace:
```python
                applied.append("outliers(remove_rows, train)")
```
with:
```python
                applied.append("outliers(remove_rows) - applied during training, inside each fold")
```

Replace this block:
```python
    class_weights = None
    if advanced._m("outlier_removal", pipeline):
        tr_out, y_tr, _n = advanced.remove_outliers(
            tr_out, y_tr, advanced._m("outlier_removal", pipeline),
            (pipeline.get("outlier_removal") or {}).get("contamination"))
        pipe_notes.append(_n)
```
with:
```python
    # Steps that add or drop TRAIN rows (outlier row removal, resampling) are not
    # written into the export: CV folds must apply them to fold-training rows only,
    # or copies of scored rows leak into training. Piece 1 reads them from
    # fitted["settings"]. class_weights changes no rows, so it is still computed.
    class_weights = None
    if advanced._m("outlier_removal", pipeline):
        pipe_notes.append(f"{advanced._m('outlier_removal', pipeline)} - applied during training, "
                          "inside each fold (not written to the export)")
```

Replace this block:
```python
    if advanced._m("imbalance", pipeline):
        tr_out, y_tr, _n, class_weights = advanced.balance(
            tr_out, y_tr, advanced._m("imbalance", pipeline), task)
        pipe_notes.append(_n)
```
with:
```python
    _imb = advanced._m("imbalance", pipeline)
    if _imb == "class_weights":
        tr_out, y_tr, _n, class_weights = advanced.balance(tr_out, y_tr, _imb, task)
        pipe_notes.append(_n)
    elif _imb:
        pipe_notes.append(f"{_imb} - applied during training, inside each fold (not written to the export)")
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python -B -W ignore preprocessing\execute.py`
Expected: prints `row-changing steps self-check passed` and ends with `split + parameter self-check passed`. Also run `.venv\Scripts\python -B -W ignore preprocessing\advanced.py` → `advanced self-check passed` (functions unchanged).

- [ ] **Step 5: Label these steps in the Preprocessing tab**

In `frontend/app/preprocess.jsx`:

Replace:
```jsx
          <label className="pp-field"><span>Outlier removal</span>
```
with:
```jsx
          <label className="pp-field"><span>Outlier removal (applied during training)</span>
```

Replace:
```jsx
          <label className="pp-field"><span>Imbalance{task !== "classification" ? " (clf only)" : ""}</span>
```
with:
```jsx
          <label className="pp-field"><span>Imbalance{task !== "classification" ? " (clf only)" : " (applied during training)"}</span>
```

Replace:
```jsx
      remove_rows: "extreme training rows removed",
```
with:
```jsx
      remove_rows: "extreme training rows removed during training (inside each fold, not in the export)",
```

Replace:
```jsx
    if (step.outliers === "remove_rows") lines.push(`train = train[within_iqr("${col}", k=${k})]  # train rows only`);
```
with:
```jsx
    if (step.outliers === "remove_rows") lines.push(`train = train[within_iqr("${col}", k=${k})]  # applied during training, inside each fold`);
```

- [ ] **Step 6: Commit**

```bash
git add backend/preprocessing/execute.py frontend/app/preprocess.jsx
git commit -m "feat: keep resampling and outlier row removal out of the export (applied per fold in training)"
```

---

### Task 4: Disk store

**Files:**
- Create: `backend/preprocessing/store.py`
- Modify: `backend/preprocessing/config.py` (add `DATA_DIR`)
- Modify: `backend/.env.example` (document `DATA_DIR`)
- Modify: `backend/.gitignore` (ignore `data/`)

**Interfaces:**
- Consumes: `config.DATA_DIR`
- Produces (all ids are 12 lowercase hex chars; invalid ids never touch a path):
  - `store.ROOT: Path` (module variable; tests reassign it)
  - `store.create(df, filename) -> str` dataset id
  - `store.save_frame(ds_id, df) -> None`
  - `store.load_frame(ds_id) -> DataFrame | None`
  - `store.meta(ds_id) -> dict | None` — `{"filename", "created", "clean_ops"}`
  - `store.add_clean_ops(ds_id, ops: list) -> None`
  - `store.clean_ops(ds_id) -> list`
  - `store.save_prep(ds_id, csv: str, fitted: dict) -> None`
  - `store.load_prep(ds_id) -> {"csv": str, "fitted": dict} | None`
  - `store.new_run(ds_id) -> str` run id (creates the folder)
  - `store.run_dir(ds_id, run_id) -> Path` — raises `KeyError` on an invalid id
  - `store.run_ids(ds_id) -> list[str]`
  - `store.all_run_dirs() -> list[Path]`
  - `store.write_json(path, obj) -> None` (atomic), `store.read_json(path) -> object | None`

- [ ] **Step 1: Add the setting**

In `backend/preprocessing/config.py`, after the line `MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 200))` add:
```python

# Where uploaded datasets, fitted pipelines and training runs are saved. A
# deployment setting (disk location), not an analysis one.
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))
```

In `backend/.env.example`, after the line `# MAX_UPLOAD_MB=200` add:
```
# Datasets, pipelines and training runs are saved here (default backend/data).
# DATA_DIR=D:/model-forge-data
```

In `backend/.gitignore`, after the line `preprocessing/report.json` add:
```
data/
```

- [ ] **Step 2: Write `store.py` with its failing self-check first**

Create `backend/preprocessing/store.py` containing only the self-check:
```python
if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    import pandas as pd
    ROOT = Path(tempfile.mkdtemp())
    df = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64"), "t": ["x", "y", None]})
    ds = create(df, "demo.csv")
    _cache.clear()                                               # simulate a restart
    pd.testing.assert_frame_equal(load_frame(ds), df)            # dtypes survive (Int64 stays Int64)
    assert meta(ds)["filename"] == "demo.csv"
    add_clean_ops(ds, [{"op": "trim_whitespace", "column": "t"}])
    add_clean_ops(ds, [{"op": "drop_duplicates"}])
    assert [o["op"] for o in clean_ops(ds)] == ["trim_whitespace", "drop_duplicates"]
    assert load_prep(ds) is None
    save_prep(ds, "a,t\n", {"features": ["a"]})
    assert load_prep(ds)["fitted"]["features"] == ["a"]
    for bad in ("../../etc", "ABCDEF123456", "", None, "0" * 13):
        assert load_frame(bad) is None and meta(bad) is None     # never touches a path outside ROOT
    run = new_run(ds)
    write_json(run_dir(ds, run) / "status.json", {"state": "done"})
    assert run_ids(ds) == [run] and read_json(run_dir(ds, run) / "status.json") == {"state": "done"}
    assert all_run_dirs() == [run_dir(ds, run)]
    try:
        run_dir(ds, "../x")
        raise AssertionError("bad run id accepted")
    except KeyError:
        pass
    for _ in range(CACHE_CAP + 2):
        create(df, "x.csv")
    assert len(_cache) == CACHE_CAP
    print("store self-check passed")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv\Scripts\python -B preprocessing\store.py`
Expected: FAIL — `NameError: name 'create' is not defined`.

- [ ] **Step 4: Implement above the self-check**

Insert at the top of `backend/preprocessing/store.py`:
```python
"""Datasets, fitted pipelines and training runs on disk, so a restart or a new
upload loses nothing. Layout under ROOT:

  <dataset_id>/frame.pkl          current table (after cleaning)
  <dataset_id>/meta.json          {"filename", "created", "clean_ops": [...]}
  <dataset_id>/prep.joblib        {"csv", "fitted"} from the last preprocessing run
  <dataset_id>/runs/<run_id>/     one folder per training run (jobs.py)

pickle/joblib keep pandas dtypes exactly, but can run code when loaded -- so this
module only loads files it wrote, and every id is validated before it becomes a
path. ponytail: no deletion/retention yet (plan §24) -- add with auth.
"""

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import joblib
import pandas as pd

import config

ROOT = Path(config.DATA_DIR)
CACHE_CAP = 5            # frames kept in memory; disk is the source of truth (our starting point)
_ID = re.compile(r"[0-9a-f]{12}")
_cache = OrderedDict()
_lock = threading.Lock()  # meta.json read-modify-write from concurrent requests


def _new_id():
    return uuid.uuid4().hex[:12]


def _check(an_id):
    if not isinstance(an_id, str) or not _ID.fullmatch(an_id):
        raise KeyError(an_id)
    return an_id


def _dir(ds_id):
    return ROOT / _check(ds_id)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves half a file


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _remember(ds_id, df):
    _cache[ds_id] = df
    _cache.move_to_end(ds_id)
    while len(_cache) > CACHE_CAP:
        _cache.popitem(last=False)


def create(df, filename):
    ds_id = _new_id()
    _dir(ds_id).mkdir(parents=True)
    write_json(_dir(ds_id) / "meta.json", {"filename": filename, "created": time.time(), "clean_ops": []})
    save_frame(ds_id, df)
    return ds_id


def save_frame(ds_id, df):
    path = _dir(ds_id) / "frame.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    df.to_pickle(tmp)
    tmp.replace(path)
    _remember(ds_id, df)


def load_frame(ds_id):
    """The current table, or None for an unknown or invalid id. Callers must not
    modify the returned frame in place (it is shared through the cache)."""
    try:
        path = _dir(ds_id) / "frame.pkl"
    except KeyError:
        return None
    if ds_id in _cache:
        _cache.move_to_end(ds_id)
        return _cache[ds_id]
    if not path.exists():
        return None
    df = pd.read_pickle(path)
    _remember(ds_id, df)
    return df


def meta(ds_id):
    try:
        return read_json(_dir(ds_id) / "meta.json")
    except KeyError:
        return None


def add_clean_ops(ds_id, ops):
    with _lock:
        m = meta(ds_id)
        m["clean_ops"] = m["clean_ops"] + list(ops)
        write_json(_dir(ds_id) / "meta.json", m)


def clean_ops(ds_id):
    return (meta(ds_id) or {}).get("clean_ops", [])


def save_prep(ds_id, csv, fitted):
    path = _dir(ds_id) / "prep.joblib"
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump({"csv": csv, "fitted": fitted}, tmp)
    tmp.replace(path)


def load_prep(ds_id):
    try:
        path = _dir(ds_id) / "prep.joblib"
    except KeyError:
        return None
    return joblib.load(path) if path.exists() else None


def new_run(ds_id):
    run_id = _new_id()
    run_dir(ds_id, run_id).mkdir(parents=True)
    return run_id


def run_dir(ds_id, run_id):
    return _dir(ds_id) / "runs" / _check(run_id)


def run_ids(ds_id):
    runs = _dir(ds_id) / "runs"
    return sorted(p.name for p in runs.iterdir() if _ID.fullmatch(p.name)) if runs.exists() else []


def all_run_dirs():
    if not ROOT.exists():
        return []
    return [p for p in ROOT.glob("*/runs/*") if _ID.fullmatch(p.parent.parent.name) and _ID.fullmatch(p.name)]


```

- [ ] **Step 5: Run it to verify it passes**

Run: `.venv\Scripts\python -B preprocessing\store.py`
Expected: `store self-check passed`

- [ ] **Step 6: Commit**

```bash
git add backend/preprocessing/store.py backend/preprocessing/config.py backend/.env.example backend/.gitignore
git commit -m "feat: disk store for datasets, pipelines and runs"
```

---

### Task 5: Background job runner

**Files:**
- Create: `backend/preprocessing/jobs.py`

**Interfaces:**
- Consumes: `store.new_run`, `store.run_dir`, `store.run_ids`, `store.all_run_dirs`, `store.write_json`, `store.read_json` (Task 4)
- Produces:
  - `jobs.DEFAULT_TIME_LIMIT_S = 3600` (spec §8, our starting point)
  - `jobs.Stopped(Exception)`
  - `jobs.Job` with `.ds_id`, `.run_id`, `.dir: Path`, `.check() -> None` (raises `Stopped("cancelled")` / `Stopped("timed_out")`), `.progress(**info) -> None`
  - `jobs.start(ds_id, fn, settings: dict, time_limit_s: float) -> str` run id; `fn(job)` runs in the background
  - `jobs.wait(run_id, timeout=None) -> None`
  - `jobs.cancel(ds_id, run_id) -> bool` (True if the run was queued/running in this process)
  - `jobs.status(ds_id, run_id) -> dict | None` — keys: `state` (`queued|running|done|failed|cancelled|timed_out|interrupted`), `progress`, `created`, `started`, `finished`, `error`, `time_limit_s`
  - `jobs.list_runs(ds_id) -> list[dict]` — each status plus `run_id`, newest first
  - `jobs.mark_interrupted() -> int` — marks runs left `queued`/`running` by a previous process

- [ ] **Step 1: Write the failing self-check**

Create `backend/preprocessing/jobs.py` containing only:
```python
if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    import pandas as pd
    store.ROOT = Path(tempfile.mkdtemp())
    ds = store.create(pd.DataFrame({"a": [1]}), "d.csv")

    def work(job):
        for i in range(3):
            job.check()
            job.progress(step=i + 1, total=3)
    r = start(ds, work, {"kind": "demo"}, time_limit_s=60)
    wait(r)
    s = status(ds, r)
    assert s["state"] == "done" and s["progress"] == {"step": 3, "total": 3}, s
    assert store.read_json(store.run_dir(ds, r) / "settings.json") == {"kind": "demo"}

    gate = threading.Event()
    def slow(job):
        gate.wait(5)
        job.check()
    r = start(ds, slow, {}, time_limit_s=60)
    assert cancel(ds, r)
    gate.set()
    wait(r)
    assert status(ds, r)["state"] == "cancelled", status(ds, r)
    assert not cancel(ds, r)                                  # finished runs can't be cancelled

    def late(job):
        time.sleep(0.3)
        job.check()
    r = start(ds, late, {}, time_limit_s=0.1)
    wait(r)
    assert status(ds, r)["state"] == "timed_out", status(ds, r)

    def boom(job):
        raise RuntimeError("bad data")
    r = start(ds, boom, {}, time_limit_s=60)
    wait(r)
    assert status(ds, r)["state"] == "failed" and "bad data" in status(ds, r)["error"], status(ds, r)

    ghost = store.new_run(ds)                                 # left "running" by a dead process
    store.write_json(store.run_dir(ds, ghost) / "status.json", {"state": "running", "created": 0})
    assert mark_interrupted() == 1
    assert status(ds, ghost)["state"] == "interrupted"
    runs = list_runs(ds)
    assert len(runs) == 5 and runs[0]["run_id"] == r and runs[-1]["run_id"] == ghost, [x["run_id"] for x in runs]
    print("jobs self-check passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -B preprocessing\jobs.py`
Expected: FAIL — `NameError: name 'store' is not defined`.

- [ ] **Step 3: Implement above the self-check**

Insert at the top of `backend/preprocessing/jobs.py`:
```python
"""Background runs: one at a time, with progress, a time limit and cancel. Status
lives in runs/<run_id>/status.json, so a restart shows "interrupted" instead of
"running forever".

A job is fn(job): job.progress(**info) reports progress, job.check() raises
Stopped when the user cancelled or the time limit passed. Both are honoured
between steps only -- a fit already running in this thread finishes first.
ponytail: a thread, not a process; hard-killing one long fit needs a process
pool plus copying the data into it.
"""

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import store

DEFAULT_TIME_LIMIT_S = 3600   # spec §8: 60 min per run, our starting point
_ACTIVE = {"queued", "running"}
_pool = ThreadPoolExecutor(max_workers=1)  # one run at a time: a run already uses the cores
_cancel = {}                  # run_id -> Event, only for runs owned by this process
_futures = {}
_lock = threading.Lock()


class Stopped(Exception):
    """Raised by Job.check(); its message becomes the run state."""


def _update(run_path, **fields):
    with _lock:
        s = store.read_json(run_path / "status.json") or {}
        s.update(fields)
        store.write_json(run_path / "status.json", s)


class Job:
    def __init__(self, ds_id, run_id, time_limit_s):
        self.ds_id, self.run_id, self.time_limit_s = ds_id, run_id, time_limit_s
        self.dir = store.run_dir(ds_id, run_id)
        self.started = None

    def check(self):
        if _cancel[self.run_id].is_set():
            raise Stopped("cancelled")
        if self.time_limit_s and time.time() - self.started > self.time_limit_s:
            raise Stopped("timed_out")

    def progress(self, **info):
        _update(self.dir, progress=info)


def start(ds_id, fn, settings, time_limit_s):
    run_id = store.new_run(ds_id)
    job = Job(ds_id, run_id, time_limit_s)
    store.write_json(job.dir / "settings.json", settings)
    _update(job.dir, state="queued", created=time.time(), time_limit_s=time_limit_s, progress={})
    _cancel[run_id] = threading.Event()
    _futures[run_id] = _pool.submit(_run, job, fn)
    return run_id


def _run(job, fn):
    state, error = "done", None
    if _cancel[job.run_id].is_set():
        state = "cancelled"
    else:
        job.started = time.time()
        _update(job.dir, state="running", started=job.started)
        try:
            fn(job)
        except Stopped as stop:
            state = str(stop)
        except Exception as exc:  # noqa: BLE001 -- any failure must end as a visible state
            traceback.print_exc()
            state, error = "failed", f"{type(exc).__name__}: {exc}"
    _update(job.dir, state=state, error=error, finished=time.time())
    _cancel.pop(job.run_id, None)


def wait(run_id, timeout=None):
    _futures[run_id].result(timeout)


def cancel(ds_id, run_id):
    store.run_dir(ds_id, run_id)  # validates both ids; KeyError on bad input
    event = _cancel.get(run_id)
    if event is None:             # finished, or owned by a previous process
        return False
    event.set()
    return True


def status(ds_id, run_id):
    try:
        return store.read_json(store.run_dir(ds_id, run_id) / "status.json")
    except KeyError:
        return None


def list_runs(ds_id):
    try:
        ids = store.run_ids(ds_id)
    except KeyError:
        return []
    runs = [{"run_id": r, **(status(ds_id, r) or {})} for r in ids]
    return sorted(runs, key=lambda s: s.get("created", 0), reverse=True)


def mark_interrupted():
    n = 0
    for run_path in store.all_run_dirs():
        s = store.read_json(run_path / "status.json") or {}
        if s.get("state") in _ACTIVE and run_path.name not in _cancel:
            _update(run_path, state="interrupted", finished=time.time())
            n += 1
    return n


```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python -B preprocessing\jobs.py`
Expected: `jobs self-check passed` (takes under 2 seconds).

- [ ] **Step 5: Commit**

```bash
git add backend/preprocessing/jobs.py
git commit -m "feat: background job runner with progress, cancel, time limit and restart recovery"
```

---

### Task 6: Wire `app.py` to the store and add run endpoints

**Files:**
- Modify: `backend/app.py` (replace whole file)
- Create: `backend/test_groundwork.py`

**Interfaces:**
- Consumes: `store.*` (Task 4), `jobs.*` (Task 5), `execute.apply_plan` result keys `csv`, `fitted`, `notes` (Task 2)
- Produces: HTTP endpoints for piece 1's Train tab:
  - `GET /api/runs?id=<ds_id>` → `list[status]`
  - `GET /api/run?id=<ds_id>&run=<run_id>` → status (404 if unknown)
  - `POST /api/run/cancel` body `{id, run}` → `{"cancelled": bool}`
  - existing endpoints unchanged in shape; they now survive restarts
  - `POST /api/train` (spec §3.5) is NOT here: it needs `train.py` and arrives with piece 1, using `jobs.start`

- [ ] **Step 1: Write the failing check**

Create `backend/test_groundwork.py`:
```python
"""Groundwork check: datasets, clean ops and pipelines survive a restart; unknown
ids 404; run endpoints answer. Run from backend/: .venv\\Scripts\\python -B test_groundwork.py"""
import io
import tempfile
import warnings
from pathlib import Path

import joblib
import pandas as pd
from fastapi import HTTPException

warnings.filterwarnings("ignore")
import app as A  # noqa: E402  (adds preprocessing/ to sys.path)
import jobs      # noqa: E402
import store     # noqa: E402

store.ROOT = Path(tempfile.mkdtemp())
df = pd.DataFrame({"review": [f" good item {i}" if i % 2 else f"bad item {i} " for i in range(40)],
                   "age": [20 + i % 30 for i in range(40)], "y": ["pos", "neg"] * 20})
ds = store.create(df, "t.csv")

A.do_clean({"id": ds, "ops": [{"op": "trim_whitespace", "column": "review"}]})
res = A.do_preprocess({"id": ds, "target": "y", "task": "classification",
                       "columns": {"review": [{"op": "encode", "method": "text"}]}})
assert "fitted" not in res and "csv" not in res and res["train_rows"] == 32, res.keys()

store._cache.clear()                                       # restart: memory gone, disk remains
assert A.download(ds).body.startswith(b"review")
fitted = joblib.load(io.BytesIO(A.download_pipeline(ds).body))
assert fitted["clean_ops"] == [{"op": "trim_whitespace", "column": "review"}], fitted["clean_ops"]
assert A.chart(ds, "histogram", x="age").status_code == 200

for call in (lambda: A.download("000000000000"), lambda: A.chart("../../x", "histogram", x="age"),
             lambda: A.run_status(ds, "../x"), lambda: A.run_status(ds, "000000000000")):
    try:
        call()
        raise AssertionError("unknown id should 404")
    except HTTPException as e:
        assert e.status_code == 404, e.status_code

r = jobs.start(ds, lambda job: job.progress(step=1), {"kind": "demo"}, time_limit_s=60)
jobs.wait(r)
assert A.run_status(ds, r)["state"] == "done" and A.list_runs(ds)[0]["run_id"] == r
assert A.cancel_run({"id": ds, "run": r}) == {"cancelled": False}   # already finished
print("groundwork check passed")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python -B test_groundwork.py`
Expected: FAIL — `KeyError` / `404` from `do_clean` because `app.py` still reads its in-memory `STORE`, not `store`.

- [ ] **Step 3: Replace `backend/app.py`**

```python
"""HTTP API for the preprocessing backend (root entrypoint).

Wires the preprocessing modules together over HTTP. It does NOT reimplement
anything -- it loads the file and calls the existing profiler + detector +
charts + eda + plot. Datasets, pipelines and runs live on disk (store.py), so a
restart loses nothing.
"""

import io
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

from fastapi import FastAPI, UploadFile, HTTPException, Response, Body   # noqa: E402
from fastapi.responses import StreamingResponse                         # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                       # noqa: E402
import joblib                                                            # noqa: E402

from profiler import load_dataset, profile_dataset               # noqa: E402
from detector import run_quality_report                          # noqa: E402
from charts import chart_data                                    # noqa: E402
from eda import render_charts                                    # noqa: E402
import plot                                                      # noqa: E402
import clean                                                     # noqa: E402
import execute                                                   # noqa: E402
import llm                                                       # noqa: E402
import config                                                    # noqa: E402
import store                                                     # noqa: E402
import jobs                                                      # noqa: E402

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
ALLOWED = {".csv", ".xlsx", ".xls"}

app = FastAPI(title="ResearchAI Studio - Preprocessing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runs left queued/running by a previous process can never finish -- say so.
jobs.mark_interrupted()


def _frame(ds_id):
    df = store.load_frame(ds_id)
    if df is None:
        raise HTTPException(404, "Dataset not found. Re-upload.")
    return df


def _payload(df, filename, ds_id):
    """The full analyze response for a frame: profile + quality report + charts +
    a small row sample (for the Cleaning tab's preview grid). Shared by
    /api/analyze and /api/clean so cleaned data re-renders identically."""
    return {"id": ds_id, "filename": filename,
            "profile": profile_dataset(df),
            "report": run_quality_report(df),
            "charts": chart_data(df),
            "eda": render_charts(df),
            "sample": json.loads(df.head(50).to_json(orient="records"))}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile):
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
        df = load_dataset(tmp_path)
        ds_id = store.create(df, file.filename)
        return _payload(df, file.filename, ds_id)
    except Exception as exc:
        raise HTTPException(422, f"Could not read file: {exc}") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/chart")
def chart(id: str, kind: str, x: str = "", y: str = "", hue: str = "", fmt: str = "png"):
    df = _frame(id)
    try:
        img, media = plot.render(df, kind, x=x or None, y=y or None, hue=hue or None, fmt=fmt)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return Response(content=img, media_type=media)


def _llm_context(df, profile, target, task):
    """Aggregated metadata + a small real-row sample for the LLM planner.
    Never raw full data: sample rows are capped by width so a wide table can't
    blow the model's context window (~100 rows, fewer for many columns)."""
    ncols = max(1, len(df.columns))
    n = min(100, max(10, 3000 // ncols))
    info = profile["columns_info"]
    cols = [{"name": c, "dtype": str(df[c].dtype),
             "missing_percent": info[c]["missing_percent"],
             "unique": info[c]["unique_values"]} for c in df.columns]
    examples = {c: [str(v) for v in df[c].dropna().unique()[:8]] for c in df.columns}
    sample = json.loads(df.head(n).to_json(orient="records"))
    return {"target": target, "task": task, "columns": cols,
            "examples": examples, "sample": sample}


@app.post("/api/plan")
def plan(payload: dict = Body(...)):
    """LLM proposes a structured clean+preprocess plan (validated against the op
    allowlist). Optional -- 503 if Ollama is down; the frontend keeps its rule
    defaults. Body: {id, target, task}."""
    df = _frame(payload.get("id"))
    # target is optional: cleaning suggestions don't need it, so the plan can load
    # on the Cleaning tab before the user has chosen an outcome column.
    context = _llm_context(df, profile_dataset(df), payload.get("target", ""),
                           payload.get("task", "classification"))
    try:
        return llm.recommend_plan(context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"AI plan unavailable ({llm.HINT()}): {exc}") from None


@app.post("/api/clean")
def do_clean(payload: dict = Body(...)):
    """Apply validated per-cell clean ops to the WHOLE frame (before split),
    save the cleaned frame, and return the re-profiled payload + a summary.
    Body: {id, ops}."""
    df = _frame(payload.get("id"))
    try:
        cleaned, summary = clean.apply_clean(df, payload.get("ops", []))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    store.save_frame(payload["id"], cleaned)  # downstream tabs now see the cleaned frame
    store.add_clean_ops(payload["id"], payload.get("ops", []))
    out = _payload(cleaned, payload.get("filename", "cleaned"), payload["id"])
    out["clean_summary"] = summary
    return out


@app.post("/api/preprocess")
def do_preprocess(payload: dict = Body(...)):
    """Apply a validated per-column op plan (fit on train only), return a preview
    + change summary. Body: {id, target, task, columns, test_size?, random_state?,
    stratify?}. Split settings fall back to 80:20 / seed 42. The cleaned CSV and
    fitted pipeline are saved for /api/download and /api/pipeline."""
    df = _frame(payload.get("id"))
    if not payload.get("target"):
        raise HTTPException(422, "No target column given.")
    try:
        result = execute.apply_plan(
            df, payload["target"], payload.get("columns", {}),
            task=payload.get("task", "classification"),
            pipeline=payload.get("pipeline"),
            test_size=payload.get("test_size"),
            random_state=payload.get("random_state"),
            stratify=payload.get("stratify", True),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    fitted = result.pop("fitted")
    fitted["clean_ops"] = store.clean_ops(payload["id"])
    store.save_prep(payload["id"], result.pop("csv"), fitted)  # big CSV stays out of the JSON response
    return result


@app.post("/api/chat")
def chat_api(payload: dict = Body(...)):
    """Multi-turn preprocessing assistant, streamed as plain-text chunks.
    Optional -- 503 if the LLM is down (raised before streaming starts)."""
    try:
        gen = llm.chat_stream(payload.get("messages", []), payload.get("context", {}))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"AI chat unavailable ({llm.HINT()}): {exc}") from None
    return StreamingResponse(gen, media_type="text/plain")


@app.post("/api/explain")
def explain(payload: dict = Body(...)):
    """Plain-English explanation of a preprocessing plan. Optional:
    if the LLM is down, the app still works -- this just returns 503."""
    try:
        return {"text": llm.explain_recipe(payload)}
    except Exception as exc:  # noqa: BLE001 -- surface any LLM/transport failure as unavailable
        raise HTTPException(503, f"AI explanation unavailable ({llm.HINT()}): {exc}") from None


def _prep(ds_id):
    prep = store.load_prep(ds_id)
    if prep is None:
        raise HTTPException(404, "No preprocessing run yet. Run preprocessing first.")
    return prep


@app.get("/api/download")
def download(id: str):
    return Response(
        content=_prep(id)["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned.csv"},
    )


@app.get("/api/pipeline")
def download_pipeline(id: str):
    """The fitted pipeline (joblib): clean ops + every train-fitted step + text
    TF-IDF. Load with joblib and use execute.transform / execute.to_matrix, same
    scikit-learn version. Only ever load a pipeline file you produced yourself --
    joblib files can run code."""
    buf = io.BytesIO()
    joblib.dump(_prep(id)["fitted"], buf)
    return Response(content=buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=pipeline.joblib"})


@app.get("/api/runs")
def list_runs(id: str):
    return jobs.list_runs(id)


@app.get("/api/run")
def run_status(id: str, run: str):
    s = jobs.status(id, run)
    if s is None:
        raise HTTPException(404, "Run not found.")
    return s


@app.post("/api/run/cancel")
def cancel_run(payload: dict = Body(...)):
    """Stops a queued/running run between steps. Body: {id, run}."""
    try:
        return {"cancelled": jobs.cancel(payload.get("id"), payload.get("run"))}
    except KeyError:
        raise HTTPException(404, "Run not found.") from None
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv\Scripts\python -B test_groundwork.py`
Expected: `groundwork check passed`

(The old "server may have restarted" wording lived only in `app.py`; verified 2026-09-13 that `frontend/app` has no copy of it, so no frontend change.)

- [ ] **Step 5: Commit**

```bash
git add backend/app.py backend/test_groundwork.py
git commit -m "feat: serve datasets and pipelines from disk; add run status and cancel endpoints"
```

---

### Task 7: Whole-piece verification

**Files:** none changed (verification only; the golden script lives in your temp folder, not the repo)

- [ ] **Step 1: All backend self-checks**

Run from `backend\preprocessing\`:
```bash
for %f in (profiler clean charts eda plot detector advanced execute store jobs llm) do @..\.venv\Scripts\python -B -W ignore %f.py
```
(Git Bash: `for f in profiler clean charts eda plot detector advanced execute store jobs llm; do ../.venv/Scripts/python -B -W ignore $f.py | tail -1; done`)
Expected: every module prints its `... self-check passed` line; no traceback.

- [ ] **Step 2: Groundwork check**

Run from `backend\`: `.venv\Scripts\python -B test_groundwork.py`
Expected: `groundwork check passed`

- [ ] **Step 3: Golden comparison against commit c3d192c**

Save as `%TEMP%\golden_groundwork.py`:
```python
"""Numeric/categorical preprocessing output must be identical to commit c3d192c
for plans without row-changing steps and without missing targets."""
import json, subprocess, sys, tempfile, warnings, importlib
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
repo, backend_py = Path(sys.argv[1]), Path(sys.argv[1]) / "backend" / "preprocessing"
old = Path(tempfile.mkdtemp())
for f in subprocess.run(["git", "-C", str(repo), "ls-tree", "--name-only", "c3d192c", "backend/preprocessing/"],
                        capture_output=True, text=True, check=True).stdout.split():
    if f.endswith(".py"):
        (old / Path(f).name).write_text(subprocess.run(["git", "-C", str(repo), "show", f"c3d192c:{f}"],
                                        capture_output=True, text=True, check=True).stdout, encoding="utf-8")

def run(code_dir):
    for m in ("profiler", "detector", "advanced", "clean", "execute", "config", "store"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(code_dir))
    ex = importlib.import_module("execute")
    sys.path.pop(0)
    rng = np.random.default_rng(0); n = 300
    df = pd.DataFrame({"age": rng.integers(18, 90, n).astype(float), "bmi": rng.normal(28, 6, n),
                       "glucose": np.r_[rng.normal(120, 30, n - 3), [900, -50, 1000]],
                       "sex": rng.choice(["M", "F"], n), "region": rng.choice(["n", "s", "e", "w"], n),
                       "label": rng.choice([0, 0, 0, 1], n), "score": rng.normal(50, 10, n)})
    df.loc[rng.choice(n, 30, replace=False), "age"] = np.nan
    plan = {"age": [{"op": "impute", "strategy": "median"}],
            "bmi": [{"op": "scale", "method": "robust"}],
            "glucose": [{"op": "outliers", "method": "clip_iqr", "k": 2}, {"op": "scale", "method": "standard"}],
            "sex": [{"op": "encode", "method": "onehot"}], "region": [{"op": "encode", "method": "ordinal"}]}
    out = {}
    for i, pipe in enumerate([{}, {"imputation": {"method": "knn"}, "feature_selection": {"method": "anova", "k": 3}},
                              {"imputation": {"method": "iterative"}, "reduction": {"method": "pca", "n": 3},
                               "imbalance": {"method": "class_weights"}}]):
        for task, tgt in (("classification", "label"), ("regression", "score")):
            r = ex.apply_plan(df, tgt, plan, task=task, pipeline=pipe)
            out[f"{i}|{task}"] = {k: r[k] for k in ("csv", "summary", "pipeline", "class_weights", "train_rows", "test_rows")}
    return json.dumps(out, sort_keys=True, default=str)

a, b = run(old), run(backend_py)
print("IDENTICAL" if a == b else "DIFFERENT")
```
Run from `backend\`: `.venv\Scripts\python -B %TEMP%\golden_groundwork.py ..`
Expected: `IDENTICAL`

- [ ] **Step 4: Frontend builds**

Run from `frontend\`: `npx next build`
Expected: `✓ Compiled successfully`

- [ ] **Step 5: Manual smoke test (user-visible)**

1. Start backend (`uvicorn app:app --reload --port 8000`) and frontend (`npm run dev`).
2. Upload a CSV, clean one column, run preprocessing with Imbalance = oversample.
3. Expected: the results panel lists "oversample - applied during training, inside each fold (not written to the export)"; train/test rows add up to the file's rows.
4. Stop and restart the backend; click "Download cleaned CSV" and "Download pipeline" without re-uploading.
5. Expected: both downloads work; `backend/data/<id>/` contains `frame.pkl`, `meta.json`, `prep.joblib`.

- [ ] **Step 6: Record in the backlog**

In `docs/backlog.md`, in "Still open from the 2026-09-13 data-pipeline audit", delete the lines about
`NaN target rows kept; oversampling written into the exported CSV; clustering blocked (target required).`
and replace with:
```
- Clustering blocked (target required) -- spec piece 4.
```
and in the "Not built" table delete the row starting `` | `CLEAN_LOG` in `app.py` `` (replaced by the disk store). Append under "Bugs found, not fixed":
```
- Dataset deletion / retention period (plan §24) not implemented -- files accumulate under backend/data.
```

- [ ] **Step 7: Commit**

```bash
git add docs/backlog.md
git commit -m "docs: backlog after ML groundwork"
```
