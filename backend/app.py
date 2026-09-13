"""HTTP API for the preprocessing backend (root entrypoint).

Wires the preprocessing modules together over HTTP. It does NOT reimplement
anything -- it loads the file and calls the existing profiler + detector +
charts + eda + plot. profiler.py / detector.py are unchanged.
"""

import io
import sys
import json
import uuid
import tempfile
from collections import OrderedDict
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

MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
ALLOWED = {".csv", ".xlsx", ".xls"}

# Uploaded dataframes kept in memory so the chart builder can request charts by
# id after upload. ponytail: in-memory + single process; a restart clears it
# (re-upload) -- move to disk/redis for persistence or multiple workers.
STORE = OrderedDict()
STORE_CAP = 5

# (cleaned CSV, fitted pipeline) per dataset id, produced by /api/preprocess and
# served by /api/download + /api/pipeline. Kept out of the JSON response.
CLEANED = OrderedDict()
# Clean ops applied per dataset id, in order -- the saved pipeline replays them
# on new rows. ponytail: grows one small list per upload until restart.
CLEAN_LOG = {}

app = FastAPI(title="ResearchAI Studio - Preprocessing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _store(df):
    ds_id = uuid.uuid4().hex[:12]
    STORE[ds_id] = df
    while len(STORE) > STORE_CAP:
        STORE.popitem(last=False)  # evict oldest
    return ds_id


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
        ds_id = _store(df)
        return _payload(df, file.filename, ds_id)
    except Exception as exc:
        raise HTTPException(422, f"Could not read file: {exc}") from None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/chart")
def chart(id: str, kind: str, x: str = "", y: str = "", hue: str = "", fmt: str = "png"):
    df = STORE.get(id)
    if df is None:
        raise HTTPException(404, "Dataset not found (server may have restarted). Re-upload.")
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
    df = STORE.get(payload.get("id"))
    if df is None:
        raise HTTPException(404, "Dataset not found (server may have restarted). Re-upload.")
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
    replace the stored frame, and return the re-profiled payload + a summary.
    Body: {id, ops}."""
    df = STORE.get(payload.get("id"))
    if df is None:
        raise HTTPException(404, "Dataset not found (server may have restarted). Re-upload.")
    try:
        cleaned, summary = clean.apply_clean(df, payload.get("ops", []))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    STORE[payload["id"]] = cleaned  # downstream tabs now see the cleaned frame
    CLEAN_LOG[payload["id"]] = CLEAN_LOG.get(payload["id"], []) + payload.get("ops", [])
    out = _payload(cleaned, payload.get("filename", "cleaned"), payload["id"])
    out["clean_summary"] = summary
    return out


@app.post("/api/preprocess")
def do_preprocess(payload: dict = Body(...)):
    """Apply a validated per-column op plan (fit on train only), return a preview
    + change summary. Body: {id, target, task, columns, test_size?, random_state?,
    stratify?}. Split settings fall back to 80:20 / seed 42. Cleaned CSV
    stashed for /api/download."""
    df = STORE.get(payload.get("id"))
    if df is None:
        raise HTTPException(404, "Dataset not found (server may have restarted). Re-upload.")
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
    fitted["clean_ops"] = CLEAN_LOG.get(payload["id"], [])
    CLEANED[payload["id"]] = (result.pop("csv"), fitted)  # keep the big CSV out of the JSON response
    while len(CLEANED) > STORE_CAP:
        CLEANED.popitem(last=False)
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


@app.get("/api/download")
def download(id: str):
    if id not in CLEANED:
        raise HTTPException(404, "No cleaned data yet. Run preprocessing first.")
    return Response(
        content=CLEANED[id][0],
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned.csv"},
    )


@app.get("/api/pipeline")
def download_pipeline(id: str):
    """The fitted pipeline (joblib): clean ops + every train-fitted step + text
    TF-IDF. Load with joblib and use execute.transform / execute.to_matrix, same
    scikit-learn version. Only ever load a pipeline file you produced yourself --
    joblib files can run code."""
    if id not in CLEANED:
        raise HTTPException(404, "No pipeline yet. Run preprocessing first.")
    buf = io.BytesIO()
    joblib.dump(CLEANED[id][1], buf)
    return Response(content=buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=pipeline.joblib"})
