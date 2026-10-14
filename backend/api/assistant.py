"""AI assistant routes: plan suggestions, chat, plain-English explanations.
All optional -- a down LLM returns 503 and the rest of the app keeps working."""

import json

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import owner_guard
from api.common import frame_or_404
from assistant import llm, results
from training.metrics import METRICS
from preprocessing.profiler import profile_dataset


router = APIRouter(dependencies=[Depends(owner_guard)])  # signed in + owns the dataset

SAMPLE_SEED = 0         # our starting point: same rows every time, so the same data gives the same prompt
MAX_CELL_CHARS = 300    # our starting point: long text cells are shortened so a text column can't fill the window
# our starting point: the row sample's JSON length. Measured 2026-10-15: 96 rows of the
# 31-column breast-cancer data were 43,503 tokens (every row repeats every column name) and
# Ollama refused them at its 16,384-token window; this budget keeps the sample to a few
# thousand tokens whatever the width.
SAMPLE_CHARS = 12_000
MIN_SAMPLE_ROWS = 5     # our starting point: fewer rows than this say too little


def _short(v):
    return v[:MAX_CELL_CHARS] if isinstance(v, str) else v


def _llm_context(df, profile, target, task):
    """Aggregated metadata + a small real-row sample for the LLM planner.
    Never raw full data: the sample is capped by width (~100 rows, fewer for many
    columns) and then halved until its JSON fits SAMPLE_CHARS, so a wide table can't
    blow the model's context window. Rows are a seeded random sample -- the first
    rows of a sorted file would mislead."""
    ncols = max(1, len(df.columns))
    n = min(100, max(10, 3000 // ncols))
    info = profile["columns_info"]
    cols = [{"name": c, "dtype": str(df[c].dtype),
             "missing_percent": info[c]["missing_percent"],
             "unique": info[c]["unique_values"]} for c in df.columns]
    examples = {c: [_short(str(v)) for v in df[c].dropna().unique()[:8]] for c in df.columns}
    while True:
        rows = df.sample(n=n, random_state=SAMPLE_SEED).sort_index() if len(df) > n else df
        text = rows.map(_short).to_json(orient="records")
        if len(text) <= SAMPLE_CHARS or n <= MIN_SAMPLE_ROWS:
            break
        n = max(MIN_SAMPLE_ROWS, n // 2)
    sample = json.loads(text)
    return {"target": target, "task": task, "columns": cols,
            "examples": examples, "sample": sample}


@router.post("/api/plan")
def plan(payload: dict = Body(...)):
    """LLM proposes a structured clean+preprocess plan (validated against the op
    allowlist). Optional -- 503 if Ollama is down; the frontend keeps its rule
    defaults. Body: {id, target, task}."""
    df = frame_or_404(payload.get("id"))
    # target is optional: cleaning suggestions don't need it, so the plan can load
    # on the Cleaning tab before the user has chosen an outcome column.
    context = _llm_context(df, profile_dataset(df), payload.get("target", ""),
                           payload.get("task", "classification"))
    try:
        return llm.recommend_plan(context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"AI plan unavailable ({llm.HINT()}): {exc}") from None


@router.post("/api/chat")
def chat_api(payload: dict = Body(...)):
    """Multi-turn preprocessing assistant, streamed as plain-text chunks.
    Optional -- 503 if the LLM is down (raised before streaming starts)."""
    context = dict(payload.get("context") or {})
    messages = payload.get("messages", [])
    question = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    # the page sends the whole run it shows; the AI gets only the part this question needs
    context["results_view"] = results.view(context.pop("results", None), question, METRICS)
    try:
        gen = llm.chat_stream(messages, context)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"AI chat unavailable ({llm.HINT()}): {exc}") from None
    return StreamingResponse(gen, media_type="text/plain")


@router.post("/api/explain")
def explain(payload: dict = Body(...)):
    """Plain-English explanation of a preprocessing plan. Optional:
    if the LLM is down, the app still works -- this just returns 503."""
    try:
        return {"text": llm.explain_recipe(payload)}
    except Exception as exc:  # noqa: BLE001 -- surface any LLM/transport failure as unavailable
        raise HTTPException(503, f"AI explanation unavailable ({llm.HINT()}): {exc}") from None
