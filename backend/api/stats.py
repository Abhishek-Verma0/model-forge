"""Statistics routes (plan §13): run an analysis on the dataset's current table,
keep it for the history / decision table / report. Routes only -- the work
happens in stats/. Reverse scoring and composite scores are cleaning ops
(POST /api/clean with reverse_score / composite)."""

from fastapi import APIRouter, Body, Depends, HTTPException

from api.auth import owner_guard
from api.common import frame_or_404
from core import store
from stats import describe, models, scales, tests

router = APIRouter(dependencies=[Depends(owner_guard)])  # signed in + owns the dataset
ALPHA_RANGE = (0.0001, 0.2)   # our starting point: accepted significance levels


def _cols(df, p, key="columns", minimum=1):
    cols = list(dict.fromkeys(p.get(key) or []))
    bad = [c for c in cols if c not in df.columns]
    if bad:
        raise ValueError(f"Not columns: {', '.join(map(str, bad))}.")
    if len(cols) < minimum:
        raise ValueError(f"Choose at least {minimum} column(s).")
    return cols


ANALYSES = {
    "describe": lambda df, p, a: describe.describe(df, _cols(df, p)),
    "frequencies": lambda df, p, a: describe.frequencies(df, _cols(df, p)),
    "normality": lambda df, p, a: describe.normality(df, _cols(df, p), p.get("by") or None, a),
    "compare": lambda df, p, a: tests.compare_groups(df, p.get("outcome"), p.get("group"), p.get("test") or "auto", a),
    "paired": lambda df, p, a: tests.paired(df, p.get("before"), p.get("after"), p.get("test") or "auto", a),
    "correlation": lambda df, p, a: tests.correlation(df, _cols(df, p, minimum=2), p.get("method") or "auto", a),
    "association": lambda df, p, a: tests.association(df, p.get("row"), p.get("col"), a),
    "regression": lambda df, p, a: (models.logistic(df, p.get("outcome"), _cols(df, p, "predictors"), p.get("event"), a)
                                    if p.get("kind") == "logistic" else
                                    models.linear(df, p.get("outcome"), _cols(df, p, "predictors"), a)),
    "reliability": lambda df, p, a: scales.reliability(df, _cols(df, p, "items", 2), p.get("reverse"),
                                                       p.get("low"), p.get("high")),
}


@router.post("/api/stats/run")
def run(payload: dict = Body(...)):
    """Body: {id, analysis, params: {...}, alpha?, save?: true} -> the result (+ its id when saved)."""
    df = frame_or_404(payload.get("id"))
    kind = payload.get("analysis")
    if kind not in ANALYSES:
        raise HTTPException(422, f"Analysis must be one of {', '.join(ANALYSES)}.")
    alpha = float(payload.get("alpha") or tests.ALPHA)
    if not ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]:
        raise HTTPException(422, f"Significance level must be between {ALPHA_RANGE[0]} and {ALPHA_RANGE[1]}.")
    try:
        result = ANALYSES[kind](df, payload.get("params") or {}, alpha)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    result["params"] = payload.get("params") or {}
    if payload.get("save", True):
        result["id"] = store.save_analysis(payload["id"], "stats", result)
    return result


@router.get("/api/stats/history")
def history(id: str):
    frame_or_404(id)
    return store.analyses(id, "stats")


@router.delete("/api/stats")
def delete(id: str, stat: str):
    try:
        store.delete_analysis(id, "stats", stat)
    except KeyError:
        raise HTTPException(404, "Not found.") from None
    return {"deleted": stat}


@router.get("/api/stats/likert")
def likert(id: str):
    return scales.likert_candidates(frame_or_404(id))


@router.get("/api/stats/decisions")
def decision_table(id: str):
    frame_or_404(id)
    return {"rows": tests.decisions(store.analyses(id, "stats")), "review": tests.REVIEW}
