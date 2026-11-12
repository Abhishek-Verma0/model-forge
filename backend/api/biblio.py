"""Bibliometrics routes (plan §14): upload a publication export (a private project
of kind "bibliometric"), read its summary, run analyses, keep them for the report.
Routes only -- the work happens in bibliometrics/."""

from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile

from api.auth import owner_guard
from api.common import read_upload
from bibliometrics import analysis, networks, parse
from core import store
from models.user import User

router = APIRouter(dependencies=[Depends(owner_guard)])  # signed in + owns the collection
ALLOWED = {".csv", ".txt", ".tsv", ".nbib", ".json"}   # Scopus / Dimensions CSV; WoS / PubMed text; Crossref / OpenAlex JSON


def _records(ds_id):
    m = store.meta(ds_id) or {}
    df = store.load_frame(ds_id)
    if df is None or m.get("kind") != "bibliometric":
        raise HTTPException(404, "Publication collection not found.")
    return df


def summary(df, m=None):
    m = m or {}
    return {"format": m.get("format"), "notes": m.get("notes", []), "completeness": parse.completeness(df),
            "overview": analysis.overview(df), "annual": analysis.annual(df)}


@router.post("/api/biblio/upload")
async def upload(file: UploadFile, user: User = Depends(owner_guard)):
    raw, _suffix = await read_upload(file, ALLOWED)
    try:
        df, fmt, notes = parse.read(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception as exc:  # noqa: BLE001 -- a malformed file is the user's to fix, not a server error
        raise HTTPException(422, f"Could not read this file: {exc}") from None
    ds_id = store.create(df, file.filename, owner=user.id, kind="bibliometric")
    store.update_meta(ds_id, format=fmt, notes=notes)
    m = store.meta(ds_id)
    return {"id": ds_id, "title": m["title"], "filename": file.filename, **summary(df, m)}


@router.get("/api/biblio/summary")
def read_summary(id: str):
    df = _records(id)
    m = store.meta(id)
    return {"id": id, "title": m.get("title"), "filename": m.get("filename"), **summary(df, m)}


def _int(p, key, default, lo, hi):
    v = int(p.get(key) if p.get(key) not in (None, "") else default)
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}.")
    return v


ANALYSES = {
    "productivity": lambda df, p: analysis.productivity(df, _int(p, "top", analysis.TOP_N, 1, 200)),
    "citations": lambda df, p: analysis.citations(df, _int(p, "top", analysis.TOP_N, 1, 200)),
    "wordcloud": lambda df, p: analysis.word_cloud(df, p.get("field", "keywords"), _int(p, "n", analysis.CLOUD_WORDS, 5, 300)),
    "trends": lambda df, p: analysis.trends(df, p.get("field", "keywords"),
                                            _int(p, "min_freq", analysis.TREND_MIN_FREQ, 1, 1000),
                                            _int(p, "per_year", analysis.TREND_PER_YEAR, 1, 50)),
    "network": lambda df, p: networks.network(df, p.get("kind", "keyword"), _int(p, "top_n", networks.TOP_N, 2, 300),
                                              _int(p, "min_weight", 1, 1, 1000), p.get("field", "keywords")),
    "thematic": lambda df, p: networks.thematic_map(df, p.get("field", "keywords"),
                                                    _int(p, "words", networks.THEME_WORDS, 10, 2000),
                                                    _int(p, "min_freq", networks.THEME_MIN_FREQ, 1, 1000)),
    "evolution": lambda df, p: networks.evolution(df, p.get("field", "keywords"), p.get("cuts"),
                                                  _int(p, "min_freq", 2, 1, 1000)),
    "three_field": lambda df, p: networks.three_field(
        df, tuple(p.get("fields") or ("authors", "keywords", "source")), _int(p, "n", networks.THREE_FIELD_N, 2, 50)),
}
FIELDS_OK = {"authors", "keywords", "index_keywords", "title", "source", "countries"}


@router.post("/api/biblio/run")
def run(payload: dict = Body(...)):
    """Body: {id, analysis, params?, save?: true}."""
    df = _records(payload.get("id"))
    kind = payload.get("analysis")
    if kind not in ANALYSES:
        raise HTTPException(422, f"Analysis must be one of {', '.join(ANALYSES)}.")
    p = payload.get("params") or {}
    fields = list(p.get("fields") or []) + ([p["field"]] if p.get("field") else [])
    if any(f not in FIELDS_OK for f in fields):
        raise HTTPException(422, f"Fields must be among {', '.join(sorted(FIELDS_OK))}.")
    try:
        result = ANALYSES[kind](df, p)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    result.update(analysis=kind, params=p)
    if payload.get("save", True):
        result["id"] = store.save_analysis(payload["id"], "biblio", result)
    return result


@router.get("/api/biblio/history")
def history(id: str):
    _records(id)
    return store.analyses(id, "biblio")


@router.delete("/api/biblio")
def delete(id: str, item: str):
    _records(id)
    try:
        store.delete_analysis(id, "biblio", item)
    except KeyError:
        raise HTTPException(404, "Not found.") from None
    return {"deleted": item}
