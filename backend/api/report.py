"""Report routes (plan §15): the project's PDF report, Excel tables and JSON
experiment record; the bibliometric report. Routes only -- the work happens in
report/."""

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import owner_guard
from core import store
from report import analysis, biblio

router = APIRouter(dependencies=[Depends(owner_guard)])  # signed in + owns the dataset
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _call(fn, *args):
    try:
        return fn(*args)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from None
    except KeyError:
        raise HTTPException(404, "Run not found.") from None


def _name(ds_id, ext):
    title = (store.meta(ds_id) or {}).get("title") or "report"
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in title).strip()[:60] or "report"
    return f'attachment; filename="{safe}.{ext}"'


@router.get("/api/report/pdf")
def report_pdf(id: str, run: str = ""):
    kind = (store.meta(id) or {}).get("kind")
    body = _call(biblio.pdf, id) if kind == "bibliometric" else _call(analysis.pdf, id, run or None)
    return Response(body, media_type="application/pdf", headers={"Content-Disposition": _name(id, "pdf")})


@router.get("/api/report/excel")
def report_excel(id: str, run: str = ""):
    kind = (store.meta(id) or {}).get("kind")
    body = _call(biblio.excel, id) if kind == "bibliometric" else _call(analysis.excel, id, run or None)
    return Response(body, media_type=XLSX, headers={"Content-Disposition": _name(id, "xlsx")})


@router.get("/api/report/json")
def report_json(id: str, run: str = ""):
    if (store.meta(id) or {}).get("kind") == "bibliometric":
        raise HTTPException(422, "The JSON record is for analysis projects; use the Excel export.")
    return _call(analysis.json_export, id, run or None)
