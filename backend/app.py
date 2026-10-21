"""HTTP entrypoint: creates the app and plugs in one router per phase.

  api/auth.py          accounts, sessions, password reset, admin; the data guard
  api/projects.py      private workspaces: list / reopen / rename / delete
  api/data.py          upload, audit, charts, cleaning, preprocessing, downloads
  api/training.py      model training runs
  api/inference.py     test trained models, save / download them
  api/explain.py       feature importance and SHAP for a trained model
  api/stats.py         descriptive + inferential statistics, reliability, regression
  api/biblio.py        bibliometrics: publication exports, citations, networks, themes
  api/report.py        PDF report, Excel tables, JSON experiment record
  api/assistant.py     optional LLM features

Every data router depends on api.auth.owner_guard: signed in, and the dataset a
request names must belong to the caller.

Run from backend/: uvicorn app:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import assistant, auth, biblio, data, explain, inference, projects, report, stats, training
from api.common import RequestLog, SafeJSONResponse
from core import config, database, jobs, logs, store  # noqa: F401 -- logs: console handler for the modelforge loggers

# Initialize database tables (User, etc.)
database.init_db()

app = FastAPI(title="ResearchAI Studio API", default_response_class=SafeJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLog)   # console: each request's arrival, answer + duration, and what a slow one is doing
app.include_router(auth.router)
app.include_router(auth.admin_router)
app.include_router(data.public)
app.include_router(projects.router)
app.include_router(data.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(explain.router)
app.include_router(stats.router)
app.include_router(biblio.router)
app.include_router(report.router)
app.include_router(assistant.router)

# Runs left queued/running by a previous process can never finish -- say so.
jobs.mark_interrupted()
# Retention period (plan §24): DATA_RETENTION_DAYS in backend/.env, 0 = keep.
_removed = store.sweep(config.DATA_RETENTION_DAYS)
if _removed:
    logging.getLogger("modelforge").info("retention: removed %d dataset(s) older than %d days",
                                         len(_removed), config.DATA_RETENTION_DAYS)


@app.get("/health")
def health():
    return {"status": "ok"}
