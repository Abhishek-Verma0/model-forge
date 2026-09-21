"""HTTP entrypoint: creates the app and plugs in one router per phase.

  api/data.py       upload, audit, charts, cleaning, preprocessing, downloads
  api/training.py   model training runs
  api/inference.py  test trained models, save / download them
  api/assistant.py  optional LLM features

Run from backend/: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import assistant, auth, data, inference, training
from core import config, database, jobs

# Initialize database tables (User, etc.)
database.init_db()

app = FastAPI(title="ResearchAI Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(assistant.router)

# Runs left queued/running by a previous process can never finish -- say so.
jobs.mark_interrupted()


@app.get("/health")
def health():
    return {"status": "ok"}
