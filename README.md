# Model Forge

## Setup

Clone, then set up backend and frontend separately.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Runs at http://localhost:8000

Optional deployment settings live in `backend/.env` — copy `backend/.env.example`
and uncomment what you change. Analysis settings (split ratio, seed, outlier
parameters) are in the UI, not here.

The AI tabs need an LLM, and the rest of the app works without one. Pick either:

```
LLM_PROVIDER=ollama                    # a model on this machine (default)
OLLAMA_MODEL=<tag from `ollama list`>
```

```
HF_TOKEN=hf_xxxx                       # huggingface.co/settings/tokens
HF_MODEL=Qwen/Qwen2.5-7B-Instruct      # token needs "Inference Providers" permission
```

```
GEMINI_API_KEY=AIza...                 # aistudio.google.com/apikey
GEMINI_MODEL=gemini-2.5-flash
```

A key selects its own provider, so `LLM_PROVIDER` is only needed to override that.
Set `LLM_FALLBACK=true` to try the other configured backends when one fails.

The hosted options send your column names and a sample of rows to a third party —
use Ollama for sensitive data.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at http://localhost:3000

The frontend defaults to `http://localhost:8000` for the API. To point elsewhere,
copy `frontend/.env.local.example` to `frontend/.env.local` and set:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the backend first — CORS is set to allow `http://localhost:3000`.
