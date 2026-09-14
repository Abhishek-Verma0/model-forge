# Model Forge

A no-code web platform for research data analysis (capstone project *ResearchAI Studio*).
Upload a tabular dataset, choose what to predict, and the app guides the whole workflow —
data checks, cleaning, exploration, preprocessing, model training and evaluation — through one
simple interface, with every step fitted on training data only so results stay leak-free and
reproducible.

## What it does

- **Upload and audit** — CSV / Excel upload with a data profile, quality score and warnings
  (missing values, duplicates, outliers, ID-like, constant and high-cardinality columns).
- **Clean** — fix whitespace, spellings, types, placeholders and duplicates; every step is saved
  and replayed on new data.
- **Explore** — automatic charts and a chart builder, downloadable as PNG or SVG.
- **Preprocess** — imputation, encoding (including free-text columns), scaling, outlier handling,
  class imbalance, feature selection and PCA, with a seeded, stratified train / test split.
- **Train and compare** — classification and regression models (scikit-learn, XGBoost, LightGBM,
  CatBoost, neural network) with cross-validation, a locked test set, editable parameters, live
  training curves and GPU support where the library allows it.
- **Test and save** — predict on a typed-in row or an uploaded file, browse the test set, and save
  trained models.
- **AI assistant (optional)** — suggests cleaning, preprocessing and models, checks whether the
  chosen target makes sense, and explains training results. Works with a local model (Ollama) or
  a hosted one (Hugging Face, Gemini); the rest of the app works without it.

Planned: clustering, ensembles (voting / stacking), hyperparameter search, result plots (confusion
matrix, ROC / PR, residuals), statistical tests, bibliometric analysis, SHAP explanations and a
PDF report.

## Tech stack

- **Frontend:** Next.js (React)
- **Backend:** FastAPI (Python)
- **Analytics:** pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost
- **Charts:** Matplotlib, Seaborn

## Project structure

```
backend/
  app.py            assembles the routers below
  api/              HTTP routes only: data.py, training.py, inference.py, assistant.py
  core/             shared: config.py, store.py (disk), jobs.py (background runs)
  preprocessing/    profiler, detector, clean, execute (fit/transform), advanced
  eda/              charts, eda, plot
  training/         models.py (model table), params.py, pipeline.py (per-fold pipeline),
                    epochs.py (live curves), metrics.py, train.py (CV runs),
                    suggest.py (AI model picks), dataview.py (preprocessed data view)
  inference/        predict.py (new rows, test set), registry.py (saved models)
  assistant/        llm.py (providers + prompts), results.py (results view + number check)
  tests/            end-to-end checks
  scripts/          manual CLI helpers (check_*.py)
frontend/
  app/              pages and components (upload, cleaning, EDA, preprocessing, training, chat)
```

Each phase lives in its own package, so a traceback names the phase that failed.

## Setup

Clone, then set up backend and frontend separately. Start the backend first.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Runs at http://localhost:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at http://localhost:3000

## Configuration

Optional deployment settings live in `backend/.env` — copy `backend/.env.example` and uncomment
what you change. Analysis settings (split ratio, seed, outlier parameters) are chosen in the UI,
not here.

The AI features need an LLM. Pick one:

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

A key selects its own provider, so `LLM_PROVIDER` is only needed to override that. Set
`LLM_FALLBACK=true` to try the other configured backends when one fails. With Ollama,
`OLLAMA_NUM_CTX` (default 16384) sets how many tokens the model reads per request; longer requests
fail with a clear error instead of being silently cut.

The hosted options send your column names and a sample of rows to a third party — use Ollama for
sensitive data.

The frontend calls `http://localhost:8000` by default. To point elsewhere, copy
`frontend/.env.local.example` to `frontend/.env.local` and set `NEXT_PUBLIC_API_URL`. CORS allows
`http://localhost:3000`.

## Tests

Every backend module has a built-in self-check. Run from `backend/` as a module, e.g.:

```bash
python -m training.train
python -m tests.test_train_api
```
