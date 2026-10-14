# Model Forge

A no-code web platform for research data analysis (capstone project *ResearchAI Studio*).
Upload a tabular dataset, choose what to predict, and the app guides the whole workflow —
data checks, cleaning, exploration, preprocessing, model training, evaluation, statistics,
explainability and a downloadable report — through one simple interface, with every step fitted
on training data only so results stay leak-free and reproducible. A separate bibliometric module
analyses publication exports from Scopus, Web of Science, PubMed and similar databases.

## What it does

- **Accounts and projects** — email / password or Google sign-in, password reset by email,
  administrator role. Every upload is a private project: only its owner can open it; projects can be
  reopened, renamed and deleted (with everything made from them), and a retention period can be set.
- **Upload and audit** — CSV, TSV, JSON or Excel upload (delimiter, encoding and decimal commas
  detected; infinite values treated as missing) with a data profile, quality score and warnings
  (missing values, duplicates, outliers, ID-like, constant and high-cardinality columns). Sample
  datasets bundled with scikit-learn load in one click.
- **Clean** — fix whitespace, spellings, types (day/month order and decimal commas decided from the
  column's own values), placeholders, zeros that mean "not measured", duplicates; questionnaire
  reverse scoring and composite scores. Every step is saved and replayed on new data.
- **Explore** — automatic charts and a chart builder, downloadable as PNG or SVG; skewness and
  kurtosis in the summary.
- **Preprocess** — imputation, encoding (including free-text columns), scaling, outlier handling,
  class imbalance, feature selection, PCA or an autoencoder, with a seeded, stratified train / test
  split (preset or custom ratio) that can keep each subject's rows together (group column).
- **Train and compare** — classification and regression models (scikit-learn, XGBoost, LightGBM,
  CatBoost, neural network), voting / stacking / weighted-average ensembles, random or grid
  hyperparameter search, cross-validation on a locked test set, live training curves, GPU where the
  library allows it. Clustering (K-Means, hierarchical, DBSCAN, Gaussian mixture) when there is no
  target.
- **Evaluate and explain** — confusion matrix, ROC, precision-recall and calibration curves;
  actual-vs-predicted, residual, error-distribution and Q-Q plots; learning and validation curves;
  permutation importance, model importance and SHAP values. Every chart downloads as SVG or PNG.
- **Statistics** — descriptive statistics, frequency tables, normality (Shapiro-Wilk, Q-Q), t-tests,
  Mann-Whitney, paired t / Wilcoxon, ANOVA / Welch / Kruskal-Wallis with Tukey post-hoc,
  chi-square / Fisher, Pearson / Spearman, linear and logistic regression, Cronbach's alpha — each with
  assumptions, effect size, confidence interval and a hypothesis decision table. The test is
  recommended from the data; the researcher confirms it.
- **Bibliometrics** — annual production and growth, productive authors / sources / countries /
  institutions, citations, h / g / i10 / m indices, co-authorship, country, keyword, co-citation and
  coupling networks, thematic map and evolution, three-field plot, word cloud, trending topics.
- **Report** — a PDF report of the whole project (or of a publication collection), Excel tables
  and a JSON experiment record.
- **Test and save** — predict on a typed-in row or an uploaded file, browse the test set, save
  trained models.
- **AI assistant (optional)** — suggests cleaning, preprocessing and models, checks whether the
  chosen target makes sense, and explains training results. Works with a local model (Ollama) or
  a hosted one (Hugging Face, Gemini); the rest of the app works without it.

Not built (plan future scope or missing libraries): LSTM / GRU / CNN / TabNet deep learning, LIME and
partial-dependence plots, SHAP for model types other than XGBoost / LightGBM / CatBoost / linear
models (needs the `shap` package), Docker images, direct subscription-database APIs.

## Tech stack

- **Frontend:** Next.js (React); charts are plain SVG
- **Backend:** FastAPI (Python), SQLAlchemy (PostgreSQL, SQLite fallback for local use)
- **Analytics:** pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost
- **Charts and reports:** Matplotlib (incl. the PDF report), Seaborn, openpyxl (Excel)

## Project structure

```
backend/
  app.py            assembles the routers below
  api/              HTTP routes only: auth.py (accounts + the data guard), projects.py, data.py,
                    training.py, inference.py, explain.py, stats.py, biblio.py, report.py, assistant.py
  core/             shared: config.py, store.py (disk), jobs.py (background runs), database.py, security.py
  preprocessing/    profiler (file reading), detector, clean, execute (fit/transform), advanced
  eda/              charts, eda, plot
  training/         models.py (model table), params.py, pipeline.py (per-fold pipeline), epochs.py,
                    metrics.py, train.py (CV runs), tuning.py, ensembles.py, clustering.py,
                    curves.py (result charts), diagnostics.py (learning / validation curves),
                    suggest.py, dataview.py
  explain/          importance.py (permutation importance, SHAP)
  stats/            describe.py, tests.py (inferential tests), models.py (regression), scales.py (reliability)
  bibliometrics/    parse.py (Scopus / WoS / PubMed / CSV), analysis.py, networks.py
  report/           layout.py (PDF pages), charts.py, analysis.py (project report), biblio.py
  inference/        predict.py (new rows, test set), registry.py (saved models)
  assistant/        llm.py (providers + prompts), results.py (results view + number check)
  tests/            end-to-end checks, several over real HTTP (server.py)
  scripts/          manual CLI helpers and deployment files
frontend/
  app/              pages and components (upload, projects, cleaning, EDA, preprocessing, training,
                    clustering, statistics, bibliometrics, report, admin, chat); chartkit.jsx = shared charts
```

Each phase lives in its own package, so a traceback names the phase that failed. Dependencies
point one way: `api → explain / stats / bibliometrics / report → training → preprocessing → core`.

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

Runs at http://localhost:8000 (API documentation at http://localhost:8000/docs).

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
not here. Notable ones:

- `GOOGLE_CLIENT_ID` — required for Google sign-in (the server only accepts tokens issued to it).
- `JWT_SECRET_KEY` — optional; otherwise a random key is generated once into `DATA_DIR/.jwt_secret`.
- `ADMIN_EMAILS` — accounts with administrator rights.
- `SMTP_*`, `FRONTEND_URL` — password-reset emails.
- `DATA_RETENTION_DAYS` — delete stale uploads at startup.

Every data route requires a signed-in user who owns the dataset. Datasets uploaded before accounts
existed have no owner and can't be opened — upload them again.

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

Every backend module has a built-in self-check, and `tests/` holds end-to-end checks — several run
the real app over HTTP (auth, the data guard, security, ML, statistics, bibliometrics, reports). Run
from `backend/` as a module, e.g.:

```bash
python -m training.train
python -m stats.tests          # reproduces R's published outputs (t-tests, ANOVA, chi-square, ...)
python -m tests.test_security
```
