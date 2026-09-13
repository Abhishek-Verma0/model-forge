# ML Training Pipeline — Design

**Date:** 2026-09-13
**Scope:** plan PDF weeks 8–14 — train, tune, compare and evaluate models on the preprocessed data (classification, regression, clustering), ensembles + MLP, optional pretrained fine-tuning. Statistics, bibliometrics, SHAP and PDF reports are out of scope.
**Status:** design approved in chat, section by section; implementation not started.
**Builds on:** the data phase as of 2026-09-13 — saved fitted pipeline (`execute.transform` / `execute.to_matrix`), raw text + word/letter TF-IDF (`docs/backlog.md`).

---

## 1. Principles

1. **Leak-free by construction.** Anything learned from data — preprocessing, resampling, tuning — is learned from the rows being trained on, never from rows being scored.
2. **One locked test set.** Split once with the preprocessing run's seed and ratio. It is used for the final score only; never for picking models or settings.
3. **No hidden constants.** Every number the pipeline uses is a setting with a source label, shown in the UI, recorded in the run (§8).
4. **Measured, not assumed.** Compatibility rules and warnings come from facts verified on the installed libraries (§4) or measured on the user's data at run time.
5. **Suggestions are starting points, not verdicts.** Only cross-validated results earn "best".

## 2. Architecture

| File (backend/preprocessing/) | Responsibility |
|---|---|
| `store.py` | Datasets, pipelines and runs on disk; small in-memory cache |
| `jobs.py` | Background run execution: progress, time limit, cancel, restart → "interrupted" |
| `models.py` | Model table: task, capability facts, search space; optional libraries loaded only if importable |
| `train.py` | Fold-refit pipeline, CV, tuning loop, locked-test evaluation, metrics, result charts, run record |
| `execute.py` (changed) | Fit part of `apply_plan` extracted into a function reused by the Preprocessing tab and every CV fold |

| Frontend | Responsibility |
|---|---|
| `train.jsx` (new) | Train tab: task, models, settings, progress, results, run history |
| `page.jsx` (changed) | Adds the tab |

**API (app.py):** start run, poll status, cancel, list runs, fetch run results / charts / best model file.

**Data flow:** saved table + preprocessing settings → target guard → locked split → per model: `Pipeline([prep, resample?, model])` → CV on train → (tuning) → refit on all train → score once on test → run record on disk.

## 3. Piece 0 — Groundwork (week 8)

1. **Pinned versions** in `requirements.txt`: the exact working set (verified 2026-09-13: pandas 3.0.5, scikit-learn 1.9.0, imbalanced-learn 0.14.2) plus user-installed `xgboost`, `lightgbm`, `catboost`, `psutil`. A saved pipeline/model loads reliably only with the same versions.
2. **Target guard.** Rows with a missing target are removed before the split and counted in the summary. Regression requires a numeric target. Classification requires ≥ 2 classes; a class with a single row gets a clear message.
3. **Resampling out of the export.** SMOTE, over/undersampling and isolation-forest row removal are no longer written into the CSV; they are recorded as settings and applied inside each fold to fold-training rows only (`imblearn.pipeline.Pipeline`). Class weights stay as information passed to models. The Preprocessing tab labels them "applied during training".
4. **Disk store.** `backend/data/<dataset_id>/` holds the current table, clean-ops log and pipeline; `runs/<run_id>/` holds settings, results, charts and the best model. Format: pickle/joblib (preserves dtypes; only files the app wrote are ever loaded). PostgreSQL and per-user ownership arrive with auth — not in this scope.
5. **Background jobs.** `POST /api/train` returns a run id; the page polls status ("model 4/12 · fold 3/5 · ~6 min left") and partial results. Time limit and cancel are checked between fits — a fit already running (thread) finishes first. Restart mid-run → run marked "interrupted".

## 4. Model table and verified compatibility

Verified 2026-09-13 on scikit-learn 1.9.0 with a sparse matrix of numeric + TF-IDF-like columns containing negative values (the shape `to_matrix` produces after scaling):

| Model | Task | Sparse input | Needs non-negative | Probabilities | Missing values |
|---|---|---|---|---|---|
| LogisticRegression | clf | ✅ | – | ✅ | ❌ ("Input X contains NaN") |
| GaussianNB | clf | ❌ dense required | – | ✅ | – |
| MultinomialNB / ComplementNB | clf | ✅ | ❌ fails on negatives | ✅ | – |
| KNeighbors | clf, reg | ✅ runs; distances degrade on wide text | – | ✅ | – |
| DecisionTree, RandomForest, ExtraTrees, AdaBoost, GradientBoosting | clf (+reg where exists) | ✅ | – | ✅ | – |
| HistGradientBoosting | clf, reg | ❌ dense required | – | ✅ | ✅ native |
| SVC / SVR | clf, reg | ✅ | – | ❌ by default (AUC from decision scores) | – |
| MLPClassifier / MLPRegressor | clf, reg | ✅ | – | ✅ | – |
| Linear, Ridge, Lasso, ElasticNet | reg | ✅ | – | – | – |
| DummyClassifier / DummyRegressor (baseline) | clf, reg | ✅ | – | ✅ | – |
| KMeans, DBSCAN | cluster | ✅ | – | – | – |
| AgglomerativeClustering, GaussianMixture | cluster | ❌ dense required | – | – | – |
| XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10 | clf (binary + multiclass), reg | ✅ | – | ✅ | ✅ native |

Boosters verified 2026-09-13 after install: sparse, NaN, multiclass and scikit-learn `cross_val_score` all pass; XGBoost `device="cuda"` and CatBoost `task_type="GPU"` fit on the RTX 5070. psutil was **not** installed at that check — parallel folds fall back to 1 worker until it is.

Also verified: SMOTE (imblearn 0.14.2) runs on the sparse matrix; `silhouette_score` accepts `sample_size`; `StackingClassifier` default `cv=None` = 5-fold internal CV; MLP has `early_stopping`.

**Compatibility rules** are computed from these facts + the data (text columns present, negatives after scaling, NaN left after preprocessing, row count). Incompatible models are listed but disabled, with the reason. Missing-value support only matters when the user left gaps unfilled.

**Slow-model warning is measured, not a fixed row count.** Before a run, each ticked model is fitted on two small samples of the user's data; the growth rate is extrapolated to full rows × folds × trials. Measured on this machine 2026-09-13: SVC fit 0.05 s / 0.21 s / 0.90 s at 2k / 4k / 8k rows (~4× per doubling); LogisticRegression ≤ 0.004 s. A warning shows when the projection exceeds the time limit.

## 5. Piece 1 — Training core (weeks 9–10)

**Task.** Pre-filled from the target and overridable. Required because `page.jsx:319` maps any numeric target to regression — a 0/1/2 or 1–5 coded class would be mis-tasked.

**Suggested models** (pre-ticked, shown with "why suggested"; never labelled best):
- always: baseline;
- classification: LogisticRegression, RandomForest, one gradient booster (LightGBM if importable and it passes the tiny-fit check; else HistGradientBoosting for dense data, GradientBoosting for sparse data); + a Naive Bayes when all inputs are non-negative; + SVC when its projected time fits the limit;
- regression: Ridge, RandomForest, one gradient booster (same rule as classification); + SVR when its projected time fits the limit.

**Run.**
1. Load table + preprocessing settings; target guard; locked split (preprocessing seed + ratio).
2. Per model `Pipeline([prep, resample-if-chosen, model])`, `prep` = our fitted-pipeline code wrapped as a scikit-learn step (fit → our fit function, transform → `transform` + `to_matrix`).
3. Stratified K-fold (classification) / K-fold (regression), shuffled with the run seed; all metrics recorded as mean ± sd across folds.
4. Refit on all train rows; score once on the locked test set.
5. Save: settings, seeds, library versions, CV table, test metrics per model, test predictions/probabilities, fit times, best model file only (others re-trainable from settings).

**Metrics (plan §11).** Classification: accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, Cohen's kappa, ROC-AUC, PR-AUC, log loss (macro-averaged for 3+ classes; decision scores for AUC when no probabilities; log loss "n/a" then). Regression: MAE, MSE, RMSE, R², adjusted R², MAPE ("n/a" when the target contains zeros).

**Primary metric** — suggested, changeable: imbalanced classification → F1 (binary) / balanced accuracy (multiclass); otherwise accuracy; regression → RMSE.

**Baseline** — most frequent class / training mean; untuned; same folds and test set; pinned reference row; never eligible for tuning or the best badge.

## 6. Pieces 2–3 — Tuning and results (week 11)

**Flow:** quick run (defaults, full CV) → tune the leaders.

**Leader selection (exact):**
1. Rank by CV mean of the primary metric (direction-aware: lower is better for RMSE, MAE, MSE, log loss). The test set is never used.
2. Eligible: not the baseline, did not fail, CV mean beats the baseline's.
3. Take top K.
4. Ties: smaller CV sd → shorter fit time → model name.
5. Shown pre-ticked; the user can add or remove before tuning.

**Search:** random search with `n_trials` per model, or grid search showing its real combination count first. Implemented as a loop over settings, each cross-validated, so progress / cancel / time limit are checked between trials. Search spaces live in the model table. Boosters and MLP use their own early stopping.

**Parallel folds — memory-aware:** workers = min(folds remaining, CPU cores, ⌊usable RAM ÷ measured per-fold peak⌋). Fold 1 of each model runs alone while process memory is tracked (psutil); its peak increase sizes the remaining folds; workers drop if memory tightens. Without psutil → 1 worker. A "max workers" field can only lower the result.

**Results screen:** comparison table sorted by the primary metric; CV mean ± sd and locked-test score, each labelled; baseline row; fit time; best settings; badge "Best by {primary metric} (CV mean)". Changing the primary metric after a run re-sorts and moves the badge, with a note naming the metric tuning optimised. Per model: confusion matrix, ROC, PR (classification); actual vs predicted, residuals (regression); learning curve on a button (retrains). Charts server-rendered with matplotlib (same pattern as `eda.py` / `plot.py`), PNG + SVG. Downloads: results CSV/JSON, best model (joblib). Run history reopened from disk.

**Known optimism:** the CV score of a tuned model is slightly optimistic (same folds choose and score); the locked-test score is the honest number and is labelled as such. Nested CV (≈ folds × cost) is not built.

## 7. Pieces 4–6

**4. Clustering (week 12).** Target optional in preprocessing; no split. Models: KMeans, DBSCAN (sparse OK), Agglomerative, GaussianMixture (dense only — disabled with text unless a reduction step is on). Metrics: silhouette (sampled), Davies-Bouldin, Calinski-Harabasz — measure compactness/separation only, stated in the UI. Charts: elbow, silhouette plot, 2-D PCA scatter, dendrogram. DBSCAN `eps` chosen with a k-distance plot. Agglomerative gets a row cap computed from available RAM: largest n with 1.2 × n(n−1)/2 × 8 bytes ≤ usable RAM (condensed pairwise distances; measured 2026-09-13 with tracemalloc: 18.7 MB at 2,000 rows, 72.0 MB at 4,000 rows = 1.13–1.17× the formula).

**5. Ensembles + MLP (week 13).** MLP models are ordinary table rows with early stopping; their loss / validation curves are the plan's DL curves. Ensembles are built from a finished run's models: voting (soft only with probability-capable models), stacking (LogisticRegression / Ridge meta-model; internal CV multiplies base-model fits by its fold count + 1 — cost warning), weighted averaging (regression). Preprocessing runs once per fold, shared by the base models. Same folds, same test set, new rows in the table.

**6. Pretrained fine-tuning (week 14, only if 8–13 on time; otherwise future scope).** One text column, classification/regression, a small Hugging Face model. User installs CUDA PyTorch for RTX 50-series + `transformers`; the model downloads once. Trained on train rows with a held-out validation share for early stopping; scored on the locked test set; row labelled "fine-tuned — test only, no CV". Hidden when no CUDA GPU is detected. Detected 2026-09-13: RTX 5070 Laptop GPU 8,151 MiB; torch not importable from `backend/.venv` or the system Python at that time.

## 8. Settings (no hidden constants)

Every entry is shown in the UI with its label, recorded in the run, and overridable within its range.

| Setting | Default | Source label |
|---|---|---|
| Test split ratio, seed | inherited from the preprocessing run | user |
| CV folds | 5 | published default (scikit-learn `cross_validate` cv=5) |
| Tuning method | off / random / grid | user |
| `n_trials` (random search) | 20, range 1–500 | our starting point |
| Leaders to tune (K) | 3, range 1–eligible | our starting point |
| Time limit per run | 60 min | our starting point |
| Parallel workers | computed (§6) | computed |
| RAM safety share | 50 % | our starting point |
| Max workers cap | none (can only lower) | user |
| Slow-model warning | projected time > time limit | computed from a timing probe |
| Imbalance cut for primary-metric suggestion | smallest class < 20 % of rows | our starting point |
| Silhouette sample | 10,000 rows | our starting point |
| Agglomerative row cap | 1.2 × n(n−1)/2 × 8 bytes ≤ usable RAM (§7) | computed; 1.2 factor from measurement |
| DBSCAN `min_samples` | 5 | published default (scikit-learn) |
| KMeans k range for elbow | 2 to min(10, rows − 1) | our starting point |
| Model search spaces | per model, listed in the model table | our starting point, each range shown |
| Fine-tune max tokens | 95th percentile of tokenised train lengths, capped at the model maximum | computed |
| Fine-tune batch size | largest that fits: start 16, halve on out-of-memory | computed at run time |
| Fine-tune learning rate / epochs | 2e-5 / up to 4 with early stopping | published (BERT paper recommends 2e-5–5e-5, 2–4 epochs) |
| Fine-tune validation share | 10 % of train | published default (scikit-learn MLP `validation_fraction`) |

## 9. Testing

Self-check style, per module:
- **Leakage:** a spy step records row indices seen by `prep.fit`; asserts none belong to that fold's scored rows.
- **Reproducibility:** same seed + settings → identical CV table.
- **Accuracy:** run metrics equal hand-written scikit-learn on the same folds.
- **Compatibility:** every allowed model × data shape trains on a tiny dataset; every disabled combination really raises.
- **Jobs:** cancel, time limit, restart → "interrupted".
- **Data phase unchanged:** golden comparison against commit 647786e keeps passing.

## 10. Delivery order

| Week | Piece |
|---|---|
| 8 | 0 Groundwork |
| 9–10 | 1 Training core + Train tab table |
| 11 | 2 Tuning + 3 results charts |
| 12 | 4 Clustering |
| 13 | 5 Ensembles + MLP |
| 14 | Buffer + tests; 6 fine-tuning only if 8–13 on time |

**Not in these weeks (stay in `docs/backlog.md`):** semicolon / decimal-comma misreads, day-first dates, dedupe-before-ID-drop order. Uploads hitting those still train on misread data.

## 11. Risks

| Risk | Control |
|---|---|
| Boosting libraries incompatible with scikit-learn 1.9.0 | Verify after install; model rows appear only when import + a tiny fit succeed |
| Fold-refit makes runs slow | Quick run first, measured time projection, time limit, memory-aware parallel folds |
| Memory blow-up with wide text matrices | Fold-1 memory measurement; fallback to 1 worker |
| Misleading "best" | CV-only selection, baseline row, locked-test score labelled, primary metric named on the badge |
| GPU stack mismatch (RTX 50-series) | Fine-tuning optional, hidden without a working CUDA device |
