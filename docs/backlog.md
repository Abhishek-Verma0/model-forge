# Backlog — deliberately not built yet

Revisit before the ML phase. Each item says why it waited and when it's needed.

## Text columns + saved pipeline (shipped 2026-09-13, "option B")

What shipped:
- `detector.text_kind` (free text vs many-valued labels); free text is no longer flagged as an ID.
- `encode method="text"` keeps the raw text column in the CSV and fits **word 1–2 + letter 3–5 TF-IDF on train**.
- **Saved fitted pipeline** for every step: `execute.transform(fitted, raw_df)` replays clean ops + all train-fitted
  values on new rows; `execute.to_matrix(fitted, frame)` builds the model input (numbers + sparse TF-IDF).
  Downloaded from `/api/pipeline` (joblib). Self-check proves replay on raw test rows == the run's own test output.
- Cleaning no longer pre-ticks high-cardinality columns for drop; Preprocessing defaults text columns to `text`.
- Fixed while in there: mutual-info feature selection was unseeded (different columns every run).
- Verified: numeric/categorical output byte-identical to commit 647786e on 12 plan × pipeline × task runs.

**Why this method (measured on UCI SMS spam, 5,171 deduped msgs, 10-fold CV, logistic regression tuned equally):**
word + letter TF-IDF uncompressed F1 0.951, 2.4 real mails flagged per 1,000 — beat full-word TF-IDF on 10/10 folds.
The earlier SVD-squashed version: size 30 → 0.910 (8.5/1,000), size 100 → 0.925. Full-word only 0.936.
Naive Bayes 0.936 (2.0/1,000). Notebook's NLTK stopwords + lemmatize, leak-free: 0.917.
Not measured: tree models, other datasets, non-English/tweets.

Not built:

| Item | Why it waited | Build when |
|---|---|---|
| **Dense fallback for models that refuse sparse input** (e.g. `HistGradientBoosting` raised "dense data is required"; SVD-squash the text for those only) | No model is chosen in the app yet | When the ML phase adds a dense-only model |
| **Memory/size limits for text** (110k TF-IDF terms on 5k SMS → 2.2 MB pipeline; long documents will be far bigger) | Not a problem on current datasets | Add `max_features` / `min_df` when a large text file hits memory |
| **Pick the encoder automatically by cross-validation** (drop vs TF-IDF vs embeddings, per text column, on train) | Needs a training loop — ML phase | Week 8+, reuse the model code |
| **Contribution card** (score of a column alone vs baseline; top words per class; leakage warning when near-perfect) | Same — needs models | With the CV item above |
| **Sentence embeddings** (sentence-transformers MiniLM ~90 MB, or Model2Vec for fast CPU) | Slow on CPU; STRABLE 2026: only pays off on free-text-heavy tables | If a demo dataset is mostly long text |
| **Tweet/informal text handling** (keep emojis and `!`/`?` as tokens, `URL`/`USER` placeholders, squeeze "soooo", split hashtags) | `TfidfVectorizer`'s word tokens drop emojis and 1-char tokens; letter n-grams catch some | Tweets/sentiment dataset |
| **User-tunable detection thresholds / settings registry** (`FREETEXT_MIN_WORDS=3`, `FREETEXT_MIN_UNIQUE=0.5`, `MAX_CATEGORICAL_UNIQUE=50`, `ID_RATIO_THRESHOLD=0.95`) | Still fixed constants in `detector.py` | With the spec's settings registry (`docs/superpowers/specs/…`) |
| **Personal-data detection** (names, emails, phones) | 2-word names are still caught only by the uniqueness ID rule | Before real user data (plan §24) |
| **Spec strings / number-with-unit extraction** (`"2.5GHz"` → 2.5, `"8GB"` → 8) | Letter n-grams capture them roughly | If laptop/product-style datasets matter |
| **List columns** (`"red;blue;green"` → one flag per item) | Not seen in current datasets | When one appears |
| **Model-native text** (CatBoost `text_features`, TabSTAR, TabPFN text adapter) | Model-phase choices | ML phase evaluation |
| **Datetime columns** | Still one-hot or dropped (>50 values); no calendar parts | Next data-phase pass |
| `CLEAN_LOG` in `app.py` keeps one small ops list per upload until restart | Tiny; in-memory store is already restart-scoped | With persistence (DB) |

## Bugs found, not fixed

- `outliers remove_rows` ignores its `k` parameter (always 1.5×IQR) while the summary label prints the chosen k (`execute.py` remove_rows loop).
- Regression with missing target values crashes ("Input y contains NaN") — same in old and new code.

## Still open from the 2026-09-13 data-pipeline audit

- HTTP 500 on a single-value numeric column, a 1-row file, or any `inf` cell (NaN/inf in JSON).
- Semicolon CSV read as one column (scores 100/100); decimal comma `3,5` → `35`; dd/mm dates swapped/lost.
- `.xls` listed as supported but `xlrd` isn't installed.
- Duplicate removal runs before the ID drop/trim/merge (Cleaning op order) → leakage; repeat subjects cross the split.
- NaN target rows kept; oversampling written into the exported CSV; clustering blocked (target required).
- Pima-style zeros-as-missing untouched.
- `reader.py` / `roles.py` exist only as `.pyc` in `__pycache__` — source was never saved.
- Unpinned `requirements.txt` (env has pandas 3.0.5, scikit-learn 1.9.0) — matters more now: a saved pipeline needs the same scikit-learn to load.
