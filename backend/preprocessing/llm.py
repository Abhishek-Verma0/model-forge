import json

import requests

from config import OLLAMA_MODEL, OLLAMA_URL
import clean as _clean
import execute as _execute
import advanced as _advanced


def analyze_dataset(profile):
    prompt = f"""
You are a data analysis assistant for a research analytics platform.

A user has uploaded a dataset.

Below is information calculated directly from the dataset:

{json.dumps(profile, indent=2)}

Your job is ONLY to explain what you can understand about the dataset.

Do NOT clean the data.
Do NOT modify the data.
Do NOT recommend preprocessing yet.
Do NOT invent information that is not present.

Explain in simple language:

1. What the dataset contains.
2. How many rows and columns it has.
3. What types of columns it contains.
4. Which columns have missing values.
5. Whether duplicate rows exist.
6. Anything unusual or worth noticing.

If something cannot be determined from the information provided,
say that clearly.

Keep the explanation concise and easy for a non-technical researcher
to understand.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False
        },
        timeout=500
    )

    response.raise_for_status()

    return response.json()["response"]


def explain_recipe(plan):
    """Plain-English explanation of a preprocessing plan for a non-technical user.

    `plan` is aggregated metadata only -- {target, task, columns:[{name, type,
    missing_percent, action}]} -- never raw rows (per the security rule)."""
    prompt = f"""You are a data-preprocessing assistant for non-technical researchers.

The user is predicting "{plan.get('target')}" ({plan.get('task')}). Below is the plan:
each entry is a column with its type, how much is missing, and the action chosen.

{json.dumps(plan.get('columns', []), indent=2)}

In 4-6 short sentences of plain language, explain what this preprocessing will do and
why. GROUP similar actions (e.g. "20 mostly-empty columns will be dropped") -- do NOT
list every column. Call out if many columns are dropped, or a free-text column is
dropped. Do not invent anything not in the plan."""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _extract_json(text):
    """Pull the first {...} object out of an LLM reply (it may wrap it in prose
    or ```json fences). Returns {} if nothing parses -- caller falls back."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _validate_plan(raw):
    """Keep only operations that pass the SAME allowlist Python executes with.
    The LLM proposes; invalid ops are dropped, never run. This is the wall
    against 'blindly execute arbitrary LLM code' -- the LLM emits data, not code."""
    clean_ops = []
    dropped = set()
    for op in raw.get("clean", []) or []:
        try:
            _clean._validate(op)
            clean_ops.append(op)
            if op.get("op") == "drop_column":
                dropped.add(op.get("column"))
        except (ValueError, AttributeError, TypeError):
            pass  # silently drop anything the executor wouldn't accept

    preprocess = {}
    for col, ops in (raw.get("preprocess", {}) or {}).items():
        valid = []
        for op in ops or []:
            # dropping is a CLEANING step: reroute any drop the LLM misfiled under
            # preprocess into clean_ops so it shows on the Cleaning tab, not lost.
            if op.get("op") == "drop_column":
                if col not in dropped:
                    clean_ops.append({"op": "drop_column", "column": col,
                                      "note": op.get("note", "not needed for the model")})
                    dropped.add(col)
                continue
            try:
                _execute._validate(op)
                valid.append(op)
            except (ValueError, AttributeError, TypeError):
                pass
        if valid:
            preprocess[col] = valid

    # dataset-level pipeline: keep only if the WHOLE thing validates, else drop it
    # (it's optional -- a bad suggestion must never block the plan).
    pipeline = raw.get("pipeline") or {}
    try:
        _advanced.validate_pipeline(pipeline)
    except (ValueError, AttributeError, TypeError):
        pipeline = {}

    return {"clean": clean_ops, "preprocess": preprocess, "pipeline": pipeline,
            "eda": str(raw.get("eda", "")).strip()}


_PLAN_SYSTEM = """You are a data-preprocessing planner. Given a dataset profile and a small
sample of real rows, choose CLEANING and PREPROCESSING operations. You do NOT write code --
you fill in a fixed menu of operations. Reply with ONLY a JSON object, no prose, no code fences.

Allowed CLEAN ops (applied to the whole table):
  {"op":"trim_whitespace","column":C}
  {"op":"merge_categories","column":C}                     # fix Male/male/" Male"
  {"op":"retype","column":C,"to":"number"|"datetime"|"category"}   # e.g. "1,200" -> number
  {"op":"nullify","column":C}                              # treat "N/A"/"-"/"?" text as missing
  {"op":"rename_column","column":C,"to":"clean_name"}      # tidy a messy header: "Age (yrs)" -> "age"
  {"op":"drop_column","column":C}
  {"op":"drop_duplicates"}
Allowed PREPROCESS ops (per column, fit on train):
  {"op":"impute","strategy":"mean"|"median"|"most_frequent"|"constant"}
  {"op":"encode","method":"onehot"|"ordinal"}
  {"op":"scale","method":"standard"|"robust"|"minmax"}
  {"op":"outliers","method":"clip_iqr"|"zscore"|"winsorize"|"remove_rows"}
Allowed DATASET-LEVEL pipeline (optional, fit on train, whole matrix):
  "imputation":{"method":"knn"|"iterative"}          # multivariate fill of numeric columns
  "outlier_removal":{"method":"isolation_forest"}    # drop anomalous train rows
  "feature_selection":{"method":"correlation"|"chi2"|"anova"|"mutual_info"|"rfe","k":N}
  "imbalance":{"method":"oversample"|"undersample"|"smote"|"class_weights"}   # classification only
  "reduction":{"method":"pca","n":N}

Rules:
- Column-aware and type-preserving. If a column only needs a missing-value fill, give ONLY impute
  -- do NOT add scale unless a model genuinely needs it. Never scale an ID or a category.
- impute numeric with median (or mean), categorical with most_frequent.
- encode categoricals (onehot for few categories, ordinal only if there is a real order).
- scale numeric ONLY when useful; leave it off otherwise.
- To DROP any unneeded column (free-text / ID / constant / mostly-empty / high-cardinality),
  put a CLEAN drop_column op in "clean". Dropping is a cleaning step, NEVER a preprocess step.
- rename_column ONLY when a header is messy/unclear (spaces, units, symbols, "col1"); use a short
  snake_case name. Do NOT rename already-clean headers, and never rename the target column.
- Add a short plain-language "note" to every op saying what it does and why, for a non-technical user.
- Do NOT include the target column in preprocess.
- Suggest a dataset-level "pipeline" ONLY when it clearly helps: feature_selection with many features,
  imbalance for a skewed classification target, reduction for many correlated features. Omit it otherwise.

Output shape:
{"clean":[{...,"note":"..."}], "preprocess":{"colname":[{...,"note":"..."}]},
 "pipeline":{...optional dataset-level...}, "eda":"1-2 sentences of EDA ideas"}"""


def recommend_plan(context):
    """LLM proposes a structured clean+preprocess plan from the profile + a small
    real-row sample. Output is validated against the execution allowlist before
    it ever reaches the user. `context` = {target, task, columns, examples, sample}."""
    user = f"""Target column: "{context.get('target')}"  (task: {context.get('task')})

Columns (name, dtype, missing %, unique count):
{json.dumps(context.get('columns', []), indent=1)}

Example values per column:
{json.dumps(context.get('examples', {}), indent=1)}

Sample rows:
{json.dumps(context.get('sample', []))}"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL,
              "prompt": _PLAN_SYSTEM + "\n\n" + user,
              "stream": False, "think": False, "format": "json"},
        timeout=300,
    )
    response.raise_for_status()
    return _validate_plan(_extract_json(response.json()["response"]))


# Ollama's chat endpoint (multi-turn), derived from the generate URL.
OLLAMA_CHAT_URL = OLLAMA_URL.replace("/generate", "/chat")


def chat_stream(messages, context):
    """Multi-turn preprocessing assistant for non-technical users, STREAMED so the
    UI can render the reply as it arrives.

    `messages` = [{role, content}] history; `context` = the recommended plan
    ({target, task, columns:[{name,type,missing_percent,action}]}) -- aggregated
    metadata only, never raw rows. Returns a generator of text chunks; the initial
    connection is made here so a dead Ollama raises before streaming starts."""
    system = f"""You are a friendly data-preprocessing assistant for NON-TECHNICAL users.
The user is preparing a dataset to predict "{context.get('target')}" ({context.get('task')}).
Here is the recommended plan -- each column with its type, how much is missing, and the action:

{json.dumps(context.get('columns', []), indent=2)}

Answer their questions about WHY and WHICH preprocessing to do, in plain, short language
(no jargon dumps). Never invent data not in the plan. If asked something unrelated, gently
steer back to preprocessing."""

    payload = [{"role": "system", "content": system}] + list(messages)
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": OLLAMA_MODEL, "messages": payload, "stream": True, "think": False},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    def gen():
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break

    return gen()