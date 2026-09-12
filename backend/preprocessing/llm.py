import json

import requests

import config
from config import OLLAMA_MODEL, OLLAMA_URL, LLM_TIMEOUT
import clean as _clean
import execute as _execute
import advanced as _advanced


# Ollama's chat endpoint (multi-turn), derived from the generate URL.
OLLAMA_CHAT_URL = OLLAMA_URL.replace("/generate", "/chat")

def HINT():
    """Named in the 503 so the message points at the backend actually in use."""
    chain = config.provider_chain()
    tried = " then ".join(chain)
    fix = {"ollama": "is Ollama running?", "huggingface": "check HF_TOKEN and HF_MODEL",
           "gemini": "check GEMINI_API_KEY and GEMINI_MODEL"}[chain[-1]]
    return f"tried {tried} -- {fix}" if len(chain) > 1 else fix


# Hugging Face and Gemini both speak OpenAI's chat-completions shape, so they
# share one request path; only (url, key, model) differ. Ollama is the odd one out.
def _openai_target(provider):
    if provider == "gemini":
        return config.GEMINI_URL, config.GEMINI_API_KEY, config.GEMINI_MODEL
    return config.HF_URL, config.HF_TOKEN, config.HF_MODEL


def _first_working(call):
    """Run `call(provider)` down the configured chain, returning the first success.
    Raises with every failure named, so a 503 says what was actually tried."""
    failures = []
    for provider in config.provider_chain():
        try:
            return call(provider)
        except Exception as exc:                      # noqa: BLE001 -- any transport/HTTP error
            failures.append(f"{provider}: {type(exc).__name__} {str(exc)[:120]}")
    raise RuntimeError("; ".join(failures) or "no LLM backend configured")


def _complete_one(provider, prompt, json_mode):
    if provider == "ollama":
        body = {"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False}
        if json_mode:
            body["format"] = "json"
        r = requests.post(config.OLLAMA_URL, json=body, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["response"]

    url, key, model = _openai_target(provider)
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _complete(prompt, json_mode=False):
    """One-shot prompt -> text, from the first backend in the chain that answers."""
    return _first_working(lambda p: _complete_one(p, prompt, json_mode))


def _parse_chunk(line):
    """(text, done) from one streamed line. Handles BOTH wire formats -- Ollama
    sends bare NDJSON, the OpenAI-compatible APIs send SSE ('data: {...}',
    'data: [DONE]') -- so the generator below stays provider-agnostic."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    line = line.strip()
    if not line:
        return "", False
    if line.startswith("data:"):
        line = line[5:].strip()
        if line == "[DONE]":
            return "", True
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return "", False
    if "choices" in d:                                    # OpenAI / Hugging Face / Gemini
        c = (d.get("choices") or [{}])[0]
        piece = (c.get("delta") or c.get("message") or {}).get("content") or ""
        return piece, c.get("finish_reason") is not None
    return d.get("message", {}).get("content", ""), bool(d.get("done"))   # Ollama


def _stream_one(provider, messages):
    if provider == "ollama":
        resp = requests.post(OLLAMA_CHAT_URL, stream=True, timeout=LLM_TIMEOUT,
                             json={"model": config.OLLAMA_MODEL, "messages": messages,
                                   "stream": True, "think": False})
    else:
        url, key, model = _openai_target(provider)
        resp = requests.post(url, headers={"Authorization": f"Bearer {key}"},
                             stream=True, timeout=LLM_TIMEOUT,
                             json={"model": model, "messages": messages, "stream": True})
    resp.raise_for_status()
    return resp


def _open_stream(messages):
    """POST a chat request with streaming on. Connects HERE, before any bytes are
    yielded, so a dead backend can still fall through to the next one."""
    return _first_working(lambda p: _stream_one(p, messages))


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

    return _complete(prompt)


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

    return _complete(prompt).strip()


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

    text = _complete(_PLAN_SYSTEM + "\n\n" + user, json_mode=True)
    return _validate_plan(_extract_json(text))


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
    resp = _open_stream(payload)

    def gen():
        for line in resp.iter_lines():
            piece, done = _parse_chunk(line)
            if piece:
                yield piece
            if done:
                break

    return gen()

if __name__ == "__main__":
    # Stream parsing is the only branchy bit -- both wire formats, no network.
    ollama = [b'{"message":{"content":"Hel"}}', b'{"message":{"content":"lo"}}',
              b'{"message":{"content":""},"done":true}']
    hf = [b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
          b'data: {"choices":[{"delta":{"content":"lo"}}]}',
          b'', b'data: [DONE]']
    for name, lines in (("ollama", ollama), ("huggingface", hf)):
        out, stopped = "", False
        for ln in lines:
            piece, done = _parse_chunk(ln)
            out += piece
            if done:
                stopped = True
                break
        assert out == "Hello", (name, out)
        assert stopped, f"{name} stream never signalled done"
    assert _parse_chunk(b"not json") == ("", False)      # junk must not kill the stream
    assert _parse_chunk(b"") == ("", False)
    assert _extract_json('prose {"a": 1} tail') == {"a": 1}
    assert _extract_json("no json here") == {}
    # the chain must try every configured backend before giving up, and say so
    seen = []
    try:
        _first_working(lambda p: seen.append(p) or (_ for _ in ()).throw(RuntimeError("boom")))
    except RuntimeError as e:
        assert seen == config.provider_chain(), (seen, config.provider_chain())
        assert all(p in str(e) for p in seen), e          # every failure named
    assert _first_working(lambda p: f"ok:{p}") == f"ok:{config.provider_chain()[0]}"
    assert isinstance(HINT(), str) and HINT()
    print(f"llm self-check passed (chain={config.provider_chain()}, both stream formats parse)")
