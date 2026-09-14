import json
import re

import requests

from assistant import results as _results
from core import config
from core.config import OLLAMA_MODEL, OLLAMA_URL, LLM_TIMEOUT
from preprocessing import advanced as _advanced, clean as _clean, execute as _execute


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


class PromptTooLong(RuntimeError):
    """Ollama refused a request larger than its context window (we send truncate=false),
    instead of silently dropping the start of the prompt."""

    def __init__(self, tokens, limit):
        super().__init__(f"AI request too long: {tokens} tokens, limit {limit} "
                         f"(raise OLLAMA_NUM_CTX in backend/.env or send less)")
        self.tokens, self.limit = tokens, limit


def _ollama_post(url, body, **kwargs):
    """POST to Ollama with our context size and truncation off (verified on Ollama
    0.32.15: an oversized request then returns HTTP 400 exceed_context_size_error)."""
    body = {**body, "truncate": False, "options": {"num_ctx": config.OLLAMA_NUM_CTX, **body.get("options", {})}}
    r = requests.post(url, json=body, timeout=LLM_TIMEOUT, **kwargs)
    if r.status_code == 400 and "exceed" in r.text:
        found = re.search(r"request \((\d+) tokens\)", r.text)
        raise PromptTooLong(int(found.group(1)) if found else None, config.OLLAMA_NUM_CTX)
    r.raise_for_status()
    return r


def _complete_one(provider, prompt, json_mode, temperature=None):
    if provider == "ollama":
        body = {"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "think": False}
        if json_mode:
            body["format"] = "json"
        if temperature is not None:
            body["options"] = {"temperature": temperature}
        return _ollama_post(config.OLLAMA_URL, body).json()["response"]

    url, key, model = _openai_target(provider)
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if temperature is not None:
        body["temperature"] = temperature
    r = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=LLM_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _complete(prompt, json_mode=False, temperature=None):
    """One-shot prompt -> text, from the first backend in the chain that answers."""
    return _first_working(lambda p: _complete_one(p, prompt, json_mode, temperature))


def identity():
    """Which AI would answer -- part of the suggestion cache key, so switching
    provider or model never reuses another model's suggestions."""
    names = {"ollama": config.OLLAMA_MODEL, "huggingface": config.HF_MODEL, "gemini": config.GEMINI_MODEL}
    return "|".join(f"{p}:{names[p]}" for p in config.provider_chain())


def suggest_models(context):
    """Ask the AI to rank models for this dataset. It sees measured facts and the
    list of models that can train on the data -- never data rows -- and may only
    pick from that list (training/suggest.py validates). Temperature 0 makes
    answers steadier; the cache is what makes them repeatable."""
    prompt = f"""You help a researcher choose machine-learning models for a tabular dataset.
Pick up to {context["max"]} models from the ALLOWED list that are most likely to do well on this data,
best first, each with one short reason tied to the facts below. Use only keys from the ALLOWED list.
Reply with ONLY JSON: {{"models": [{{"key": "<allowed key>", "reason": "<one sentence>"}}]}}

Dataset facts (measured on the preprocessed training rows):
{json.dumps(context["facts"], indent=1)}

ALLOWED models:
{json.dumps(context["models"], indent=1)}"""
    return _extract_json(_complete(prompt, json_mode=True, temperature=0))


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
        msgs = list(messages)
        while True:
            try:
                return _ollama_post(OLLAMA_CHAT_URL, {"model": config.OLLAMA_MODEL, "messages": msgs,
                                                      "stream": True, "think": False}, stream=True)
            except PromptTooLong:
                # A long conversation drops its OLDEST turns -- never the system message
                # (rules + facts) and never the newest question.
                if len(msgs) <= 2:
                    raise
                del msgs[1]
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
  {"op":"encode","method":"onehot"|"ordinal"|"text"}           # text = kept raw, word+letter tf-idf fit at training
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
- encode free-text and many-valued text columns (reviews, notes, cities, product names) with
  method "text". Do NOT drop a column just because it is text or has many values.
- scale numeric ONLY when useful; leave it off otherwise.
- To DROP any unneeded column (ID / constant / mostly-empty),
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
    ({target, task, rows, columns:[{name,type,missing_percent,action}], dataset_facts}) --
    aggregated metadata only, never raw rows. `dataset_facts` = measured profile facts
    for every column (type, unique values, missing %, constant, ID-like, top values
    or statistics). `results_view` (optional) = the part of the training run on the page
    this question needs (assistant/results.py). Returns a generator of text chunks; the
    initial connection is made here so a dead Ollama raises before streaming starts.
    When the reply ends, any number in it that was not in what the AI was sent is
    listed in a final "Not verified" line."""
    results_view = context.get("results_view")
    results_part = "" if not results_view else f"""

Training results currently on the page (numbers computed by the app):
{json.dumps(results_view, indent=1, default=str)}

When asked about models or results, use ONLY these numbers and name the metric you use.
"train" = score on the model's own training rows; "cv_mean"/"cv_sd" = average and spread over held-out
folds; "test" = the locked test set, never used to choose the model. Train far better than CV means the
model memorised its rows (overfitting); CV close to test means the score holds on unseen rows. "Best" means
the best CV mean on the main metric -- not proof it is best in real use. A number you need that is not
listed is "not available" -- say so. Say clearly when advice on what to try next is only a suggestion."""
    system = f"""You are an honest assistant for NON-TECHNICAL users, helping with data preprocessing and training results.

Dataset: {context.get('rows')} rows. Measured facts for every column (no data rows are shared):
{json.dumps(context.get('dataset_facts'), indent=1, default=str)}

The user's CURRENT selection is "{context.get('target')}" as the column to predict ({context.get('task')}).
That is only their current choice -- it may be a mistake. Do not defend it because it is selected.

Judging a target (use the facts above and what the column names mean):
- Unusable: constant, looks like an ID, almost unique per row (free text), mostly missing,
  or a class with too few rows to learn from.
- Usable but not natural: a column that describes the rows rather than an outcome people
  usually want to know, when another column in the table looks like that outcome.
  Say which column looks like the more natural target and why, but add that the right target
  depends on the user's goal, which only they know.
- Give the same answer to "which column should I predict?" whatever is currently selected.

Recommended plan for the current selection -- each feature column with its type, missing %, and action:
{json.dumps(context.get('columns', []), indent=2)}

Answer in plain, short language (no jargon dumps). Never invent data not in the facts, the plan
or the results. If asked something unrelated, gently steer back to preprocessing and training.{results_part}"""

    payload = [{"role": "system", "content": system}] + list(messages)
    resp = _open_stream(payload)

    def gen():
        answer = ""
        for line in resp.iter_lines():
            piece, done = _parse_chunk(line)
            if piece:
                answer += piece
                yield piece
            if done:
                break
        missing = _results.unverified(answer, payload)
        if missing:
            yield ("\n\n⚠️ Not verified: " + ", ".join(missing) +
                   " — not found in the data the AI was given. Check these before trusting them.")

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

    # context window: our size + truncation off on every Ollama request; an oversized
    # request is a clear error, and a long chat drops its oldest turns only
    class _Resp:
        def __init__(self, status, text):
            self.status_code, self.text = status, text

        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "ok"}

    sent = []

    def fake_post(url, json=None, **kw):
        sent.append(json)
        size = len(json.get("messages") or [json.get("prompt")])
        if size > 3:  # pretend anything over 3 messages exceeds the window
            return _Resp(400, '{"error":{"message":"request (9999 tokens) exceeds the available context size",'
                              '"type":"exceed_context_size_error"}}')
        return _Resp(200, "")

    real_post, requests.post = requests.post, fake_post
    try:
        assert _complete_one("ollama", "short", False) == "ok"
        assert sent[-1]["truncate"] is False and sent[-1]["options"]["num_ctx"] == config.OLLAMA_NUM_CTX, sent[-1]
        assert _complete_one("ollama", "t", False, temperature=0) == "ok"
        assert sent[-1]["options"] == {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0}, sent[-1]
        history = [{"role": "system", "content": "rules"}] + [{"role": "user", "content": f"q{i}"} for i in range(6)]
        _stream_one("ollama", history)
        kept = sent[-1]["messages"]
        assert kept[0]["content"] == "rules" and kept[-1]["content"] == "q5" and len(kept) == 3, kept
        requests.post = lambda url, json=None, **kw: _Resp(400, '{"error":"request (9999 tokens) exceeds ..."}')
        try:
            _complete_one("ollama", "huge", False)
            raise AssertionError("an oversized prompt must raise, not be cut")
        except PromptTooLong as e:
            assert e.tokens == 9999 and "OLLAMA_NUM_CTX" in str(e), e
    finally:
        requests.post = real_post
    print(f"llm self-check passed (chain={config.provider_chain()}, both stream formats parse)")
