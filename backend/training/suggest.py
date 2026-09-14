"""Model suggestions for a dataset (spec 2026-09-15). The model registry decides
what CAN train; the AI only ranks among those; code validates. Same inputs always
give the same list:

  1. cache hit (same data fingerprint + profile + compatible models + AI identity)
     -> the cached AI list, source "ai_cached"
  2. otherwise ask the AI (temperature 0); keep allowed keys in its order, drop
     duplicates, cap at MAX_SUGGESTIONS; >= 1 left -> source "ai", cached
  3. AI unreachable / error / nothing usable -> the registry rules (models.suggest),
     source "rules", with the reason; failures are not cached, so the AI is retried

The AI never receives data rows -- only measured facts and the allowed models.
"""

import hashlib
import json

from assistant import llm
from core import store
from training import train

MAX_SUGGESTIONS = 5  # our starting point
_ask = llm.suggest_models  # replaced by tests with a fake AI


def _allowed(opt):
    return [m for m in opt["models"] if m["compatible"] and not m["baseline"]]


def _profile(opt):
    return {k: v for k, v in opt["facts"].items() if k != "prep_seconds"}  # timing varies run to run


def cache_key(opt, identity):
    key = {"fingerprint": opt["fingerprint"], "task": opt["task"], "target": opt["target"],
           "profile": _profile(opt), "allowed": [m["key"] for m in _allowed(opt)], "ai": identity}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]


def validate(raw, allowed):
    """The AI's reply -> [{key, why}] using only allowed keys, first mention wins."""
    items = raw.get("models") if isinstance(raw, dict) else None
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        key = item.get("key") if isinstance(item, dict) else None
        if key in allowed and key not in seen:
            seen.add(key)
            out.append({"key": key, "why": str(item.get("reason") or "").strip()[:200] or None})
        if len(out) == MAX_SUGGESTIONS:
            break
    return out


def suggest(ds_id):
    opt = train.options(ds_id)
    allowed = _allowed(opt)
    path = store.suggestion_path(ds_id, cache_key(opt, llm.identity()))
    cached = store.read_json(path)
    if cached:
        return {**cached, "source": "ai_cached"}
    try:
        facts = {"task": opt["task"], "target": opt["target"], **_profile(opt)}
        models = [{"key": m["key"], "label": m["label"], "what": m["about"]["what"],
                   "good_for": m["about"]["good"], "watch_out": m["about"]["watch"]} for m in allowed]
        picks = validate(_ask({"facts": facts, "models": models, "max": MAX_SUGGESTIONS}),
                         {m["key"] for m in allowed})
        reason = None if picks else "AI returned no model from the allowed list"
    except Exception as exc:  # noqa: BLE001 -- any AI failure falls back to the rules
        picks, reason = [], f"AI unavailable: {type(exc).__name__}: {str(exc)[:160]}"
    if picks:
        result = {"source": "ai", "reason": None, "models": picks}
        store.write_json(path, result)
        return result
    return {"source": "rules", "reason": reason,
            "models": [{"key": m["key"], "why": m["why"]} for m in opt["models"] if m["suggested"]]}


if __name__ == "__main__":
    import tempfile
    import warnings
    from pathlib import Path
    import pandas as pd
    from preprocessing import execute
    warnings.filterwarnings("ignore")
    store.ROOT = Path(tempfile.mkdtemp())
    n = 120
    df = pd.DataFrame({"msg": [f"{'win prize' if i % 4 == 0 else 'see you'} {i}" for i in range(n)],
                       "price": [float(i % 13) for i in range(n)], "y": ["spam" if i % 4 == 0 else "ham" for i in range(n)]})
    ds = store.create(df, "t.csv")
    r = execute.apply_plan(df, "y", {"msg": [{"op": "encode", "method": "text"}]})
    r["fitted"]["clean_ops"] = store.clean_ops(ds)
    store.save_prep(ds, r["csv"], r["fitted"])
    calls = []

    def fake(reply):
        def ask(context):
            calls.append(context)
            assert "rows" in context["facts"] and all("key" in m for m in context["models"])
            assert not any(isinstance(v, list) and len(v) > 50 for v in context["facts"].values()), "no data rows"
            if isinstance(reply, Exception):
                raise reply
            return reply
        return ask

    def fresh():
        for f in (store.ROOT / ds / "suggestions").glob("*.json"):
            f.unlink()
        calls.clear()

    rules = [m["key"] for m in train.options(ds)["models"] if m["suggested"]]

    # unknown, incompatible (gaussian_nb: text data) and duplicate keys are dropped; order kept; cap 5
    _ask = fake({"models": [{"key": "lightgbm", "reason": "fast on wide text"}, {"key": "nope"},
                            {"key": "gaussian_nb", "reason": "x"}, {"key": "lightgbm", "reason": "dup"},
                            {"key": "logistic_regression", "reason": "strong linear text baseline"},
                            {"key": "random_forest"}, {"key": "svm"}, {"key": "mlp"}, {"key": "xgboost"}]})
    first = suggest(ds)
    assert first["source"] == "ai" and [m["key"] for m in first["models"]] == \
        ["lightgbm", "logistic_regression", "random_forest", "svm", "mlp"], first
    assert first["models"][0]["why"] == "fast on wide text" and len(calls) == 1

    # same inputs -> cached list, the AI is not asked again
    _ask = fake(RuntimeError("must not be called"))
    again = suggest(ds)
    assert again["source"] == "ai_cached" and again["models"] == first["models"] and len(calls) == 1

    # AI down -> deterministic rule list with the reason; failure not cached (AI retried next time)
    fresh()
    _ask = fake(ConnectionError("ollama not running"))
    down = suggest(ds)
    assert down["source"] == "rules" and [m["key"] for m in down["models"]] == rules and "ollama not running" in down["reason"]
    assert suggest(ds)["source"] == "rules" and len(calls) == 2 and not list((store.ROOT / ds / "suggestions").glob("*.json"))

    # malformed or empty replies -> rules
    for reply in ({}, {"models": "lightgbm"}, {"models": [{"key": "gaussian_nb"}]}, []):
        fresh()
        _ask = fake(reply)
        res = suggest(ds)
        assert res["source"] == "rules" and res["reason"] == "AI returned no model from the allowed list", (reply, res)

    # a different AI model or changed data gives a different cache key
    opt = train.options(ds)
    assert cache_key(opt, "ollama:a") != cache_key(opt, "ollama:b")
    assert cache_key(opt, "x") != cache_key({**opt, "fingerprint": "different"}, "x")
    assert cache_key(opt, "x") != cache_key({**opt, "target": "other"}, "x"), "a new target must ask the AI again"
    assert cache_key(opt, "x") == cache_key(train.options(ds), "x"), "the key must not change between calls"
    print("suggest self-check passed")
