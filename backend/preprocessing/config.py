"""Deployment settings -- things that differ per MACHINE, never per analysis.

Analysis parameters (split ratio, seed, outlier fences, detector thresholds)
deliberately do NOT live here: they change the result, so they travel with the
request and get recorded in the run summary. A hidden env var that silently
alters output would break the reproducibility the project is built on.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Which LLM backend the AI tabs talk to: "ollama" (a model on this machine),
# "huggingface" or "gemini" (hosted APIs -- these send your column names and
# sample rows to a third party).
#
# A credential is unambiguous intent, so HF_TOKEN / GEMINI_API_KEY select their
# provider on their own; an explicit LLM_PROVIDER still wins. Without that,
# filling in the hosted settings and commenting out the Ollama ones silently
# kept talking to localhost:11434.
HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_DEFAULT_PROVIDER = ("huggingface" if HF_TOKEN else "gemini" if GEMINI_API_KEY else "ollama")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 500))

# When the chosen backend errors or is unreachable, fall through to the other
# CONFIGURED ones instead of failing. Off by default: a fallback can move your
# data from a local model to a hosted one, which must be a decision, not an accident.
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")  # match the model actually pulled in Ollama

HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_URL = os.getenv("HF_URL", "https://router.huggingface.co/v1/chat/completions")

# Gemini speaks OpenAI's chat-completions shape at this path, so it shares the
# whole HF code path -- only the URL, key and model differ.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = os.getenv("GEMINI_URL",
                       "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")

# Fallback order: the hosted ones first, then Ollama -- it is the only provider
# that always looks "configured", so it should not shadow a real credential.
PROVIDERS = ("huggingface", "gemini", "ollama")
_CREDENTIAL = {"huggingface": ("HF_TOKEN", HF_TOKEN, "https://huggingface.co/settings/tokens"),
               "gemini": ("GEMINI_API_KEY", GEMINI_API_KEY, "https://aistudio.google.com/apikey")}

if LLM_PROVIDER not in PROVIDERS:
    raise SystemExit(f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)} (got '{LLM_PROVIDER}').")
if LLM_PROVIDER in _CREDENTIAL and not _CREDENTIAL[LLM_PROVIDER][1]:
    name, _, where = _CREDENTIAL[LLM_PROVIDER]
    raise SystemExit(f"LLM_PROVIDER={LLM_PROVIDER} needs {name} in backend/.env (get one at {where}).")


def configured(provider):
    """True if this provider has what it needs. Ollama needs no credential --
    whether it is actually running is only discoverable by calling it."""
    return bool(_CREDENTIAL[provider][1]) if provider in _CREDENTIAL else True


def provider_chain():
    """Providers to try, best first. Just the chosen one unless LLM_FALLBACK is on."""
    if not LLM_FALLBACK:
        return [LLM_PROVIDER]
    ordered = [LLM_PROVIDER] + [p for p in PROVIDERS if p != LLM_PROVIDER]
    return [p for p in ordered if configured(p)]


CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 200))


def dataset_path():
    """The dataset to work on: CLI argument first, then $DATASET_PATH."""
    path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATASET_PATH")
    if not path:
        raise SystemExit(
            "No dataset given. Pass a path as an argument or set DATASET_PATH."
        )
    return path
