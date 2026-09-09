"""Runtime configuration. Nothing here is hard-coded to one dataset."""

import os
import sys

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")  # match the model actually pulled in Ollama


def dataset_path():
    """The dataset to work on: CLI argument first, then $DATASET_PATH."""
    path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATASET_PATH")
    if not path:
        raise SystemExit(
            "No dataset given. Pass a path as an argument or set DATASET_PATH."
        )
    return path