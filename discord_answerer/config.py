"""Central configuration. Everything is overridable via environment variables
to keep swaps (embedding model, LLM backend) trivial."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
DATA_DIR = ROOT / "data"


def _load_dotenv() -> None:
    """Load a .env file (KEY=VALUE) from the project root, with no external
    dependency. Variables already set in the environment are NOT overwritten."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# --- Embeddings (local, free) ---
# Trivial swap: "Qwen/Qwen3-Embedding-0.6B" (default) <-> "BAAI/bge-m3".
EMBED_MODEL = os.environ.get("DA_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
# Qwen3-Embedding is instruction-aware: prefix the QUERY only.
QUERY_INSTRUCTION = (
    "Instruct: Given a gaming question, retrieve relevant Discord messages "
    "that answer it.\nQuery: "
)

# --- Chunking (conversation windows, not one embedding per message) ---
CHUNK_MAX_MESSAGES = 12
CHUNK_OVERLAP_MESSAGES = 3
CHUNK_TIME_GAP_MINUTES = 30  # split between two distinct conversations

# --- Search ---
DEFAULT_TOP_K = 30
DEFAULT_SCORE_CUTOFF = 0.35  # trims obvious noise; the real anti-hallucination guard is the LLM

# --- LLM backend (synthesis) ---
LLM_BACKEND = os.environ.get("DA_LLM_BACKEND", "gemini")  # "gemini" | "ollama"
GEMINI_MODEL = os.environ.get("DA_GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("DA_OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# EXACT fallback sentence when the info is not in the Discord.
NOT_FOUND_MESSAGE = "Not found in the Discord."
