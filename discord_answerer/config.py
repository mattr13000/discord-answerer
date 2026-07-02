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
# Device for embedding: "" = auto (CUDA if available, else CPU). Force with
# DA_EMBED_DEVICE=cuda / cpu. GPU is strongly recommended past ~10k messages.
EMBED_DEVICE = os.environ.get("DA_EMBED_DEVICE", "").strip()
# Encode batch size. 64 fits comfortably alongside other GPU work on an 8 GB card
# (model runs in fp16 on CUDA); lower it (e.g. 8-16) if you game while indexing.
EMBED_BATCH_SIZE = int(os.environ.get("DA_EMBED_BATCH_SIZE", "64"))
# Qwen3-Embedding is instruction-aware: prefix the QUERY only.
QUERY_INSTRUCTION = (
    "Instruct: Given a gaming question, retrieve relevant Discord messages "
    "that answer it.\nQuery: "
)

# --- Chunking (conversation windows, not one embedding per message) ---
CHUNK_MAX_MESSAGES = 12
CHUNK_OVERLAP_MESSAGES = 3
CHUNK_TIME_GAP_MINUTES = 30  # split between two distinct conversations

# --- Retrieval (staged: adaptive cosine pool -> rerank -> constant final_k) ---
# Stage 1 — candidate POOL (recall). Its size scales with the corpus so the right
# chunk isn't elbowed out of a fixed top-k by hundreds of cross-channel false
# friends. pool_size() = clamp(POOL_MIN, n_chunks * POOL_FRACTION, POOL_MAX).
POOL_FRACTION = float(os.environ.get("DA_POOL_FRACTION", "0.05"))
POOL_MIN = int(os.environ.get("DA_POOL_MIN", "100"))
POOL_MAX = int(os.environ.get("DA_POOL_MAX", "2000"))
# Coarse pre-filter applied on the pool (NOT the fine ranking — the reranker does
# that). Kept loose: the real anti-hallucination guard is the LLM. In hybrid mode
# this filters the DENSE side only — a low-cosine chunk can still be rescued by BM25.
DEFAULT_SCORE_CUTOFF = float(os.environ.get("DA_SCORE_CUTOFF", "0.35"))

# Stage 1b — lexical BM25 signal fused with the dense pool (recall). Dense
# embeddings miss exact tokens (item/skill/boss names, patch numbers like "+9 STR")
# that dominate a game Discord; BM25 catches them. The two rankings are combined by
# Reciprocal Rank Fusion (RRF): fused(d) = sum_r 1 / (RRF_K + rank_r(d)). No new
# dependency — BM25 is implemented in-repo (lexical.py), numpy-only.
HYBRID_ENABLED = os.environ.get("DA_HYBRID", "1").strip().lower() not in ("0", "false", "no", "")
RRF_K = int(os.environ.get("DA_RRF_K", "60"))  # RRF damping; larger = flatter rank weighting
BM25_K1 = float(os.environ.get("DA_BM25_K1", "1.5"))  # term-frequency saturation
BM25_B = float(os.environ.get("DA_BM25_B", "0.75"))  # document-length normalization

# Stage 3 — what Gemini actually sees (precision). CONSTANT: does not scale with
# the corpus, so the context stays tight (no "lost in the middle", strong lock).
FINAL_K = int(os.environ.get("DA_FINAL_K", "12"))

# Stage 2 — local cross-encoder reranker (precision). Reorders the pool before
# trimming to FINAL_K. Full local for now (BAAI/bge-reranker-v2-m3, ~568M,
# multilingual); device auto = CUDA+fp16 / CPU, mirroring embed.py.
RERANK_ENABLED = os.environ.get("DA_RERANK", "1").strip().lower() not in ("0", "false", "no", "")
RERANK_MODEL = os.environ.get("DA_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.environ.get("DA_RERANK_DEVICE", "").strip()
# Junk-pool floor: if the BEST chunk's rerank_score is below this, the whole pool is
# treated as off-topic and retrieval returns [] -> exact NOT_FOUND, no LLM call. The
# measured separation is wide (in-scope ~0.95, off-topic ~0.01), so 0.05 is a
# conservative cut that strengthens the lock without risking in-scope answers. Only
# applied when the reranker actually ran (cosine-fallback path keeps current behaviour).
# Set DA_RERANK_FLOOR=0 to disable.
RERANK_FLOOR = float(os.environ.get("DA_RERANK_FLOOR", "0.05"))

# Stage 2b — result diversification (precision/coverage). Picks the FINAL_K answer set
# from the reranked pool with (a) Maximal Marginal Relevance to drop near-duplicate
# chunks and (b) a per-channel quota so one chatty channel can't monopolize the answer.
DIVERSITY_ENABLED = os.environ.get("DA_DIVERSITY", "1").strip().lower() not in ("0", "false", "no", "")
# MMR trade-off: score = MMR_LAMBDA*relevance - (1-MMR_LAMBDA)*max_similarity_to_picked.
# 1.0 = pure relevance (no diversity), 0.0 = pure diversity. 0.7 leans on relevance.
MMR_LAMBDA = float(os.environ.get("DA_MMR_LAMBDA", "0.7"))
# Max share of the FINAL_K answer one channel may take (cap = ceil(k * fraction)).
# Relaxed automatically when the pool can't otherwise fill k (e.g. single-channel scope).
PER_CHANNEL_FRACTION = float(os.environ.get("DA_CHANNEL_FRACTION", "0.5"))


def pool_size(n_chunks: int) -> int:
    """Adaptive candidate-pool size: clamp(POOL_MIN, n_chunks*POOL_FRACTION, POOL_MAX)."""
    target = int(n_chunks * POOL_FRACTION)
    return max(POOL_MIN, min(target, POOL_MAX))

# --- LLM backend (synthesis) ---
LLM_BACKEND = os.environ.get("DA_LLM_BACKEND", "gemini")  # "gemini" | "ollama"
# Default to 3.1-flash-lite: free tier with ~25x higher rate limits than 2.5-flash.
# Override with DA_GEMINI_MODEL (e.g. "gemini-2.5-flash", "gemini-3.1-flash-lite-preview").
GEMINI_MODEL = os.environ.get("DA_GEMINI_MODEL", "gemini-3.1-flash-lite")
OLLAMA_MODEL = os.environ.get("DA_OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# EXACT fallback sentence when the info is not in the Discord.
NOT_FOUND_MESSAGE = "Not found in the Discord."
