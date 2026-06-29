"""Local multilingual embeddings (free, runs on the GPU).

Default model: Qwen/Qwen3-Embedding-0.6B (strong multilingual, incl. Korean).
Swap to BAAI/bge-m3 by changing `DA_EMBED_MODEL` — nothing else to touch.
Vectors are L2-normalized -> cosine = plain dot product.
"""

import numpy as np

from . import config

_model = None


def _resolve_device() -> str:
    """`DA_EMBED_DEVICE` wins; otherwise auto-pick CUDA if available, else CPU."""
    if config.EMBED_DEVICE:
        return config.EMBED_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    global _model
    if _model is None:
        # Lazy import: avoids loading torch as long as we only parse.
        from sentence_transformers import SentenceTransformer

        device = _resolve_device()
        _model = SentenceTransformer(config.EMBED_MODEL, device=device)
        # fp16 on GPU: ~2x less VRAM (leaves headroom for other GPU work) and faster,
        # with no meaningful quality loss for retrieval. Skip on CPU (no fp16 speedup).
        # NB: post-load `.half()` is safe for SentenceTransformer (bi-encoder), but
        # NOT for the CrossEncoder in rerank.py — there it breaks the forward
        # dispatch under sentence-transformers 5.x, so that path sets fp16 via
        # model_kwargs at load instead. Keep the two in sync if you refactor.
        if device == "cuda":
            _model = _model.half()
    return _model


def _is_instruction_aware() -> bool:
    return "qwen3-embedding" in config.EMBED_MODEL.lower()


def embed_documents(texts: list[str]) -> np.ndarray:
    model = _get_model()
    emb = model.encode(
        texts,
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 256,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    if _is_instruction_aware():
        text = config.QUERY_INSTRUCTION + text
    emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)[0]
