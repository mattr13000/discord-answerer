"""Cross-encoder reranker (stage 2 of retrieval): precision pass on the candidate
pool from search.py before it is trimmed to FINAL_K for the LLM.

A bi-encoder (embed.py) scores query and document independently — fast, but blunt.
A cross-encoder reads (query, document) together and scores their relevance
directly, which is far sharper at separating the truly-on-topic chunk from the
hundreds of cross-channel false friends that cosine ranks just as high.

Mirrors embed.py: lazy model load, device auto (CUDA+fp16 / CPU). Full local for
now (BAAI/bge-reranker-v2-m3, multilingual). Never crashes retrieval: if the
model is disabled or unavailable, it falls back to the cosine order untouched.
"""

from . import config

_model = None
_load_failed = False


def _resolve_device() -> str:
    """`DA_RERANK_DEVICE` wins; otherwise auto-pick CUDA if available, else CPU."""
    if config.RERANK_DEVICE:
        return config.RERANK_DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    """Lazy-load the cross-encoder. Returns None (and remembers) on any failure so
    a missing model / no torch degrades to the cosine order instead of crashing."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        # Lazy import: keeps torch/sentence-transformers out of the parse path.
        from sentence_transformers import CrossEncoder

        device = _resolve_device()
        # fp16 on GPU: ~2x less VRAM + faster, no meaningful quality loss for
        # ranking (same trade-off as embed.py). Set at load via model_kwargs —
        # mutating `.model` after load breaks sentence-transformers 5.x's forward
        # dispatch (features get passed positionally as input_ids).
        model_kwargs = {}
        if device == "cuda":
            import torch

            model_kwargs["torch_dtype"] = torch.float16
        _model = CrossEncoder(config.RERANK_MODEL, device=device, model_kwargs=model_kwargs)
    except Exception:  # noqa: BLE001 — any failure -> fallback, never break retrieval
        _load_failed = True
        _model = None
    return _model


def rerank(query, chunks, top_k=None):
    """Reorder `chunks` by cross-encoder relevance to `query`, keep the best `top_k`.

    Each returned item is the ORIGINAL chunk dict (link/text/channel_name/score…
    preserved) with an added `rerank_score`. Pure precision pass — no chunk is
    invented or mutated beyond that key.

    Fallbacks (never raises): empty input -> []; reranker disabled or model
    unavailable -> the first `top_k` chunks in their incoming (cosine) order.
    """
    top_k = config.FINAL_K if top_k is None else top_k
    if not chunks:
        return []
    if not config.RERANK_ENABLED:
        return chunks[:top_k]

    model = _get_model()
    if model is None:
        return chunks[:top_k]

    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda cs: cs[1], reverse=True)
    out = []
    for chunk, score in ranked[:top_k]:
        item = dict(chunk)
        item["rerank_score"] = float(score)
        out.append(item)
    return out
