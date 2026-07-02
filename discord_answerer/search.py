"""Hybrid search (stage 1 of retrieval): build an adaptive candidate POOL — not the
final answer set — by fusing a dense semantic ranking with a lexical BM25 ranking.

Stage 1a (dense): encode the query, cosine against the index. Since all vectors are
L2-normalized, cosine reduces to a dot product (embeddings @ qvec).
Stage 1b (lexical): BM25 over the same chunks (lexical.py) — catches exact tokens the
embeddings blur (item/skill/boss names, patch numbers).
Fusion: Reciprocal Rank Fusion (RRF) merges the two rankings by rank, not by
incomparable raw scores. The coarse cosine `cutoff` is applied to the DENSE side only,
so a low-cosine chunk can still be rescued into the pool by BM25.

The pool is then handed to the reranker (precision) before being trimmed to FINAL_K
for the LLM. When `k is None`, the pool size scales with the corpus (config.pool_size)
so the right chunk isn't elbowed out at large scale.
"""

import numpy as np

from . import config, embed


def _dense_order(scores, k, cutoff):
    """Top-`k` document indices by cosine, sorted desc, keeping only score >= cutoff.

    Partial selection (argpartition, O(n)) then sort just those k, instead of fully
    sorting the whole corpus on every query — this is the hot path as the index
    scales to tens of thousands of chunks."""
    k = min(k, len(scores))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    order = top[np.argsort(-scores[top])]
    return [int(i) for i in order if scores[i] >= cutoff]


def _rrf_fuse(rankings, k):
    """Reciprocal Rank Fusion of several ordered index lists -> top-`k` indices.

    fused(d) = sum over rankings of 1 / (RRF_K + rank_d), rank 0-based. Robust to the
    two signals living on different scales: only positions matter."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)
    ordered = sorted(fused, key=lambda i: fused[i], reverse=True)
    return ordered[:k]


def search(query, embeddings, chunks_data, k=None, cutoff=None):
    k = k or config.pool_size(len(chunks_data))
    cutoff = config.DEFAULT_SCORE_CUTOFF if cutoff is None else cutoff

    qvec = embed.embed_query(query)
    scores = embeddings @ qvec

    dense = _dense_order(scores, k, cutoff)

    if config.HYBRID_ENABLED:
        # Lazy import: keeps BM25/regex off the dense-only path.
        from . import lexical

        lexical_order = [idx for idx, _ in lexical.top_n(query, chunks_data, k)]
        order = _rrf_fuse([dense, lexical_order], k)
    else:
        order = dense

    results = []
    for idx in order:
        item = dict(chunks_data[idx])
        item["score"] = float(scores[idx])  # cosine, for display continuity
        item["_row"] = int(idx)  # global row -> lets diversify.py fetch the vector
        results.append(item)
    return results
