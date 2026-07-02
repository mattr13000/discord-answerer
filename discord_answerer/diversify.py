"""Result diversification (stage 2b): pick the FINAL_K answer set from the reranked
pool with Maximal Marginal Relevance (MMR) and a per-channel quota.

The reranker gives a sharp relevance order, but the top of it is often several
near-duplicate chunks of the *same* conversation (overlapping windows) and, across a
multi-channel server, a chatty channel can crowd out a better answer sitting in a
quieter one. Both waste the tight FINAL_K context.

MMR re-selects greedily, trading relevance against novelty:
    mmr(c) = lambda * relevance(c) - (1 - lambda) * max_similarity(c, already_picked)
so each pick is relevant *and* adds something new. A per-channel cap then bounds how
much of the answer any single channel may take — relaxed automatically when the pool
can't otherwise fill k (e.g. a single-channel scope, where the cap is moot).

Degrades safely: similarity needs each item's embedding (via its `_row` from
search.py); if that's missing the similarity term is 0 (relevance + quota still
apply). Disabled or k<=1 -> the reranked order untouched.
"""

import math

import numpy as np

from . import config


def _relevances(ranked) -> np.ndarray:
    """Relevance term of MMR, one value per item, roughly in [0, 1].

    When the reranker ran, its `rerank_score` is the sharp signal — use it. On the
    cosine-fallback path (reranker disabled/unavailable) there is NO per-item score
    that matches the incoming order: in hybrid mode that order is RRF-fused, and the
    raw cosine `score` would demote exactly the low-cosine chunks BM25 rescued into
    the pool — silently undoing the hybrid pass. So there we trust the incoming
    *order* itself and use a positional relevance (1 at the top, ->0 at the bottom),
    which keeps MMR/quota active without re-ranking by a stale signal."""
    if "rerank_score" in ranked[0]:
        return np.array([float(it["rerank_score"]) for it in ranked], dtype=np.float32)
    n = len(ranked)
    return np.array([1.0 - i / n for i in range(n)], dtype=np.float32)


def select(ranked, embeddings, k):
    """Pick `k` items from `ranked` (sorted by relevance desc) via MMR + channel quota."""
    if not ranked:
        return []
    k = min(k, len(ranked))
    if not config.DIVERSITY_ENABLED or k <= 1:
        return ranked[:k]

    lam = config.MMR_LAMBDA
    rel = _relevances(ranked)

    # Candidate vectors (L2-normalized -> dot product == cosine similarity). Only
    # available when every item carries its global row index from search.py.
    rows = [it.get("_row") for it in ranked]
    vecs = None
    if embeddings is not None and all(r is not None for r in rows):
        vecs = embeddings[rows]

    cap = max(1, math.ceil(k * config.PER_CHANNEL_FRACTION))
    chan_count: dict = {}
    selected: list = []
    picked_pos: list[int] = []
    remaining = list(range(len(ranked)))

    def _best(respect_cap: bool):
        """Index (into `ranked`) of the best remaining candidate by MMR, or None if
        `respect_cap` filtered everything out."""
        best_i, best_score = None, -math.inf
        for i in remaining:
            if respect_cap and chan_count.get(ranked[i].get("channel_name"), 0) >= cap:
                continue
            if vecs is not None and picked_pos:
                sim = float(np.max(vecs[picked_pos] @ vecs[i]))
            else:
                sim = 0.0
            mmr = lam * rel[i] - (1.0 - lam) * sim
            if mmr > best_score:
                best_i, best_score = i, mmr
        return best_i

    while len(selected) < k and remaining:
        # Respect the per-channel cap first; if it blocks every remaining candidate
        # (low-diversity pool / single channel), relax it to keep filling k.
        choice = _best(respect_cap=True)
        if choice is None:
            choice = _best(respect_cap=False)
        selected.append(ranked[choice])
        picked_pos.append(choice)
        remaining.remove(choice)
        ch = ranked[choice].get("channel_name")
        chan_count[ch] = chan_count.get(ch, 0) + 1

    return selected
