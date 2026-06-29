"""Semantic search (stage 1 of retrieval): encode the query, cosine against the
index, return an adaptive candidate POOL — not the final answer set.

Since all vectors are L2-normalized, cosine reduces to a dot product
(embeddings @ qvec). The pool is then handed to the reranker (precision) before
being trimmed to FINAL_K for the LLM. When `k is None`, the pool size scales with
the corpus (config.pool_size) so the right chunk isn't elbowed out at large scale.
"""

import numpy as np

from . import config, embed


def search(query, embeddings, chunks_data, k=None, cutoff=None):
    k = k or config.pool_size(len(chunks_data))
    cutoff = config.DEFAULT_SCORE_CUTOFF if cutoff is None else cutoff

    qvec = embed.embed_query(query)
    scores = embeddings @ qvec

    # Top-k via partial selection (O(n)) then sort only those k, instead of fully
    # sorting the whole corpus on every query — this is the hot path as the index
    # scales to tens of thousands of chunks.
    k = min(k, len(scores))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    order = top[np.argsort(-scores[top])]

    results = []
    for idx in order:
        score = float(scores[idx])
        if score < cutoff:
            continue
        item = dict(chunks_data[idx])
        item["score"] = score
        results.append(item)
    return results
