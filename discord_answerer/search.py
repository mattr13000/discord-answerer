"""Semantic search: encode the query, cosine against the index, return top-k.

Since all vectors are L2-normalized, cosine reduces to a dot product
(embeddings @ qvec).
"""

import numpy as np

from . import config, embed


def search(query, embeddings, chunks_data, k=None, cutoff=None):
    k = k or config.DEFAULT_TOP_K
    cutoff = config.DEFAULT_SCORE_CUTOFF if cutoff is None else cutoff

    qvec = embed.embed_query(query)
    scores = embeddings @ qvec
    order = np.argsort(-scores)[:k]

    results = []
    for idx in order:
        score = float(scores[idx])
        if score < cutoff:
            continue
        item = dict(chunks_data[idx])
        item["score"] = score
        results.append(item)
    return results
