"""High-level query pipeline — the single headless entry point that wires the
retrieval stages so callers (Streamlit UI, CLI, tests) never re-implement the flow:

    search.search          stage 1:  dense cosine + BM25 -> RRF pool   (recall)
      -> rerank.rerank      stage 2:  cross-encoder rerank             (precision)
        -> [floor]          stage 2a: reject junk pools before the LLM
          -> diversify      stage 2b: MMR + per-channel quota -> FINAL_K
            -> synthesize    stage 3:  bounded LLM over FINAL_K         (answer)

`retrieve` returns the ranked chunks; `ask` runs the whole thing against a saved
index and also synthesizes the answer. The Streamlit app wraps these in its own
caching layer — the library itself stays free of any UI dependency.
"""

from . import config, diversify, index_build, rerank, search, synthesize


def _below_floor(ranked) -> bool:
    """True if the reranker ran and even the best chunk falls under RERANK_FLOOR.

    `ranked` is sorted by rerank_score desc, so ranked[0] is the max. We only gate on
    `rerank_score` (the cosine-fallback path has none and keeps its old behaviour), so
    an off-topic pool that the cross-encoder confidently scores ~0.01 short-circuits to
    [] -> exact NOT_FOUND, with no Gemini call and no weakening of in-scope answers."""
    if config.RERANK_FLOOR <= 0 or not ranked:
        return False
    top = ranked[0]
    return "rerank_score" in top and top["rerank_score"] < config.RERANK_FLOOR


def retrieve(question, embeddings, chunks_data, *, k=None, cutoff=None):
    """Stages 1+2+2a+2b: hybrid candidate pool, cross-encoder rerank, junk-pool floor,
    then MMR + per-channel quota down to the best `k` chunks (config.FINAL_K when None).

    Returns chunk dicts carrying `score` (cosine) and, when the reranker is active,
    `rerank_score`. Returns [] on an empty corpus or a sub-floor (off-topic) pool —
    never raises."""
    k = k or config.FINAL_K
    pool = search.search(question, embeddings, chunks_data, cutoff=cutoff)
    # Diversify chooses k from a relevance-ordered set, so rerank the whole pool when
    # it's on; otherwise rerank straight to k.
    ranked = rerank.rerank(question, pool, top_k=len(pool) if config.DIVERSITY_ENABLED else k)
    if _below_floor(ranked):
        return []
    selected = diversify.select(ranked, embeddings, k=k)
    for item in selected:  # `_row` is pipeline plumbing (search -> diversify), not API
        item.pop("_row", None)
    return selected


def ask(question, guild_id, *, scope=None, k=None, cutoff=None, backend=None):
    """Full pipeline against a saved index: load -> retrieve -> synthesize.

    `scope=None` searches the whole server; otherwise a single channel id. Returns
    {"answer": str, "results": [chunk dicts]}. With no matches the answer is the
    exact NOT_FOUND fallback (synthesize handles the empty case), so the 0-web /
    0-assumption lock holds for headless callers too.
    """
    if scope is None:
        embeddings, chunks_data, _ = index_build.load_server(guild_id)
    else:
        embeddings, chunks_data, _ = index_build.load_channel(guild_id, scope)
    results = retrieve(question, embeddings, chunks_data, k=k, cutoff=cutoff)
    answer = synthesize.synthesize(question, results, backend=backend)
    return {"answer": answer, "results": results}
