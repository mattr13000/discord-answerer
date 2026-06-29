"""High-level query pipeline — the single headless entry point that wires the
three retrieval stages so callers (Streamlit UI, CLI, tests) never re-implement
the flow:

    search.search        stage 1: adaptive cosine pool      (recall)
      -> rerank.rerank   stage 2: cross-encoder rerank      (precision)
        -> synthesize    stage 3: bounded LLM over FINAL_K  (answer)

`retrieve` returns the ranked chunks; `ask` runs the whole thing against a saved
index and also synthesizes the answer. The Streamlit app wraps these in its own
caching layer — the library itself stays free of any UI dependency.
"""

from . import index_build, rerank, search, synthesize


def retrieve(question, embeddings, chunks_data, *, k=None, cutoff=None):
    """Stage 1 + 2: cosine candidate pool, then cross-encoder rerank to the best
    `k` chunks (config.FINAL_K when None).

    Returns chunk dicts carrying `score` (cosine) and, when the reranker is
    active, `rerank_score`. Never raises on an empty corpus — returns []."""
    pool = search.search(question, embeddings, chunks_data, cutoff=cutoff)
    return rerank.rerank(question, pool, top_k=k)


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
