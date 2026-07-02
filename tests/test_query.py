"""Pipeline orchestration (query.py): junk-pool floor, `_row` hygiene, and an
end-to-end retrieve through real search/lexical/diversify with a stubbed
embedder and a keyword-overlap fake reranker (no model loads)."""

import numpy as np
import pytest

from discord_answerer import config, query, rerank
from .conftest import make_chunk, unit


@pytest.fixture(autouse=True)
def pinned_config(monkeypatch):
    """Pin the knobs this suite asserts on, so DA_* env overrides can't skew it."""
    monkeypatch.setattr(config, "RERANK_FLOOR", 0.05)
    monkeypatch.setattr(config, "DIVERSITY_ENABLED", True)
    monkeypatch.setattr(config, "HYBRID_ENABLED", True)


def with_rerank(chunks, scores):
    out = []
    for c, s in zip(chunks, scores):
        item = dict(c)
        item["rerank_score"] = s
        out.append(item)
    return sorted(out, key=lambda it: it["rerank_score"], reverse=True)


def test_floor_rejects_offtopic_pool(monkeypatch):
    pool = [make_chunk("a"), make_chunk("b")]
    monkeypatch.setattr(query.search, "search", lambda *a, **kw: pool)
    monkeypatch.setattr(
        query.rerank, "rerank", lambda q, chunks, top_k=None: with_rerank(chunks, [0.01, 0.005])
    )
    assert query.retrieve("off topic", None, pool) == []


def test_floor_ignores_cosine_fallback(monkeypatch):
    """No rerank_score (reranker fell back) -> the floor must NOT fire."""
    pool = [make_chunk("a") | {"score": 0.4}, make_chunk("b") | {"score": 0.38}]
    monkeypatch.setattr(query.search, "search", lambda *a, **kw: pool)
    monkeypatch.setattr(query.rerank, "rerank", lambda q, chunks, top_k=None: chunks[:top_k])
    results = query.retrieve("q", None, pool, k=2)
    assert len(results) == 2


def test_retrieve_strips_row(monkeypatch):
    pool = [make_chunk("a") | {"_row": 0}, make_chunk("b") | {"_row": 1}]
    monkeypatch.setattr(query.search, "search", lambda *a, **kw: pool)
    monkeypatch.setattr(
        query.rerank, "rerank", lambda q, chunks, top_k=None: with_rerank(chunks, [0.9, 0.8])
    )
    for item in query.retrieve("q", None, pool, k=2):
        assert "_row" not in item


def test_empty_corpus_returns_empty(monkeypatch):
    monkeypatch.setattr(query.search, "search", lambda *a, **kw: [])
    assert query.retrieve("q", np.zeros((0, 3)), []) == []


def test_end_to_end_bm25_rescue_survives_to_final_k(fake_embed, monkeypatch):
    """The scenario the hybrid pass exists for: the only chunk naming the queried
    item has a near-zero cosine. It must enter the pool via BM25, win the (fake)
    rerank, and come out of retrieve() first."""
    chunks = [make_chunk(f"generic endgame chat number {i}", channel="legend") for i in range(9)]
    chunks.append(make_chunk("Mjolnir drops from Thor at 1%", channel="market"))
    vecs = [unit([1.0, 0.05 * i, 0.0]) for i in range(9)] + [unit([0.0, 0.0, 1.0])]
    embeddings = np.stack(vecs)
    fake_embed([1.0, 0.0, 0.0])

    def fake_rerank(q, pool, top_k=None):
        q_tokens = set(q.lower().split())
        scores = [
            len(q_tokens & set(c["text"].lower().split())) / len(q_tokens) for c in pool
        ]
        return with_rerank(pool, scores)[:top_k]

    monkeypatch.setattr(rerank, "rerank", fake_rerank)
    monkeypatch.setattr(query.rerank, "rerank", fake_rerank)

    results = query.retrieve("Mjolnir drops", embeddings, chunks, k=3, cutoff=0.35)
    assert results and "Mjolnir" in results[0]["text"]
    assert all("_row" not in r for r in results)
