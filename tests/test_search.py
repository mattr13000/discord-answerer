"""Hybrid stage-1 search: dense cutoff, BM25 rescue via RRF, pool metadata."""

import numpy as np
import pytest

from discord_answerer import config, search
from .conftest import make_chunk, unit


@pytest.fixture
def corpus():
    """3 chunks: two semantically close to the query axis, one orthogonal but
    holding the exact rare token 'Mjolnir'."""
    chunks = [
        make_chunk("best endgame weapon discussion", channel="legend"),
        make_chunk("endgame gear overview thread", channel="legend"),
        make_chunk("Mjolnir sells for 50m on the market", channel="market"),
    ]
    embeddings = np.stack([
        unit([1.0, 0.1, 0.0]),
        unit([1.0, 0.2, 0.0]),
        unit([0.0, 0.0, 1.0]),  # cosine ~0 vs the query -> below any cutoff
    ])
    return embeddings, chunks


def test_dense_cutoff_filters_low_cosine(fake_embed, corpus, monkeypatch):
    monkeypatch.setattr(config, "HYBRID_ENABLED", False)
    embeddings, chunks = corpus
    fake_embed([1.0, 0.0, 0.0])
    results = search.search("endgame weapon", embeddings, chunks, k=10, cutoff=0.35)
    texts = [r["text"] for r in results]
    assert len(results) == 2
    assert all("Mjolnir" not in t for t in texts)


def test_bm25_rescues_low_cosine_exact_token(fake_embed, corpus, monkeypatch):
    monkeypatch.setattr(config, "HYBRID_ENABLED", True)
    embeddings, chunks = corpus
    fake_embed([1.0, 0.0, 0.0])
    results = search.search("Mjolnir price", embeddings, chunks, k=10, cutoff=0.35)
    assert any("Mjolnir" in r["text"] for r in results)


def test_results_carry_score_and_row(fake_embed, corpus, monkeypatch):
    monkeypatch.setattr(config, "HYBRID_ENABLED", False)
    embeddings, chunks = corpus
    fake_embed([1.0, 0.0, 0.0])
    results = search.search("endgame", embeddings, chunks, k=10, cutoff=0.0)
    for r in results:
        assert isinstance(r["score"], float)
        assert r["text"] == chunks[r["_row"]]["text"]
    # originals not mutated
    assert "_row" not in chunks[0] and "score" not in chunks[0]


def test_rrf_fuse_prefers_doc_ranked_high_in_both():
    fused = search._rrf_fuse([[0, 1, 2], [1, 0, 2]], k=3)
    assert set(fused[:2]) == {0, 1}
    assert fused[2] == 2
