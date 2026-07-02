"""MMR + per-channel quota (diversify.py), incl. the cosine-fallback relevance."""

import numpy as np
import pytest

from discord_answerer import config, diversify
from .conftest import make_chunk, unit


@pytest.fixture(autouse=True)
def diversity_on(monkeypatch):
    monkeypatch.setattr(config, "DIVERSITY_ENABLED", True)
    monkeypatch.setattr(config, "MMR_LAMBDA", 0.7)
    monkeypatch.setattr(config, "PER_CHANNEL_FRACTION", 0.5)


def ranked_item(text, rerank_score=None, score=None, channel="general", row=None):
    item = make_chunk(text, channel=channel)
    if rerank_score is not None:
        item["rerank_score"] = rerank_score
    if score is not None:
        item["score"] = score
    if row is not None:
        item["_row"] = row
    return item


def test_disabled_returns_head(monkeypatch):
    monkeypatch.setattr(config, "DIVERSITY_ENABLED", False)
    ranked = [ranked_item(f"c{i}", rerank_score=1.0 - i / 10) for i in range(5)]
    assert diversify.select(ranked, None, k=2) == ranked[:2]


def test_mmr_skips_near_duplicate():
    embeddings = np.stack([
        unit([1.0, 0.0]),
        unit([1.0, 0.01]),  # near-duplicate of row 0
        unit([0.0, 1.0]),   # distinct
    ])
    ranked = [
        ranked_item("dup A", rerank_score=0.95, row=0),
        ranked_item("dup B", rerank_score=0.94, row=1),
        ranked_item("novel", rerank_score=0.80, row=2),
    ]
    picked = diversify.select(ranked, embeddings, k=2)
    assert [p["text"] for p in picked] == ["dup A", "novel"]


def test_per_channel_quota_caps_chatty_channel():
    # k=2, fraction 0.5 -> cap 1 per channel; 3 top items share a channel.
    ranked = [
        ranked_item("a1", rerank_score=0.9, channel="chatty"),
        ranked_item("a2", rerank_score=0.8, channel="chatty"),
        ranked_item("b1", rerank_score=0.2, channel="quiet"),
    ]
    picked = diversify.select(ranked, None, k=2)
    assert {p["channel_name"] for p in picked} == {"chatty", "quiet"}


def test_quota_relaxes_when_single_channel():
    ranked = [ranked_item(f"c{i}", rerank_score=1.0 - i / 10, channel="only") for i in range(4)]
    picked = diversify.select(ranked, None, k=3)
    assert len(picked) == 3  # cap (ceil(3*0.5)=2) relaxed to fill k


def test_fallback_uses_incoming_order_not_raw_cosine():
    """Cosine-fallback path (no rerank_score): a BM25-rescued chunk sits early in
    the RRF order despite a LOW cosine. Positional relevance must keep it ahead of
    a late high-cosine chunk — raw-cosine relevance would demote it."""
    ranked = [
        ranked_item("rescued exact-token chunk", score=0.05),  # RRF rank 0
        ranked_item("dense match", score=0.60),
        ranked_item("weaker dense match", score=0.55),
    ]
    picked = diversify.select(ranked, None, k=2)
    assert picked[0]["text"] == "rescued exact-token chunk"


def test_reranked_relevance_prefers_rerank_score():
    ranked = [
        ranked_item("top", rerank_score=0.9, score=0.1),
        ranked_item("bottom", rerank_score=0.1, score=0.9),
    ]
    picked = diversify.select(ranked, None, k=2)
    assert picked[0]["text"] == "top"
