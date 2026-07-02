"""BM25 (lexical.py): ranking on exact tokens + the identity-pinned index memo."""

import pytest

from discord_answerer import lexical


@pytest.fixture(autouse=True)
def clean_cache():
    lexical._cache.clear()
    yield
    lexical._cache.clear()


def docs(*texts):
    return [{"text": t} for t in texts]


def test_exact_rare_token_ranks_first():
    corpus = docs(
        "anyone tried the new dungeon yet",
        "the Excalibur drop rate from Baphomet is around 1%",
        "what a great patch overall, lots of balance changes",
    )
    top = lexical.top_n("Excalibur drop rate", corpus, 3)
    assert top and top[0][0] == 1


def test_no_term_overlap_scores_nothing():
    corpus = docs("healing build discussion", "tank rotation guide")
    assert lexical.top_n("Baphomet card price", corpus, 5) == []


def test_top_n_caps_and_sorts_desc():
    corpus = docs("sword sword sword", "sword shield", "sword", "bow only")
    top = lexical.top_n("sword", corpus, 2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]


def test_memo_returns_same_index_for_same_list():
    corpus = docs("alpha", "beta")
    first = lexical._get_index(corpus)
    assert lexical._get_index(corpus) is first


def test_memo_rejects_different_list_with_same_length():
    a = docs("alpha", "beta")
    b = docs("gamma", "delta")  # same length, different object/content
    index_a = lexical._get_index(a)
    index_b = lexical._get_index(b)
    assert index_a is not index_b
    # b must actually be indexed from b's texts, not a stale hit
    assert lexical.top_n("gamma", b, 2)[0][0] == 0


def test_memo_pins_cached_list_and_evicts_lru():
    kept = docs("pinned corpus")
    lexical._get_index(kept)
    for i in range(lexical._CACHE_MAX + 3):
        lexical._get_index(docs(f"filler {i}"))
    assert len(lexical._cache) <= lexical._CACHE_MAX
    # the oldest entry (kept) was evicted, a fresh call rebuilds without error
    assert lexical.top_n("pinned", kept, 1)[0][0] == 0
