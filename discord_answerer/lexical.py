"""Lexical BM25 retrieval (stage 1b): a sparse keyword signal fused with the dense
cosine pool in search.py.

Dense embeddings are great at meaning but blunt on *exact tokens* — item names,
skill/boss names, patch numbers like "+9 STR" — which are exactly the vocabulary a
game Discord turns on. BM25 scores those term overlaps directly, so a chunk that is
the only one mentioning a queried item gets surfaced even when its cosine is low.

Implemented in-repo (numpy + stdlib), no extra dependency — same "brute-force, no
vector DB" ethos as the rest of the index. The model is a textbook BM25 Okapi over
an inverted index, so a query only touches the postings of its own terms (fast as the
corpus scales to tens of thousands of chunks).
"""

import math
import re
from collections import Counter, OrderedDict

import numpy as np

from . import config

# Unicode word tokens: keeps EN/FR words and KR syllables, drops punctuation. The
# chunk text carries "author (timestamp): line" prefixes; those tokens are harmless
# noise that BM25's idf naturally down-weights (they're common across every chunk).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Inverted-index BM25 Okapi over a fixed list of documents."""

    def __init__(self, texts: list[str]):
        self.n_docs = len(texts)
        self.doc_len = np.zeros(self.n_docs, dtype=np.float32)
        self.postings: dict[str, list[tuple[int, int]]] = {}  # token -> [(doc_idx, tf)]
        df: Counter = Counter()
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            self.doc_len[i] = len(tokens)
            for token, tf in Counter(tokens).items():
                self.postings.setdefault(token, []).append((i, tf))
                df[token] += 1
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        # idf = ln(1 + (N - df + 0.5)/(df + 0.5)) — the "+1" keeps it non-negative.
        self.idf = {
            token: math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5))
            for token, d in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query` (0 where no query term matches)."""
        k1, b = config.BM25_K1, config.BM25_B
        out = np.zeros(self.n_docs, dtype=np.float32)
        if self.avgdl == 0.0:
            return out
        for token in set(_tokenize(query)):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf[token]
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                denom = tf + k1 * (1.0 - b + b * dl / self.avgdl)
                out[doc_idx] += idf * (tf * (k1 + 1.0)) / denom
        return out


# Building the index tokenizes the whole corpus, so we memoize it per loaded scope.
# Keyed by id(chunks_data), BUT each entry also holds a strong reference to that
# exact list: while cached it can't be garbage-collected, so its id can never be
# recycled to a different list — the `is` check makes a stale hit impossible (a
# plain id()+len key could, rarely, match a new list allocated at a freed address).
# Bounded LRU (evict oldest, keep hot scopes) so switching across many servers in
# one session can't hoard indexes or pin unbounded chunk lists.
_cache: OrderedDict[int, tuple[object, BM25Index]] = OrderedDict()
_CACHE_MAX = 8


def _get_index(chunks_data) -> BM25Index:
    key = id(chunks_data)
    cached = _cache.get(key)
    if cached is not None and cached[0] is chunks_data:
        _cache.move_to_end(key)
        return cached[1]
    index = BM25Index([c["text"] for c in chunks_data])
    _cache[key] = (chunks_data, index)
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return index


def top_n(query: str, chunks_data, n: int) -> list[tuple[int, float]]:
    """Top-`n` documents by BM25, as (index, score) sorted desc. Only positive scores
    (a doc sharing no query term scores 0 and is useless to fuse)."""
    if n <= 0 or not chunks_data:
        return []
    scores = _get_index(chunks_data).scores(query)
    n = min(n, len(scores))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0.0]
