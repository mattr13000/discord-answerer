"""Shared fixtures. The suite never loads a model: `fake_embed` monkeypatches
`embed.embed_query` (and tests stub `rerank.rerank` where needed), mirroring the
offline validation style — fast, deterministic, CI-safe."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_chunk(text, channel="general", link="https://discord.com/channels/1/2/3"):
    return {"text": text, "link": link, "channel_name": channel, "channel_id": "2"}


def unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def fake_embed(monkeypatch):
    """Make embed.embed_query return a caller-chosen unit vector, no model load."""
    from discord_answerer import embed

    state = {"vec": unit([1.0, 0.0, 0.0])}

    def set_query_vec(vec):
        state["vec"] = unit(vec)

    monkeypatch.setattr(embed, "embed_query", lambda text: state["vec"])
    return set_query_vec
