"""Index library (index_build.py): server stacking, channel tagging, and the
embedding-model consistency guard. Uses a temp index dir; no model loads."""

import json

import numpy as np
import pytest

from discord_answerer import config, index_build


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    return config.INDEX_DIR


def write_channel(guild_id, channel_id, channel_name, texts, embed_model="model-a", dim=3):
    d = config.INDEX_DIR / guild_id / channel_id
    d.mkdir(parents=True)
    rng = np.random.default_rng(int(channel_id))
    np.save(d / "embeddings.npy", rng.random((len(texts), dim), dtype=np.float32))
    chunks = [{"text": t, "link": "l", "message_ids": [], "timestamp": "", "author_span": ""} for t in texts]
    (d / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    meta = {
        "embed_model": embed_model,
        "guild_id": guild_id,
        "guild_name": "Guild",
        "channel_id": channel_id,
        "channel_name": channel_name,
        "num_messages": len(texts),
        "num_chunks": len(texts),
        "dim": dim,
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_load_server_stacks_and_tags_channels(index_dir):
    write_channel("10", "1", "legend", ["a", "b"])
    write_channel("10", "2", "market", ["c"])
    embeddings, chunks, meta = index_build.load_server("10")
    assert embeddings.shape == (3, 3)
    assert len(chunks) == 3
    assert {c["channel_name"] for c in chunks} == {"legend", "market"}
    assert meta["num_channels"] == 2 and meta["num_messages"] == 3


def test_load_server_rejects_mixed_embed_models(index_dir):
    write_channel("10", "1", "legend", ["a"], embed_model="model-a")
    write_channel("10", "2", "market", ["b"], embed_model="model-b")
    with pytest.raises(ValueError, match="different embedding models"):
        index_build.load_server("10")


def test_load_server_missing_guild_raises(index_dir):
    with pytest.raises(FileNotFoundError):
        index_build.load_server("nope")


def test_load_channel_tags_rows(index_dir):
    write_channel("10", "1", "legend", ["a"])
    _, chunks, meta = index_build.load_channel("10", "1")
    assert chunks[0]["channel_name"] == "legend"
    assert meta["channel_id"] == "1"


def test_delete_channel_drops_empty_server(index_dir):
    write_channel("10", "1", "legend", ["a"])
    index_build.delete_channel("10", "1")
    assert not (index_dir / "10").exists()
