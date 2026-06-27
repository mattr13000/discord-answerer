"""Build and load the vector index (numpy brute-force).

Persists into `index/`: embeddings.npy (matrix), chunks.json (aligned metadata),
meta.json (model, guild/channel, counters).
"""

import json

import numpy as np

from . import chunk, config, embed, parse


def build_index(json_path) -> dict:
    export = parse.parse_export(json_path)
    chunks = chunk.build_chunks(export)
    if not chunks:
        raise ValueError("No usable message in the export.")

    embeddings = embed.embed_documents([c.text for c in chunks])

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", embeddings)

    chunks_data = [
        {
            "text": c.text,
            "message_ids": c.message_ids,
            "link": c.link,
            "timestamp": c.timestamp,
            "author_span": c.author_span,
        }
        for c in chunks
    ]
    with open(config.INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    meta = {
        "embed_model": config.EMBED_MODEL,
        "guild_id": export.guild_id,
        "guild_name": export.guild_name,
        "channel_id": export.channel_id,
        "channel_name": export.channel_name,
        "num_messages": len(export.messages),
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }
    with open(config.INDEX_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def index_exists() -> bool:
    return (config.INDEX_DIR / "embeddings.npy").exists() and (
        config.INDEX_DIR / "chunks.json"
    ).exists()


def load_index():
    embeddings = np.load(config.INDEX_DIR / "embeddings.npy")
    with open(config.INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks_data = json.load(f)
    with open(config.INDEX_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, chunks_data, meta
