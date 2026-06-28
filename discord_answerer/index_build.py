"""Build, list, load and delete vector indexes (numpy brute-force).

The index is a *library*: each ingested Discord export lives in its own
subfolder `index/<key>/` (key = "<guild_id>_<channel_id>"), holding:
- embeddings.npy : the vector matrix
- chunks.json    : aligned metadata (one row per vector)
- meta.json      : model, guild/channel, counters

This lets the app keep several Discords and switch between them.
"""

import json
import shutil

import numpy as np

from . import chunk, config, embed, parse


def _key(guild_id, channel_id) -> str:
    return f"{guild_id or 'guild'}_{channel_id or 'channel'}"


def _index_dir(key: str):
    return config.INDEX_DIR / key


def _has_files(d) -> bool:
    return (
        (d / "embeddings.npy").exists()
        and (d / "chunks.json").exists()
        and (d / "meta.json").exists()
    )


def _migrate_legacy() -> None:
    """Move a pre-library flat index (files directly in index/) into index/<key>/."""
    flat_meta = config.INDEX_DIR / "meta.json"
    if not flat_meta.exists() or not (config.INDEX_DIR / "embeddings.npy").exists():
        return
    try:
        meta = json.loads(flat_meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    dest = _index_dir(_key(meta.get("guild_id"), meta.get("channel_id")))
    if _has_files(dest):  # already migrated
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("embeddings.npy", "chunks.json", "meta.json"):
        src = config.INDEX_DIR / name
        if src.exists():
            shutil.move(str(src), str(dest / name))


def build_index(json_path) -> dict:
    export = parse.parse_export(json_path)
    chunks = chunk.build_chunks(export)
    if not chunks:
        raise ValueError("No usable message in the export.")

    embeddings = embed.embed_documents([c.text for c in chunks])

    key = _key(export.guild_id, export.channel_id)
    dest = _index_dir(key)
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "embeddings.npy", embeddings)

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
    with open(dest / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    meta = {
        "key": key,
        "embed_model": config.EMBED_MODEL,
        "guild_id": export.guild_id,
        "guild_name": export.guild_name,
        "channel_id": export.channel_id,
        "channel_name": export.channel_name,
        "num_messages": len(export.messages),
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }
    with open(dest / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def list_indexes() -> list[dict]:
    """Return the meta dict of every indexed Discord, sorted by display name."""
    _migrate_legacy()
    if not config.INDEX_DIR.exists():
        return []
    metas = []
    for d in config.INDEX_DIR.iterdir():
        if not (d.is_dir() and _has_files(d)):
            continue
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta.setdefault("key", d.name)
        metas.append(meta)
    metas.sort(key=lambda m: (m.get("guild_name", ""), m.get("channel_name", "")))
    return metas


def has_any_index() -> bool:
    return bool(list_indexes())


def load_index(key: str):
    d = _index_dir(key)
    embeddings = np.load(d / "embeddings.npy")
    with open(d / "chunks.json", encoding="utf-8") as f:
        chunks_data = json.load(f)
    with open(d / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, chunks_data, meta


def delete_index(key: str) -> None:
    d = _index_dir(key)
    if d.exists():
        shutil.rmtree(d)
