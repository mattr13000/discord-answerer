"""Build, list, load and delete vector indexes (numpy brute-force).

The index is a *library of servers*. Each ingested Discord channel lives in its
own folder, grouped by server (guild):

    index/<guild_id>/<channel_id>/
        embeddings.npy : the vector matrix
        chunks.json    : aligned metadata (one row per vector)
        meta.json      : model, guild/channel, counters

A *server* is just the set of channel folders under `index/<guild_id>/`. Grouping
channels never recomputes embeddings: `load_server` simply `np.vstack`-es the
already-saved per-channel matrices and concatenates their chunks (each chunk gets
tagged in memory with the channel it came from). Re-embedding only happens when a
channel's content or `DA_EMBED_MODEL` changes.
"""

import json
import shutil

import numpy as np

from . import chunk, config, embed, parse


def _server_dir(guild_id):
    return config.INDEX_DIR / (str(guild_id) or "guild")


def _channel_dir(guild_id, channel_id):
    return _server_dir(guild_id) / (str(channel_id) or "channel")


def _has_files(d) -> bool:
    return (
        (d / "embeddings.npy").exists()
        and (d / "chunks.json").exists()
        and (d / "meta.json").exists()
    )


def _read_meta(d) -> dict | None:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


_migrated = False


def _migrate_layout() -> None:
    """Bring older index layouts into `index/<guild_id>/<channel_id>/`.

    Handles two legacy shapes, idempotently:
    - flat files directly in `index/` (the very first single-index layout);
    - flat per-channel folders `index/<guild>_<channel>/` (the first library).
    Pure folder moves — embeddings are never recomputed.

    Legacy folders only ever come from older builds, never created at runtime, so
    once a process has migrated there is nothing new to scan. The `_migrated`
    guard makes this a no-op on subsequent calls (list_servers runs on every
    Streamlit rerun).
    """
    global _migrated
    if _migrated or not config.INDEX_DIR.exists():
        return

    # 1. files-directly-in-index/  ->  index/<guild>/<channel>/
    if _has_files(config.INDEX_DIR):
        meta = _read_meta(config.INDEX_DIR)
        if meta:
            dest = _channel_dir(meta.get("guild_id"), meta.get("channel_id"))
            if not _has_files(dest):
                dest.mkdir(parents=True, exist_ok=True)
                for name in ("embeddings.npy", "chunks.json", "meta.json"):
                    src = config.INDEX_DIR / name
                    if src.exists():
                        shutil.move(str(src), str(dest / name))

    # 2. flat index/<guild>_<channel>/  ->  index/<guild>/<channel>/
    for d in list(config.INDEX_DIR.iterdir()):
        if not (d.is_dir() and _has_files(d)):
            continue
        meta = _read_meta(d)
        if not meta:
            continue
        dest = _channel_dir(meta.get("guild_id"), meta.get("channel_id"))
        if dest.resolve() == d.resolve() or _has_files(dest):
            continue  # already nested, or destination taken
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("embeddings.npy", "chunks.json", "meta.json"):
            src = d / name
            if src.exists():
                shutil.move(str(src), str(dest / name))
        shutil.rmtree(d, ignore_errors=True)

    _migrated = True


def build_index(json_path) -> dict:
    export = parse.parse_export(json_path)
    chunks = chunk.build_chunks(export)
    if not chunks:
        raise ValueError("No usable message in the export.")

    embeddings = embed.embed_documents([c.text for c in chunks])

    dest = _channel_dir(export.guild_id, export.channel_id)
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


def list_servers() -> list[dict]:
    """Return one entry per indexed server (guild), each grouping its channels.

    Shape: {guild_id, guild_name, channels:[meta…], num_channels,
    num_messages_total}. Channels are sorted by name; servers by guild name.
    """
    _migrate_layout()
    if not config.INDEX_DIR.exists():
        return []

    servers: dict[str, dict] = {}
    for sd in config.INDEX_DIR.iterdir():
        if not sd.is_dir():
            continue
        for cd in sd.iterdir():
            if not (cd.is_dir() and _has_files(cd)):
                continue
            meta = _read_meta(cd)
            if not meta:
                continue
            gid = str(meta.get("guild_id") or sd.name)
            srv = servers.setdefault(
                gid,
                {
                    "guild_id": gid,
                    "guild_name": meta.get("guild_name", ""),
                    "channels": [],
                    "num_channels": 0,
                    "num_messages_total": 0,
                },
            )
            srv["channels"].append(meta)
            srv["num_channels"] += 1
            srv["num_messages_total"] += int(meta.get("num_messages", 0) or 0)
            if not srv["guild_name"]:
                srv["guild_name"] = meta.get("guild_name", "")

    result = list(servers.values())
    for srv in result:
        srv["channels"].sort(key=lambda m: m.get("channel_name", ""))
    result.sort(key=lambda s: (s.get("guild_name", ""), s.get("guild_id", "")))
    return result


def has_any_index() -> bool:
    return bool(list_servers())


def _load_channel_dir(cd):
    """Load one channel folder, tagging each chunk with its channel meta."""
    embeddings = np.load(cd / "embeddings.npy")
    with open(cd / "chunks.json", encoding="utf-8") as f:
        chunks_data = json.load(f)
    meta = _read_meta(cd) or {}
    cid, cname = meta.get("channel_id", ""), meta.get("channel_name", "")
    for row in chunks_data:
        row["channel_id"] = cid
        row["channel_name"] = cname
    return embeddings, chunks_data, meta


def load_channel(guild_id, channel_id):
    """Load a single channel index (embeddings, channel-tagged chunks, meta)."""
    return _load_channel_dir(_channel_dir(guild_id, channel_id))


def load_server(guild_id):
    """Stack every channel of a server into one searchable index.

    Returns (embeddings, chunks_data, server_meta). No recompute: the per-channel
    matrices are vstacked and the channel-tagged chunks concatenated.
    """
    sd = _server_dir(guild_id)
    mats, chunks_data, channels = [], [], []
    if sd.exists():
        channel_dirs = sorted(
            (cd for cd in sd.iterdir() if cd.is_dir() and _has_files(cd)),
            key=lambda cd: (_read_meta(cd) or {}).get("channel_name", ""),
        )
        for cd in channel_dirs:
            emb, rows, meta = _load_channel_dir(cd)
            mats.append(emb)
            chunks_data.extend(rows)
            channels.append(meta)

    if not mats:
        raise FileNotFoundError(f"No indexed channel for server {guild_id!r}.")

    # Vectors from different embedding models live in incompatible spaces; stacking
    # them would either crash on a dim mismatch or, worse, silently rank garbage.
    # (Happens when DA_EMBED_MODEL changed between two channel builds.)
    models = {str(c.get("embed_model")) for c in channels if c.get("embed_model")}
    if len(models) > 1:
        detail = "; ".join(
            f"#{c.get('channel_name', '?')}: {c.get('embed_model', '?')}" for c in channels
        )
        raise ValueError(
            f"Channels of server {guild_id!r} were indexed with different embedding "
            f"models ({detail}). Their vectors are not comparable — remove and "
            "re-index the outdated channels so the whole server uses one model."
        )

    embeddings = np.vstack(mats)
    server_meta = {
        "guild_id": str(guild_id),
        "guild_name": next((c.get("guild_name", "") for c in channels if c.get("guild_name")), ""),
        "channels": channels,
        "num_channels": len(channels),
        "num_messages": sum(int(c.get("num_messages", 0) or 0) for c in channels),
    }
    return embeddings, chunks_data, server_meta


def delete_channel(guild_id, channel_id) -> None:
    cd = _channel_dir(guild_id, channel_id)
    if cd.exists():
        shutil.rmtree(cd, ignore_errors=True)
    sd = _server_dir(guild_id)  # drop the server folder if it's now empty
    if sd.exists() and not any(sd.iterdir()):
        shutil.rmtree(sd, ignore_errors=True)


def delete_server(guild_id) -> None:
    sd = _server_dir(guild_id)
    if sd.exists():
        shutil.rmtree(sd, ignore_errors=True)
