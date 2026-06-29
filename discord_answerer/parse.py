"""Reads a DiscordChatExporter export (JSON format).

Expected structure: a root object with `guild{id,name}`, `channel{id,name}`,
`messages[]`. Each message: `id`, `timestamp`, `content`, `author{name,nickname}`.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone


def parse_timestamp(ts: str):
    """ISO-8601 timestamp -> aware datetime, or None if absent/unparseable.

    Shared by chunking (time-gap segmentation) and parsing (chronological sort)
    so the two never drift. Naive timestamps are assumed UTC so all results are
    comparable.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Message:
    id: str
    timestamp: str
    author: str
    content: str


@dataclass
class ParsedExport:
    guild_id: str
    guild_name: str
    channel_id: str
    channel_name: str
    messages: list[Message]


def parse_export(path) -> ParsedExport:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    guild = data.get("guild") or {}
    channel = data.get("channel") or {}

    messages: list[Message] = []
    for m in data.get("messages", []):
        content = (m.get("content") or "").strip()
        if not content:
            continue  # MVP: skip attachment/embed-only messages
        author = m.get("author") or {}
        author_name = author.get("nickname") or author.get("name") or "unknown"
        messages.append(
            Message(
                id=str(m.get("id")),
                timestamp=m.get("timestamp", ""),
                author=author_name,
                content=content,
            )
        )

    # Chunking assumes chronological order to split on time gaps. Exports are
    # normally already ascending, but sort defensively (stable: messages with no
    # timestamp keep their relative position) so one out-of-order export can't
    # silently corrupt chunk boundaries.
    messages.sort(key=lambda m: parse_timestamp(m.timestamp) or datetime.min.replace(tzinfo=timezone.utc))

    return ParsedExport(
        guild_id=str(guild.get("id", "")),
        guild_name=guild.get("name", ""),
        channel_id=str(channel.get("id", "")),
        channel_name=channel.get("name", ""),
        messages=messages,
    )


def message_link(guild_id: str, channel_id: str, message_id: str) -> str:
    """Rebuild the clickable Discord jump-link to a message."""
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
