"""Reads a DiscordChatExporter export (JSON format).

Expected structure: a root object with `guild{id,name}`, `channel{id,name}`,
`messages[]`. Each message: `id`, `timestamp`, `content`, `author{name,nickname}`,
`reference{messageId}` (for replies).
"""

import json
from dataclasses import dataclass


@dataclass
class Message:
    id: str
    timestamp: str
    author: str
    content: str
    reply_to: str | None


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
        ref = m.get("reference") or {}
        reply_to = ref.get("messageId")
        messages.append(
            Message(
                id=str(m.get("id")),
                timestamp=m.get("timestamp", ""),
                author=author_name,
                content=content,
                reply_to=str(reply_to) if reply_to else None,
            )
        )

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
