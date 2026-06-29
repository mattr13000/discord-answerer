"""Groups messages into *conversation windows*.

Discord messages are short and conversational: a build, an answer to a question,
etc. are often spread across several messages. So we chunk by sliding windows
(with overlap), splitting on large time gaps to avoid mixing two distinct
conversations.
"""

from dataclasses import dataclass

from . import config, parse


@dataclass
class Chunk:
    text: str
    message_ids: list[str]
    link: str
    timestamp: str
    author_span: str


def _segments(messages):
    """Split the message list into segments separated by a large time gap."""
    seg: list = []
    last_ts = None
    for m in messages:
        ts = parse.parse_timestamp(m.timestamp)
        if last_ts and ts:
            gap_min = (ts - last_ts).total_seconds() / 60.0
            if gap_min > config.CHUNK_TIME_GAP_MINUTES and seg:
                yield seg
                seg = []
        seg.append(m)
        last_ts = ts or last_ts
    if seg:
        yield seg


def _make_chunk(window, export) -> Chunk:
    lines = []
    for m in window:
        ts = (m.timestamp or "")[:16].replace("T", " ")
        lines.append(f"{m.author} ({ts}): {m.content}")
    first = window[0]
    return Chunk(
        text="\n".join(lines),
        message_ids=[m.id for m in window],
        link=parse.message_link(export.guild_id, export.channel_id, first.id),
        timestamp=first.timestamp,
        author_span=", ".join(dict.fromkeys(m.author for m in window)),
    )


def build_chunks(export: parse.ParsedExport) -> list[Chunk]:
    step = max(1, config.CHUNK_MAX_MESSAGES - config.CHUNK_OVERLAP_MESSAGES)
    chunks: list[Chunk] = []
    for seg in _segments(export.messages):
        if len(seg) <= config.CHUNK_MAX_MESSAGES:
            chunks.append(_make_chunk(seg, export))
            continue
        i = 0
        while i < len(seg):
            window = seg[i : i + config.CHUNK_MAX_MESSAGES]
            chunks.append(_make_chunk(window, export))
            if i + config.CHUNK_MAX_MESSAGES >= len(seg):
                break
            i += step
    return chunks
