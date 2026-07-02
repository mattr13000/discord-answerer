"""Export parsing (parse.py) and conversation-window chunking (chunk.py)."""

import json

import pytest

from discord_answerer import chunk, config, parse


def export_json(tmp_path, messages):
    data = {
        "guild": {"id": "10", "name": "Guild"},
        "channel": {"id": "20", "name": "general"},
        "messages": messages,
    }
    p = tmp_path / "export.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def msg(mid, ts, content, name="alice", nickname=None):
    return {
        "id": mid,
        "timestamp": ts,
        "content": content,
        "author": {"name": name, "nickname": nickname},
    }


def test_parse_skips_empty_sorts_and_prefers_nickname(tmp_path):
    p = export_json(tmp_path, [
        msg("2", "2026-01-01T10:05:00Z", "second"),
        msg("3", "2026-01-01T10:10:00Z", "   "),  # empty after strip -> skipped
        msg("1", "2026-01-01T10:00:00Z", "first", name="bob", nickname="Bobby"),
    ])
    export = parse.parse_export(p)
    assert export.guild_id == "10" and export.channel_name == "general"
    assert [m.content for m in export.messages] == ["first", "second"]
    assert export.messages[0].author == "Bobby"


def test_parse_timestamp_handles_z_naive_and_garbage():
    aware = parse.parse_timestamp("2026-01-01T10:00:00Z")
    naive = parse.parse_timestamp("2026-01-01T10:00:00")
    assert aware is not None and aware.tzinfo is not None
    assert naive is not None and naive.tzinfo is not None  # assumed UTC
    assert parse.parse_timestamp("not a date") is None
    assert parse.parse_timestamp("") is None


def _export_obj(messages):
    return parse.ParsedExport(
        guild_id="10", guild_name="Guild", channel_id="20",
        channel_name="general", messages=messages,
    )


def _messages(n, start_hour=10, gap_minutes=1):
    out = []
    for i in range(n):
        total = start_hour * 60 + i * gap_minutes
        ts = f"2026-01-01T{total // 60:02d}:{total % 60:02d}:00Z"
        out.append(parse.Message(id=str(i), timestamp=ts, author="alice", content=f"m{i}"))
    return out


def test_time_gap_splits_conversations(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_TIME_GAP_MINUTES", 30)
    early = _messages(3)
    late = [
        parse.Message(id="99", timestamp="2026-01-01T15:00:00Z", author="bob", content="hours later")
    ]
    chunks = chunk.build_chunks(_export_obj(early + late))
    assert len(chunks) == 2
    assert "hours later" in chunks[1].text and "hours later" not in chunks[0].text


def test_windows_overlap_and_cover_all_messages(monkeypatch):
    monkeypatch.setattr(config, "CHUNK_MAX_MESSAGES", 4)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MESSAGES", 1)
    chunks = chunk.build_chunks(_export_obj(_messages(10)))
    covered = {mid for c in chunks for mid in c.message_ids}
    assert covered == {str(i) for i in range(10)}
    assert all(len(c.message_ids) <= 4 for c in chunks)
    # consecutive windows share the overlap
    assert set(chunks[0].message_ids) & set(chunks[1].message_ids)


def test_chunk_link_points_to_first_message():
    chunks = chunk.build_chunks(_export_obj(_messages(2)))
    assert chunks[0].link == "https://discord.com/channels/10/20/0"
