"""Command-line entry point: `python -m discord_answerer <command>`.

Two commands, both thin wrappers over the library so the CLI, the Streamlit UI
and any test share the exact same pipeline (see query.py):

    python -m discord_answerer build data/export.json
    python -m discord_answerer servers
    python -m discord_answerer ask "best end-game build?" --guild 123 [--channel 456]

`ask` needs a Gemini key (GEMINI_API_KEY in .env or env) unless --backend ollama.
"""

import argparse
import sys

from . import config, index_build, query


def _cmd_build(args) -> int:
    meta = index_build.build_index(args.json_path)
    print(
        f"Indexed #{meta.get('channel_name')} ({meta.get('num_messages')} messages, "
        f"{meta.get('num_chunks')} chunks) into {meta.get('guild_name')} [{meta.get('guild_id')}]."
    )
    return 0


def _cmd_servers(_args) -> int:
    servers = index_build.list_servers()
    if not servers:
        print("No indexed server yet. Build one with: python -m discord_answerer build <export.json>")
        return 0
    for s in servers:
        print(f"{s['guild_name']} [{s['guild_id']}] - {s['num_channels']} channels, "
              f"{s['num_messages_total']} messages")
        for ch in s["channels"]:
            print(f"    #{ch.get('channel_name')} [{ch.get('channel_id')}] - {ch.get('num_messages')} msgs")
    return 0


def _cmd_ask(args) -> int:
    out = query.ask(
        args.question,
        args.guild,
        scope=args.channel,
        k=args.k,
        cutoff=args.cutoff,
        backend=args.backend,
    )
    print(out["answer"])
    if args.show_sources:
        print("\n--- sources ---")
        for i, r in enumerate(out["results"], 1):
            chan = f" #{r['channel_name']}" if r.get("channel_name") else ""
            print(f"[Message {i}]{chan} {r.get('link', '')}")
    return 0


def main(argv=None) -> int:
    # LLM answers and Discord names are multilingual (FR/KR/…); force UTF-8 so the
    # Windows console code page doesn't mojibake them. Harmless elsewhere.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="discord_answerer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="index a DiscordChatExporter JSON export")
    p_build.add_argument("json_path", help="path to the .json export")
    p_build.set_defaults(func=_cmd_build)

    p_servers = sub.add_parser("servers", help="list indexed servers and channels")
    p_servers.set_defaults(func=_cmd_servers)

    p_ask = sub.add_parser("ask", help="ask a question against a saved index")
    p_ask.add_argument("question")
    p_ask.add_argument("--guild", required=True, help="guild id (see `servers`)")
    p_ask.add_argument("--channel", default=None, help="restrict to one channel id (default: whole server)")
    p_ask.add_argument("--k", type=int, default=None, help=f"messages used in the answer (default {config.FINAL_K})")
    p_ask.add_argument("--cutoff", type=float, default=None, help="coarse score pre-filter")
    p_ask.add_argument("--backend", default=None, choices=["gemini", "ollama"], help="LLM backend")
    p_ask.add_argument("--show-sources", action="store_true", help="print the cited source links")
    p_ask.set_defaults(func=_cmd_ask)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
