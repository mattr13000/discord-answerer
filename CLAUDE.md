# CLAUDE.md — Discord Answerer

Context for future Claude Code sessions on this project.

## Project language convention (IMPORTANT)
**The entire project is in English** — code, identifiers, comments, docstrings,
Markdown docs, the Streamlit UI, the system prompt, and `config.NOT_FOUND_MESSAGE`
(`"Not found in the Discord."`). Do not regress to French.
LLM answers follow the language of the user's question (enforced in the system prompt).

## Goal
A RAG pipeline **strictly bounded** to the messages of an exported Discord. The user
asks a question (e.g. "best end-game build?"); the app semantically retrieves the
relevant messages and synthesizes an answer via an LLM.

**Non-negotiable constraint: 0 web, 0 LLM assumption.** If the info isn't in the
Discord, the answer must be EXACTLY `Not found in the Discord.`
(`config.NOT_FOUND_MESSAGE`). This is the product's core value — never weaken it.
The three locks live in `synthesize.py`: no search tool passed to the model, context =
only the retrieved chunks, strict system prompt.

## Current status (MVP complete & validated)
Working end-to-end and validated on a real export (Echoes of Morroc, a Ragnarok-like
private server; 4434 messages -> 846 chunks). Confirmed: Raw retrieval, cross-lingual
search (EN/FR/KR), Gemini synthesis with `[Message N]` citations, and the "Not found"
lock (out-of-scope question -> exact fallback). The Streamlit app runs at
http://localhost:8501. A local `.venv` is set up (Python 3.14, **torch CPU build**); the
user's `GEMINI_API_KEY` lives in `.env` (gitignored) — or can be pasted directly in the
UI. Default search cutoff = 0.35, but the real anti-hallucination guard is the LLM
(out-of-scope queries can still score ~0.46).

**UX pass done** (non-tech friendly): drag & drop upload of the export (`st.file_uploader`),
a **multi-Discord library** (each export indexed under `index/<guild_id>_<channel_id>/`,
switchable via a sidebar selector), in-UI Gemini key entry, advanced knobs folded into an
expander, plain-language mode labels, and **hover-card tooltips** revealing a cited
message on hover (over inline `[Message N]` citations and the source list).

## Architecture
```
app.py                      # Streamlit UI (Raw + LLM modes)
discord_answerer/
  config.py                 # everything overridable via env var; loads .env
  parse.py                  # DiscordChatExporter JSON -> normalized messages
  chunk.py                  # conversation windows (NOT 1 embedding/message)
  embed.py                  # local embeddings (Qwen3-0.6B default, bge-m3 alt)
  index_build.py            # build/list/load/delete numpy indexes (library of Discords)
  search.py                 # cosine top-k (normalized vectors -> dot product)
  synthesize.py             # bounded LLM synthesis, pluggable backend (gemini/ollama)
data/                       # JSON exports (gitignored except sample_export.json)
index/                      # generated index library (gitignored)
  <guild_id>_<channel_id>/  #   one subfolder per Discord: embeddings.npy + chunks.json + meta.json
```
`index_build` auto-migrates a legacy flat index (files directly under `index/`) into the
per-Discord subfolder layout on first `list_indexes()` call.

## Key decisions (and why)
- **Input = exported JSON**, no Discord API in the code: decoupled POC, no Discord
  secrets, no ToS risk inside the app itself.
- **Conversation windows** instead of 1 embedding/message: on Discord the info is
  spread across a Q->A thread; chunking in isolation loses context.
- **Keep** short messages and questions (do NOT filter them out): useful niche info.
- **Citations** via rebuilt Discord jump-links (`parse.message_link`).
- **Local embeddings** (free, multilingual, private) + **numpy brute-force**
  (no Milvus/vector DB: unnecessary at this volume).
- **Gemini free tier** by default (no credit card); **Ollama** local as a private alt.
- Intentional divergences vs the reference project `github.com/Jeet-Chugh/ask-discord`.

## Commands
```bash
streamlit run app.py                       # launch the UI
python -c "from discord_answerer import index_build as i; print(i.build_index('data/sample_export.json'))"  # index via CLI
```

## Environment
- Targets **Python 3.12+** (3.14 also works; PyTorch ships cp314 wheels).
- Available GPU: RTX 4060 Ti 16 GB. Installed torch is CPU-only by default; reinstall
  the CUDA build for GPU speed.

## Guardrails for future changes
- Any change to `synthesize.py` must **preserve** the `NOT_FOUND_MESSAGE` fallback and
  never enable a web search tool (nor Gemini's Google Search grounding).
- Keep the embedding-model swap trivial (a single `DA_EMBED_MODEL` variable).

## Hover tooltip on cited sources (DONE)
Implemented in `app.py` as a custom-CSS **hover card** (`.da-tt` / `.da-tt-box`, injected
once). It reveals the full source message when hovering over inline `[Message N]`
citations in the LLM answer and over the entries in the source expander. UI-only. Helpers:
`_tt_text` (HTML-escape + neutralize markdown + `\n`→`<br>`), `_tt`, `_answer_with_tooltips`
(whole answer is HTML-escaped first so Discord HTML can't inject; markdown links left
untouched via negative lookahead). Known limit: a citation at the very top of an
`overflow:hidden` container can clip the box — move it below the citation if it bites.

## Possible next steps
- Drag & drop currently writes the upload to a temp file then calls `build_index`; the
  `data/` path-based flow is gone from the UI (CLI build still works).
- Bigger ideas: automated fetching / incremental sync, reranking, multi-channel per guild.
