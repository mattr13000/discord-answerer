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
user's `GEMINI_API_KEY` lives in `.env` (gitignored). Default search cutoff = 0.35, but
the real anti-hallucination guard is the LLM (out-of-scope queries can still score ~0.46).

## Architecture
```
app.py                      # Streamlit UI (Raw + LLM modes)
discord_answerer/
  config.py                 # everything overridable via env var; loads .env
  parse.py                  # DiscordChatExporter JSON -> normalized messages
  chunk.py                  # conversation windows (NOT 1 embedding/message)
  embed.py                  # local embeddings (Qwen3-0.6B default, bge-m3 alt)
  index_build.py            # build/load numpy index (embeddings.npy + chunks.json)
  search.py                 # cosine top-k (normalized vectors -> dot product)
  synthesize.py             # bounded LLM synthesis, pluggable backend (gemini/ollama)
data/                       # JSON exports (gitignored except sample_export.json)
index/                      # generated index (gitignored)
```

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

## Next feature (planned): hover tooltip on cited source messages
Goal: when hovering over a cited source — the `[Message N]` / "Discord link" entries in
the LLM answer's sources, and/or rows in Raw mode — show the full message text in a
**tooltip** box positioned above the link (richer styled variant = "hover card").

This is a **UI-only** change in `app.py`: no backend change needed. Each search result
already carries everything required — `text`, `link`, `score`, `author_span`,
`message_ids` (see `search.search()`); the answer's sources currently live in an
`st.expander`.

Streamlit implementation notes (simplest first):
- **HTML `title=""`** via `st.markdown(..., unsafe_allow_html=True)`: native browser
  tooltip, plain text only, no styling, position not controllable. Quickest.
- **Custom CSS tooltip**: a `<span class="tt">link<span class="tt-box">message</span></span>`
  with `:hover` CSS to reveal a box above; inject the CSS once via `st.markdown`. Gives
  the desired "box above on hover" look. Must HTML-escape the message text and preserve
  newlines (`<br>` or `white-space: pre-wrap`).
- `st.popover` opens on **click**, not hover — not a match, but a possible fallback.
- For a polished hover card, consider a small custom HTML component / `streamlit-extras`.
