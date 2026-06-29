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
private server). The library now holds **4 channels of that server** (#legend 4434,
#black-plague 2205, #satsujin 1490, #blast-juggler 1169 = 9298 msgs, 1995 chunks total).
Confirmed: Raw retrieval, cross-lingual search (EN/FR/KR), Gemini synthesis with
`[Message N]` citations, and the "Not found" lock (out-of-scope question -> exact fallback).
The Streamlit app runs at
http://localhost:8501. A local `.venv` is set up (Python 3.14, **torch CUDA build —
cu126**, GPU active; see Environment); the user's `GEMINI_API_KEY` lives in `.env`
(gitignored) — or can be pasted directly in the UI. Default search cutoff = 0.35, but the real anti-hallucination guard is the LLM
(out-of-scope queries can still score ~0.46).

**UX pass done** (non-tech friendly): drag & drop upload of the export (`st.file_uploader`),
a **multi-server library** (channels grouped per server — see below), in-UI Gemini key
entry, advanced knobs folded into an expander, plain-language mode labels, and **hover-card
tooltips** revealing a cited message on hover (over inline `[Message N]` citations and the
source list).

**Multi-channel done** (backend fully validated; UI not yet browser-tested — see below):
a *server* groups several channels (one JSON export each, sharing the same `guild.id`).
Retrieval spans all channels by default (the channel of each cited message is shown), with
a "Search in" toggle to narrow to one channel. Grouping never re-embeds: `load_server` just
`np.vstack`-es the already-saved per-channel matrices and tags each chunk with its channel
(from that channel's `meta.json`) at load time. Multi-file drag & drop indexes several
channels at once. (Retrieval scaling was since reworked — see "Staged retrieval done" below;
the old fixed `top_k=60` is gone, replaced by an adaptive pool + reranker + constant `FINAL_K`.)

**Committed (branch `ux+limit-test`):** the multi-channel work + the UX pass + `gemini-3.1`
default are now committed (`a22fa7f`), and `ARCHITECTURE.md` was added (`2b36fcd`). Backend
was validated by script (migration of the 4 flat indexes -> nested layout,
`load_server`/`load_channel`, cross-channel search) and the app boots clean headless — but
**the Streamlit UI itself was NOT clicked through in a browser**. That browser pass + UX
polish is the next job the user flagged ("du boulot sur l'UI a faire"). Things to eyeball:
the server selector + per-channel remove buttons, the "Search in" scope radio (whole server
vs one channel), the multi-file upload progress bar, and the `#channel` tag shown on each
citation/source.

**Staged retrieval done (browser-validated, 2026-06-29; branch `big-context-optimization`).**
The single-stage `DEFAULT_TOP_K = 60` was the scaling bottleneck (one knob doubling as both
candidate handle *and* LLM context; the good chunk got elbowed out at 57k msgs). Replaced by a
3-stage pipeline: **(1)** `search.py` returns an **adaptive candidate pool** sized to the corpus
(`config.pool_size(n) = clamp(POOL_MIN 100, n*POOL_FRACTION 0.05, POOL_MAX 2000)`; ~610 at the
12k-chunk server) — recall. **(2)** `rerank.py` (NEW) runs a **local cross-encoder**
(`BAAI/bge-reranker-v2-m3`, ~568M, multilingual, CUDA+fp16, mirrors `embed.py`) — precision.
**(3)** trims to a **constant `FINAL_K = 12`** for Gemini — tight context, strong lock.
Measured separation on the real server: in-scope chunks rerank to ~0.95, off-topic to ~0.01
(despite near-identical cosine ~0.35–0.65). New config knobs (`DA_*`): `POOL_FRACTION/MIN/MAX`,
`FINAL_K`, `RERANK_ENABLED/MODEL/DEVICE`; `DEFAULT_SCORE_CUTOFF` (0.35) is now only a **coarse
pre-filter** on the pool, not the fine ranking. The Advanced slider is re-semantized as
"Messages used in the answer" (5–25, def 12) = FINAL_K; the pool is internal/adaptive.
`synthesize.py` was **NOT** touched — the 3 locks + exact `NOT_FOUND` fallback are intact; we
just feed it a better packet. Gotcha recorded in code: with sentence-transformers 5.x, set fp16
via `model_kwargs={"torch_dtype": float16}` at load — mutating `.model.half()` after load breaks
CrossEncoder's forward dispatch. Rerank load failures **silently fall back** to cosine order
(no crash, but the precision gain vanishes) → consider pinning ST/transformers in `requirements.txt`.
*Next passes already scoped (deferred): hybrid BM25, per-channel quota + MMR, query
routing/decomposition; and a cheap `rerank_score` floor to reject junk pools before Gemini
(saves an API call on out-of-scope — the 0.95-vs-0.01 gap makes it near-zero-risk).*

**In flight / uncommitted (as of 2026-06-28):** **GPU enablement** — torch swapped to the
cu126 CUDA build and `embed.py` now runs the model on GPU (device auto + fp16, see
Environment), so indexing big exports (the user's target is a 30k+ msg server) is fast.
The README got a clear "Requirements" section (local embedding model is mandatory, GPU
strongly recommended past ~10k msgs) and stale facts fixed (cu124->cu126,
gemini-2.5->3.1). Plus other working-tree edits to `app.py`/`index_build.py`/`synthesize.py`
(branch `ux+limit-test`, not yet reviewed here).

## Architecture
```
app.py                      # Streamlit UI (Raw + LLM modes)
discord_answerer/
  config.py                 # everything overridable via env var; loads .env
  parse.py                  # DiscordChatExporter JSON -> normalized messages
  chunk.py                  # conversation windows (NOT 1 embedding/message)
  embed.py                  # local embeddings (Qwen3-0.6B default, bge-m3 alt); device auto (CUDA+fp16/CPU)
  index_build.py            # build + list_servers/load_server/load_channel + delete (library of servers)
  search.py                 # stage 1: cosine -> adaptive candidate POOL (config.pool_size)
  rerank.py                 # stage 2: local cross-encoder (bge-reranker-v2-m3); CUDA+fp16; safe fallback
  synthesize.py             # stage 3: bounded LLM synthesis of FINAL_K chunks, pluggable backend (gemini/ollama)
data/                       # JSON exports (gitignored except sample_export.json)
index/                      # generated index library (gitignored)
  <guild_id>/               #   one folder per server (guild)
    <channel_id>/           #     one subfolder per channel: embeddings.npy + chunks.json + meta.json
```
`index_build` auto-migrates older layouts (files directly under `index/`, or flat
`index/<guild_id>_<channel_id>/` folders) into the nested `index/<guild_id>/<channel_id>/`
layout on first `list_servers()` call — a pure folder move, no re-embed.

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
- GPU: **RTX 4060 Ti 8 GB**. **torch is now the CUDA build (`2.12.1+cu126`)** on this
  machine — `torch.cuda.is_available()` is True. `embed.py` auto-selects `cuda` (fp16) and
  falls back to CPU elsewhere. To rebuild the env from scratch with GPU:
  `pip install torch==2.12.1+cu126 --index-url https://download.pytorch.org/whl/cu126`
  (the default `requirements.txt` still pulls the CPU build).
- Embedding knobs (config.py): `DA_EMBED_DEVICE` ("" = auto), `DA_EMBED_BATCH_SIZE`
  (default 64; lower to 8-16 to keep VRAM free for other GPU work on the 8 GB card).

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
