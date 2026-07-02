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

**Hybrid + floor + diversify done (logic-validated 2026-06-30; browser pass pending).** The three
deferred retrieval passes are now shipped, all behind `DA_*` env flags (default ON), with
`synthesize.py` and `app.py` **untouched** (the floor surfaces via app's existing `if not results
-> NOT_FOUND`). **(#1) Junk-pool floor** (`query._below_floor`): if the best `rerank_score` <
`RERANK_FLOOR` (0.05) the pool is off-topic → `retrieve` returns `[]` → exact `NOT_FOUND`, **no
Gemini call**. Only fires when the reranker actually ran (cosine-fallback path has no
`rerank_score`, keeps old behaviour). **(#2) Hybrid BM25** (`lexical.py`, in-repo numpy-only BM25
Okapi, **no new dependency**; memoised per scope by `id(chunks_data)`+len): dense + BM25 fused by
**RRF** in `search.py`; the `0.35` cosine cutoff now filters the **dense side only**, so a
low-cosine exact-token match (item/skill/boss name, "+9 STR") is *rescued* by BM25. **(#3) MMR +
per-channel quota** (`diversify.py`): picks `FINAL_K` from the *whole* reranked pool by
`λ·rel − (1−λ)·max_sim` (λ=0.7, sim via each item's `_row` into the embedding matrix) under a
per-channel cap `ceil(k·0.5)`, relaxed when the pool can't otherwise fill `k` (single-channel
scope). `query.retrieve` now reranks the **whole pool** when diversify is on (was: straight to
`FINAL_K`). New knobs: `DA_RERANK_FLOOR`, `DA_HYBRID`, `DA_RRF_K`, `DA_BM25_K1/B`, `DA_DIVERSITY`,
`DA_MMR_LAMBDA`, `DA_CHANNEL_FRACTION`. Validated by an offline scratchpad script (BM25 ranking,
RRF rescue, MMR de-dup, quota + relaxation, floor, end-to-end `retrieve`) that monkeypatches
`embed_query`/`rerank` so no model loads — **the in-browser pass on the real server is still the
user's to run** (in-scope answer, exact-name BM25 rescue, off-topic instant NOT_FOUND, no
single-channel domination). New knobs are env-only by decision; UI exposure is a later pass.
*Still open: query routing/decomposition; per-corpus BM25 weight tuning. **#5 (dev/"noob"
distribution split) is deferred** to its own pass — see [[distribution-model-two-audiences]].*

**Review-fixes pass + test suite (2026-07-02; branch `many-fixes`).** A code-review sweep
fixed: **(a)** `app.py` now freezes the **guild** into the `asked` snapshot too (switching
the active server with an answer on screen used to re-fire retrieval/LLM against the new
corpus, or crash on a stale channel id); **(b)** `diversify.py` uses **positional relevance**
(incoming-order rank) instead of raw cosine on the reranker-fallback path — raw cosine
silently undid the BM25 rescue by demoting exactly the low-cosine rescued chunks;
**(c)** `lexical.py`'s BM25 memo is now an **identity-pinned LRU** (cached entry holds the
chunk list itself, `is`-checked — a recycled `id()` can no longer produce a stale index);
**(d)** `load_server` **rejects mixed embedding models** across channels with an actionable
error; **(e)** prompt-injection hardening: system prompt + `build_prompt` declare message
content *data, never instructions* (locks intact, only strengthened); **(f)** `_gemini`
raises on an **empty answer** (safety block) instead of rendering a silent blank, with a
friendly UI message; **(g)** a pasted Gemini key lives in `st.session_state`, **not**
`os.environ` (process-global env would leak the key across sessions if ever hosted) —
`synthesize()` grew an optional `api_key` param for this; **(h)** the internal `_row` key is
stripped from `retrieve()` results; **(i)** the cutoff slider tooltip says it's dense-only.
**Tests:** `tests/` (pytest, `requirements-dev.txt`) — 35 tests covering BM25/RRF rescue,
MMR/quota/relaxation/fallback-relevance, floor, parse/chunk, index library incl. the
mixed-model guard, and the exact NOT_FOUND lock; `embed_query`/`rerank` are monkeypatched
(no model loads, <1s) and config knobs are pinned per-test so `DA_*` env can't skew runs.
Browser pass on these UI-visible bits (frozen guild, session key) is still the user's to run.

**In flight / uncommitted (as of 2026-06-28):** **GPU enablement** — torch swapped to the
cu126 CUDA build and `embed.py` now runs the model on GPU (device auto + fp16, see
Environment), so indexing big exports (the user's target is a 30k+ msg server) is fast.
The README got a clear "Requirements" section (local embedding model is mandatory, GPU
strongly recommended past ~10k msgs) and stale facts fixed (cu124->cu126,
gemini-2.5->3.1). Plus other working-tree edits to `app.py`/`index_build.py`/`synthesize.py`
(branch `ux+limit-test`, not yet reviewed here).

## Architecture
```
app.py                      # Streamlit UI (Raw + LLM modes); wraps query.* in st caches
discord_answerer/
  __main__.py               # CLI: python -m discord_answerer {build,servers,ask} (same pipeline, headless)
  query.py                  # pipeline orchestration: retrieve (search->rerank->floor->diversify) + ask (load->retrieve->synthesize)
  config.py                 # everything overridable via env var; loads .env
  parse.py                  # DiscordChatExporter JSON -> normalized messages (sorted; parse_timestamp shared w/ chunk)
  chunk.py                  # conversation windows (NOT 1 embedding/message)
  embed.py                  # local embeddings (Qwen3-0.6B default, bge-m3 alt); device auto (CUDA+fp16/CPU)
  index_build.py            # build + list_servers/load_server/load_channel + delete (library of servers)
  search.py                 # stage 1: dense cosine + BM25, fused by RRF -> adaptive candidate POOL (config.pool_size)
  lexical.py                # stage 1b: in-repo BM25 Okapi (inverted index, numpy-only, memoised); top_n()
  rerank.py                 # stage 2: local cross-encoder (bge-reranker-v2-m3); CUDA+fp16; safe fallback + fallback_active()
  diversify.py              # stage 2b: MMR + per-channel quota -> the diverse, de-duped FINAL_K; select()
  synthesize.py             # stage 3: bounded LLM synthesis of FINAL_K chunks, pluggable backend (gemini/ollama)
tests/                      # pytest suite (models stubbed, <1s; pip install -r requirements-dev.txt)
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
streamlit run app.py                                         # launch the UI
python -m discord_answerer build data/sample_export.json     # index an export (CLI)
python -m discord_answerer servers                           # list indexed servers/channels + ids
python -m discord_answerer ask "best end-game build?" --guild <id> [--channel <id>] [--show-sources]
```
The CLI is a thin wrapper over `query.ask`/`index_build` — same pipeline as the UI,
headless (useful for tests/scripts). `ask` needs a Gemini key unless `--backend ollama`.

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
- Bigger ideas: automated fetching / incremental sync, query routing/decomposition,
  patch-obsolescence (recency/version) filtering, and a dev-vs-"noob" distribution split
  (#5, deferred). (Reranking, multi-channel, hybrid BM25, junk-pool floor and MMR+quota are done.)
