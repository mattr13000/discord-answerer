# Discord Answerer

Ask a natural-language question about a **very niche** game Discord server
(private-server MMO, old Korean games, FR/Dofus communities…) and get a
**synthesized answer** built **only** from that Discord's messages.

> **Core constraint: 0 web, 0 assumption.** If the info isn't in the Discord, the app
> answers `Not found in the Discord.` — it will never make something up or look
> elsewhere. That's exactly the failure mode of mainstream LLMs on these niches.

Discord's native search is keyword-based: if your question doesn't literally match
any message, it returns nothing. Here, **multilingual semantic search** finds the
relevant messages by *meaning*, then a **strictly bounded LLM** synthesizes an answer
with **citations** back to the source messages.

---

## Requirements (read this first)

This app **runs the embedding model on your own machine** — it is not a thin cloud
client. Concretely:

- **A local embedding model** (`Qwen3-Embedding-0.6B`, ~1.2 GB) is downloaded on first
  run and used to vectorize every message. **This is mandatory** for both modes (Raw and
  LLM) — there is no cloud-embedding fallback.
- **Indexing is the heavy step.** A small server indexes in seconds; a large one
  (tens of thousands of messages) can take a *long* time on CPU. **A CUDA GPU is strongly
  recommended past ~10k messages** — it cuts indexing from long minutes to a couple
  (see the GPU note under [Install](#install)).
- **An LLM key is optional** — only the *LLM / synthesis* mode needs one (free Gemini
  tier, or fully-local Ollama). Raw mode needs no key at all.

> TL;DR: a local model does the search; the cloud LLM (if any) only writes the final
> answer from the messages that local model already found.

---

## Quick start

New here? Five steps to get running. (Each step is detailed further down.)

**1. Prerequisites** — install [Python 3.12+](https://www.python.org/downloads/) (3.14 works too).
An NVIDIA GPU is optional but **strongly recommended for large servers** (see [Install](#install)).

**2. Install the app**
```bash
git clone <this-repo> && cd "discord answerer"   # or download the ZIP and open the folder
python -m venv .venv
.venv\Scripts\activate                            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
> First run downloads the **local embedding model** (~1.2 GB), cached afterwards. It runs
> on **your** machine (GPU if available, else CPU) and is required for every search.

**3. Get a Discord export (JSON)**
- **Someone shared a ready-made export with you?** Just drop the `.json` file into the
  `data/` folder and jump to step 4.
- **Make your own:** use [**DiscordChatExporter**](https://github.com/Tyrrrz/DiscordChatExporter)
  to export a channel as **JSON** into `data/`. (See the ToS note below.)
- **Just want to try it?** A tiny `data/sample_export.json` is bundled — works out of the box.

**4. Add a free LLM key** (only needed for the **LLM / synthesis** mode)
- Get a free key (no credit card) at [**Google AI Studio**](https://aistudio.google.com).
- Copy `.env.example` to `.env` and fill it in:
  ```
  GEMINI_API_KEY=your_key
  ```
- Want it fully offline/private instead? Use [Ollama](https://ollama.com) (set `DA_LLM_BACKEND=ollama`).
- The **Raw** mode (semantic search only) needs **no key at all**.

**5. Run**
```bash
streamlit run app.py
```
Opens http://localhost:8501 → point to your JSON in the sidebar, click **(Re)index**, ask away.

---

## How it works

```
JSON export (DiscordChatExporter)
   -> parse -> chunking into conversation windows
   -> local embeddings (Qwen3-Embedding-0.6B)
   -> numpy index -> cosine top-k search
   -> bounded LLM synthesis (Gemini free tier / local Ollama)
   -> answer + Discord links of the cited messages
```

Two modes in the UI:
- **Raw**: shows the raw messages that matched (score + link). No LLM, no key needed —
  perfect to validate/debug retrieval.
- **LLM**: bounded synthesis with cited sources, or `Not found in the Discord.`

---

# Setup in detail

## Install

Targets **Python 3.12+** (3.14 works too — PyTorch ships `cp314` wheels).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

On first run, the embedding model (~1.2 GB for Qwen3-0.6B) is downloaded and cached
automatically. It then runs **locally** on your GPU (if available) or CPU to vectorize
every message — this indexing step is the compute-heavy part of the app.

### GPU acceleration (recommended for large servers)

`requirements.txt` installs a **CPU-only** torch by default, which works everywhere but is
slow to index big exports (tens of thousands of messages can take a long time). On an
NVIDIA card, swap in the CUDA build for a large speedup (indexing drops from long minutes
to a couple):

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Then check the GPU is seen:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

> The CUDA wheel (~2.5 GB) bundles its own CUDA runtime — no system CUDA toolkit needed.
> The same wheel works on any VRAM size; on a small (e.g. 8 GB) card, keep the embedding
> batch size modest if you want to use the GPU for other things at the same time.

---

## 1) Get the messages (JSON export)

The app does **not** connect to Discord: it reads an exported file.

1. Install [**DiscordChatExporter**](https://github.com/Tyrrrz/DiscordChatExporter).
2. Export the channel as **JSON** to `data/export.json`.

> ⚠️ **ToS warning.** For a server where you're just a member (typical public game
> Discords), exporting uses **your user token**, which is technically a *self-bot*
> against Discord's terms. Use sparingly (no request spam). If you **own/admin** the
> server, you can export via a **bot token** (compliant).

A small synthetic export (`data/sample_export.json`) is included so you can test
right away, without real data.

## 2) LLM key

Default backend: **Gemini** (free tier, no credit card).

1. Create a free key at [**Google AI Studio**](https://aistudio.google.com).
2. Put it in a `.env` file at the project root (copy `.env.example`):
   ```
   GEMINI_API_KEY=your_key
   ```

> The free API key is **independent** of any Gemini app subscription.
> The default `gemini-3.1-flash-lite` has a generous free tier (~25× the request limits of
> `gemini-2.5-flash`) — plenty for personal use.
> Note: on the free tier, Google may use your requests to improve its models.

**Fully local & private alternative (no key):** [Ollama](https://ollama.com).
```bash
ollama pull qwen2.5:14b
# in .env: DA_LLM_BACKEND=ollama
```

> **Raw mode works with no key and no LLM at all.**

## 3) Run

```bash
streamlit run app.py
```
-> http://localhost:8501 — point to your export, click **(Re)index**, ask away.

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DA_EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model (swap: `BAAI/bge-m3`) |
| `DA_LLM_BACKEND` | `gemini` | `gemini` or `ollama` |
| `DA_GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model |
| `GEMINI_API_KEY` | — | Google AI Studio key |
| `DA_OLLAMA_MODEL` | `qwen2.5:14b` | Local Ollama model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |

### Choosing the embedding model
- **Qwen3-Embedding-0.6B** (default): light (~1.2 GB), excellent multilingual (incl. Korean).
- **BAAI/bge-m3**: hybrid dense + lexical, useful for niche jargon (item names).

Test both on your real export and keep whichever retrieves the best messages.

---

## MVP limits / next steps
- Manual ingestion (no live Discord connection). Possible next steps: automated
  fetching, incremental sync, reranking.
