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

## Quick start

New here? Five steps to get running. (Each step is detailed further down.)

**1. Prerequisites** — install [Python 3.12+](https://www.python.org/downloads/) (3.14 works too).

**2. Install the app**
```bash
git clone <this-repo> && cd "discord answerer"   # or download the ZIP and open the folder
python -m venv .venv
.venv\Scripts\activate                            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
> First run downloads the embedding model (~1.2 GB), cached locally afterwards.

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
automatically. It runs locally on your GPU/CPU.

> Installed torch is CPU-only by default. For GPU acceleration on an NVIDIA card:
> `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124`

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
> Free tier ≈ 250 req/day for `gemini-2.5-flash` — plenty for personal use.
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
| `DA_GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
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
  fetching, incremental sync, reranking, multi-channel.
