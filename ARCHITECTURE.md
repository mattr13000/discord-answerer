# Architecture — Discord Answerer

A bird's-eye view of the project: the RAG pipeline, the code map, and the
features shipped so far. For conventions, rationale and guardrails, see
[`CLAUDE.md`](CLAUDE.md).

> **One-line summary:** a RAG pipeline **strictly bounded** to an exported
> Discord. Ask a question → it semantically retrieves the relevant messages →
> an LLM synthesizes an answer **only** from them. If the info isn't there, the
> answer is exactly `Not found in the Discord.` — **0 web, 0 assumption.**

---

## 1. The RAG pipeline (how data flows)

Two phases share the same embedding model: **indexing** (offline, once per
Discord export) and **querying** (every question).

```mermaid
flowchart TD
    subgraph INGEST["🗂️ Indexing — once per Discord export"]
        A["DiscordChatExporter<br/>JSON export"] -->|parse.py| B["normalized messages<br/>(id, ts, author, content, reply_to)"]
        B -->|chunk.py| C["conversation windows<br/>12 msgs, 3 overlap,<br/>split on 30-min gaps"]
        C -->|embed.py<br/>Qwen3-Embedding-0.6B| D["L2-normalized vectors"]
        D -->|index_build.py| E[("index/&lt;guild&gt;_&lt;channel&gt;/<br/>embeddings.npy<br/>chunks.json + meta.json")]
    end

    subgraph QUERY["❓ Querying — every question"]
        Q["user question"] -->|embed.py<br/>embed_query + instruction| QV["query vector"]
        QV -->|search.py<br/>cosine = dot product| TOPK["top-k = 30<br/>cutoff ≥ 0.35"]
        E -.loaded.-> TOPK
        TOPK -->|synthesize.py| LLM{"LLM backend<br/>gemini / ollama"}
        LLM -->|grounded| ANS["✅ synthesized answer<br/>+ [Message N] citations"]
        LLM -->|not in context| NF["⛔ Not found in the Discord."]
    end
```

### The 3 anti-hallucination locks (in `synthesize.py`)

The product's core value. **Never weaken these.**

```mermaid
flowchart LR
    L1["🔒 1 · No search tool<br/>passed to the model<br/>(no Google grounding)"]
    L2["🔒 2 · Context =<br/>only the retrieved chunks"]
    L3["🔒 3 · Strict system prompt<br/>+ exact fallback sentence"]
    L1 --- L2 --- L3
```

> Note: the score cutoff (`0.35`) only trims obvious noise — out-of-scope
> queries can still score ~0.46. **The real guard is the LLM**, held by the 3
> locks above.

---

## 2. Code map (who calls who)

```mermaid
flowchart TD
    APP["app.py<br/>Streamlit UI (Raw + LLM modes)"]

    APP --> CFG["config.py<br/>env-overridable settings + .env loader"]
    APP --> IB["index_build.py<br/>build · list · load · delete<br/>(library of Discords)"]
    APP --> SE["search.py<br/>cosine top-k"]
    APP --> SY["synthesize.py<br/>bounded LLM synthesis"]

    IB --> PA["parse.py<br/>JSON → Message / ParsedExport"]
    IB --> CH["chunk.py<br/>conversation windows"]
    IB --> EM["embed.py<br/>local embeddings (lazy torch)"]
    CH --> PA
    SE --> EM
    SY --> GEM["Gemini API<br/>(google-genai)"]
    SY --> OLL["Ollama<br/>(local HTTP)"]

    EM -. "SentenceTransformer<br/>Qwen3 / bge-m3" .-> HF["🤗 model"]

    classDef ext fill:#2d2d2d,stroke:#888,color:#ddd;
    class GEM,OLL,HF ext;
```

| Module | Role | Key entry points |
|---|---|---|
| `app.py` | Streamlit UI — ingestion, library switch, Raw & LLM modes, tooltips | — |
| `config.py` | Central config, all env-overridable; loads `.env` with no dep | constants |
| `parse.py` | DiscordChatExporter JSON → normalized messages | `parse_export`, `message_link` |
| `chunk.py` | Group messages into overlapping conversation windows | `build_chunks` |
| `embed.py` | Local multilingual embeddings (lazy-imports torch) | `embed_documents`, `embed_query` |
| `index_build.py` | Build/list/load/delete the per-Discord index library | `build_index`, `list_indexes`, `load_index`, `delete_index` |
| `search.py` | Encode query, cosine vs. index, return top-k | `search` |
| `synthesize.py` | Bounded LLM synthesis (gemini/ollama), the 3 locks | `synthesize`, `build_prompt` |

### The index library on disk

```
index/                              # gitignored
  <guild_id>_<channel_id>/          # one folder per ingested Discord
    embeddings.npy                  #   the vector matrix
    chunks.json                     #   aligned metadata (one row per vector)
    meta.json                       #   model, guild/channel, counters
```

`index_build.list_indexes()` auto-migrates a legacy flat index (files directly
under `index/`) into this per-Discord layout on first call.

---

## 3. Features shipped

```mermaid
mindmap
  root((Discord<br/>Answerer))
    Core RAG
      Bounded synthesis
      3 anti-hallucination locks
      "Not found" exact fallback
      [Message N] citations w/ jump-links
    Retrieval
      Local multilingual embeddings
      Cross-lingual EN/FR/KR
      Conversation-window chunking
      numpy brute-force cosine
    UX pass non-tech
      Drag and drop upload
      Multi-Discord library + sidebar switch
      In-UI Gemini key entry
      Hover-cards on citations
      Grouped citations Message 1, 2, 3
      Answer caching no re-billed call
      Human error messages 429 / key / Ollama
    Backends
      Gemini free tier default
      Ollama local private alt
      Trivial embed-model swap
```

**Done & validated** on a real export (Echoes of Morroc — 4434 messages → 846
chunks): raw retrieval, cross-lingual search (EN/FR/KR), Gemini synthesis with
citations, and the "Not found" lock on out-of-scope questions.

---

## 4. Next phase (noted, not yet implemented)

1. **Keep leveling up the UX/UI** beyond the non-tech pass already done.
2. **Scale to a bigger target Discord** — a semi-popular game whose knowledge
   lives on its Discord, **45k+ messages** (vs. the 4434-msg test export).

> ⚠️ **New constraint from #2 — patch obsolescence.** The game ships regular
> patches, so old messages can describe outdated mechanics/builds. The pipeline
> will need **recency / version filtering** (time-weighting at search,
> patch-version awareness, or filtering pre-latest-patch messages). On a
> frequently-patched game, **"grounded but obsolete" is a failure mode as bad as
> hallucination.**
