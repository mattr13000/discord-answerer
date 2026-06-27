"""Streamlit interface — Discord Answerer.

Run with:  streamlit run app.py   ->  http://localhost:8501

Two modes:
- Raw: shows the raw messages that matched (score + Discord link). No LLM,
  no API key required. Ideal to validate/debug retrieval.
- LLM: strictly bounded synthesis (Gemini free tier by default, or local Ollama).
"""

import html
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from discord_answerer import config, index_build
from discord_answerer import search as search_mod
from discord_answerer import synthesize as synth_mod

st.set_page_config(page_title="Discord Answerer", page_icon="🎮", layout="wide")

# --- Hover tooltip ("hover card") styling, injected once per run ---
# UI-only: reveals the full text of a cited source message in a box positioned
# above the link/citation on hover. <span> is an inline tag (not an HTML block),
# so Streamlit still parses the surrounding markdown.
_TOOLTIP_CSS = """
<style>
.da-tt { position: relative; display: inline-block; border-bottom: 1px dotted; cursor: help; }
.da-tt .da-tt-box {
    visibility: hidden; opacity: 0;
    position: absolute; left: 0; bottom: 135%; z-index: 1000;
    width: max-content; max-width: 460px; max-height: 320px; overflow-y: auto;
    padding: 10px 12px; border-radius: 8px;
    background: #1f2030; color: #e6e6e6;
    border: 1px solid #3a3b52; box-shadow: 0 6px 22px rgba(0, 0, 0, .4);
    font-size: .85rem; line-height: 1.45; font-weight: 400;
    white-space: normal; text-align: left;
    transition: opacity .12s ease-in-out;
}
.da-tt:hover .da-tt-box { visibility: visible; opacity: 1; }
.da-tt .da-tt-box::after {
    content: ""; position: absolute; top: 100%; left: 18px;
    border: 6px solid transparent; border-top-color: #1f2030;
}
</style>
"""
st.markdown(_TOOLTIP_CSS, unsafe_allow_html=True)


def _tt_text(message: str) -> str:
    """Escape a Discord message for safe display inside the tooltip box.

    The tooltip HTML is injected into a string that Streamlit still parses as
    inline markdown, so we both HTML-escape and backslash-neutralize markdown
    metacharacters (otherwise `*`, `_`, `[`… would be reinterpreted), then turn
    newlines into <br>.
    """
    safe = html.escape(message)
    safe = re.sub(r"([\\`*_~\[\]])", r"\\\1", safe)
    return safe.replace("\n", "<br>")


def _tt(label_html: str, message: str) -> str:
    """Wrap a (trusted) label in a hover card revealing `message`."""
    return (
        f'<span class="da-tt">{label_html}'
        f'<span class="da-tt-box">{_tt_text(message)}</span></span>'
    )


def _answer_with_tooltips(answer: str, results) -> str:
    """Add a hover card on each inline `[Message N]` citation in the LLM answer.

    The whole answer is HTML-escaped first so that any raw HTML carried over
    from a Discord message cannot be injected; markdown still renders.
    `[Message N](link)` markdown links are left untouched (negative lookahead).
    """
    esc = html.escape(answer)

    def repl(m):
        n = int(m.group(1))
        if 1 <= n <= len(results):
            return _tt(f"[Message {n}]", results[n - 1]["text"])
        return m.group(0)

    return re.sub(r"\[Message (\d+)\](?!\s*\()", repl, esc)


@st.cache_resource(show_spinner="Loading the index…")
def _load_index():
    return index_build.load_index()


def _clear_cache():
    _load_index.clear()


st.title("🎮 Discord Answerer")
st.caption(
    "Synthesized answers strictly bounded to the messages of an exported Discord. "
    "**0 web, 0 assumption**: if it's not in the Discord, it doesn't exist."
)

# --- Sidebar: ingestion + settings ---
with st.sidebar:
    st.header("⚙️ Settings")

    default_path = str(config.DATA_DIR / "export.json")
    json_path = st.text_input("Export file (DiscordChatExporter JSON)", value=default_path)
    if st.button("📥 (Re)index", use_container_width=True):
        try:
            with st.spinner("Indexing (embeddings)…"):
                meta = index_build.build_index(json_path)
            _clear_cache()
            st.success(f"Indexed: {meta['num_chunks']} chunks / {meta['num_messages']} messages.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Indexing failed: {e}")

    st.divider()
    k = st.slider("Top-k messages", 5, 60, config.DEFAULT_TOP_K, step=5)
    cutoff = st.slider("Similarity cutoff", 0.0, 1.0, float(config.DEFAULT_SCORE_CUTOFF), 0.05)
    backend = st.selectbox(
        "LLM backend (LLM mode)",
        options=["gemini", "ollama"],
        index=0 if config.LLM_BACKEND == "gemini" else 1,
    )

# --- Index required ---
if not index_build.index_exists():
    st.info("No index yet. Enter a JSON export in the sidebar and click **(Re)index**.")
    st.stop()

embeddings, chunks_data, meta = _load_index()
st.caption(
    f"Index: **{meta.get('guild_name','?')}** / #{meta.get('channel_name','?')} — "
    f"{meta['num_chunks']} chunks, model `{meta['embed_model']}`."
)

# --- Search ---
mode = st.radio("Mode", ["LLM (synthesis)", "Raw (raw messages)"], horizontal=True)
question = st.text_input("Your question", placeholder="e.g. what is the best end-game build?")

if question:
    results = search_mod.search(question, embeddings, chunks_data, k=k, cutoff=cutoff)

    if mode.startswith("LLM"):
        if not results:
            st.warning(config.NOT_FOUND_MESSAGE)
        else:
            try:
                with st.spinner(f"Synthesizing via {backend}…"):
                    answer = synth_mod.synthesize(question, results, backend=backend)
                st.markdown("### Answer")
                st.caption("Hover a [Message N] citation to preview its source message.")
                st.markdown(_answer_with_tooltips(answer, results), unsafe_allow_html=True)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
            with st.expander(f"View the {len(results)} source messages used"):
                items = []
                for i, r in enumerate(results, 1):
                    link = html.escape(r["link"], quote=True)
                    items.append(
                        f'{_tt(f"<b>[Message {i}]</b>", r["text"])} · '
                        f'score {r["score"]:.3f} · '
                        f'<a href="{link}" target="_blank">Discord link</a>'
                    )
                st.markdown("<br><br>".join(items), unsafe_allow_html=True)
    else:
        st.markdown(f"### {len(results)} most relevant messages")
        if not results:
            st.warning("No message above the cutoff.")
        for i, r in enumerate(results, 1):
            st.markdown(
                f"**[{i}]** · score **{r['score']:.3f}** · _{r.get('author_span','')}_ · "
                f"[Discord link]({r['link']})"
            )
            st.text(r["text"])
            st.divider()
