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
import tempfile
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
    """Add a hover card on each message number inside an LLM citation.

    Handles both single (`[Message 1]`) and grouped (`[Message 1, 2, 3]`)
    citations: every *number* becomes its own hover target, so the brackets and
    the "Message(s)" label stay plain text (lighter UI). The whole answer is
    HTML-escaped first so raw HTML from a Discord message can't be injected;
    markdown still renders, and `[Message N](link)` markdown links are left
    untouched (negative lookahead).
    """
    esc = html.escape(answer)

    def _wrap_number(nm):
        n = int(nm.group(0))
        if 1 <= n <= len(results):
            return _tt(nm.group(0), results[n - 1]["text"])
        return nm.group(0)

    def _wrap_citation(m):
        return re.sub(r"\d+", _wrap_number, m.group(0))

    return re.sub(r"\[Messages?\b[^\]]*\](?!\s*\()", _wrap_citation, esc)


def _friendly_error(exc, backend: str) -> str:
    """Turn a raw exception into a calm, actionable message for non-tech users."""
    msg = str(exc)
    if backend == "ollama" and ("Cannot reach Ollama" in msg or "urlopen" in msg.lower()):
        return (
            "Couldn't reach the local Ollama server. Make sure Ollama is running "
            "and the model is installed — or switch the answer engine back to "
            "**Gemini** in Advanced settings."
        )
    if "API key" in msg or "GEMINI_API_KEY" in msg:
        return (
            "The Gemini API key seems missing or invalid. Add a valid free key in "
            "the left panel (step 2) — [get one here](https://aistudio.google.com/apikey)."
        )
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low or "rate limit" in low:
        return (
            "⏳ The free Gemini quota / rate limit was hit. Wait a minute and try "
            "again — or switch the answer engine to **ollama** (fully local, no quota) "
            "in Advanced settings."
        )
    return f"Something went wrong while writing the answer:\n\n{msg}"


@st.cache_data(show_spinner=False, max_entries=512)
def _synthesize_cached(question: str, backend: str, sources: tuple) -> str:
    """Cache LLM answers by (question, backend, retrieved sources).

    Spamming Send, or any rerun (toggling a display option, moving a slider),
    re-executes the script — but an identical request is served from cache
    instead of firing a new, rate-limited and metered, API call. Exceptions are
    not cached, so a 429/quota error still surfaces normally.
    """
    chunks = [{"link": link, "text": text} for link, text in sources]
    return synth_mod.synthesize(question, chunks, backend=backend)


@st.cache_resource(show_spinner="Loading your Discord…")
def _load_index(key: str):
    return index_build.load_index(key)


def _clear_cache():
    _load_index.clear()


st.title("🎮 Discord Answerer")
st.caption(
    "Ask anything about your Discord. Answers come **only** from its messages — "
    "no web, no guessing. If it's not in the Discord, you'll be told so."
)

# --- Sidebar: setup ---
with st.sidebar:
    st.header("Setup")

    libraries = index_build.list_indexes()

    # Step 1 — pick a saved Discord (recall)
    st.markdown("**1 · Your Discords**")
    if not libraries:
        st.caption("None yet — add one below 👇")
    else:
        keys = [m["key"] for m in libraries]
        labels = {
            m["key"]: f'{m.get("guild_name", "?")} · #{m.get("channel_name", "?")}'
            for m in libraries
        }
        default_key = st.session_state.get("active_key", keys[0])
        if default_key not in keys:
            default_key = keys[0]
        active_key = st.selectbox(
            "Active Discord",
            options=keys,
            index=keys.index(default_key),
            format_func=lambda kk: labels[kk],
        )
        st.session_state["active_key"] = active_key
        with st.popover("🗑️ Remove", use_container_width=True):
            st.caption(f"Remove **{labels[active_key]}** from your library?")
            if st.button("Yes, remove it", use_container_width=True):
                index_build.delete_index(active_key)
                _clear_cache()
                st.session_state.pop("active_key", None)
                st.rerun()

    # Step 2 — add a Discord (drag & drop)
    st.markdown("**2 · Add a Discord**")
    uploaded = st.file_uploader(
        "Drop your export (.json)",
        type=["json"],
        help="The JSON file exported with DiscordChatExporter.",
    )
    if uploaded is not None and st.button("📥 Index this export", use_container_width=True):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            with st.spinner("Reading and embedding the messages…"):
                meta = index_build.build_index(tmp_path)
            os.unlink(tmp_path)
            _clear_cache()
            st.session_state["active_key"] = meta["key"]
            st.success(
                f"Indexed {meta['num_messages']} messages from "
                f"**{meta.get('guild_name', '?')}**."
            )
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't index this file: {e}")

    # Step 3 — Gemini key (needed for AI answers only)
    st.markdown("**3 · AI answer key**")
    has_env_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    key_input = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Already set ✓" if has_env_key else "Paste your free key here",
        help="Free, no credit card. Needed only for AI answers — not for browsing.",
    )
    if key_input:
        os.environ["GEMINI_API_KEY"] = key_input.strip()
    key_ready = has_env_key or bool(key_input)
    if not key_ready:
        st.caption("🔑 No key yet → [get a free one](https://aistudio.google.com/apikey)")

    with st.expander("⚙️ Advanced settings"):
        k = st.slider(
            "Messages to retrieve", 5, 60, config.DEFAULT_TOP_K, step=5,
            help="How many Discord messages to pull in before answering.",
        )
        cutoff = st.slider(
            "Match strictness", 0.0, 1.0, float(config.DEFAULT_SCORE_CUTOFF), 0.05,
            help="Higher = keep only very close matches.",
        )
        backend = st.selectbox(
            "Answer engine", options=["gemini", "ollama"],
            index=0 if config.LLM_BACKEND == "gemini" else 1,
            help="gemini = free cloud (needs a key). ollama = fully local (needs Ollama running).",
        )
        show_scores = st.checkbox(
            "Show match scores", value=False,
            help="Display the technical relevance score on each message.",
        )

# --- Onboarding until a Discord is added ---
if not libraries:
    st.info(
        "👋 **Welcome!** Three steps to get started:\n\n"
        "1. Export a Discord channel to JSON with **DiscordChatExporter**.\n"
        "2. In the left panel, **drop the .json file** and click **Index this export**.\n"
        "3. For AI answers, paste a free Gemini API key — then ask your question below."
    )
    st.stop()

active_key = st.session_state.get("active_key") or libraries[0]["key"]
embeddings, chunks_data, meta = _load_index(active_key)
st.caption(
    f"📚 Reading from **{meta.get('guild_name', '?')}** · "
    f"#{meta.get('channel_name', '?')}  ·  {meta.get('num_messages', '?')} messages"
)

# --- Ask ---
mode = st.radio(
    "What do you want to do?",
    ["💬 Get an answer", "🔍 Browse messages"],
    horizontal=True,
    captions=["AI summary with cited sources", "Read the raw matching messages"],
)
ask_col, send_col = st.columns([6, 1], vertical_alignment="bottom")
with ask_col:
    question = st.text_input(
        "Your question", placeholder="e.g. what is the best end-game build?"
    )
with send_col:
    st.button(
        "", icon=":material/send:", type="primary",
        use_container_width=True, help="Send (or just press Enter)",
    )

if question:
    with st.spinner("🔎 Searching the Discord…"):
        results = search_mod.search(question, embeddings, chunks_data, k=k, cutoff=cutoff)

    if mode.startswith("💬"):
        if not results:
            st.warning(config.NOT_FOUND_MESSAGE)
        elif backend == "gemini" and not key_ready:
            st.info(
                "🔑 To get an AI answer with **Gemini**, add a free API key in the "
                "left panel (step 2) — no credit card needed, "
                "[get one here](https://aistudio.google.com/apikey).\n\n"
                "Or switch the answer engine to **ollama** in Advanced settings, or use "
                "**🔍 Browse messages** to read the raw matches."
            )
        else:
            try:
                sources = tuple((r["link"], r["text"]) for r in results)
                with st.status(
                    "✍️ Reading the Discord and writing your answer…", expanded=False
                ) as status:
                    answer = _synthesize_cached(question, backend, sources)
                    status.update(label="✅ Answer ready", state="complete")
                st.markdown("### Answer")
                st.caption("Hover a [Message N] citation to preview its source message.")
                st.markdown(_answer_with_tooltips(answer, results), unsafe_allow_html=True)
            except Exception as e:  # noqa: BLE001
                st.error(_friendly_error(e, backend))
            with st.expander(f"View the {len(results)} source messages used"):
                items = []
                for i, r in enumerate(results, 1):
                    link = html.escape(r["link"], quote=True)
                    score = f' · score {r["score"]:.3f}' if show_scores else ""
                    items.append(
                        f'{_tt(f"<b>[Message {i}]</b>", r["text"])}{score} · '
                        f'<a href="{link}" target="_blank">Discord link</a>'
                    )
                st.markdown("<br><br>".join(items), unsafe_allow_html=True)
    else:
        if not results:
            st.warning("No matching message found. Try rephrasing your question.")
        else:
            st.markdown(f"### {len(results)} matching messages")
            for i, r in enumerate(results, 1):
                score = f" · match **{r['score']:.3f}**" if show_scores else ""
                st.markdown(
                    f"**{i}.** _{r.get('author_span', '')}_{score} · "
                    f"[Discord link]({r['link']})"
                )
                st.text(r["text"])
                st.divider()
