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
from discord_answerer import query as query_mod
from discord_answerer import rerank as rerank_mod
from discord_answerer import synthesize as synth_mod

st.set_page_config(page_title="Discord Answerer", page_icon="🎮", layout="wide")

# --- Hover tooltip ("hover card") styling, injected once per run ---
# UI-only: reveals the full text of a cited source message in a box positioned
# above the link/citation on hover. <span> is an inline tag (not an HTML block),
# so Streamlit still parses the surrounding markdown.
# Palette follows the active Streamlit theme (light/dark) so the card never
# clashes with the page. `st.context.theme` is available since Streamlit 1.46.
_theme = getattr(getattr(st, "context", None), "theme", None)
_is_light = getattr(_theme, "type", "dark") == "light"
_TT_BG, _TT_FG, _TT_BORDER = (
    ("#ffffff", "#1f2030", "#d3d3df") if _is_light else ("#1f2030", "#e6e6e6", "#3a3b52")
)
# Token-replaced (not an f-string) so the CSS braces stay literal.
_TOOLTIP_CSS = """
<style>
.da-tt { position: relative; display: inline-block; border-bottom: 1px dotted; cursor: help; }
.da-tt .da-tt-box {
    visibility: hidden; opacity: 0;
    position: absolute; left: 0; bottom: 135%; z-index: 1000;
    width: max-content; max-width: 460px; max-height: 320px; overflow-y: auto;
    padding: 10px 12px; border-radius: 8px;
    background: __BG__; color: __FG__;
    border: 1px solid __BORDER__; box-shadow: 0 6px 22px rgba(0, 0, 0, .4);
    font-size: .85rem; line-height: 1.45; font-weight: 400;
    white-space: normal; text-align: left;
    transition: opacity .12s ease-in-out;
}
.da-tt:hover .da-tt-box { visibility: visible; opacity: 1; }
.da-tt .da-tt-box::after {
    content: ""; position: absolute; top: 100%; left: 18px;
    border: 6px solid transparent; border-top-color: __BG__;
}
</style>
""".replace("__BG__", _TT_BG).replace("__FG__", _TT_FG).replace("__BORDER__", _TT_BORDER)
st.markdown(_TOOLTIP_CSS, unsafe_allow_html=True)

# --- Ask/answer panel: a solid filled box that visually separates the question +
# answer zone from the scope selector sitting above it. Theme-agnostic (neutral
# semi-transparent fill, readable on both light and dark Streamlit themes).
_PANEL_CSS = """
<style>
.st-key-ask_panel {
    background: rgba(130, 130, 150, .08);
    border: 1px solid rgba(130, 130, 150, .20);
    border-radius: 12px;
    padding: 18px 22px;
}
/* Keep the answer + results at a comfortable reading width on wide layout. */
.st-key-answer_box { max-width: 860px; }
/* The "Answer ready" status is an st.status (an expander under the hood) used
   purely as a spinner→checkmark indicator — its body is always empty. Hide the
   chevron and kill the toggle so it reads as a plain status line, not a dropdown.
   The spinner/check icon has its own testid (stExpanderIcon*) and is untouched. */
.st-key-answer_status summary [data-testid="stIconMaterial"] { display: none; }
.st-key-answer_status summary { cursor: default; pointer-events: none; list-style: none; }
.st-key-answer_status summary::-webkit-details-marker { display: none; }
/* Drop the built-in white "complete" check — the label's green ✅ already says it.
   Keep the running spinner (stExpanderIconSpinner) untouched. */
.st-key-answer_status [data-testid="stExpanderIconCheck"] { display: none; }
</style>
"""
st.markdown(_PANEL_CSS, unsafe_allow_html=True)


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


def _display_score(r) -> float:
    """The score that actually drove the ordering: the cross-encoder's
    `rerank_score` when present, else the stage-1 cosine `score` (fallback path).
    Showing cosine while ranking by rerank would print numbers that contradict
    the order the user sees."""
    return r.get("rerank_score", r.get("score", 0.0))


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
            "the left panel (step 3) — [get one here](https://aistudio.google.com/apikey)."
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


@st.cache_data(show_spinner=False, max_entries=256)
def _retrieve_cached(question: str, guild_id: str, scope: str | None, k: int, cutoff: float):
    """Cache retrieval (stage 1 search + stage 2 rerank) by its inputs.

    Like _synthesize_cached, this shields an answer already on screen from
    Streamlit's rerun-on-every-click model. The cross-encoder rerank runs on the
    GPU and is the slow step; without this, clicking the scope radio or nudging a
    slider re-ran it every time (freezing the page) only to reproduce the
    identical frozen result. Cached here, those clicks are instant cache hits and
    configure only the *next* question.
    """
    a_emb, a_chunks, _ = _load_scope(guild_id, scope)
    return query_mod.retrieve(question, a_emb, a_chunks, k=k, cutoff=cutoff)


@st.cache_resource(show_spinner="Loading your Discord…")
def _load_scope(guild_id: str, channel_id: str | None):
    """Load the active scope: the whole server, or a single channel.

    `channel_id is None` -> stack every channel of the server (np.vstack of the
    already-saved matrices, no recompute); otherwise load just that channel.
    """
    if channel_id is None:
        return index_build.load_server(guild_id)
    return index_build.load_channel(guild_id, channel_id)


def _clear_cache():
    _load_scope.clear()
    # Retrieval is cached by (question, scope, …); after a re-index or delete its
    # chunks can be stale, so drop it alongside the loaded matrices.
    _retrieve_cached.clear()


st.title("🎮 Discord Answerer")
st.caption(
    "Ask anything about your Discord. Answers come **only** from its messages — "
    "no web, no guessing. If it's not in the Discord, you'll be told so."
)

# --- Sidebar: setup ---
with st.sidebar:
    st.header("Setup")

    servers = index_build.list_servers()

    # Step 1 — pick a saved server (recall)
    st.markdown("**1 · Your servers**")
    if not servers:
        st.caption("None yet — add one below 👇")
    else:
        gids = [s["guild_id"] for s in servers]
        by_gid = {s["guild_id"]: s for s in servers}
        labels = {
            s["guild_id"]: f'{s.get("guild_name") or "?"}  ({s["num_channels"]} ch.)'
            for s in servers
        }
        default_gid = st.session_state.get("active_key", gids[0])
        if default_gid not in gids:
            default_gid = gids[0]
        active_key = st.selectbox(
            "Active server",
            options=gids,
            index=gids.index(default_gid),
            format_func=lambda g: labels[g],
        )
        st.session_state["active_key"] = active_key

        srv = by_gid[active_key]
        with st.expander(f"📂 {srv['num_channels']} channels · {srv['num_messages_total']} msgs"):
            for ch in srv["channels"]:
                col_a, col_b = st.columns([5, 1], vertical_alignment="center")
                col_a.caption(f"#{ch.get('channel_name', '?')} · {ch.get('num_messages', '?')} msgs")
                if col_b.button("🗑️", key=f"del_{ch.get('channel_id')}", help="Remove this channel", use_container_width=True):
                    index_build.delete_channel(active_key, ch.get("channel_id"))
                    _clear_cache()
                    st.session_state.pop("active_key", None)
                    st.rerun()
            with st.popover("🗑️ Remove whole server", use_container_width=True):
                st.caption(f"Remove **{srv.get('guild_name') or '?'}** and all its channels?")
                if st.button("Yes, remove it", use_container_width=True):
                    index_build.delete_server(active_key)
                    _clear_cache()
                    st.session_state.pop("active_key", None)
                    st.rerun()

    # Step 2 — add channels (drag & drop, one JSON per channel). Once at least one
    # server exists, fold it into an expander so it stops competing with the
    # daily "just ask" flow; on first launch it stays open for onboarding.
    if servers:
        add_ctx = st.expander("➕ Add channels")
    else:
        st.markdown("**2 · Add channels**")
        add_ctx = st.container()
    with add_ctx:
        uploaded = st.file_uploader(
            "Drop your exports (.json)",
            type=["json"],
            accept_multiple_files=True,
            help="One JSON per channel, exported with DiscordChatExporter. Channels of the "
            "same server are grouped automatically.",
        )
        if uploaded and st.button(f"📥 Index {len(uploaded)} file(s)", use_container_width=True):
            progress = st.progress(0.0)
            done, failures, last_meta = 0, [], None
            for i, up in enumerate(uploaded, 1):
                progress.progress((i - 1) / len(uploaded), text=f"Channel {i}/{len(uploaded)}…")
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                        tmp.write(up.getbuffer())
                        tmp_path = tmp.name
                    last_meta = index_build.build_index(tmp_path)
                    done += 1
                except Exception as e:  # noqa: BLE001
                    failures.append(f"{up.name}: {e}")
                finally:
                    if tmp_path:
                        os.unlink(tmp_path)
            progress.progress(1.0, text="Done")
            _clear_cache()
            if last_meta:
                st.session_state["active_key"] = str(last_meta.get("guild_id", ""))
            if done:
                st.success(f"Indexed {done} channel(s) into **{last_meta.get('guild_name', '?')}**.")
            for f in failures:
                st.error(f"Couldn't index {f}")
            if done:
                st.rerun()

    # Step 3 — Gemini key (needed for AI answers only). Folded once a key is set
    # (env var or already pasted), shown plainly otherwise.
    has_env_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if has_env_key:
        key_ctx = st.expander("🔑 AI key · active ✓")
    else:
        st.markdown("**3 · AI answer key**")
        key_ctx = st.container()
    with key_ctx:
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
            "Messages used in the answer", 5, 25, config.FINAL_K, step=1,
            help="How many of the best-matching Discord messages the answer is built "
            "from. The candidate pool searched beforehand is sized automatically.",
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

# --- Onboarding until a server is added ---
if not servers:
    st.info(
        "👋 **Welcome!** Three steps to get started:\n\n"
        "1. Export your Discord channels to JSON with **DiscordChatExporter** (one file each).\n"
        "2. In the left panel, **drop the .json files** and click **Index** — channels of the "
        "same server are grouped automatically.\n"
        "3. For AI answers, paste a free Gemini API key — then ask your question below."
    )
    st.stop()

active_key = st.session_state.get("active_key") or servers[0]["guild_id"]
active_srv = next((s for s in servers if s["guild_id"] == active_key), servers[0])

# Scope: the whole server, or one of its channels.
WHOLE = "🌐 Whole server"
scope_options = [WHOLE] + [c.get("channel_id") for c in active_srv["channels"]]
scope_labels = {c.get("channel_id"): f"#{c.get('channel_name', '?')}" for c in active_srv["channels"]}
scope = st.radio(
    "Search in",
    options=scope_options,
    format_func=lambda o: WHOLE if o == WHOLE else scope_labels.get(o, f"#{o}"),
    horizontal=True,
)
scope_channel = None if scope == WHOLE else scope
# Live load drives only the caption below; the answer retrieves against the scope
# frozen at submit time (see the ask panel) so changing category never re-prompts.
_, _, meta = _load_scope(active_key, scope_channel)

scope_detail = (
    f"{meta.get('num_channels', '?')} channels"
    if scope_channel is None
    else f"#{meta.get('channel_name', '?')}"
)
st.caption(
    f"📚 Reading from **{meta.get('guild_name', '?')}** · "
    f"{scope_detail} · {meta.get('num_messages', '?')} messages"
)

# --- Ask --- (wrapped in a solid panel that separates it from the scope selector)
ANSWER, BROWSE = "💬 Get an answer", "🔍 Browse messages"
with st.container(key="ask_panel"):
    # Mode stays outside the form so switching it re-renders the persisted
    # question immediately (no need to re-submit).
    mode = st.radio(
        "What do you want to do?",
        [ANSWER, BROWSE],
        horizontal=True,
        captions=["AI summary with cited sources", "Read the raw matching messages"],
    )
    # Single-shot chat look: the current question is shown back as a user bubble
    # and the result as an assistant bubble (no multi-turn memory — each question
    # triggers an independent retrieval). The question persists in session_state
    # so the exchange stays visible across reruns (sliders, mode switch) until a
    # new question is sent.
    question = st.session_state.get("question", "")

    # Input pinned to the TOP of the panel (above the exchange): the user types
    # and the new answer renders right below — no scrolling down past the sources
    # to type, then back up to read. st.chat_input draws the send button inside
    # the text box and submits on Enter or click. Placed inside this container
    # (not the app root) it renders inline here instead of pinned to the page
    # bottom. On submit we stash the text and rerun so the new question's bubbles
    # render immediately, with no one-rerun lag.
    placeholder = "Ask another question…" if question else "e.g. what is the best end-game build?"
    prompt = st.chat_input(placeholder)
    if prompt:
        # Lock the retrieval settings to THIS question. Streamlit reruns on every
        # widget click, so without this snapshot, clicking another category (or
        # nudging an Advanced slider) would re-retrieve against the new scope and
        # re-prompt the LLM on the answer already on screen. Frozen here, those
        # widgets instead configure the *next* question.
        st.session_state["question"] = prompt
        st.session_state["asked"] = {"scope": scope_channel, "k": k, "cutoff": cutoff}
        st.rerun()

    if question:
        with st.container(key="answer_box"):
            with st.chat_message("user"):
                st.markdown(question)

            # Retrieve against the settings frozen when the question was sent, not
            # the live widgets — so toggling scope/sliders afterwards never fires a
            # new LLM call. Falls back to live settings if state is somehow missing.
            asked = st.session_state.get(
                "asked", {"scope": scope_channel, "k": k, "cutoff": cutoff}
            )
            with st.spinner("🔎 Searching the Discord…"):
                # Stage 1: adaptive cosine pool (recall). Stage 2: cross-encoder
                # rerank trims it to the asked FINAL_K best (precision). Cached by
                # (question, scope, k, cutoff) so reruns don't re-fire the GPU.
                results = _retrieve_cached(
                    question, active_key, asked["scope"], asked["k"], asked["cutoff"]
                )
            # Reranker unavailable -> retrieval silently fell back to cosine order.
            # Warn once so the quality drop isn't invisible (see rerank.fallback_active).
            if rerank_mod.fallback_active() and not st.session_state.get("_rerank_warned"):
                st.toast(
                    "⚠️ Reranker unavailable — using basic ranking. Answers may be less precise.",
                    icon="⚠️",
                )
                st.session_state["_rerank_warned"] = True

            with st.chat_message("assistant"):
                if mode == ANSWER:
                    if not results:
                        st.warning(config.NOT_FOUND_MESSAGE)
                    elif backend == "gemini" and not key_ready:
                        st.info(
                            "🔑 To get an AI answer with **Gemini**, add a free API key in the "
                            "left panel — no credit card needed, "
                            "[get one here](https://aistudio.google.com/apikey).\n\n"
                            "Or switch the answer engine to **ollama** in Advanced settings, or use "
                            "**🔍 Browse messages** to read the raw matches."
                        )
                    else:
                        try:
                            sources = tuple((r["link"], r["text"]) for r in results)
                            # Keyed wrapper so the CSS below strips the (always-empty)
                            # expand chevron from this status, keeping just the
                            # spinner→checkmark. Nothing is written into the status body.
                            with st.container(key="answer_status"):
                                with st.status(
                                    "✍️ Reading the Discord and writing your answer…",
                                    expanded=False,
                                ) as status:
                                    answer = _synthesize_cached(question, backend, sources)
                                    status.update(label="✅ Answer ready", state="complete")
                            st.caption("Hover a [Message N] citation to preview its source message.")
                            st.markdown(_answer_with_tooltips(answer, results), unsafe_allow_html=True)
                        except Exception as e:  # noqa: BLE001
                            st.error(_friendly_error(e, backend))
                        with st.expander(f"View the {len(results)} source messages used"):
                            items = []
                            for i, r in enumerate(results, 1):
                                link = html.escape(r["link"], quote=True)
                                score = f' · score {_display_score(r):.3f}' if show_scores else ""
                                chan = r.get("channel_name")
                                chan = f' · <code>#{html.escape(chan)}</code>' if chan else ""
                                items.append(
                                    f'{_tt(f"<b>[Message {i}]</b>", r["text"])}{chan}{score} · '
                                    f'<a href="{link}" target="_blank">Discord link</a>'
                                )
                            st.markdown("<br><br>".join(items), unsafe_allow_html=True)
                else:
                    if not results:
                        st.warning("No matching message found. Try rephrasing your question.")
                    else:
                        st.markdown(f"**{len(results)} matching messages**")
                        for i, r in enumerate(results, 1):
                            score = f" · match **{_display_score(r):.3f}**" if show_scores else ""
                            chan = f" · `#{r['channel_name']}`" if r.get("channel_name") else ""
                            st.markdown(
                                f"**{i}.** _{r.get('author_span', '')}_{chan}{score} · "
                                f"[Discord link]({r['link']})"
                            )
                            st.text(r["text"])
                            st.divider()
