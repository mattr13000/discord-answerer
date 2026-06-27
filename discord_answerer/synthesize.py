"""LLM synthesis strictly bounded to the provided messages. Pluggable backend.

Anti-hallucination locks:
1. No search tool is passed to the model (no Google Search grounding; a local
   Ollama LLM has no network access anyway).
2. The model only sees the retrieved Discord chunks.
3. Strict system prompt + exact fallback "Not found in the Discord.".
"""

import json
import os
import urllib.request

from . import config

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about a niche video game "
    "EXCLUSIVELY from the Discord messages provided below.\n"
    "Absolute rules:\n"
    "- Use NO external knowledge, NO assumptions, NO information from the web.\n"
    "- Rely only on the content of the provided messages.\n"
    "- If the requested information is not present in these messages, answer "
    f'EXACTLY and only: "{config.NOT_FOUND_MESSAGE}"\n'
    "- Otherwise, write a clear synthesis, then cite the source messages used "
    "([Message N] numbers and their links).\n"
    "- LANGUAGE: ALWAYS write your answer in the same language as the user's "
    "question, regardless of the language of these instructions or the Discord "
    "messages. (English question -> English answer; French -> French.)"
)


def _format_context(chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        link = c.get("link", "")
        blocks.append(f"[Message {i}] {link}\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks) -> str:
    context = _format_context(chunks)
    return (
        f"Question: {question}\n\n"
        f"=== AVAILABLE DISCORD MESSAGES ===\n{context}\n"
        f"=== END OF MESSAGES ===\n\n"
        "Answer following the rules strictly."
    )


def synthesize(question: str, chunks, backend: str | None = None) -> str:
    backend = backend or config.LLM_BACKEND
    if not chunks:
        return config.NOT_FOUND_MESSAGE

    user = build_prompt(question, chunks)
    if backend == "gemini":
        return _gemini(user)
    if backend == "ollama":
        return _ollama(user)
    raise ValueError(f"Unknown LLM backend: {backend!r}")


def _gemini(user: str) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Get a free key (no credit card) at "
            "https://aistudio.google.com and put it in .env"
        )
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            # Deliberately NO tool -> no Google Search grounding -> no web access.
        ),
    )
    return (resp.text or "").strip()


def _ollama(user: str) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_HOST}. "
            "Is Ollama running and the model pulled (ollama pull ...)?"
        ) from e
    return (data.get("message", {}).get("content") or "").strip()
