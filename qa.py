"""
qa.py — Generate grounded answers with source citations using Claude.
Supports streaming and multi-turn history with prompt caching.
"""
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are an expert assistant for industrial plant floor supervisors.
Answer questions using ONLY the provided document excerpts.
Be precise and operational — supervisors need actionable answers.
Always cite which source document(s) you drew from using [Source N] notation inline.
If the answer cannot be found in the provided excerpts, say so clearly rather than guessing.
Do not add information not present in the sources."""

_ANSWER_SUFFIX = (
    "Provide a clear, direct answer. Use [Source N] citations inline where relevant. "
    "If multiple sources agree, synthesize them. If sources conflict, note the discrepancy."
)

_NO_DOCS_MSG = (
    "No relevant documents were found for your question. "
    "Please try rephrasing or check that the topic is covered in the available manuals."
)


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source {i}: {chunk['source']}, chunk {chunk['chunk_index']}]"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def get_sources(chunks: list[dict]) -> list[str]:
    seen: set = set()
    sources = []
    for chunk in chunks:
        src = chunk["source"]
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def _build_messages(question: str, context: str, history: list[dict]) -> list[dict]:
    messages = []
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"DOCUMENT EXCERPTS:\n{context}",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"QUESTION: {question}\n\n{_ANSWER_SUFFIX}",
            },
        ],
    })
    return messages


def answer_question_stream(
    question: str,
    chunks: list[dict],
    client: anthropic.Anthropic,
    history: list[dict] | None = None,
):
    if not chunks:
        yield _NO_DOCS_MSG
        return

    context = format_context(chunks)
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=_build_messages(question, context, history or []),
    ) as stream:
        yield from stream.text_stream


def answer_question(
    question: str,
    chunks: list[dict],
    client: anthropic.Anthropic,
    history: list[dict] | None = None,
) -> dict:
    if not chunks:
        return {"answer": _NO_DOCS_MSG, "sources": []}

    context = format_context(chunks)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=_build_messages(question, context, history or []),
    )
    return {
        "answer": response.content[0].text.strip(),
        "sources": get_sources(chunks),
    }
