"""System prompt builders for RAG-grounded chat."""

from __future__ import annotations

_BASE_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's question clearly and concisely."
)


def build_rag_prompt(sources: list[dict]) -> str:
    """Build the system prompt, embedding retrieved chunks as numbered context.

    Chunks are numbered ``[1]``, ``[2]`` ... and the model is instructed to
    cite them inline. With no sources, the prompt degrades to a plain
    assistant prompt so the model answers from its own knowledge.
    """
    if not sources:
        return (
            _BASE_INSTRUCTIONS
            + " If you don't know the answer, say so honestly."
        )
    context = "\n\n".join(f"[{s['rank']}] {s['text']}" for s in sources)
    return (
        "You are a helpful assistant. Use the numbered context below to answer "
        "the user's question.\n\n"
        f"<context>\n{context}\n</context>\n\n"
        "Rules:\n"
        "- Ground every factual claim in the context and cite sources inline like [1], [2].\n"
        "- If the context does not contain the answer, say so, then answer from "
        "general knowledge without citations.\n"
        "- Be concise."
    )
