"""POST /v1/chat/completions — OpenAI-compatible chat with RAG + SSE streaming."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import metrics
from ..limits import rate_limit
from ..ollama_client import OllamaError
from ..rag.prompts import build_rag_prompt

router = APIRouter(tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_k: int = Field(default=4, ge=0, le=20)
    session_id: str | None = None
    use_rag: bool = True


def _last_user_message(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _public_sources(chunks: list[dict]) -> list[dict]:
    """Strip raw chunk text; the API only exposes excerpts."""
    return [
        {
            "rank": c["rank"],
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "chunk_index": c["chunk_index"],
            "score": c["score"],
            "excerpt": c["excerpt"],
        }
        for c in chunks
    ]


@router.post("/chat/completions")
@rate_limit()
async def chat_completions(body: ChatRequest, request: Request):  # type: ignore[no-untyped-def]
    settings = request.app.state.settings
    pipeline = request.app.state.pipeline
    memory = request.app.state.memory
    ollama = request.app.state.ollama

    query = _last_user_message(body.messages)
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must include at least one user message.",
        )

    chunks: list[dict] = []
    if body.use_rag and body.top_k > 0:
        try:
            chunks = pipeline.retrieve(query, top_k=body.top_k)
        except OllamaError:
            raise HTTPException(
                status_code=502, detail="Retrieval upstream unavailable."
            ) from None
    sources = _public_sources(chunks)

    system_prompt = build_rag_prompt(chunks)
    history = (
        memory.get(body.session_id, settings.HISTORY_WINDOW) if body.session_id else []
    )
    model_messages = [{"role": "system", "content": system_prompt}]
    model_messages.extend(history)
    model_messages.extend(m.model_dump() for m in body.messages)
    model = body.model or settings.OLLAMA_MODEL
    metrics.chat_requests_total.labels(stream=str(body.stream).lower()).inc()

    if body.stream:
        return StreamingResponse(
            _stream_events(ollama, model_messages, model, body, memory, query, sources),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    try:
        result = ollama.chat(
            model_messages, model=model, temperature=body.temperature
        )
    except OllamaError:
        raise HTTPException(
            status_code=502, detail="LLM upstream unavailable."
        ) from None

    content = result["content"]
    if body.session_id:
        memory.add(body.session_id, "user", query)
        memory.add(body.session_id, "assistant", content)

    prompt_text = system_prompt + "".join(m["content"] for m in model_messages[1:])
    prompt_tokens = _estimate_tokens(prompt_text)
    completion_tokens = _estimate_tokens(content)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "sources": sources,
    }


def _stream_events(  # type: ignore[no-untyped-def]
    ollama, model_messages, model, body: ChatRequest, memory, query: str, sources: list[dict]
) -> Iterator[str]:
    """SSE generator: sources first, then content deltas, then [DONE]."""
    yield f"data: {json.dumps({'sources': sources})}\n\n"
    parts: list[str] = []
    try:
        for piece in ollama.chat_stream(
            model_messages, model=model, temperature=body.temperature
        ):
            parts.append(piece)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n"
    except OllamaError:
        yield (
            "data: "
            + json.dumps(
                {
                    "error": {
                        "code": "upstream_error",
                        "message": "LLM upstream unavailable.",
                    }
                }
            )
            + "\n\n"
        )
    else:
        if body.session_id:
            memory.add(body.session_id, "user", query)
            memory.add(body.session_id, "assistant", "".join(parts))
    yield "data: [DONE]\n\n"
