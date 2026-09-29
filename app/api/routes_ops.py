"""Operations routes: liveness, readiness, models, Prometheus metrics."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["ops"])  # /health, /ready, /metrics — always open
models_router = APIRouter(tags=["models"])  # /v1/models — auth applies via /v1/*


@router.get("/health")
def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Readiness: Ollama and ChromaDB both answer. 503 when degraded."""
    ollama_ok = request.app.state.ollama.ping()
    chroma_ok = request.app.state.store.ping()
    ready_ok = ollama_ok and chroma_ok
    return JSONResponse(
        status_code=200 if ready_ok else 503,
        content={
            "status": "ready" if ready_ok else "degraded",
            "checks": {"ollama": ollama_ok, "chroma": chroma_ok},
        },
    )


@models_router.get("/models")
def list_models(request: Request) -> dict:
    """OpenAI-style model list for the configured chat + embedding models."""
    settings = request.app.state.settings
    return {
        "object": "list",
        "data": [
            {"id": settings.OLLAMA_MODEL, "object": "model", "owned_by": "ollama"},
            {
                "id": settings.OLLAMA_EMBED_MODEL,
                "object": "model",
                "owned_by": "ollama",
            },
        ],
    }


@router.get("/metrics")
def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint (no auth)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
