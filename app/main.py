"""FastAPI application factory, middleware, and error handlers."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, metrics
from .api import routes_chat, routes_docs, routes_ops
from .config import Settings, get_settings
from .limits import configure_rate_limit, limiter
from .memory import ConversationMemory
from .ollama_client import OllamaClient
from .rag.pipeline import RagPipeline
from .rag.store import DocumentStore
from .security import require_api_key

log = logging.getLogger("ollama_rag_api")


class JsonFormatter(logging.Formatter):
    """Single-line JSON log records to stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str) -> None:
    """Route structured logs to stdout at the configured level."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


class RequestIDMiddleware:
    """Assign every request a UUID; echo it back as ``X-Request-ID``."""

    def __init__(self, app):  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope["state"]["request_id"] = request_id

        async def send_with_id(message):  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_id)


class AccessLogMiddleware:
    """JSON access log: request_id, method, path, status, latency_ms.

    Bodies and headers are never logged, so API keys and document contents
    stay out of the logs.
    """

    def __init__(self, app):  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status_holder: dict[str, int] = {}

        async def send_and_capture(message):  # type: ignore[no-untyped-def]
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_and_capture)
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            status = status_holder.get("status", 500)
            path = scope.get("path", "-")
            route = scope.get("route")
            path_label = getattr(route, "path", path) or path
            metrics.http_requests_total.labels(
                method=scope.get("method", "-"),
                path=path_label,
                status=str(status),
            ).inc()
            metrics.http_request_duration_seconds.labels(
                method=scope.get("method", "-"), path=path_label
            ).observe(latency_ms / 1000)
            log.info(
                "request",
                extra={
                    "request_id": scope["state"].get("request_id", "-"),
                    "method": scope.get("method", "-"),
                    "path": path,
                    "status": status,
                    "latency_ms": round(latency_ms, 2),
                },
            )


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    502: "upstream_error",
}


async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else code
    return _error(code, message, exc.status_code)


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _error("validation_error", "Invalid request body.", 422)


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return _error("rate_limited", "Rate limit exceeded. Try again later.", 429)


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception(
        "unhandled error",
        extra={"request_id": getattr(request.state, "request_id", "-")},
    )
    # Never leak stack traces or internals to callers.
    return _error("internal_error", "An unexpected error occurred.", 500)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application (pass ``settings`` to override env)."""
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL)
    configure_rate_limit(settings.RATE_LIMIT_PER_MIN)

    app = FastAPI(
        title="ollama-rag-api",
        version=__version__,
        description="Production-grade Chat API with RAG, powered by Ollama.",
    )
    app.state.settings = settings
    app.state.limiter = limiter
    app.state.memory = ConversationMemory()
    app.state.ollama = OllamaClient(
        host=settings.OLLAMA_HOST,
        chat_model=settings.OLLAMA_MODEL,
        embed_model=settings.OLLAMA_EMBED_MODEL,
    )
    app.state.store = DocumentStore(persist_dir=settings.CHROMA_DIR)
    app.state.pipeline = RagPipeline(
        store=app.state.store,
        ollama_client=app.state.ollama,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(AccessLogMiddleware)
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    auth = [Depends(require_api_key)]
    app.include_router(routes_chat.router, prefix="/v1", dependencies=auth)
    app.include_router(routes_docs.router, prefix="/v1", dependencies=auth)
    app.include_router(routes_ops.models_router, prefix="/v1", dependencies=auth)
    app.include_router(routes_ops.router)  # /health, /ready, /metrics stay open

    return app


# Module-level app for uvicorn: `uvicorn app.main:app`
app = create_app()
