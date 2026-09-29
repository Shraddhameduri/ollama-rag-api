"""Thin Ollama wrapper with retries and timeouts.

Uses the ``ollama`` python client pointed at ``OLLAMA_HOST``. All network
failures are retried with exponential backoff and surfaced as
:exc:`OllamaError` so routes can map them to a 502 without leaking details.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import ollama

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or fails after retries."""


class OllamaClient:
    """Chat + embeddings client for an Ollama server."""

    def __init__(
        self,
        host: str,
        chat_model: str,
        embed_model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._client = ollama.Client(host=host, timeout=timeout)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout
        self.max_retries = max_retries

    def _with_retries(self, fn, operation: str):  # type: ignore[no-untyped-def]
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 — retried, then wrapped
                last_exc = exc
                log.warning(
                    "ollama %s failed (attempt %d/%d): %s",
                    operation,
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                )
                time.sleep(min(2 ** (attempt - 1), 8))
        raise OllamaError(
            f"Ollama {operation} failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> dict[str, str]:
        """Non-streaming chat completion. Returns ``{"content", "model"}``."""

        def _do() -> dict[str, str]:
            resp = self._client.chat(
                model=model or self.chat_model,
                messages=messages,
                stream=False,
                options={"temperature": temperature},
            )
            content = resp.message.content if resp.message else None
            return {"content": content or "", "model": resp.model or (model or self.chat_model)}

        return self._with_retries(_do, "chat")

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Streaming chat completion yielding content deltas.

        A stream cannot be resumed mid-flight, so this is single-attempt:
        failures raise :exc:`OllamaError` for the caller to surface as an
        SSE error event.
        """
        try:
            stream = self._client.chat(
                model=model or self.chat_model,
                messages=messages,
                stream=True,
                options={"temperature": temperature},
            )
            for chunk in stream:
                content = chunk.message.content if chunk.message else None
                if content:
                    yield content
        except Exception as exc:
            raise OllamaError(f"Ollama chat stream failed: {exc}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""

        def _do() -> list[list[float]]:
            try:
                resp = self._client.embed(model=self.embed_model, input=texts)
                return [list(map(float, vec)) for vec in resp.embeddings]
            except AttributeError:
                # Older ollama clients only expose the single-prompt endpoint.
                out: list[list[float]] = []
                for text in texts:
                    single = self._client.embeddings(model=self.embed_model, prompt=text)
                    out.append(list(map(float, single["embedding"])))
                return out

        return self._with_retries(_do, "embed")

    def ping(self) -> bool:
        """True when the Ollama server answers (used by /ready)."""
        try:
            self._client.list()
            return True
        except Exception:  # noqa: BLE001 — ping is best-effort by design
            return False
