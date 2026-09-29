"""Shared pytest fixtures: isolated app with a mocked Ollama client.

No test in this suite touches the network or a real Ollama server:
``OllamaClient.chat`` / ``chat_stream`` / ``embed`` / ``ping`` are patched,
and ChromaDB uses a per-test temporary directory.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app, limiter


def _fake_chat(self, messages, model=None, temperature=0.2):  # type: ignore[no-untyped-def]
    return {"content": "Test answer with citation [1].", "model": model or "test-model"}


def _fake_chat_stream(self, messages, model=None, temperature=0.2):  # type: ignore[no-untyped-def]
    yield "Hello "
    yield "world"


def _fake_embed(self, texts):  # type: ignore[no-untyped-def]
    # Constant vectors -> cosine distance 0 -> similarity 1.0 for every chunk.
    return [[0.1] * 8 for _ in texts]


def _fake_ping(self):  # type: ignore[no-untyped-def]
    return True


@pytest.fixture
def settings(tmp_path):
    return Settings(
        CHROMA_DIR=str(tmp_path / "chroma"),
        API_KEY="test-key-123",
        RATE_LIMIT_PER_MIN=10_000,
    )


@pytest.fixture
def settings_no_auth(tmp_path):
    return Settings(
        CHROMA_DIR=str(tmp_path / "chroma"),
        API_KEY="",
        RATE_LIMIT_PER_MIN=10_000,
    )


@pytest.fixture
def app(settings, monkeypatch):
    monkeypatch.setattr("app.ollama_client.OllamaClient.chat", _fake_chat)
    monkeypatch.setattr("app.ollama_client.OllamaClient.chat_stream", _fake_chat_stream)
    monkeypatch.setattr("app.ollama_client.OllamaClient.embed", _fake_embed)
    monkeypatch.setattr("app.ollama_client.OllamaClient.ping", _fake_ping)
    return create_app(settings)


@pytest.fixture
def app_no_auth(settings_no_auth, monkeypatch):
    monkeypatch.setattr("app.ollama_client.OllamaClient.chat", _fake_chat)
    monkeypatch.setattr("app.ollama_client.OllamaClient.chat_stream", _fake_chat_stream)
    monkeypatch.setattr("app.ollama_client.OllamaClient.embed", _fake_embed)
    monkeypatch.setattr("app.ollama_client.OllamaClient.ping", _fake_ping)
    return create_app(settings_no_auth)


@pytest.fixture
def client(app):
    limiter.enabled = False
    try:
        yield TestClient(app)
    finally:
        limiter.enabled = True


@pytest.fixture
def client_no_auth(app_no_auth):
    limiter.enabled = False
    try:
        yield TestClient(app_no_auth)
    finally:
        limiter.enabled = True


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key-123"}
