"""API tests with a mocked Ollama client (no network, no real Ollama)."""

from __future__ import annotations

import json


def _upload(client, headers, name="doc.md", content=b"# Doc\n\nRAG content here."):
    files = {"file": (name, content, "text/markdown")}
    return client.post("/v1/documents", files=files, headers=headers)


def test_chat_completion_with_rag(client, auth_headers):
    up = _upload(client, auth_headers)
    assert up.status_code == 200
    doc_id = up.json()["doc_id"]

    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "messages": [{"role": "user", "content": "What is this about?"}],
            "use_rag": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "usage" in body and body["usage"]["total_tokens"] > 0
    assert body["sources"], "expected retrieved sources"
    assert body["sources"][0]["doc_id"] == doc_id
    assert "excerpt" in body["sources"][0]
    assert "text" not in body["sources"][0], "raw chunk text must not leak"


def test_chat_without_rag_has_empty_sources(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "use_rag": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_chat_requires_user_message(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions",
        headers=auth_headers,
        json={"messages": [{"role": "system", "content": "hi"}]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


def test_chat_invalid_body_envelope(client, auth_headers):
    resp = client.post(
        "/v1/chat/completions", headers=auth_headers, json={"messages": []}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


def test_chat_streaming_sse(client, auth_headers):
    _upload(client, auth_headers)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        json={
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        events = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert events, "expected SSE data events"
    first = json.loads(events[0][len("data: ") :])
    assert "sources" in first, "sources must be emitted first"
    assert events[-1] == "data: [DONE]"
    deltas = "".join(
        json.loads(e[len("data: ") :])["choices"][0]["delta"]["content"]
        for e in events[1:-1]
    )
    assert deltas == "Hello world"


def test_session_history_roundtrip(client, auth_headers):
    payload = {
        "messages": [{"role": "user", "content": "first question"}],
        "session_id": "sess-1",
    }
    assert client.post("/v1/chat/completions", headers=auth_headers, json=payload).status_code == 200
    memory = client.app.state.memory
    history = memory.get("sess-1", 10)
    assert [t["role"] for t in history] == ["user", "assistant"]


def test_documents_crud(client, auth_headers):
    up = _upload(client, auth_headers, name="notes.txt", content=b"alpha beta gamma")
    assert up.status_code == 200
    body = up.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks"] >= 1 and body["chars"] > 0

    listed = client.get("/v1/documents", headers=auth_headers).json()["documents"]
    assert any(d["doc_id"] == body["doc_id"] for d in listed)

    deleted = client.delete(
        f"/v1/documents/{body['doc_id']}", headers=auth_headers
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == body["doc_id"]

    again = client.delete(f"/v1/documents/{body['doc_id']}", headers=auth_headers)
    assert again.status_code == 404
    assert again.json()["error"]["code"] == "not_found"


def test_upload_rejects_bad_extension(client, auth_headers):
    files = {"file": ("evil.exe", b"MZ...", "application/octet-stream")}
    resp = client.post("/v1/documents", files=files, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


def test_upload_rejects_empty_file(client, auth_headers):
    files = {"file": ("empty.txt", b"   ", "text/plain")}
    resp = client.post("/v1/documents", files=files, headers=auth_headers)
    assert resp.status_code == 400


def test_health_ready_metrics_open(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert b"http_requests_total" in metrics_resp.content


def test_unknown_route_envelope(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    body = resp.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message"}


def test_request_id_header_present(client, auth_headers):
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.headers.get("x-request-id"), "X-Request-ID must be echoed"
