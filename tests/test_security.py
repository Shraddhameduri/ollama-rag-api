"""API security tests: X-API-Key auth behaviour."""

from __future__ import annotations


def test_open_routes_need_no_key(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_v1_requires_key_when_configured(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "unauthorized"


def test_wrong_key_rejected(client):
    resp = client.get("/v1/models", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_correct_key_accepted(client, auth_headers):
    resp = client.get("/v1/models", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert any(m["id"] == "llama3.2:1b" for m in data)


def test_auth_disabled_when_no_key_configured(client_no_auth):
    # No X-API-Key header at all, yet /v1/* works.
    resp = client_no_auth.get("/v1/models")
    assert resp.status_code == 200
    resp = client_no_auth.get("/v1/documents")
    assert resp.status_code == 200


def test_upload_without_key_rejected(client):
    files = {"file": ("doc.txt", b"hello", "text/plain")}
    resp = client.post("/v1/documents", files=files)
    assert resp.status_code == 401
