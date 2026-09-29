"""Prometheus metrics shared by middleware and routes."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled.",
    ["method", "path", "status"],
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
chat_requests_total = Counter(
    "chat_requests_total",
    "Chat completion requests.",
    ["stream"],
)
documents_ingested_total = Counter(
    "documents_ingested_total",
    "Documents successfully ingested.",
)
