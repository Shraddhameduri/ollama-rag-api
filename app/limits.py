"""Shared slowapi rate limiter.

The limit value is configured at app startup from ``Settings`` via
:func:`configure_rate_limit`; the per-request key is the caller's API key
(hashed) or, when auth is disabled, the client IP address.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from slowapi import Limiter
from slowapi.util import get_remote_address

_rate_limit_per_min = 60


def configure_rate_limit(per_min: int) -> None:
    """Set the requests-per-minute budget (called once in the app factory)."""
    global _rate_limit_per_min
    _rate_limit_per_min = max(1, int(per_min))


def _rate_limit_value() -> str:
    return f"{_rate_limit_per_min}/minute"


def _key_func(request) -> str:  # type: ignore[no-untyped-def]
    api_key = request.headers.get("X-API-Key")
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"api-key:{digest}"
    return get_remote_address(request)


limiter = Limiter(key_func=_key_func)


def rate_limit() -> Callable:  # type: ignore[type-arg]
    """Decorator factory applying the configured per-minute limit."""
    return limiter.limit(_rate_limit_value)
