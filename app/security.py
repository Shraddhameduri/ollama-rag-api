"""API-key authentication.

When ``Settings.API_KEY`` is set, every ``/v1/*`` route requires the
``X-API-Key`` header. The comparison is constant-time. When no API key is
configured, authentication is disabled entirely (documented behaviour).
``/health``, ``/ready`` and ``/metrics`` never require a key.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"


async def require_api_key(request: Request) -> str | None:
    """FastAPI dependency enforcing the API key when one is configured."""
    settings = request.app.state.settings
    if not settings.auth_enabled:
        return None
    provided = request.headers.get(API_KEY_HEADER, "")
    if not provided or not secrets.compare_digest(provided, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return provided
