"""API-key authentication. Every endpoint except GET /health requires it."""

from __future__ import annotations

from fastapi import Header, HTTPException, status


def require_api_key(x_api_key: str | None, expected: str) -> str:
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY is not configured",
        )
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return x_api_key


def api_key_dep(expected: str):
    async def _dep(x_api_key: str | None = Header(default=None)) -> str:
        return require_api_key(x_api_key, expected)

    return _dep
