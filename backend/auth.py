"""
backend/auth.py — API key authentication for DocuLingua.

All endpoints except GET /health require a valid X-API-Key header.

Usage in endpoints:
    from backend.auth import verify_api_key
    from fastapi import Depends

    @app.post("/upload", dependencies=[Depends(verify_api_key)])
    async def upload_document(...):
        ...

The key is read from the API_KEY environment variable.  Set it in .env:
    API_KEY=<output of: python -c "import secrets; print(secrets.token_urlsafe(32))">
"""

import os
import secrets
import logging

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger("DocuLingua.Auth")

# FastAPI will extract the header value automatically; auto_error=False lets us
# distinguish "header missing" (403) from "header present but wrong" (401).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency that validates the X-API-Key request header.

    Raises:
        HTTPException 403  — header is absent entirely
        HTTPException 401  — header is present but the value is wrong
        HTTPException 500  — API_KEY env var is not configured on the server

    Returns the validated key string so endpoints can log it if needed.
    """
    expected_key = os.getenv("API_KEY")

    if not expected_key:
        # Server mis-configuration — the operator forgot to set API_KEY.
        logger.error(
            "API_KEY environment variable is not set. "
            "Add API_KEY to .env and restart the service."
        )
        raise HTTPException(
            status_code=500,
            detail="Server authentication is not configured. Contact the administrator.",
        )

    if not api_key:
        raise HTTPException(
            status_code=403,
            detail="Missing X-API-Key header. Include your API key with every request.",
        )

    # Use constant-time comparison to prevent timing-based key discovery.
    if not secrets.compare_digest(api_key, expected_key):
        logger.warning("Rejected request with invalid API key.")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return api_key
