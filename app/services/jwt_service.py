"""JWT issuance + verification.

Uses HS256 with FLASK_SECRET_KEY (or JWT_SECRET_KEY if explicitly set).
Tokens carry: sub (user_id), email, plan, exp, iat.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt  # PyJWT

logger = logging.getLogger(__name__)

ALGO = "HS256"
DEFAULT_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "24"))


def _secret() -> str:
    return (
        os.getenv("JWT_SECRET_KEY")
        or os.getenv("FLASK_SECRET_KEY")
        or "change-me-in-production"
    )


def issue(user_id: int, email: str, name: str, plan: Optional[str] = None, 
          ttl_hours: Optional[int] = None) -> str:
    """Issue an access token for a verified user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "uid": user_id,
        "email": email,
        "name": name,
        "plan": plan or "free",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=ttl_hours or DEFAULT_TTL_HOURS)).timestamp()),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGO)


def decode(token: str) -> dict:
    """Decode and validate a token. Raises jwt exceptions on failure."""
    return jwt.decode(token, _secret(), algorithms=[ALGO])


def decode_safe(token: str) -> Optional[dict]:
    """Decode without raising — returns None on any failure."""
    try:
        return decode(token)
    except jwt.PyJWTError as e:
        logger.debug("JWT decode failed: %s", e)
        return None
