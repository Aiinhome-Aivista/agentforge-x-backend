"""Auth helpers: password hashing + @require_auth Flask decorator.

Pure additive — no existing route signatures are touched. New routes opt
into auth by decorating themselves with ``@require_auth``.
"""

import logging
from functools import wraps
from typing import Optional

import bcrypt
from flask import request, jsonify, g

from app.services import jwt_service

logger = logging.getLogger(__name__)


# ── Password hashing ─────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    if not plain or not hashed:
        return False
    try:
        # Legacy plaintext rows (the original users.password column held
        # plaintext) — accept exact match so we don't lock anyone out.
        if not hashed.startswith(("$2a$", "$2b$", "$2y$")):
            return plain == hashed
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as e:
        logger.warning("Password verify error: %s", e)
        return False


# ── JWT extraction ───────────────────────────────────────────────────────────
def _extract_token() -> Optional[str]:
    h = request.headers.get("Authorization", "")
    if h.lower().startswith("bearer "):
        return h.split(" ", 1)[1].strip()
    # Fallback: token query param (avoid using in production, useful for SSE)
    return request.args.get("token")


def current_user() -> Optional[dict]:
    """Return decoded claims, or None if not authenticated."""
    tok = _extract_token()
    if not tok:
        return None
    return jwt_service.decode_safe(tok)


def require_auth(fn):
    """Reject the request if the caller has no valid JWT."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        claims = current_user()
        if not claims:
            return jsonify({
                "status": False, "statuscode": 401,
                "message": "Authentication required"
            }), 401
        g.user = claims
        return fn(*args, **kwargs)
    return wrapper
