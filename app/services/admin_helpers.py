"""Admin helpers — ``@require_admin`` decorator.

Companion to ``auth_helpers.require_auth``. Reads ``users.is_admin`` for
the authenticated user (from ``g.user.uid`` set by ``require_auth``) and
rejects with 403 if the flag is unset.

Usage:
    @bp.get("/admin/...")
    @require_auth
    @require_admin
    def some_admin_endpoint(): ...
"""

import logging
from functools import wraps

from flask import g, jsonify

from app.db.db_connection import get_mysql_connection

logger = logging.getLogger(__name__)


def is_admin(user_id: int) -> bool:
    """Return True iff the user has the is_admin flag set."""
    if not user_id:
        return False
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT is_admin FROM users WHERE id=%s LIMIT 1",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return False
        # mysql-connector returns a tuple here.
        v = row[0] if not isinstance(row, dict) else row.get("is_admin")
        return bool(v)
    except Exception as e:
        logger.error("is_admin lookup failed: %s", e, exc_info=True)
        return False
    finally:
        cur.close()
        conn.close()


def require_admin(fn):
    """Reject the request unless the authenticated user is an admin."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        claims = getattr(g, "user", None) or {}
        uid = claims.get("uid")
        if not uid or not is_admin(uid):
            return jsonify({
                "status": False, "statuscode": 403,
                "message": "Admin access required",
            }), 403
        return fn(*args, **kwargs)
    return wrapper
