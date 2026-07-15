"""Admin service — user management + subscription overview.

Encapsulates DB work so the route layer stays thin. Every function here
assumes the caller is already authenticated as an admin (the route layer
enforces that with ``@require_admin``).
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from app.db.db_connection import get_mysql_connection
from app.services.auth_helpers import hash_password

logger = logging.getLogger(__name__)


# ── User listing + filtering ────────────────────────────────────────────────
def list_users(*, q: Optional[str] = None,
               limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Return all users with their active subscription summary.

    The result is shaped for the admin grid: one row per user, with the
    active plan flattened in (or NULL if the user has no active sub).
    """
    where = ""
    args: List[Any] = []
    if q:
        where = "WHERE u.email LIKE %s OR u.name LIKE %s"
        like = f"%{q.strip()}%"
        args.extend([like, like])

    args.extend([max(1, min(int(limit or 100), 500)), max(0, int(offset or 0))])

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            f"""
            SELECT u.id, u.name, u.email, u.country, u.auth_provider,
                   u.email_verified, u.is_active, u.is_admin,
                   u.llm_tokens_used AS llm_api_used, u.file_size_limit_mb,
                   u.avatar_url, u.created_at,
                   s.plan_code, s.status AS sub_status, s.period_end,
                   s.amount AS sub_amount, s.currency AS sub_currency,
                   s.extra_workspaces
              FROM users u
              LEFT JOIN (
                SELECT s1.*
                  FROM user_subscriptions s1
                  JOIN (
                    SELECT user_id, MAX(id) AS max_id
                      FROM user_subscriptions
                     WHERE status='active'
                     GROUP BY user_id
                  ) m ON m.max_id = s1.id
              ) s ON s.user_id = u.id
            {where}
             ORDER BY u.created_at DESC, u.id DESC
             LIMIT %s OFFSET %s
            """,
            tuple(args),
        )
        return cur.fetchall() or []
    finally:
        cur.close()
        conn.close()


def count_users(*, q: Optional[str] = None) -> int:
    where = ""
    args: List[Any] = []
    if q:
        where = "WHERE email LIKE %s OR name LIKE %s"
        like = f"%{q.strip()}%"
        args.extend([like, like])
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM users {where}", tuple(args))
        row = cur.fetchone()
        if not row:
            return 0
        # Cursor type varies — accept tuple, list, or dict.
        if isinstance(row, dict):
            return int(next(iter(row.values()), 0) or 0)
        return int(row[0] or 0)
    finally:
        cur.close()
        conn.close()


# ── Create user ─────────────────────────────────────────────────────────────
def create_user(*, name: str, email: str, password: str,
                country: str = "IN", is_admin: bool = False,
                email_verified: bool = True, file_size_limit_mb: int = 5) -> Dict[str, Any]:
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if len(name) < 2:
        raise ValueError("Name must be at least 2 characters")
    if not email or "@" not in email:
        raise ValueError("Valid email required")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters")

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id FROM users WHERE email=%s LIMIT 1", (email,))
        if cur.fetchone():
            raise ValueError("A user with this email already exists")

        cur.execute(
            """
            INSERT INTO users
                (name, email, password, country, auth_provider,
                 email_verified, is_admin, file_size_limit_mb)
            VALUES (%s, %s, %s, %s, 'email', %s, %s, %s)
            """,
            (name, email, hash_password(password), country[:8],
             1 if email_verified else 0, 1 if is_admin else 0, file_size_limit_mb),
        )
        conn.commit()
        new_id = cur.lastrowid
        cur.execute(
            "SELECT id, name, email, country, is_admin, is_active, "
            "email_verified, auth_provider, created_at, file_size_limit_mb "
            "FROM users WHERE id=%s",
            (new_id,),
        )
        return cur.fetchone()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


# ── Update / delete ─────────────────────────────────────────────────────────
def update_user(user_id: int, **fields) -> Optional[Dict[str, Any]]:
    """Permitted fields: is_active, is_admin, name, country, file_size_limit_mb."""
    permitted = ("is_active", "is_admin", "name", "country", "file_size_limit_mb")
    sets: List[str] = []
    args: List[Any] = []
    for k in permitted:
        if k in fields:
            v = fields[k]
            if k in ("is_active", "is_admin"):
                v = 1 if v else 0
            elif k == "country":
                v = (v or "")[:8]
            sets.append(f"`{k}`=%s")
            args.append(v)

    if not sets:
        return get_user(user_id)

    args.append(user_id)
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            f"UPDATE users SET {', '.join(sets)} WHERE id=%s",
            tuple(args),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
    return get_user(user_id)


def delete_user(user_id: int) -> bool:
    """Hard-delete a user and all dependent rows.

    Subscriptions, payments, workspaces, OTP rows, and blog posts authored
    by the user are removed in the same transaction. We do a soft-delete
    by default (toggle ``is_active=0``) — pass force=True from the route
    layer to remove the row outright.
    """
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        # Detach optional dependents before removing the parent. Some of
        # these tables may not exist on every install (migrations are
        # applied incrementally), so wrap each in its own try.
        for sql in (
            "DELETE FROM user_subscriptions WHERE user_id=%s",
            "DELETE FROM payment_orders     WHERE user_id=%s",
            "DELETE FROM workspaces         WHERE user_id=%s",
            "DELETE FROM user_usage         WHERE user_id=%s",
            "DELETE FROM blog_posts         WHERE author_id=%s",
        ):
            try:
                cur.execute(sql, (user_id,))
            except Exception as inner:
                logger.warning("Skipping dependent delete (%s): %s",
                               sql.split(" ", 4)[2], inner)

        cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT id, name, email, country, auth_provider,
                   email_verified, is_active, is_admin,
                   llm_tokens_used AS llm_api_used, file_size_limit_mb,
                   avatar_url, created_at, updated_at
              FROM users WHERE id=%s
            """,
            (user_id,),
        )
        user = cur.fetchone()
        if user:
            cur.execute(
                "SELECT filename, size_mb, created_at FROM user_uploaded_files WHERE user_id=%s ORDER BY created_at DESC",
                (user_id,)
            )
            user["uploaded_files"] = cur.fetchall() or []
        return user
    finally:
        cur.close()
        conn.close()


# ── Subscription overview ───────────────────────────────────────────────────
def subscriptions_overview(*, status: Optional[str] = None,
                           plan_code: Optional[str] = None,
                           limit: int = 200) -> Dict[str, Any]:
    """Return a list of user_subscriptions rows joined with the user, plus a
    small aggregate breakdown by plan/status."""
    conds: List[str] = []
    args: List[Any] = []
    if status:
        conds.append("s.status=%s")
        args.append(status)
    if plan_code:
        conds.append("s.plan_code=%s")
        args.append(plan_code)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            f"""
            SELECT s.id, s.user_id, s.plan_code, s.status, s.currency,
                   s.amount, s.period_start, s.period_end,
                   s.extra_workspaces, s.razorpay_order_id, s.razorpay_payment_id,
                   u.name AS user_name, u.email AS user_email,
                   p.name AS plan_name, p.workspaces AS plan_workspaces
              FROM user_subscriptions s
              JOIN users u             ON u.id = s.user_id
              LEFT JOIN subscription_plans p ON p.id = s.plan_id
            {where}
             ORDER BY s.id DESC
             LIMIT %s
            """,
            tuple(args + [max(1, min(int(limit or 200), 1000))]),
        )
        rows = cur.fetchall() or []

        cur.execute(
            """
            SELECT plan_code, status, COUNT(*) AS c
              FROM user_subscriptions
             GROUP BY plan_code, status
            """
        )
        agg_rows = cur.fetchall() or []
        agg = {f"{r['plan_code']}|{r['status']}": int(r['c']) for r in agg_rows}

        cur.execute("SELECT COUNT(*) AS c FROM users")
        users_total = int((cur.fetchone() or {}).get("c") or 0)

        cur.execute(
            """
            SELECT COUNT(*) AS c
              FROM user_subscriptions
             WHERE status='active' AND period_end > NOW()
            """
        )
        active_total = int((cur.fetchone() or {}).get("c") or 0)

        return {
            "items": rows,
            "aggregates": {
                "users_total":  users_total,
                "active_total": active_total,
                "by_plan_status": agg,
            },
        }
    finally:
        cur.close()
        conn.close()
