"""Workspace service — encapsulates plan-aware workspace CRUD.

Pure additive module. Nothing in app/core or app/api/routes.py is modified
except for the 2-line blueprint registration in routes.py (mirrors the way
auth_routes / subscription_routes are wired).

A "workspace" is a persisted snapshot of the user's last analyze response
(the JSON dict returned by /api/analyze) plus an optional friendly name.
The number of workspaces a user may keep at any one time is determined by:

    allowed = plan.workspaces  +  sum(extra_workspaces from active subscription)

where:
  * plan.workspaces        →  3 (free) | 10 (basic) | 20 (premium)
  * extra_workspaces       →  +10 per addon packet (Razorpay, ₹500 each)

If the user has no active subscription the free-tier baseline (3) is used,
matching the pricing matrix.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from app.db.db_connection import get_mysql_connection

logger = logging.getLogger(__name__)


# ── Plan limit lookup ────────────────────────────────────────────────────────
def _free_baseline(cur) -> int:
    """Workspace count granted to a user with no active subscription."""
    cur.execute(
        "SELECT workspaces FROM subscription_plans WHERE code='free' LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        return 3
    # row is a dict when called via dict cursor, tuple otherwise.
    return int(row["workspaces"] if isinstance(row, dict) else row[0])


def get_quota(user_id: int) -> Dict[str, Any]:
    """Return the user's workspace quota and current usage.

    Shape:
        {
            "plan_code":         "basic" | "free" | ...,
            "plan_workspaces":   10,
            "extra_workspaces":  20,    # from purchased addon packets
            "allowed":           30,
            "used":              7,
            "remaining":         23,
            "subscription_active": True,
        }
    """
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT s.plan_code, s.extra_workspaces, p.workspaces AS plan_workspaces
              FROM user_subscriptions s
              JOIN subscription_plans p ON p.id = s.plan_id
             WHERE s.user_id=%s AND s.status='active' AND s.period_end > NOW()
             ORDER BY s.id DESC LIMIT 1
            """,
            (user_id,),
        )
        sub = cur.fetchone()

        if sub:
            plan_code = sub["plan_code"]
            plan_workspaces = int(sub["plan_workspaces"] or 0)
            extra = int(sub["extra_workspaces"] or 0)
            sub_active = True
        else:
            plan_code = "free"
            plan_workspaces = _free_baseline(cur)
            extra = 0
            sub_active = False

        allowed = plan_workspaces + extra

        cur.execute(
            "SELECT COUNT(*) AS c FROM workspaces WHERE user_id=%s", (user_id,)
        )
        used = int((cur.fetchone() or {}).get("c") or 0)

        return {
            "plan_code":           plan_code,
            "plan_workspaces":     plan_workspaces,
            "extra_workspaces":    extra,
            "allowed":             allowed,
            "used":                used,
            "remaining":           max(0, allowed - used),
            "subscription_active": sub_active,
        }
    finally:
        cur.close()
        conn.close()


# ── CRUD ─────────────────────────────────────────────────────────────────────
def list_workspaces(user_id: int) -> List[Dict[str, Any]]:
    """Latest-first list of workspaces. Excludes the heavy `analysis_data` blob
    so the listing endpoint stays cheap; use `get_workspace` for the full
    payload."""
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT id, user_id, name, session_id, process_key, user_input,
                   data_size_mb, created_at, updated_at,
                   (analysis_data IS NOT NULL) AS has_analysis
              FROM workspaces
             WHERE user_id=%s
             ORDER BY COALESCE(updated_at, created_at) DESC, id DESC
            """,
            (user_id,),
        )
        return cur.fetchall() or []
    finally:
        cur.close()
        conn.close()


def get_workspace(user_id: int, workspace_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a workspace + its full analysis payload, ownership-checked."""
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            SELECT id, user_id, name, session_id, process_key, user_input,
                   analysis_data, chat_history, data_size_mb, created_at, updated_at
              FROM workspaces
             WHERE id=%s AND user_id=%s LIMIT 1
            """,
            (workspace_id, user_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        # MySQL returns JSON columns as a string in some driver versions and as
        # a Python object in others. Normalise to a Python object.
        ad = row.get("analysis_data")
        if isinstance(ad, (str, bytes, bytearray)):
            try:
                row["analysis_data"] = json.loads(ad)
            except (ValueError, TypeError):
                # Leave as-is if it isn't valid JSON for any reason.
                pass
        
        ch = row.get("chat_history")
        if isinstance(ch, (str, bytes, bytearray)):
            try:
                row["chat_history"] = json.loads(ch)
            except (ValueError, TypeError):
                pass

        return row
    finally:
        cur.close()
        conn.close()


def _coerce_size_mb(analysis_data: Any) -> float:
    """Approximate the size of the analysis payload in MB, for usage tracking."""
    try:
        s = json.dumps(analysis_data or {}, default=str)
        return round(len(s.encode("utf-8")) / (1024 * 1024), 3)
    except Exception:
        return 0.0


def _process_key_from_analysis(analysis_data: Any) -> Optional[str]:
    """Best-effort: pull `process._key` (or similar) out of the analyze blob."""
    if not isinstance(analysis_data, dict):
        return None
    proc = analysis_data.get("process") or {}
    if isinstance(proc, dict):
        for k in ("process_key", "_key", "key", "id"):
            v = proc.get(k)
            if v:
                return str(v)
    for k in ("process_key", "_key"):
        v = analysis_data.get(k)
        if v:
            return str(v)
    return None


class QuotaExceeded(Exception):
    """Raised when a workspace creation would push the user over the plan limit."""
    def __init__(self, quota: Dict[str, Any]):
        super().__init__("Workspace quota exceeded")
        self.quota = quota


def create_workspace(
    user_id: int,
    name: str,
    *,
    session_id: Optional[str] = None,
    user_input: Optional[str] = None,
    analysis_data: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create a workspace, enforcing plan-based quota.

    Returns ``(workspace_row, quota_after)``. Raises :class:`QuotaExceeded`
    if the caller is already at the cap.
    """
    name = (name or "").strip() or f"Workspace {datetime.utcnow():%Y-%m-%d %H:%M}"
    process_key = _process_key_from_analysis(analysis_data)
    size_mb = _coerce_size_mb(analysis_data)
    payload = json.dumps(analysis_data, default=str) if analysis_data is not None else None
    chat_payload = json.dumps(chat_history, default=str) if chat_history is not None else None

    quota_before = get_quota(user_id)
    if quota_before["used"] >= quota_before["allowed"]:
        raise QuotaExceeded(quota_before)

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            INSERT INTO workspaces
                (user_id, name, session_id, process_key, user_input,
                 analysis_data, chat_history, data_size_mb)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, name, session_id, process_key, user_input, payload, chat_payload, size_mb),
        )
        new_id = cur.lastrowid
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    ws = get_workspace(user_id, new_id) or {"id": new_id, "name": name}
    return ws, get_quota(user_id)


def delete_workspace(user_id: int, workspace_id: int) -> bool:
    """Delete a workspace owned by ``user_id``. Returns ``True`` on success,
    ``False`` if the row was not found (or not owned by the caller)."""
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM workspaces WHERE id=%s AND user_id=%s",
            (workspace_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def rename_workspace(user_id: int, workspace_id: int, new_name: str) -> bool:
    new_name = (new_name or "").strip()
    if not new_name:
        return False
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE workspaces SET name=%s WHERE id=%s AND user_id=%s",
            (new_name, workspace_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def update_chat_history(user_id: int, workspace_id: int, chat_history: List[Dict[str, Any]]) -> bool:
    """Updates the chat_history for a specific workspace."""
    chat_payload = json.dumps(chat_history, default=str) if chat_history is not None else None
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE workspaces SET chat_history=%s WHERE id=%s AND user_id=%s",
            (chat_payload, workspace_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def update_workspace_analysis(user_id: int, workspace_id: int, analysis_data: Dict[str, Any]) -> bool:
    """Updates the analysis_data for a specific workspace (used when re-analyzing)."""
    payload = json.dumps(analysis_data, default=str) if analysis_data is not None else None
    size_mb = _coerce_size_mb(analysis_data)
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "UPDATE workspaces SET analysis_data=%s, data_size_mb=%s WHERE id=%s AND user_id=%s",
            (payload, size_mb, workspace_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
