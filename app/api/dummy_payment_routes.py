"""Dummy payment routes — activate any plan or addon packet INSTANTLY without
Razorpay, for testing purposes only.

Mounted under /api/billing/dummy via the main api blueprint. New file —
does not modify the existing Razorpay flow in subscription_routes.py.

EVERY endpoint here is gated by the env flag DUMMY_PAYMENTS_ENABLED. When
the flag is unset (the production default), every request returns 403 so
the dummy surface cannot be abused on a live deployment.

Routes:

  GET   /billing/dummy/status                 — public: is dummy mode on?
  POST  /billing/dummy/activate               — auth: instantly activate a plan
                                                 body: {plan_code}
  POST  /billing/dummy/addon                  — auth: instantly grant addon packets
                                                 body: {packets}
  POST  /billing/dummy/cancel                 — auth: cancel the active plan
                                                 (handy to re-test the upgrade flow)

Records inserted into ``payment_orders`` are tagged with status='paid' and
``razorpay_order_id='dummy_<uuid>'`` so they're trivially distinguishable
from real Razorpay rows.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify, g

from app.db.db_connection import get_mysql_connection
from app.services.auth_helpers import require_auth

logger = logging.getLogger(__name__)
dummy_billing_bp = Blueprint("dummy_billing", __name__, url_prefix="/billing/dummy")


def _enabled() -> bool:
    """Master switch — defaults to OFF unless explicitly enabled in env.

    Accepts: '1', 'true', 'yes', 'on' (case-insensitive).
    """
    raw = (os.getenv("DUMMY_PAYMENTS_ENABLED") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})


def _err(message, code=400, **extra):
    return jsonify({"status": False, "statuscode": code, "message": message, **extra}), code


def _guard():
    """Return a 403 response object when dummy mode is off, or None when ok."""
    if not _enabled():
        return _err(
            "Dummy payments are not enabled on this server. "
            "Set DUMMY_PAYMENTS_ENABLED=true to enable for testing.",
            code=403,
        )
    return None


def _fetch_plan(cur, code: str):
    cur.execute(
        "SELECT * FROM subscription_plans WHERE code=%s AND is_active=1",
        (code,),
    )
    return cur.fetchone()


def _active_subscription(cur, user_id: int):
    cur.execute(
        """
        SELECT s.*, p.name AS plan_name, p.datasize_mb, p.workspaces,
               p.has_market_research, p.has_deep_insights, p.features
          FROM user_subscriptions s
          JOIN subscription_plans p ON p.id = s.plan_id
         WHERE s.user_id=%s AND s.status='active' AND s.period_end > NOW()
         ORDER BY s.id DESC LIMIT 1
        """,
        (user_id,),
    )
    return cur.fetchone()


# ── Public: is dummy mode on? ───────────────────────────────────────────────
@dummy_billing_bp.get("/status")
def dummy_status():
    return _ok("OK", data={"enabled": _enabled()})


# ── Auth: instantly activate any plan ───────────────────────────────────────
@dummy_billing_bp.post("/activate")
@require_auth
def dummy_activate():
    g_resp = _guard()
    if g_resp is not None:
        return g_resp

    body = request.get_json(silent=True) or {}
    plan_code = (body.get("plan_code") or "").lower().strip()
    if plan_code not in ("free", "basic", "premium"):
        return _err("plan_code must be one of: free, basic, premium")

    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        plan = _fetch_plan(cur, plan_code)
        if not plan:
            return _err("Plan not found", 404)

        # Mirror the real verify_order flow exactly: expire any prior active
        # sub, then insert a fresh one. The only difference is that the
        # razorpay_* columns carry synthetic dummy identifiers.
        cur.execute(
            """
            UPDATE user_subscriptions SET status='expired'
             WHERE user_id=%s AND status='active'
            """,
            (uid,),
        )

        # Pick the same currency the regular flow would have picked.
        country = (g.user.get("country") or "IN").upper()
        if plan_code == "free":
            currency, amount = "INR", 0.0
        elif country == "IN":
            currency, amount = "INR", float(plan["price_inr"])
        else:
            currency, amount = "USD", float(plan["price_usd"])

        dummy_id = f"dummy_{uuid.uuid4().hex[:12]}"
        notes = {"user_id": uid, "kind": "subscription",
                 "plan_code": plan_code, "_dummy": True}

        # Record a "paid" payment_order for audit symmetry with the real flow.
        cur.execute(
            """
            INSERT INTO payment_orders
                (user_id, plan_id, kind, addon_packets, currency, amount,
                 razorpay_order_id, razorpay_payment_id, razorpay_signature,
                 status, notes)
            VALUES (%s, %s, 'subscription', 0, %s, %s,
                    %s, %s, 'dummy_signature', 'paid', %s)
            """,
            (uid, plan["id"], currency, amount,
             dummy_id, f"pay_{dummy_id}", json.dumps(notes)),
        )

        start = datetime.utcnow()
        end = start + timedelta(days=int(plan["period_days"]))
        cur.execute(
            """
            INSERT INTO user_subscriptions
                (user_id, plan_id, plan_code, status, currency, amount,
                 period_start, period_end,
                 razorpay_order_id, razorpay_payment_id, razorpay_signature)
            VALUES (%s, %s, %s, 'active', %s, %s, %s, %s, %s, %s, 'dummy_signature')
            """,
            (uid, plan["id"], plan_code, currency, amount,
             start, end, dummy_id, f"pay_{dummy_id}"),
        )

        conn.commit()
        sub = _active_subscription(cur, uid)
        return _ok(f"[TEST MODE] Plan '{plan_code}' activated without payment",
                   data=sub, dummy=True)
    except Exception as e:
        conn.rollback()
        logger.error("dummy_activate failed: %s", e, exc_info=True)
        return _err("Could not activate plan in test mode", 500)
    finally:
        cur.close()
        conn.close()


# ── Auth: instantly grant addon packets (+packets*10 workspaces) ────────────
@dummy_billing_bp.post("/addon")
@require_auth
def dummy_addon():
    g_resp = _guard()
    if g_resp is not None:
        return g_resp

    body = request.get_json(silent=True) or {}
    try:
        packets = int(body.get("packets") or 0)
    except (TypeError, ValueError):
        return _err("packets must be an integer")
    if packets < 1 or packets > 100:
        return _err("packets must be between 1 and 100")

    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        sub = _active_subscription(cur, uid)
        if not sub or sub["plan_code"] == "free":
            return _err(
                "Addon packets require an active Basic/Premium plan",
                403,
            )

        plan = _fetch_plan(cur, sub["plan_code"])
        unit = float(plan["addon_packet_inr"] or 0)
        amount = unit * packets

        dummy_id = f"dummy_{uuid.uuid4().hex[:12]}"
        notes = {"user_id": uid, "kind": "addon",
                 "plan_id": plan["id"], "packets": packets, "_dummy": True}

        cur.execute(
            """
            INSERT INTO payment_orders
                (user_id, plan_id, kind, addon_packets, currency, amount,
                 razorpay_order_id, razorpay_payment_id, razorpay_signature,
                 status, notes)
            VALUES (%s, %s, 'addon', %s, 'INR', %s,
                    %s, %s, 'dummy_signature', 'paid', %s)
            """,
            (uid, plan["id"], packets, amount,
             dummy_id, f"pay_{dummy_id}", json.dumps(notes)),
        )

        cur.execute(
            """
            UPDATE user_subscriptions
               SET extra_workspaces = extra_workspaces + (%s * 10)
             WHERE id=%s
            """,
            (packets, sub["id"]),
        )

        conn.commit()
        sub = _active_subscription(cur, uid)
        return _ok(
            f"[TEST MODE] Added {packets * 10} workspaces without payment",
            data=sub, dummy=True,
        )
    except Exception as e:
        conn.rollback()
        logger.error("dummy_addon failed: %s", e, exc_info=True)
        return _err("Could not grant addon in test mode", 500)
    finally:
        cur.close()
        conn.close()


# ── Auth: cancel the active plan (handy to re-test the upgrade flow) ────────
@dummy_billing_bp.post("/cancel")
@require_auth
def dummy_cancel():
    g_resp = _guard()
    if g_resp is not None:
        return g_resp

    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            """
            UPDATE user_subscriptions SET status='cancelled'
             WHERE user_id=%s AND status='active'
            """,
            (uid,),
        )
        conn.commit()
        return _ok("[TEST MODE] Active subscription cancelled", dummy=True)
    except Exception as e:
        conn.rollback()
        logger.error("dummy_cancel failed: %s", e, exc_info=True)
        return _err("Could not cancel in test mode", 500)
    finally:
        cur.close()
        conn.close()
