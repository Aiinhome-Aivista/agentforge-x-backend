"""Subscription + payment routes (Razorpay).

Mounted under /api/billing via the main api blueprint. New file — does not
modify any existing endpoint. Routes:

  GET   /billing/plans                     — public plan catalogue
  GET   /billing/subscription              — auth: caller's active plan
  POST  /billing/subscribe/free            — auth: activate free trial
  POST  /billing/orders                    — auth: create Razorpay order
                                              body: {plan_code, country?, kind?, packets?}
  POST  /billing/orders/verify             — auth: verify checkout result + activate
                                              body: {razorpay_order_id, razorpay_payment_id, razorpay_signature}
  GET   /billing/usage                     — auth: rolling usage vs plan limits
  GET   /billing/config                    — public: razorpay key_id (for SPA)
"""

import logging
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify, g

from app.db.db_connection import get_mysql_connection
from app.services import razorpay_service
from app.services.auth_helpers import require_auth
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)
billing_bp = Blueprint("billing", __name__, url_prefix="/billing")


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})

def _err(message, code=400):
    return jsonify({"status": False, "statuscode": code, "message": message}), code


def _fetch_plan(cur, code: str):
    cur.execute("SELECT * FROM subscription_plans WHERE code=%s AND is_active=1",
                (code,))
    return cur.fetchone()


def _active_subscription(cur, user_id: int):
    cur.execute("""
        SELECT s.*, p.name AS plan_name, p.datasize_mb, p.workspaces,
               p.has_market_research, p.has_deep_insights, p.features
          FROM user_subscriptions s
          JOIN subscription_plans p ON p.id = s.plan_id
         WHERE s.user_id=%s AND s.status='active' AND s.period_end > NOW()
         ORDER BY s.id DESC LIMIT 1
    """, (user_id,))
    return cur.fetchone()


# ── Public: plan catalogue ──────────────────────────────────────────────────
@billing_bp.get("/plans")
@require_auth
def plans():
    uid = g.user["uid"]

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)

    try:
        # GET ALL ACTIVE PLANS
        cur.execute("""
            SELECT 
                id,
                code,
                name,
                period_days,
                price_inr,
                price_usd,
                datasize_mb,
                workspaces,
                addon_packet_inr,
                has_market_research,
                has_deep_insights,
                features
            FROM subscription_plans
            WHERE is_active = 1
            ORDER BY price_inr ASC, id ASC
        """)
        plans = cur.fetchall()

        # GET USER ACTIVE SUBSCRIPTION
        cur.execute("""
            SELECT 
                us.id,
                us.plan_id,
                us.plan_code,
                us.status,
                us.period_start,
                us.period_end
            FROM user_subscriptions us
            WHERE us.user_id = %s
              AND us.status = 'active'
              AND us.period_end >= NOW()
            ORDER BY us.period_end DESC
            LIMIT 1
        """, (uid,))

        active_subscription = cur.fetchone()

        # MARK CURRENT PLAN + REMAINING DAYS
        for plan in plans:
            plan["is_current_plan"] = False
            plan["remaining_days"] = 0

            if active_subscription and plan["id"] == active_subscription["plan_id"]:

                plan["is_current_plan"] = True

                end_date = active_subscription["period_end"]

                if isinstance(end_date, str):
                    end_date = datetime.strptime(
                        end_date,
                        "%Y-%m-%d %H:%M:%S"
                    )

                remaining_days = (end_date - datetime.now()).days

                plan["remaining_days"] = max(remaining_days, 0)

                plan["subscription"] = {
                    "subscription_id": active_subscription["id"],
                    "status": active_subscription["status"],
                    "period_start": active_subscription["period_start"],
                    "period_end": active_subscription["period_end"]
                }

        return _ok("OK", data=plans)

    finally:
        cur.close()
        conn.close()

@billing_bp.get("/config")
def config():
    return _ok("OK", data={
        "razorpay_key_id": razorpay_service.public_key(),
        "configured":      razorpay_service.is_configured(),
    })


# ── Auth: current subscription ──────────────────────────────────────────────
@billing_bp.get("/subscription")
@require_auth
def my_subscription():

    uid = g.user["uid"]

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)

    try:
        sub = _active_subscription(cur, uid)

        # CALCULATE REMAINING DAYS
        if sub and sub.get("period_end"):

            end_date = sub["period_end"]

            # HANDLE GMT STRING DATE
            if isinstance(end_date, str):
                end_date = parsedate_to_datetime(end_date)

            # HANDLE TIMEZONE
            if end_date.tzinfo is not None:
                now = datetime.now(timezone.utc)
            else:
                now = datetime.now()

            remaining_days = max(
                (end_date - now).days,
                0
            )

            sub["remaining_days"] = remaining_days

        return _ok("OK", data=sub)

    finally:
        cur.close()
        conn.close()

# ── Auth: activate the free 3-day trial ────────────────────────────────────
@billing_bp.post("/subscribe/free")
@require_auth
def subscribe_free():
    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        plan = _fetch_plan(cur, "free")
        if not plan:
            return _err("Free plan unavailable", 500)

        # If a paid sub is active, refuse to overwrite it.
        existing = _active_subscription(cur, uid)
        if existing and existing["plan_code"] != "free":
            return _err("You already have an active paid plan", 409)

        # Allow only one free trial per user (rare-tolerant guard).
        cur.execute("""
            SELECT 1 FROM user_subscriptions
             WHERE user_id=%s AND plan_code='free' LIMIT 1
        """, (uid,))
        if cur.fetchone():
            return _err("Free trial already used", 409)

        start = datetime.utcnow()
        end = start + timedelta(days=int(plan["period_days"]))
        cur.execute("""
            INSERT INTO user_subscriptions
                (user_id, plan_id, plan_code, status, currency, amount,
                 period_start, period_end)
            VALUES (%s, %s, 'free', 'active', 'INR', 0, %s, %s)
        """, (uid, plan["id"], start, end))
        conn.commit()
        sub = _active_subscription(cur, uid)
        return _ok("Free trial activated", data=sub)
    except Exception as e:
        conn.rollback()
        logger.error("subscribe_free error: %s", e, exc_info=True)
        return _err("Could not activate free trial", 500)
    finally:
        cur.close(); conn.close()


# ── Auth: create a Razorpay order for a paid plan or addon ──────────────────
@billing_bp.post("/orders")
@require_auth
def create_order():
    if not razorpay_service.is_configured():
        return _err("Payments are not configured. Contact admin.", 503)

    body = request.get_json(silent=True) or {}
    kind = (body.get("kind") or "subscription").lower()
    plan_code = (body.get("plan_code") or "").lower()
    country = (body.get("country") or g.user.get("country") or "IN").upper()
    packets = int(body.get("packets") or 0)

    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        if kind == "subscription":
            if plan_code not in ("basic", "premium"):
                return _err("Only basic/premium can be purchased")
            plan = _fetch_plan(cur, plan_code)
            if not plan:
                return _err("Plan not found", 404)
            if country == "IN":
                amount = float(plan["price_inr"]); currency = "INR"
            else:
                amount = float(plan["price_usd"]); currency = "USD"
            notes = {"user_id": uid, "kind": "subscription",
                     "plan_code": plan_code}
            receipt = f"sub_{uid}_{plan_code}_{int(datetime.utcnow().timestamp())}"
        elif kind == "addon":
            if packets < 1 or packets > 100:
                return _err("packets must be between 1 and 100")
            sub = _active_subscription(cur, uid)
            if not sub or sub["plan_code"] == "free":
                return _err("Addon packets require an active Basic/Premium plan", 403)
            plan = _fetch_plan(cur, sub["plan_code"])
            unit = float(plan["addon_packet_inr"] or 0)
            if unit <= 0:
                return _err("Addons not available on this plan", 400)
            # Addons are priced in INR per the spec.
            amount = unit * packets; currency = "INR"
            notes = {"user_id": uid, "kind": "addon",
                     "plan_id": plan["id"], "packets": packets}
            receipt = f"addon_{uid}_{packets}_{int(datetime.utcnow().timestamp())}"
        else:
            return _err("Unsupported kind")

        order = razorpay_service.create_order(
            razorpay_service.amount_in_minor(amount),
            currency=currency, receipt=receipt, notes=notes,
        )
        if not order:
            return _err("Failed to create order with payment provider", 502)

        cur.execute("""
            INSERT INTO payment_orders
                (user_id, plan_id, kind, addon_packets, currency, amount,
                 razorpay_order_id, status, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'created', %s)
        """, (uid, plan["id"], kind, packets if kind == "addon" else 0,
              currency, amount, order["id"],
              __import__("json").dumps(notes)))
        conn.commit()

        return _ok("Order created", data={
            "order_id":     order["id"],
            "amount_minor": order["amount"],
            "currency":     order["currency"],
            "key_id":       razorpay_service.public_key(),
            "kind":         kind,
            "plan_code":    plan_code if kind == "subscription" else None,
            "packets":      packets if kind == "addon" else 0,
        })
    except Exception as e:
        conn.rollback()
        logger.error("create_order error: %s", e, exc_info=True)
        return _err("Could not create order", 500)
    finally:
        cur.close(); conn.close()


# ── Auth: verify Razorpay signature + activate plan/addon ───────────────────
@billing_bp.post("/orders/verify")
@require_auth
def verify_order():
    body = request.get_json(silent=True) or {}
    order_id   = body.get("razorpay_order_id")
    payment_id = body.get("razorpay_payment_id")
    signature  = body.get("razorpay_signature")
    if not all((order_id, payment_id, signature)):
        return _err("Missing payment verification fields")

    if not razorpay_service.verify_signature(order_id, payment_id, signature):
        return _err("Payment signature invalid", 400)

    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT * FROM payment_orders
             WHERE razorpay_order_id=%s AND user_id=%s LIMIT 1
        """, (order_id, uid))
        po = cur.fetchone()
        if not po:
            return _err("Order not found", 404)
        if po["status"] == "paid":
            return _ok("Already activated", data={"order_id": order_id})

        cur.execute("""
            UPDATE payment_orders
               SET status='paid', razorpay_payment_id=%s, razorpay_signature=%s
             WHERE id=%s
        """, (payment_id, signature, po["id"]))

        if po["kind"] == "subscription":
            cur.execute("SELECT * FROM subscription_plans WHERE id=%s",
                        (po["plan_id"],))
            plan = cur.fetchone()
            # Expire any prior active sub
            cur.execute("""
                UPDATE user_subscriptions SET status='expired'
                 WHERE user_id=%s AND status='active'
            """, (uid,))
            start = datetime.utcnow()
            end = start + timedelta(days=int(plan["period_days"]))
            cur.execute("""
                INSERT INTO user_subscriptions
                    (user_id, plan_id, plan_code, status, currency, amount,
                     period_start, period_end,
                     razorpay_order_id, razorpay_payment_id, razorpay_signature)
                VALUES (%s,%s,%s,'active',%s,%s,%s,%s,%s,%s,%s)
            """, (uid, plan["id"], plan["code"], po["currency"], po["amount"],
                  start, end, order_id, payment_id, signature))
        elif po["kind"] == "addon":
            sub_id = None
            cur.execute("""
                SELECT id FROM user_subscriptions
                 WHERE user_id=%s AND status='active' AND period_end>NOW()
                 ORDER BY id DESC LIMIT 1
            """, (uid,))
            row = cur.fetchone()
            if row:
                sub_id = row["id"]
                cur.execute("""
                    UPDATE user_subscriptions
                       SET extra_workspaces = extra_workspaces + (%s * 10)
                     WHERE id=%s
                """, (int(po["addon_packets"]), sub_id))

        conn.commit()
        # Fetch fresh sub
        sub = _active_subscription(cur, uid)
        return _ok("Payment verified and plan activated", data=sub)
    except Exception as e:
        conn.rollback()
        logger.error("verify_order error: %s", e, exc_info=True)
        return _err("Verification failed", 500)
    finally:
        cur.close(); conn.close()


# ── Auth: usage ─────────────────────────────────────────────────────────────
@billing_bp.get("/usage")
@require_auth
def usage():
    uid = g.user["uid"]
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        sub = _active_subscription(cur, uid)
        cur.execute("""
            SELECT * FROM user_usage
             WHERE user_id=%s
             ORDER BY id DESC LIMIT 1
        """, (uid,))
        u = cur.fetchone() or {"data_used_mb": 0, "analyses_run": 0}
        cur.execute("SELECT COUNT(*) AS c FROM workspaces WHERE user_id=%s", (uid,))
        ws = (cur.fetchone() or {}).get("c", 0)
        return _ok("OK", data={
            "subscription": sub,
            "usage":        u,
            "workspaces":   ws,
        })
    finally:
        cur.close(); conn.close()
