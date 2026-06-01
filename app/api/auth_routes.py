"""Authentication routes (signup, OTP verify, signin, Google).

Mounted under /api/auth via the main api blueprint. New file — no existing
route is altered.
"""

import os
import re
import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify

from app.db.db_connection import get_mysql_connection
from app.services import jwt_service, email_service, google_oauth_service, geo_service
from app.services.auth_helpers import (
    hash_password, verify_password, require_auth, current_user,
)

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
OTP_TTL_MIN = int(os.getenv("OTP_TTL_MINUTES", "10"))
MAX_OTP_ATTEMPTS = 5


# ── Helpers ─────────────────────────────────────────────────────────────────
def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})

def _err(message, code=400):
    return jsonify({"status": False, "statuscode": code, "message": message}), code

def _hash_otp(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def _gen_otp() -> str:
    # 6-digit zero-padded
    return f"{secrets.randbelow(10**6):06d}"

def _get_user_by_email(cur, email: str):
    cur.execute("SELECT * FROM users WHERE email = %s LIMIT 1", (email,))
    return cur.fetchone()

def _active_plan(cur, user_id: int):
    cur.execute("""
        SELECT s.plan_code, s.period_end, s.status
          FROM user_subscriptions s
         WHERE s.user_id = %s AND s.status = 'active'
              AND s.period_end > NOW()
         ORDER BY s.id DESC LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    return row["plan_code"] if row else None


# ── Signup: send OTP ────────────────────────────────────────────────────────
@auth_bp.post("/signup/request-otp")
def signup_request_otp():
    data = request.get_json(silent=True) or {}
    name     = (data.get("name") or "").strip()
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    # Country is auto-detected from the caller's IP. A client-supplied value
    # is only honoured when it looks like a valid override (testing only).
    override = (data.get("country") or "").strip().upper()
    country  = override if (len(override) == 2 and override.isalpha()) \
                       else geo_service.detect_country(request)
    country  = country[:8]

    if not name or len(name) < 2:
        return _err("Name is required (min 2 characters)")
    if not EMAIL_RE.match(email):
        return _err("Please enter a valid email")
    if len(password) < 8:
        return _err("Password must be at least 8 characters")

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        existing = _get_user_by_email(cur, email)
        if existing and existing.get("email_verified"):
            return _err("An account with this email already exists. Please sign in.", 409)

        # Pre-create / update user as unverified
        pwd_hash = hash_password(password)
        if existing:
            cur.execute("""
                UPDATE users SET name=%s, password=%s, country=%s,
                                 auth_provider='email', email_verified=0
                 WHERE id=%s
            """, (name, pwd_hash, country, existing["id"]))
        else:
            cur.execute("""
                INSERT INTO users (name, email, password, country,
                                   auth_provider, email_verified)
                VALUES (%s, %s, %s, %s, 'email', 0)
            """, (name, email, pwd_hash, country))

        # Generate + store OTP (hash it, never store the code itself)
        code = _gen_otp()
        cur.execute("""
            DELETE FROM email_otps
             WHERE email=%s AND purpose='signup' AND verified=0
        """, (email,))
        cur.execute("""
            INSERT INTO email_otps (email, otp_hash, purpose, expires_at)
            VALUES (%s, %s, 'signup', %s)
        """, (email, _hash_otp(code),
              datetime.utcnow() + timedelta(minutes=OTP_TTL_MIN)))
        conn.commit()

        sent = email_service.send_otp_email(email, code, "signup")
        resp = {"email": email, "ttl_minutes": OTP_TTL_MIN, "delivered": sent}
        # In dev (no SMTP), expose the OTP so the developer can proceed.
        if email_service.is_dev_mode():
            resp["debug_otp"] = code
            logger.warning("DEV mode: returning OTP in API response for %s", email)
        return _ok("OTP sent. Please check your email.", **resp)
    except Exception as e:
        conn.rollback()
        logger.error("signup_request_otp error: %s", e, exc_info=True)
        return _err("Could not send OTP, please try again", 500)
    finally:
        cur.close(); conn.close()


# ── Signup: verify OTP and complete ─────────────────────────────────────────
@auth_bp.post("/signup/verify-otp")
def signup_verify_otp():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    code  = (data.get("otp") or "").strip()

    if not EMAIL_RE.match(email):
        return _err("Invalid email")
    if not code.isdigit() or len(code) != 6:
        return _err("OTP must be a 6-digit code")

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT * FROM email_otps
             WHERE email=%s AND purpose='signup' AND verified=0
             ORDER BY id DESC LIMIT 1
        """, (email,))
        rec = cur.fetchone()
        if not rec:
            return _err("No active OTP for this email. Please request a new one.", 410)
        if rec["attempts"] >= MAX_OTP_ATTEMPTS:
            return _err("Too many incorrect attempts. Request a new OTP.", 429)
        if rec["expires_at"] < datetime.utcnow():
            return _err("OTP expired. Please request a new one.", 410)

        if rec["otp_hash"] != _hash_otp(code):
            cur.execute("UPDATE email_otps SET attempts=attempts+1 WHERE id=%s",
                        (rec["id"],))
            conn.commit()
            return _err("Incorrect OTP", 401)

        # Mark OTP used + user verified
        cur.execute("UPDATE email_otps SET verified=1 WHERE id=%s", (rec["id"],))
        cur.execute("UPDATE users SET email_verified=1 WHERE email=%s", (email,))
        conn.commit()

        user = _get_user_by_email(cur, email)
        if not user:
            return _err("User record missing", 500)
        plan = _active_plan(cur, user["id"]) or None

        token = jwt_service.issue(user["id"], user["email"], user["name"], plan)
        return _ok("Account verified", token=token, data={
            "id": user["id"], "name": user["name"], "email": user["email"],
            "country": user.get("country", "IN"),
            "avatar_url": user.get("avatar_url"),
            "plan": plan,
        })
    except Exception as e:
        conn.rollback()
        logger.error("signup_verify_otp error: %s", e, exc_info=True)
        return _err("Verification failed", 500)
    finally:
        cur.close(); conn.close()


# ── Signin: email + password ────────────────────────────────────────────────
@auth_bp.post("/signin")
def signin():
    data = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not EMAIL_RE.match(email) or not password:
        return _err("Email and password are required")

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        user = _get_user_by_email(cur, email)
        if not user:
            return _err("No account found with this email", 404)
        if user.get("auth_provider") == "google" and not user.get("password"):
            return _err("This account uses Google sign-in. Please continue with Google.", 400)
        if not user.get("is_active", 1):
            return _err("Account disabled. Contact support.", 403)
        if not verify_password(password, user.get("password") or ""):
            return _err("Incorrect password", 401)
        if not user.get("email_verified", 0):
            return _err("Email not verified. Please complete signup OTP.", 403)

        plan = _active_plan(cur, user["id"]) or None
        token = jwt_service.issue(user["id"], user["email"], user["name"], plan)
        return _ok("Signed in", token=token, data={
            "id": user["id"], "name": user["name"], "email": user["email"],
            "country": user.get("country", "IN"),
            "avatar_url": user.get("avatar_url"),
            "plan": plan,
        })
    except Exception as e:
        logger.error("signin error: %s", e, exc_info=True)
        return _err("Sign-in failed", 500)
    finally:
        cur.close(); conn.close()


# ── Google sign-in / sign-up ────────────────────────────────────────────────
@auth_bp.post("/google")
def google_signin():
    data = request.get_json(silent=True) or {}
    id_token_str = data.get("id_token") or data.get("credential")
    override = (data.get("country") or "").strip().upper()
    country  = override if (len(override) == 2 and override.isalpha()) \
                       else geo_service.detect_country(request)
    country  = country[:8]

    if not id_token_str:
        return _err("Google id_token is required")

    info = google_oauth_service.verify_google_id_token(id_token_str)
    if not info:
        return _err("Invalid Google token", 401)

    email = info["email"].lower()
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        user = _get_user_by_email(cur, email)
        if user:
            # Link the google_id if missing; mark verified.
            if not user.get("google_id"):
                cur.execute("""
                    UPDATE users SET google_id=%s, avatar_url=%s,
                                     email_verified=1, auth_provider=%s
                     WHERE id=%s
                """, (info["sub"], info.get("picture"),
                      user.get("auth_provider") or "google", user["id"]))
                conn.commit()
                user = _get_user_by_email(cur, email)
        else:
            cur.execute("""
                INSERT INTO users (name, email, google_id, avatar_url, country,
                                   auth_provider, email_verified)
                VALUES (%s, %s, %s, %s, %s, 'google', 1)
            """, (info["name"], email, info["sub"],
                  info.get("picture"), country))
            conn.commit()
            user = _get_user_by_email(cur, email)

        plan = _active_plan(cur, user["id"]) or None
        token = jwt_service.issue(user["id"], user["email"], user["name"], plan)
        return _ok("Signed in with Google", token=token, data={
            "id": user["id"], "name": user["name"], "email": user["email"],
            "country": user.get("country", "IN"),
            "avatar_url": user.get("avatar_url"),
            "plan": plan,
        })
    except Exception as e:
        conn.rollback()
        logger.error("google_signin error: %s", e, exc_info=True)
        return _err("Google sign-in failed", 500)
    finally:
        cur.close(); conn.close()


# ── Resend OTP (rate-limited to once per 30s in DB by recreate timestamp) ───
@auth_bp.post("/signup/resend-otp")
def signup_resend_otp():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not EMAIL_RE.match(email):
        return _err("Invalid email")

    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT id, created_at FROM email_otps
             WHERE email=%s AND purpose='signup'
             ORDER BY id DESC LIMIT 1
        """, (email,))
        last = cur.fetchone()
        if last and (datetime.utcnow() - last["created_at"]).total_seconds() < 30:
            return _err("Please wait a moment before requesting another OTP", 429)

        code = _gen_otp()
        cur.execute("""
            INSERT INTO email_otps (email, otp_hash, purpose, expires_at)
            VALUES (%s, %s, 'signup', %s)
        """, (email, _hash_otp(code),
              datetime.utcnow() + timedelta(minutes=OTP_TTL_MIN)))
        conn.commit()

        sent = email_service.send_otp_email(email, code, "signup")
        resp = {"delivered": sent, "ttl_minutes": OTP_TTL_MIN}
        if email_service.is_dev_mode():
            resp["debug_otp"] = code
        return _ok("OTP resent", **resp)
    except Exception as e:
        conn.rollback()
        logger.error("resend_otp error: %s", e, exc_info=True)
        return _err("Could not resend OTP", 500)
    finally:
        cur.close(); conn.close()


# ── Whoami (used by frontend on hard reload) ────────────────────────────────
@auth_bp.get("/me")
@require_auth
def me():
    from flask import g
    uid = g.user.get("uid")
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, email, country, avatar_url, "
                    "auth_provider, email_verified, created_at "
                    "FROM users WHERE id=%s", (uid,))
        u = cur.fetchone()
        if not u:
            return _err("User not found", 404)
        plan = _active_plan(cur, uid)
        return _ok("OK", data={**u, "plan": plan})
    finally:
        cur.close(); conn.close()


# ── Public IP-based geolocation (for showing currency on /signup, /pricing) ─
@auth_bp.get("/locate")
def locate():
    country = geo_service.detect_country(request)
    return _ok("OK", data={
        "country":  country,
        "currency": geo_service.currency_for(country),
    })
