"""Captcha routes — issue a challenge, and gate the existing auth endpoints.

Mounted under /api/captcha via the main api blueprint. New file — does not
modify the existing auth_routes.py.

How the gating works (without touching auth_routes.py):
  * This module exposes ``register_auth_gate(api_bp)`` which attaches a
    ``before_request`` handler to the **api blueprint**. The handler only
    inspects requests for the captcha-protected paths (signin, signup
    request-otp, signup resend-otp). For every other route it returns
    ``None`` immediately.
  * If the captcha header / body field is missing or invalid, the handler
    returns a 400 JSON error and the request never reaches the original
    auth handler. Otherwise the request proceeds untouched.

Routes:
  GET  /captcha/new         — public: returns {prompt, token, ttl_seconds}
  GET  /captcha/status      — public: {disabled: bool}  (matches the
                              dummy-payments status convention)
"""

import logging

from flask import Blueprint, request, jsonify

from app.services import captcha_service

logger = logging.getLogger(__name__)
captcha_bp = Blueprint("captcha", __name__, url_prefix="/captcha")


# Endpoints that must carry a valid captcha (request path RELATIVE to the
# main api blueprint, i.e. without the /api prefix). If you add new
# captcha-protected endpoints, list them here.
CAPTCHA_PROTECTED_PATHS = {
    "/auth/signin",
    "/auth/signup/request-otp",
    "/auth/signup/resend-otp",
}


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})


def _err(message, code=400, **extra):
    return jsonify({"status": False, "statuscode": code, "message": message, **extra}), code


# ── Public: issue a challenge ───────────────────────────────────────────────
@captcha_bp.get("/new")
def captcha_new():
    return _ok("OK", data=captcha_service.issue())


# ── Public: is captcha disabled? (handy for the SPA test harness) ───────────
@captcha_bp.get("/status")
def captcha_status():
    return _ok("OK", data={"disabled": captcha_service.is_disabled()})


# ── Before-request gate (registered by the api blueprint owner) ─────────────
def _extract_captcha() -> tuple:
    """Pull the captcha (token, answer) from the request, in this order:
       * JSON body fields ``captcha_token`` / ``captcha_answer``
       * HTTP headers ``X-Captcha-Token`` / ``X-Captcha-Answer``
    Returns ('','') if neither source provides them.
    """
    body = request.get_json(silent=True) or {}
    tok = body.get("captcha_token") or request.headers.get("X-Captcha-Token") or ""
    ans = body.get("captcha_answer") or request.headers.get("X-Captcha-Answer") or ""
    return str(tok), str(ans)


def auth_captcha_gate():
    """``before_request`` handler. Returns a Flask response to short-circuit
    the request, or ``None`` to let it through."""
    # Skip non-protected endpoints. Note: request.path is the FULL path,
    # e.g. '/api/auth/signin'. We strip the api blueprint's url_prefix
    # ('/api') manually to compare against CAPTCHA_PROTECTED_PATHS.
    path = request.path or ""
    if path.startswith("/api"):
        rel = path[len("/api"):] or "/"
    else:
        rel = path
    if rel not in CAPTCHA_PROTECTED_PATHS:
        return None

    if captcha_service.is_disabled():
        return None

    if request.method == "OPTIONS":   # CORS preflight
        return None

    token, answer = _extract_captcha()
    try:
        captcha_service.verify(token, answer, raise_on_failure=True)
    except captcha_service.CaptchaError as e:
        return _err(str(e), code=400)
    except Exception as e:
        logger.error("Captcha gate error: %s", e, exc_info=True)
        return _err("Captcha verification failed", code=400)
    return None


def register_auth_gate(api_bp):
    """Attach the ``before_request`` hook. Idempotent: calling this twice
    won't register the handler twice."""
    existing = getattr(api_bp, "_afx_captcha_gate_registered", False)
    if existing:
        return
    api_bp.before_request(auth_captcha_gate)
    setattr(api_bp, "_afx_captcha_gate_registered", True)
