"""Admin routes — user management and subscription overview.

Mounted under /api/admin via the main api blueprint. Every endpoint
requires both ``@require_auth`` (a valid JWT) and ``@require_admin``
(``users.is_admin = 1`` for the JWT's user). Anything blog-related lives
in ``blog_routes.py``; this module covers users + subscriptions.

Routes:
  GET    /admin/whoami                   — confirms the caller is an admin
  GET    /admin/users                    — list users with their active sub
  POST   /admin/users                    — create a new user
  GET    /admin/users/<int:uid>          — fetch one user
  PATCH  /admin/users/<int:uid>          — update is_active/is_admin/name/country
  DELETE /admin/users/<int:uid>          — delete user + dependent rows
  GET    /admin/subscriptions            — subscriptions list + aggregates
"""

import logging

from flask import Blueprint, request, jsonify, g

from app.services.auth_helpers import require_auth
from app.services.admin_helpers import require_admin
from app.services import admin_service

logger = logging.getLogger(__name__)
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})


def _err(message, code=400, **extra):
    return jsonify({"status": False, "statuscode": code, "message": message, **extra}), code


# ── Whoami (admin) ──────────────────────────────────────────────────────────
@admin_bp.get("/whoami")
@require_auth
@require_admin
def admin_whoami():
    return _ok("OK", data={"uid": g.user["uid"], "is_admin": True})


# ── Users ───────────────────────────────────────────────────────────────────
@admin_bp.get("/users")
@require_auth
@require_admin
def admin_list_users():
    q = (request.args.get("q") or "").strip() or None
    try:
        limit  = int(request.args.get("limit") or 100)
        offset = int(request.args.get("offset") or 0)
    except ValueError:
        return _err("limit and offset must be integers")

    items = admin_service.list_users(q=q, limit=limit, offset=offset)
    total = admin_service.count_users(q=q)
    return _ok("OK", data=items, total=total)


@admin_bp.post("/users")
@require_auth
@require_admin
def admin_create_user():
    body = request.get_json(silent=True) or {}
    try:
        user = admin_service.create_user(
            name           = body.get("name", ""),
            email          = body.get("email", ""),
            password       = body.get("password", ""),
            country        = (body.get("country") or "IN"),
            is_admin       = bool(body.get("is_admin", False)),
            email_verified = bool(body.get("email_verified", True)),
            file_size_limit_mb = int(body.get("file_size_limit_mb", 50))
        )
    except ValueError as ve:
        return _err(str(ve), 400)
    except Exception as e:
        logger.error("admin_create_user failed: %s", e, exc_info=True)
        return _err("Could not create user", 500)
    return _ok("User created", data=user)


@admin_bp.get("/users/<int:uid>")
@require_auth
@require_admin
def admin_get_user(uid):
    u = admin_service.get_user(uid)
    if not u:
        return _err("User not found", 404)
    return _ok("OK", data=u)


@admin_bp.patch("/users/<int:uid>")
@require_auth
@require_admin
def admin_update_user(uid):
    body = request.get_json(silent=True) or {}
    # Only forward whitelisted fields. The service layer also enforces this.
    permitted = {k: body[k] for k in ("is_active", "is_admin", "name", "country", "file_size_limit_mb")
                 if k in body}
    try:
        u = admin_service.update_user(uid, **permitted)
    except Exception as e:
        logger.error("admin_update_user failed: %s", e, exc_info=True)
        return _err("Could not update user", 500)
    if not u:
        return _err("User not found", 404)
    return _ok("User updated", data=u)


@admin_bp.post("/users/<int:uid>/reset_storage")
@require_auth
@require_admin
def admin_reset_user_storage(uid):
    try:
        admin_service.reset_user_storage(uid)
    except Exception as e:
        logger.error("admin_reset_user_storage failed: %s", e, exc_info=True)
        return _err("Could not reset storage", 500)
    return _ok("Storage reset")



@admin_bp.delete("/users/<int:uid>")
@require_auth
@require_admin
def admin_delete_user(uid):
    if uid == g.user["uid"]:
        return _err("Refusing to delete the currently signed-in admin", 400)
    try:
        ok = admin_service.delete_user(uid)
    except Exception as e:
        logger.error("admin_delete_user failed: %s", e, exc_info=True)
        return _err("Could not delete user", 500)
    if not ok:
        return _err("User not found", 404)
    return _ok("User deleted", data={"id": uid})


# ── Subscriptions overview ──────────────────────────────────────────────────
@admin_bp.get("/subscriptions")
@require_auth
@require_admin
def admin_subscriptions():
    status    = (request.args.get("status") or "").strip() or None
    plan_code = (request.args.get("plan_code") or "").strip() or None
    try:
        limit = int(request.args.get("limit") or 200)
    except ValueError:
        return _err("limit must be an integer")
    data = admin_service.subscriptions_overview(
        status=status, plan_code=plan_code, limit=limit,
    )
    return _ok("OK", data=data)


# -- LLM Settings -----------------------------------------------------------
from app.services.settings_service import get_all_settings, update_settings, get_llm_change_logs

@admin_bp.get('/settings')
@require_auth
@require_admin
def admin_get_settings():
    settings = get_all_settings()
    logs = get_llm_change_logs()
    return _ok('OK', data={'settings': settings, 'logs': logs})

@admin_bp.post('/settings')
@require_auth
@require_admin
def admin_update_settings():
    body = request.get_json(silent=True) or {}
    updates = {}
    for k in ['LLM_PROVIDER', 'LLM_MODEL_NAME', 'LLM_API_KEY', 'LLM_BASE_URL']:
        if k in body:
            updates[k] = body[k]
    
    try:
        update_settings(g.user['uid'], updates)
    except Exception as e:
        logger.error(f'Failed to update settings: {e}')
        return _err('Could not update settings', 500)
    
    return _ok('Settings updated')

