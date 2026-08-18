"""Workspace routes — store, list, view, and delete saved analyze responses.

Mounted under /api/workspaces via the main api blueprint. New file — does
not modify any existing endpoint. Routes:

  GET    /workspaces                — auth: list caller's workspaces (latest first)
  GET    /workspaces/quota          — auth: workspace quota for current plan
  POST   /workspaces                — auth: create a workspace from an analyze
                                       response. Body:
                                         {
                                           name?: "My Workspace",
                                           session_id?: "...",
                                           user_input?: "...",
                                           analysis: { ...full analyze JSON... }
                                         }
                                       Returns 403 if the user is at quota.
  GET    /workspaces/<int:ws_id>    — auth: full workspace incl. analysis blob
  PATCH  /workspaces/<int:ws_id>    — auth: rename a workspace ({name})
  DELETE /workspaces/<int:ws_id>    — auth: remove a workspace
"""

import logging

from flask import Blueprint, request, jsonify, g

from app.services.auth_helpers import require_auth
from app.services import workspace_service
from app.services.workspace_service import QuotaExceeded

logger = logging.getLogger(__name__)
workspace_bp = Blueprint("workspaces", __name__, url_prefix="/workspaces")


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})


def _err(message, code=400, **extra):
    return jsonify(
        {"status": False, "statuscode": code, "message": message, **extra}
    ), code


# ── Quota ───────────────────────────────────────────────────────────────────
@workspace_bp.get("/quota")
@require_auth
def quota():
    uid = g.user["uid"]
    return _ok("OK", data=workspace_service.get_quota(uid))


# ── List ────────────────────────────────────────────────────────────────────
@workspace_bp.get("")
@require_auth
def list_workspaces():
    uid = g.user["uid"]
    items = workspace_service.list_workspaces(uid)
    return _ok("OK", data=items, quota=workspace_service.get_quota(uid))


# ── Create (saves the last analyze response) ────────────────────────────────
@workspace_bp.post("")
@require_auth
def create_workspace():
    uid = g.user["uid"]
    body = request.get_json(silent=True) or {}

    name = body.get("name") or ""
    session_id = body.get("session_id")
    user_input = body.get("user_input")
    analysis = body.get("analysis")
    chat_history = body.get("chat_history")
    uploaded_files_log = body.get("uploaded_files_log")

    # Be lenient: accept the analyze response either nested under "analysis"
    # (the explicit contract) or unwrapped at the top level.
    if analysis is None and any(
        k in body for k in ("process", "steps", "suggestions")
    ):
        analysis = {k: v for k, v in body.items()
                    if k not in ("name", "session_id", "user_input")}

    if not isinstance(analysis, dict) or not analysis:
        return _err("Field 'analysis' is required and must be a non-empty object")

    try:
        ws, quota_after = workspace_service.create_workspace(
            uid, name,
            session_id=session_id,
            user_input=user_input,
            analysis_data=analysis,
            chat_history=chat_history,
            uploaded_files_log=uploaded_files_log,
        )
    except QuotaExceeded as qe:
        return _err(
            "Workspace limit reached for your current plan. "
            "Delete an older workspace or buy an addon packet to add 10 more.",
            code=403,
            data={"quota": qe.quota},
        )
    except Exception as e:
        logger.error("create_workspace failed: %s", e, exc_info=True)
        return _err("Could not save workspace", 500)

    return _ok("Workspace saved", data=ws, quota=quota_after)


# ── Get one ─────────────────────────────────────────────────────────────────
@workspace_bp.get("/<int:ws_id>")
@require_auth
def get_workspace(ws_id: int):
    uid = g.user["uid"]
    ws = workspace_service.get_workspace(uid, ws_id)
    if not ws:
        return _err("Workspace not found", 404)
    return _ok("OK", data=ws)


# ── Rename ──────────────────────────────────────────────────────────────────
@workspace_bp.patch("/<int:ws_id>")
@require_auth
def rename_workspace(ws_id: int):
    uid = g.user["uid"]
    body = request.get_json(silent=True) or {}
    new_name = (body.get("name") or "").strip()
    if not new_name:
        return _err("Field 'name' is required")
    if not workspace_service.rename_workspace(uid, ws_id, new_name):
        return _err("Workspace not found", 404)
    return _ok("Workspace renamed", data=workspace_service.get_workspace(uid, ws_id))


# ── Delete (user can delete older workspaces) ───────────────────────────────
@workspace_bp.delete("/<int:ws_id>")
@require_auth
def delete_workspace(ws_id: int):
    uid = g.user["uid"]
    try:
        deleted = workspace_service.delete_workspace(uid, ws_id)
    except Exception as e:
        logger.error("delete_workspace failed: %s", e, exc_info=True)
        return _err("Could not delete workspace", 500)
    if not deleted:
        return _err("Workspace not found", 404)
    return _ok("Workspace deleted", data={"id": ws_id},
               quota=workspace_service.get_quota(uid))

# ── Update Chat History ─────────────────────────────────────────────────────
@workspace_bp.patch("/<int:ws_id>/chat")
@require_auth
def update_workspace_chat(ws_id: int):
    uid = g.user["uid"]
    body = request.get_json(silent=True) or {}
    chat_history = body.get("chat_history")
    
    if chat_history is None:
        return _err("Field 'chat_history' is required", 400)
        
    try:
        updated = workspace_service.update_chat_history(uid, ws_id, chat_history)
        if not updated:
            return _err("Workspace not found", 404)
        return _ok("Chat history updated")
    except Exception as e:
        logger.error("update_workspace_chat failed: %s", e, exc_info=True)
        return _err("Could not update chat history", 500)


# ── Update Analysis Data ────────────────────────────────────────────────────
@workspace_bp.patch("/<int:ws_id>/analysis")
@require_auth
def update_workspace_analysis(ws_id: int):
    uid = g.user["uid"]
    body = request.get_json(silent=True) or {}
    analysis = body.get("analysis")
    silent = body.get("silent", False)
    
    if analysis is None:
        return _err("Field 'analysis' is required", 400)
        
    try:
        updated = workspace_service.update_workspace_analysis(uid, ws_id, analysis, silent=silent)
        if not updated:
            return _err("Workspace not found", 404)
        return _ok("Analysis data updated")
    except Exception as e:
        logger.error("update_workspace_analysis failed: %s", e, exc_info=True)
        return _err("Could not update analysis data", 500)
