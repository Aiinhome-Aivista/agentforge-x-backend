"""Web search routes — search the live web from the analysis input panel.

Mounted under /api/search via the main api blueprint. New file — does not
modify any existing endpoint.

Routes:
  POST  /search/web             — auth: search the web. body: {query, max_results?}
  GET   /search/config          — auth: which provider is in use
"""

import logging

from flask import Blueprint, request, jsonify, g

from app.services.auth_helpers import require_auth
from app.services import web_search_service

logger = logging.getLogger(__name__)
search_bp = Blueprint("search", __name__, url_prefix="/search")


def _ok(message="OK", **extra):
    return jsonify({"status": True, "statuscode": 200, "message": message, **extra})


def _err(message, code=400, **extra):
    return jsonify({"status": False, "statuscode": code, "message": message, **extra}), code


@search_bp.get("/config")
@require_auth
def search_config():
    return _ok("OK", data={
        "provider": web_search_service.provider_name(),
    })


@search_bp.post("/web")
@require_auth
def search_web():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return _err("Field 'query' is required")
    if len(query) > 500:
        return _err("'query' is too long (max 500 chars)")

    try:
        max_results = int(body.get("max_results") or 8)
    except (TypeError, ValueError):
        max_results = 8

    try:
        result = web_search_service.search(query, max_results=max_results)
    except Exception as e:
        logger.error("web search failed for uid=%s: %s",
                     g.user.get("uid"), e, exc_info=True)
        return _err("Web search failed. Please try again.", 502)

    return _ok("OK", data=result)
