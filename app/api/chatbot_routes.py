"""
app/api/chatbot_routes.py — UPDATED for the Yes/No confirmation flow.

Endpoints:
    POST /api/chatbot/ask
        body: {
                "query":            "...",
                "process_key":      "<optional>",
                "pending_context":  "<echoed from previous bot turn when
                                      awaiting_confirmation was true>"
              }
        returns:
        {
          "status":              true,
          "answer":              "...",
          "in_scope":            true|false,
          "intent":              "show_steps" | "provide_context"
                                | "confirm_reanalyze_yes" | "confirm_reanalyze_no" | ... ,
          "process_key":         "...",
          "offer_reanalyze":      true|false,
          "awaiting_confirmation": true|false,
          "confirmed":           true|false,
          "captured_context":    "..."
        }

    POST /api/chatbot/reanalyze
        body: {
                "process_key":        "<required>",
                "additional_context": "<the captured context note>",
              }

    GET /api/chatbot/<process_key>/context-revisions
"""

from __future__ import annotations

import logging
from flask import Blueprint, request, jsonify

from app.services.chatbot_service   import answer_query, OUT_OF_SCOPE_MESSAGE
from app.services.reanalysis_service import reanalyze_process, list_context_revisions

logger = logging.getLogger(__name__)

chatbot_bp = Blueprint("chatbot", __name__)


@chatbot_bp.post("/chatbot/ask")
def ask_chatbot():
    try:
        data = request.get_json(silent=True) or {}
        query           = (data.get("query") or data.get("message") or "").strip()
        process_key     = data.get("process_key") or data.get("processKey") or None
        pending_context = (
            data.get("pending_context")
            or data.get("pendingContext")
            or None
        )

        if not query:
            return jsonify({
                "status":      False,
                "message":     "Query is required.",
                "answer":      "",
                "in_scope":    True,
                "intent":      None,
                "process_key": process_key,
            }), 400

        result = answer_query(query, process_key, pending_context=pending_context)

        # Build a flat payload — only include the optional keys when present
        payload = {
            "status":      True,
            "answer":      result["answer"],
            "in_scope":    result["in_scope"],
            "intent":      result["intent"],
            "process_key": result["process_key"],
        }
        for opt in ("offer_reanalyze", "awaiting_confirmation",
                    "confirmed", "captured_context"):
            if opt in result:
                payload[opt] = result[opt]

        return jsonify(payload), 200

    except Exception as e:
        logger.error(f"[chatbot] failure: {e}", exc_info=True)
        return jsonify({
            "status":      True,
            "answer":      OUT_OF_SCOPE_MESSAGE,
            "in_scope":    False,
            "intent":      None,
            "process_key": None,
        }), 200


@chatbot_bp.post("/chatbot/reanalyze")
def reanalyze_chatbot():
    """Triggered when the user clicks Yes on the confirmation prompt."""
    try:
        data = request.get_json(silent=True) or {}
        process_key = (data.get("process_key") or data.get("processKey") or "").strip()
        additional_context = (
            data.get("additional_context")
            or data.get("captured_context")
            or data.get("context")
            or ""
        ).strip()

        if not process_key:
            return jsonify({
                "status":  False,
                "message": "process_key is required.",
            }), 400
        if not additional_context:
            return jsonify({
                "status":  False,
                "message": "additional_context is required.",
            }), 400

        result = reanalyze_process(process_key, additional_context, source="chatbot")
        ok = result.get("status") == "ok"
        return jsonify({
            "status":         ok,
            "message":        result.get("message", ""),
            "process_key":    result.get("process_key", process_key),
            "revision_count": result.get("revision_count", 0),
            "refreshed":      result.get("refreshed", False),
            "process":        result.get("process"),
        }), (200 if ok else 500)

    except Exception as e:
        logger.error(f"[chatbot/reanalyze] failure: {e}", exc_info=True)
        return jsonify({
            "status":  False,
            "message": "Re-analysis failed unexpectedly. Please try again.",
        }), 500


@chatbot_bp.get("/chatbot/<process_key>/context-revisions")
def get_context_revisions(process_key: str):
    try:
        revisions = list_context_revisions(process_key)
        return jsonify({
            "status":      True,
            "process_key": process_key,
            "count":       len(revisions),
            "revisions":   revisions,
        }), 200
    except Exception as e:
        logger.error(f"[chatbot/context-revisions] failure: {e}", exc_info=True)
        return jsonify({
            "status":    False,
            "message":   "Could not load context revisions.",
            "revisions": [],
        }), 500
