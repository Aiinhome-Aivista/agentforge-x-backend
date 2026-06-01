"""
app/api/blueprint_export_routes.py — UPDATED
─────────────────────────────────────────────────────────────────────────────
Public GET endpoints for the Process-Agentification Blueprint export.

Endpoints:
    GET /api/processes/<process_key>/blueprint-export        (existing)
    GET /api/suggestions/<suggestion_id>/blueprint-export    (NEW per spec)

Both return the same payload shape — every block is one of:
  {"type": "heading3",  "text": "..."}
  {"type": "paragraph", "text": "..."}
  {"type": "bullets",   "items": ["..."]}
  {"type": "table",     "headers": [...], "rows": [[...]]}

The payload mirrors the reference AgentForge_P2P_Agentic_Blueprint_v2.docx:
  - cover           (brand + title + subtitle + tagline + footer line)
  - sections[]      (§0..§9, each with title + lead + blocks[])
  - closing         (final paragraphs)

The frontend generators iterate `blocks` and render them with the formatting
from the reference document.  No content is hardcoded in the UI.

Difference between the two endpoints:
  - /processes/<key>/blueprint-export     → blueprint covers the whole process
  - /suggestions/<id>/blueprint-export    → blueprint is FOCUSED on the chosen
                                            suggestion's process step (the one
                                            with the higher agentic intervention)
"""

from __future__ import annotations

import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

blueprint_export_bp = Blueprint("blueprint_export", __name__)


@blueprint_export_bp.get("/processes/<process_key>/blueprint-export")
def get_blueprint_export(process_key: str):
    """
    Returns the full blueprint payload for the given process_key.  Always
    returns 200 with a complete payload — the builder fills missing data
    with deterministic defaults so the exported document is never empty.
    """
    try:
        from app.services.blueprint_builder_service import build_blueprint
        payload = build_blueprint(process_key)
        return jsonify({
            "status": True,
            "data":   payload,
        }), 200
    except Exception as e:
        logger.error(f"[blueprint-export] failure for {process_key}: {e}", exc_info=True)
        return jsonify({
            "status":  False,
            "message": "Could not build blueprint payload.",
            "data":    None,
        }), 500


@blueprint_export_bp.get("/suggestions/<suggestion_id>/blueprint-export")
def get_suggestion_blueprint_export(suggestion_id: str):
    """
    NEW — Returns the full blueprint payload FOCUSED ON the given suggestion.

    The payload shape is identical to the /processes/.../blueprint-export
    response (so the same frontend generators render it), but every section is
    written around the SPECIFIC suggestion the user selected.  The builder
    resolves the suggestion → its process step → its parent process, then
    composes the blueprint around that step.  The result is what the user
    expects when they click "EXPORT BLUEPRINT PDF/WORD/PPT" from a suggestion
    detail page.
    """
    try:
        from app.services.blueprint_builder_service import build_blueprint_for_suggestion
        payload = build_blueprint_for_suggestion(suggestion_id)
        return jsonify({
            "status": True,
            "data":   payload,
        }), 200
    except Exception as e:
        logger.error(
            f"[blueprint-export] suggestion failure for {suggestion_id}: {e}",
            exc_info=True,
        )
        return jsonify({
            "status":  False,
            "message": "Could not build suggestion blueprint payload.",
            "data":    None,
        }), 500
