"""
app/api/code_generation_routes.py
─────────────────────────────────────────────────────────────────────────────
Code generation endpoint — implements the new "Download the Code" dropdown
entry on the Suggestion Details page.

Endpoint:
    GET /api/suggestions/<suggestion_key>/download-code
        returns: application/zip (attachment)
        filename: agentforgex_<suggestion_slug>.zip
"""

from __future__ import annotations

import io
import logging
import re
from flask import Blueprint, send_file, jsonify

from app.db.arango import get_db, COLLECTIONS
from app.services.code_generator_service import build_code_zip
from app.services.source_detector_service import _DEFAULT_SOURCE

logger = logging.getLogger(__name__)

code_gen_bp = Blueprint("code_gen", __name__)


def _slug(s: str) -> str:
    s = (s or "agentforgex").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "agentforgex"


@code_gen_bp.get("/suggestions/<suggestion_key>/download-code")
def download_code(suggestion_key: str):
    """
    Builds a ZIP scaffold tailored to the suggestion + its parent process +
    its technical design.  Returned inline as an attachment.
    """
    try:
        db = get_db()
        col = db.collection

        suggestion = col(COLLECTIONS["suggestions"]).get(suggestion_key)
        if not suggestion:
            return jsonify({"status": False, "message": "Suggestion not found."}), 404

        process_key = suggestion.get("process_key")
        process = col(COLLECTIONS["documents"]).get(process_key) if process_key else None
        steps = list(db.aql(
            "FOR s IN process_steps FILTER s.process_key == @key SORT s.step_number RETURN s",
            {"key": process_key or ""},
        ))

        # We deliberately call the technical-design route's builder lazily.
        # If that's not available for some reason, we still produce a usable
        # scaffold from suggestion + process + steps alone.
        try:
            from app.api.technical_design_routes import (
                _resolve_doc_context, _derive_header_fields,
                _build_full_prompt, _call_llm_full, _build_technical_design,
            )
            ctx = _resolve_doc_context(suggestion_key)
            header = _derive_header_fields(ctx)
            dyn = _call_llm_full(_build_full_prompt(ctx, header)) if ctx["found"] else {}
            technical_design = _build_technical_design(ctx, header, dyn)
        except Exception as e:
            logger.warning(f"[code-gen] technical-design fetch failed, using minimal: {e}")
            technical_design = {"sections": []}

        # Use the stored data lineage if present, else ADF default.
        data_lineage = (process or {}).get("data_lineage") or {
            "data_source": {
                "name": _DEFAULT_SOURCE["name"],
                "type": _DEFAULT_SOURCE["type"],
            },
            "data_target": {
                "name": "Downstream Process System",
                "type": "Downstream System",
            },
        }

        zip_bytes = build_code_zip(
            suggestion=suggestion,
            process=process or {},
            steps=steps,
            technical_design=technical_design,
            data_lineage=data_lineage,
        )

        title = suggestion.get("title") or "agentforgex"
        filename = f"agentforgex_{_slug(title)}.zip"

        return send_file(
            io.BytesIO(zip_bytes),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
            max_age=0,
        )

    except Exception as e:
        logger.error(f"[code-gen] failure for {suggestion_key}: {e}", exc_info=True)
        return jsonify({
            "status": False,
            "message": "Could not generate code bundle.",
        }), 500
