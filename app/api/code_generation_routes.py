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


@code_gen_bp.post("/suggestions/<suggestion_key>/download-code")
def download_code(suggestion_key: str):
    """
    Builds a ZIP scaffold tailored to the suggestion + its parent process +
    its technical design.  Returned inline as an attachment.
    """
    try:
        from flask import request
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
            try:
                from app.api.technical_design_routes import _DEF_AGENTS, _DEF_TOOLS
                technical_design = {"sections": [], "agents": _DEF_AGENTS, "tools": _DEF_TOOLS}
            except Exception:
                technical_design = {"sections": [], "agents": [], "tools": []}

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

        # Attempt to override tech stack based on chat history
        chat_history = []
        if request.is_json:
            chat_history = request.get_json(silent=True).get("chat_history", [])
        
        if chat_history:
            try:
                from app.core.mistral_client import get_mistral_client
                llm = get_mistral_client()
                chat_text = "\n".join([f"{msg.get('sender') or msg.get('role')}: {msg.get('text') or msg.get('content')}" for msg in chat_history])
                prompt = (
                    "Extract the technology stack mentioned in this chat history.\n"
                    f"Chat:\n{chat_text}\n\n"
                    "Return ONLY JSON strictly in this format (and ONLY include what is explicitly mentioned, else use defaults):\n"
                    "{\"data_source\": {\"name\": \"...\", \"type\": \"...\"}, \"data_target\": {\"name\": \"...\", \"type\": \"...\"}}"
                )
                resp = llm._chat_json(prompt, "Return valid JSON", expect="object")
                if resp and "data_source" in resp and "name" in resp["data_source"]:
                    data_lineage = resp
            except Exception as e:
                logger.warning(f"[code-gen] Failed to extract tech stack from chat history: {e}")

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
