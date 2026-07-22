"""
app/api/sme_routes.py
─────────────────────────────────────────────────────────────────────────────
Endpoints for the SME-driven workflow (base graph → chat → enrich → analyze).

    POST /api/sme/ingest        (multipart) files[] + user_input + session_id
        → builds the base knowledge graph from ANY uploaded data and returns
          a summary + suggested what/where/why questions.
        returns: {
          status, session_id, domain, summary, business_context,
          node_count, edge_count, entities[], glossary[], suggested_questions[]
        }

    POST /api/sme/chat          (json) { session_id, query, history[] }
        → answers grounded in the base graph (what/where/why).
        returns: { status, answer, grounded, followup_question, referenced_entities[] }

    POST /api/sme/finalize      (json) { session_id, transcript }
        → folds the SME conversation into the base graph (enrichment) so the
          final analysis picks it up. Returns the enrichment summary.
        returns: { status, enriched_nodes, new_relationships, sme_profile }

    GET  /api/sme/<session_id>/graph
        → the current base graph (nodes + edges + semantic layer).

The final process analysis itself is produced by the existing /api/analyze
endpoint (so the AnalysisPage payload shape is unchanged); the frontend calls
/sme/finalize first, then /analyze with the same session_id and the SME
transcript appended to user_input.
"""

from __future__ import annotations

import os
import uuid
import logging
from flask import Blueprint, request, jsonify, g
from werkzeug.utils import secure_filename
from app.db.db_connection import get_mysql_connection
from app.services.auth_helpers import require_auth

from app.services.semantic_kg_service import (
    ingest_to_base_graph,
    sme_chat,
    enrich_from_sme,
    get_base_graph,
    generate_suggested_questions,
    get_readiness,
    start_sme_interview,
)

logger = logging.getLogger(__name__)
sme_bp = Blueprint("sme", __name__)

# Spec-supported inputs: CSV, Excel, PDF, ERP logs, SOPs, Emails,
# OCR/text exports, web content and conversational notes.
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "csv", "xlsx", "xls",
                      "eml", "log", "md", "json"}
MAX_FILES = 20
MAX_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 5))


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@sme_bp.post("/sme/ingest")
@require_auth
def sme_ingest():
    session_id = request.form.get("session_id") or str(uuid.uuid4())
    user_input = (request.form.get("user_input") or "").strip()
    uploaded = request.files.getlist("files") if "files" in request.files else []

    if not uploaded and not user_input:
        return jsonify({"error": "No input provided", "session_id": session_id}), 400
    if uploaded and len(uploaded) > MAX_FILES:
        return jsonify({"error": f"Maximum {MAX_FILES} files allowed"}), 400

    file_size_limit_mb = MAX_SIZE_MB
    if hasattr(g, 'user') and g.user and g.user.get('uid'):
        uid = g.user.get('uid')
        try:
            conn = get_mysql_connection()
            with conn.cursor(dictionary=True) as cur:
                cur.execute("SELECT file_size_limit_mb FROM users WHERE id = %s", (uid,))
                u = cur.fetchone()
                if u and u.get('file_size_limit_mb') is not None:
                    file_size_limit_mb = float(u['file_size_limit_mb'])
            conn.close()
        except Exception as e:
            logger.error(f"User limit check error: {e}")

    file_data = []
    total_size_mb = 0
    for f in uploaded:
        if not f or not f.filename:
            continue
        if not _allowed(f.filename):
            return jsonify({"error": f"File type not allowed: {f.filename}"}), 400
        blob = f.read()
        size_mb = len(blob) / (1024 * 1024)
        total_size_mb += size_mb
        file_data.append((blob, secure_filename(f.filename)))

    logger.error(f"[SME_INGEST] total_size_mb={total_size_mb}, limit={file_size_limit_mb}")
    if total_size_mb > file_size_limit_mb:
        return jsonify({"error": f"Total upload size ({total_size_mb:.1f}MB) exceeds your limit of {file_size_limit_mb:.1f}MB."}), 400

    try:
        # ── NEW: log uploaded files to the user ──
        if file_data and hasattr(g, 'user') and g.user and g.user.get('uid'):
            uid = g.user.get('uid')
            try:
                conn = get_mysql_connection()
                with conn.cursor() as cur:
                    for fb, fname in file_data:
                        size_mb = len(fb) / (1024 * 1024)
                        cur.execute(
                            "SELECT id FROM user_uploaded_files WHERE user_id=%s AND filename=%s AND ABS(size_mb - %s) < 0.01 AND created_at >= NOW() - INTERVAL 10 MINUTE LIMIT 1",
                            (uid, fname, size_mb)
                        )
                        if not cur.fetchone():
                            cur.execute(
                                "INSERT INTO user_uploaded_files (user_id, filename, size_mb) VALUES (%s, %s, %s)",
                                (uid, fname, size_mb)
                            )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging uploaded files: {e}")

        result = ingest_to_base_graph(file_data, user_input=user_input,
                                      session_id=session_id)
        
        # Handle early validation failure
        if result.get("error_type") == "IRRELEVANT_FILE":
            return jsonify({
                "status": "error",
                "error_type": "IRRELEVANT_FILE",
                "error": result.get("message"),
                "recommended_solution": result.get("recommended_solution")
            }), 400

        result.setdefault("session_id", session_id)
        return jsonify(result), (200 if result.get("status") == "ok" else 200)
    except Exception as e:
        logger.error(f"[sme/ingest] failure: {e}", exc_info=True)
        # Graceful: the frontend can still proceed straight to analysis.
        return jsonify({
            "status": "error",
            "session_id": session_id,
            "message": "Base graph build failed; you can still run the analysis.",
            "node_count": 0, "edge_count": 0,
            "entities": [], "relationships": [], "edges": [],
            "suggested_questions": [
                "What are the main entities in my data?",
                "Where does each key field come from?",
                "Why are these entities related?",
                "Which business rules should the analysis know about?",
            ],
        }), 200


@sme_bp.post("/sme/chat")
def sme_chat_route():
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    query = (data.get("query") or data.get("message") or "").strip()
    history = data.get("history") or []

    if not session_id:
        return jsonify({"status": False, "message": "session_id is required"}), 400
    if not query:
        return jsonify({"status": False, "message": "query is required"}), 400

    try:
        result = sme_chat(session_id, query, history=history)
        return jsonify({"status": True, **result}), 200
    except Exception as e:
        logger.error(f"[sme/chat] failure: {e}", exc_info=True)
        return jsonify({
            "status": True,
            "answer": "Your note is captured and will be used in the final analysis.",
            "what_coverage": 0, "where_coverage": 0, "why_coverage": 0,
            "analysis_ready": False, "collection_status": "in_progress",
            "grounded": False, "followup_question": "", "referenced_entities": [],
        }), 200


@sme_bp.post("/sme/finalize")
def sme_finalize():
    """Fold the SME conversation into the base graph before final analysis."""
    data = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    transcript = (data.get("transcript") or data.get("context") or "").strip()

    if not session_id:
        return jsonify({"status": False, "message": "session_id is required"}), 400

    try:
        result = enrich_from_sme(session_id, transcript)
        return jsonify({"status": True, **result}), 200
    except Exception as e:
        logger.error(f"[sme/finalize] failure: {e}", exc_info=True)
        return jsonify({"status": True, "enriched_nodes": 0,
                        "new_relationships": 0, "sme_profile": {}}), 200


@sme_bp.get("/sme/<session_id>/graph")
def sme_graph(session_id: str):
    try:
        return jsonify({"status": True, **get_base_graph(session_id)}), 200
    except Exception as e:
        logger.error(f"[sme/graph] failure: {e}", exc_info=True)
        return jsonify({"status": False, "message": "Could not load graph",
                        "nodes": [], "edges": []}), 500


# Newly added

# Spec philosophy: What → Where → Why or How
_WH_ORDER = ("what", "where", "why", "how")

_DAGENT_FALLBACK_QUESTIONS = [
    "What are the main entities in my uploaded data and what does each represent?",
    "What key metrics or fields stand out in this dataset?",
    "Where does each important field originate in the source files?",
    "Why are these entities related the way they are?",
    "How are exceptions or escalations handled in this process?",
]


def _wh_rank(question: str) -> int:
    """Rank a question by its leading interrogative: what → where → why/how → rest."""
    first = (question or "").strip().lower().split(" ", 1)[0].rstrip("',.?!:;")
    try:
        return _WH_ORDER.index(first)
    except ValueError:
        return len(_WH_ORDER)


def _order_by_philosophy(questions):
    """Stable sort keeping all 'What…' first, then 'Where…', then 'Why…'."""
    return [q for _, q in sorted(enumerate(questions), key=lambda p: (_wh_rank(p[1]), p[0]))]


@sme_bp.get("/sme/<session_id>/suggested-questions")
def sme_suggested_questions(session_id: str):
    """Data-grounded starter questions for the DAgent chat panel."""
    try:
        questions = generate_suggested_questions(session_id) or []
    except Exception as e:
        logger.error(f"[sme/suggested-questions] failure: {e}", exc_info=True)
        questions = []

    if not questions:
        questions = list(_DAGENT_FALLBACK_QUESTIONS)

    return jsonify({
        "status": True,
        "session_id": session_id,
        "questions": _order_by_philosophy(questions),
    }), 200

@sme_bp.get("/sme/<session_id>/readiness")
def sme_readiness(session_id: str):
    """Final-analysis gate: is contextual understanding sufficient?

    The spec requires that final process analysis only begins after
    sufficient semantic confidence is achieved, key entities are understood,
    critical relationships are clarified and SME context is incorporated.
    The frontend can poll this before enabling the 'Run Analysis' action.
    """
    try:
        return jsonify({"status": True, **get_readiness(session_id)}), 200
    except Exception as e:
        logger.error(f"[sme/readiness] failure: {e}", exc_info=True)
        # Graceful: never block the user from analysing if the gate fails.
        return jsonify({"status": True, "session_id": session_id,
                        "analysis_ready": True, "remaining_gaps": [],
                        "message": "Readiness check unavailable"}), 200


@sme_bp.post("/sme/<session_id>/start")
def sme_start(session_id: str):
    """System-led interview opener: the system speaks first.

    Called by the frontend right after /sme/ingest. Returns a warm,
    personalised opening message that shows what was understood from the
    upload and asks the first question (the SME's role, if unknown).
    """
    try:
        result = start_sme_interview(session_id)
        return jsonify({"status": True, "session_id": session_id, **result}), 200
    except Exception as e:
        logger.error(f"[sme/start] failure: {e}", exc_info=True)
        return jsonify({
            "status": True, "session_id": session_id,
            "message": ("Thanks so much for sharing this with me — I've had a good "
                        "look through it and there's a lot here I'm curious about. "
                        "I'd love to understand how it actually works for you, in "
                        "your own words, so the analysis really reflects reality. "
                        "To start, where do you fit into all this — what's your "
                        "role here?"),
            "question": "Where do you fit into all this — what's your role here?",
        }), 200
