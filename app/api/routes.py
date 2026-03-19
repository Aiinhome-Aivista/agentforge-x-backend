"""
Flask REST API routes for Process Agentifier.
"""

import os
import logging
from app.core.rag_service import rag_query
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.core.analysis_service import analysis_service

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"}
MAX_FILES = 20
MAX_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Health ────────────────────────────────────────────────────────────────────

@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "process-agentifier"})


# ── Upload & Analyze ──────────────────────────────────────────────────────────

@api_bp.post("/analyze")
def analyze():
    """
    Accepts multipart/form-data with one or more files.
    Runs the full analysis pipeline and returns the result.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "Empty file list"}), 400
    if len(uploaded) > MAX_FILES:
        return jsonify({"error": f"Maximum {MAX_FILES} files allowed"}), 400

    file_data = []
    for f in uploaded:
        if not f.filename:
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"File type not allowed: {f.filename}"}), 400

        file_bytes = f.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            return jsonify({"error": f"File too large: {f.filename} ({size_mb:.1f}MB)"}), 400

        file_data.append((file_bytes, secure_filename(f.filename)))

    if not file_data:
        return jsonify({"error": "No valid files found"}), 400

    try:
        result = analysis_service.analyze(file_data)
        return jsonify(result.to_api()), 200
    except ValueError as e:
        logger.error(f"Analysis config error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"error": "Analysis failed. Please try again."}), 500


# ── Process CRUD ──────────────────────────────────────────────────────────────

@api_bp.get("/processes")
def list_processes():
    try:
        processes = analysis_service.list_processes()
        return jsonify({"processes": processes})
    except Exception as e:
        logger.error(f"List processes error: {e}")
        return jsonify({"error": "Could not fetch processes"}), 500


@api_bp.get("/processes/<process_key>")
def get_process(process_key: str):
    try:
        result = analysis_service.get_process(process_key)
        if not result:
            return jsonify({"error": "Process not found"}), 404
        return jsonify(result)
    except Exception as e:
        logger.error(f"Get process error: {e}")
        return jsonify({"error": "Could not fetch process"}), 500


@api_bp.get("/processes/<process_key>/steps")
def get_steps(process_key: str):
    try:
        result = analysis_service.get_process(process_key)
        if not result:
            return jsonify({"error": "Process not found"}), 404
        return jsonify({"steps": result["steps"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.get("/processes/<process_key>/automation")
def get_automation(process_key: str):
    try:
        result = analysis_service.get_process(process_key)
        if not result:
            return jsonify({"error": "Process not found"}), 404
        return jsonify({"suggestions": result["suggestions"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── RAG Chat ──────────────────────────────────────────────────────────────

@api_bp.post("/chat")
def chat():
    data = request.json

    query = data.get("query")
    process_key = data.get("process_key")

    if not query:
        return jsonify({"error": "Query required"}), 400

    response = rag_query(query, process_key)
    # optional: graph url build
    graph_url = None
    if process_key:
        BASE_URL = os.getenv("BASE_URL")
        graph_url = f"{BASE_URL}/graphs/{process_key}/graph.html"

    return jsonify({
        "query": query,
        "answer": response,
        "graph_url": graph_url   
    })