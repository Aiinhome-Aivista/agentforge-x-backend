"""
app/api/routes.py
─────────────────────────────────────────────────────────────────────────────
UPDATED (drop-in replacement)

Changes vs previous version:
  • Registers `chatbot_bp`  (/api/chatbot/*)
  • Registers `code_gen_bp` (/api/suggestions/<id>/download-code)
  • The `/analyze` endpoint now ALSO runs the new CSV / document
    source-target detection over the uploaded files and attaches the result
    to the response so the front-end can persist it alongside the analysis.
  • `/chat` route is preserved (legacy) but new clients should use
    /api/chatbot/ask which is scoped + intent-aware.
"""
import os
import logging
from app.core.rag_service import rag_query
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.db.db_connection import get_mysql_connection
from app.core.analysis_service import analysis_service
from app.core.simulation_service import SimulationService
from app.api.agent_routes import agent_bp
from app.api.auth_routes import auth_bp
from app.api.subscription_routes import billing_bp
from app.api.workspace_routes import workspace_bp
from app.api.dummy_payment_routes import dummy_billing_bp
from app.api.web_search_routes import search_bp
from app.api.captcha_routes import captcha_bp, register_auth_gate
from app.api.blog_routes import blog_bp
from app.api.admin_routes import admin_bp
from app.api.technical_design_routes import technical_design_bp
from app.api.chatbot_routes import chatbot_bp                # ⬅️ NEW
from app.api.code_generation_routes import code_gen_bp       # ⬅️ NEW
from app.api.blueprint_export_routes import blueprint_export_bp  # ⬅️ NEW (blueprint export API)
from app.api.sme_routes import sme_bp                          # ⬅️ NEW (SME-driven workflow)
from app.services.auth_helpers import require_auth, current_user
from flask import g

from app.services.source_detector_service import (             # ⬅️ NEW
    build_source_target_report,
)

import uuid

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")
api_bp.register_blueprint(agent_bp)
api_bp.register_blueprint(auth_bp)
api_bp.register_blueprint(billing_bp)
api_bp.register_blueprint(workspace_bp)
api_bp.register_blueprint(dummy_billing_bp)
api_bp.register_blueprint(search_bp)
api_bp.register_blueprint(captcha_bp)
api_bp.register_blueprint(blog_bp)
api_bp.register_blueprint(admin_bp)
api_bp.register_blueprint(technical_design_bp)
api_bp.register_blueprint(chatbot_bp)        # ⬅️ NEW
api_bp.register_blueprint(code_gen_bp)       # ⬅️ NEW
api_bp.register_blueprint(blueprint_export_bp)  # ⬅️ NEW (GET /processes/<key>/blueprint-export)
api_bp.register_blueprint(sme_bp)               # ⬅️ NEW (/sme/ingest, /sme/chat, /sme/finalize)
register_auth_gate(api_bp)

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"}
MAX_FILES = 20
MAX_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 5))

simulation_service = SimulationService()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Health ────────────────────────────────────────────────────────────────────
@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "process-agentifier"})


@api_bp.route("/test-db", methods=["GET"])
def test_db():
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return {
        "status": "success",
        "statuscode": 200,
        "message": "Database connection successful",
        "data": result
    }


# ── Login (legacy) ────────────────────────────────────────────────────────────
@api_bp.route("/login", methods=['POST'])
def login():
    data = request.get_json()
    email = data["email"]
    password = str(data["password"])

    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM users where email = %s"
    cursor.execute(sql, (email,))
    user = cursor.fetchone()
    if not user:
        return jsonify({"status": False, "statuscode": 404, "message": "Email not found"})
    if user["password"] != password:
        return jsonify({"status": False, "statuscode": 401, "message": "Incorrect password"})
    cursor.close()
    conn.close()
    return jsonify({
        "status": True, "statuscode": 200, "message": "Login Successfully!!",
        "data": {"id": user["id"], "name": user["name"]},
    })


# ── Upload & Analyze ──────────────────────────────────────────────────────────
@api_bp.post("/analyze")
@require_auth
def analyze():
    session_id = request.form.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        
    # Fetch user file size limit
    file_size_limit_mb = MAX_SIZE_MB
    if hasattr(g, 'user') and g.user and g.user.get('uid'):
        uid = g.user.get('uid')
        conn = get_mysql_connection()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute("SELECT file_size_limit_mb FROM users WHERE id = %s", (uid,))
                u = cur.fetchone()
                if u and u.get('file_size_limit_mb'):
                    file_size_limit_mb = float(u['file_size_limit_mb'])
        except Exception as e:
            logger.error(f"User limit check error: {e}")
        finally:
            conn.close()

    user_input = request.form.get("user_input", "").strip()
    mission_vision = request.form.get("mission_vision_context", "").strip()
    if mission_vision:
        user_input = f"Company Mission & Vision:\n{mission_vision}\n\n{user_input}".strip()

    uploaded = request.files.getlist("files") if "files" in request.files else []

    if not uploaded and not user_input:
        return jsonify({"error": "No input provided"}), 400
    if uploaded and len(uploaded) > MAX_FILES:
        return jsonify({"error": f"Maximum {MAX_FILES} files allowed"}), 400

    file_data = []
    total_size_mb = 0
    for f in uploaded:
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            return jsonify({"error": f"File type not allowed: {f.filename}"}), 400
        file_bytes = f.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        total_size_mb += size_mb
        file_data.append((file_bytes, secure_filename(f.filename)))

    if total_size_mb > file_size_limit_mb:
        return jsonify({"error": f"Total upload size ({total_size_mb:.1f}MB) exceeds your limit of {file_size_limit_mb:.1f}MB."}), 400

    if not file_data and not user_input:
        return jsonify({"error": "No valid input found"}), 400

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
                            "INSERT INTO user_uploaded_files (user_id, filename, size_mb) VALUES (%s, %s, %s)",
                            (uid, fname, size_mb)
                        )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error logging uploaded files: {e}")

        result = analysis_service.analyze(
            file_data,
            user_input=user_input,
            session_id=session_id,
        )

        # ── NEW: build source/target detection report from uploaded files ──
        combined_text = user_input or ""
        try:
            from app.parsers.file_parser import parse_file
            for fb, fname in file_data:
                if fname.lower().endswith((".csv",)):
                    continue  # CSV handled inside detector
                try:
                    text, _meta = parse_file(fb, fname)
                    combined_text += "\n\n" + (text or "")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            source_target_report = build_source_target_report(file_data, combined_text)
        except Exception as e:
            logger.warning(f"[/analyze] source-target detection failed: {e}")
            source_target_report = {
                "csv_source_detection": [],
                "document_data_lineage": {
                    "data_source": {
                        "name": "ADF (Azure Data Factory)",
                        "type": "Data Integration Service",
                        "evidence": "Fallback (detector failure).",
                    },
                    "data_target": {
                        "name": "Downstream Process System",
                        "type": "Downstream System",
                        "evidence": None,
                    },
                    "detection_method": "fallback",
                    "fallback_applied": True,
                    "fallback_reason": "Source/target detector raised an exception.",
                },
            }

        uploaded_files_log = [{"filename": fname, "size_mb": round(len(fb) / (1024 * 1024), 3)} for fb, fname in file_data]

        api_dict = result.to_api()
        api_dict["csv_source_detection"]  = source_target_report["csv_source_detection"]
        api_dict["document_data_lineage"] = source_target_report["document_data_lineage"]
        api_dict["uploaded_files_log"] = uploaded_files_log
        return jsonify(api_dict), 200

    except ValueError as e:
        logger.error(f"Analysis config error: {e}")
        error_str = str(e)
        if error_str.startswith("IRRELEVANT_FILE|"):
            parts = error_str.split("|")
            return jsonify({
                "status": "error",
                "error_type": "IRRELEVANT_FILE",
                "error": parts[1] if len(parts) > 1 else "Irrelevant file.",
                "recommended_solution": parts[2] if len(parts) > 2 else "Please upload a relevant file."
            }), 400
        return jsonify({"error": error_str}), 400
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


@api_bp.get("/processes/<_key>/flow")
def get_react_flow(_key: str):
    """Returns the process data mapped specifically for the React Flow frontend."""
    try:
        flow_data = analysis_service.get_react_flow_data(_key)
        if not flow_data.get("lanes") or not flow_data.get("flow"):
            return jsonify({"error": "No flow data"}), 404
        return jsonify(flow_data)
    except Exception as e:
        logger.error(f"Get React flow error: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch process flow data"}), 500


# ── RAG Chat (LEGACY — kept for backwards-compat; prefer /chatbot/ask) ───────
@api_bp.post("/chat")
def chat():
    data = request.json
    query = data.get("query")
    process_key = data.get("process_key")
    if not query:
        return jsonify({"error": "Query required"}), 400

    # Route legacy callers through the new scoped chatbot service so the
    # behaviour is identical regardless of which endpoint the client uses.
    from app.services.chatbot_service import answer_query
    result = answer_query(query, process_key)

    graph_url = None
    if process_key:
        BASE_URL = os.getenv("BASE_URL")
        graph_url = f"{BASE_URL}/graphs/{process_key}/graph.html"

    return jsonify({
        "query":       query,
        "answer":      result["answer"],
        "in_scope":    result["in_scope"],
        "intent":      result["intent"],
        "graph_url":   graph_url,
    })


@api_bp.get("/suggestions/<suggestion_key>/architecture")
def get_agent_architecture(suggestion_key: str):
    try:
        result = analysis_service.get_agent_architecture(suggestion_key)
        if not result:
            return jsonify({"error": "Suggestion not found"}), 404
        return jsonify(result)
    except Exception as e:
        logger.error(f"Architecture fetch error: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch architecture"}), 500


@api_bp.post("/simulate/<process_key>")
def simulate(process_key):
    try:
        data = analysis_service.get_process(process_key)
        if not data:
            return jsonify({"error": "Process not found"}), 404
        steps = data["steps"]
        suggestions = data["suggestions"]
        result = simulation_service.run_simulation(steps, suggestions)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
