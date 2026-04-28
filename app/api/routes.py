import os
import logging
from app.core.rag_service import rag_query
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from app.db.db_connection import get_mysql_connection
from app.core.analysis_service import analysis_service
from app.core.simulation_service import SimulationService
from app.api.agent_routes import agent_bp
import uuid

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")
api_bp.register_blueprint(agent_bp)

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"}
MAX_FILES = 20
MAX_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50))

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


# ── Upload & Analyze ──────────────────────────────────────────────────────────
@api_bp.route("/login",methods=['POST'])
def login():
    data = request.get_json()
    email = data["email"]
    password = str(data["password"])

    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    # Email check
    sql = "SELECT * FROM users where email = %s"
    cursor.execute(sql,(email,))
    user = cursor.fetchone()
    if not user:
        return jsonify({
            "status": False,
            "statuscode": 404,
            "message": "Email not found"
        })

    # Password check
    if user["password"] != password:
        return jsonify({
            "status": False,
            "statuscode": 401,
            "message": "Incorrect password"
        })    


    cursor.close()
    conn.close()

    return jsonify({
        "status":True,
        "statuscode":200,
        "message":"Login Successfully!!",
        "data":{
            "id": user["id"],
            "name": user["name"]
        }
    })



# @api_bp.post("/analyze")
# def analyze():
#     """
#     Accepts multipart/form-data with one or more files.
#     Runs the full analysis pipeline and returns the result.
#     """
#     session_id = request.form.get("session_id")

#     if not session_id:
#         session_id = str(uuid.uuid4())

#     user_input = request.form.get("user_input", "").strip()

#     uploaded = request.files.getlist("files") if "files" in request.files else []

#     # 🔥 NEW: allow text-only input
#     if not uploaded and not user_input:
#         return jsonify({"error": "No input provided"}), 400

#     # uploaded = request.files.getlist("files")
#     file_data = []

#     for f in uploaded:
#         if not f.filename:
#             continue
#         if not allowed_file(f.filename):
#             return jsonify({"error": f"File type not allowed: {f.filename}"}), 400

#         file_bytes = f.read()
#         file_data.append((file_bytes, secure_filename(f.filename)))
#     if len(uploaded) > MAX_FILES:
#         return jsonify({"error": f"Maximum {MAX_FILES} files allowed"}), 400
#     user_input = request.form.get("user_input", "").strip()

#     file_data = []
#     for f in uploaded:
#         if not f.filename:
#             continue
#         if not allowed_file(f.filename):
#             return jsonify({"error": f"File type not allowed: {f.filename}"}), 400

#         file_bytes = f.read()
#         size_mb = len(file_bytes) / (1024 * 1024)
#         if size_mb > MAX_SIZE_MB:
#             return jsonify({"error": f"File too large: {f.filename} ({size_mb:.1f}MB)"}), 400

#         file_data.append((file_bytes, secure_filename(f.filename)))

#     if not file_data:
#         return jsonify({"error": "No valid files found"}), 400

#     try:
#         result = analysis_service.analyze(
#             file_data,
#             user_input=user_input,
#             session_id=session_id
#         )
#         return jsonify(result.to_api()), 200
#     except ValueError as e:
#         logger.error(f"Analysis config error: {e}")
#         return jsonify({"error": str(e)}), 400
#     except Exception as e:
#         logger.error(f"Analysis failed: {e}", exc_info=True)
#         return jsonify({"error": "Analysis failed. Please try again."}), 500

@api_bp.post("/analyze")
def analyze():
    print("🔥 ANALYZE API HIT")
    print("FILES:", request.files)
    print("FORM:", request.form)

    session_id = request.form.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())

    user_input = request.form.get("user_input", "").strip()

    uploaded = request.files.getlist("files") if "files" in request.files else []

    # ✅ Allow text-only OR file-only
    if not uploaded and not user_input:
        return jsonify({"error": "No input provided"}), 400

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
            return jsonify({
                "error": f"File too large: {f.filename} ({size_mb:.1f}MB)"
            }), 400

        file_data.append((file_bytes, secure_filename(f.filename)))

    # ❗ IMPORTANT: Only block if BOTH empty
    if not file_data and not user_input:
        return jsonify({"error": "No valid input found"}), 400

    try:
        result = analysis_service.analyze(
            file_data,
            user_input=user_input,
            session_id=session_id
        )
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


@api_bp.get("/processes/<_key>/flow")
def get_react_flow(_key: str):
    """
    Returns the process data mapped specifically for the React Flow frontend.
    """
    try:
        # Pass the _key value to your service
        flow_data = analysis_service.get_react_flow_data(_key)
        
        if not flow_data.get("lanes") or not flow_data.get("flow"):
            return jsonify({"error": "No flow data"}), 404
            
        return jsonify(flow_data)
    except Exception as e:
        logger.error(f"Get React flow error: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch process flow data"}), 500

    
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

