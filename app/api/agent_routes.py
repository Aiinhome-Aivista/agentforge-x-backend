from flask import Blueprint, request, jsonify
from app.db.db_connection import get_mysql_connection

agent_bp = Blueprint("agent", __name__, url_prefix="/agent")


@agent_bp.route("/run", methods=["POST"])
def run_agent_process():
    try:
        data = request.get_json()
        step_key = data.get("step_key")
        session_id = data.get("session_id")

        if not session_id:
            return jsonify({
                "status": "error",
                "message": "session_id is required"
            }), 400



        if not step_key:
            return jsonify({
                "status": "error",
                "message": "step_key is required"
            }), 400

        conn = get_mysql_connection()
        cursor = conn.cursor(dictionary=True)

        # 🔹 1. Get latest analysis_results
        cursor.execute(
            "SELECT id, steps FROM analysis_results WHERE session_id = %s ORDER BY id DESC LIMIT 1",
            (session_id,)
        )
        row = cursor.fetchone()
        
        step = None

        if row and row["steps"]:
            import json
            steps_json = row["steps"]

            if isinstance(steps_json, str):
                steps_json = json.loads(steps_json)

            # 🔹 2. Find step
            for s in steps_json:
                if (
                    s.get("step_key") == step_key or
                    s.get("_key") == step_key or
                    s.get("id") == step_key
                ):
                    step = s
                    break
            
            # 🔹 3. Update automation_potential = 0 in JSON
            if step:
                updated_flag = False
                for s in steps_json:
                    if (
                        s.get("step_key") == step_key or
                        s.get("_key") == step_key or
                        s.get("id") == step_key
                    ):
                        s["automation_potential"] = 0
                        updated_flag = True

                if updated_flag:
                    cursor.execute(
                        "UPDATE analysis_results SET steps = %s WHERE id = %s",
                        (json.dumps(steps_json), row["id"])
                    )

        # 🔹 4. Insert / Update agent_runs table
        cursor.execute("""
            INSERT INTO agent_runs (step_key, process_key, status, automation_potential)
            VALUES (%s, %s, 'executed', 0)
            ON DUPLICATE KEY UPDATE
                status = 'executed',
                automation_potential = 0
        """, (step_key, step.get("process_key") if step else None))

        conn.commit()

        return jsonify({
            "status": "success",
            "message": "Process Automation Complete",
            "step_key": step_key,
            "automation_potential": 0
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

