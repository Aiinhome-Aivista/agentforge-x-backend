import os
from flask import Flask, send_from_directory
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    # ✅ Updated CORS (specific origin + credentials)
    CORS(
        app,
        resources={
            r"/agentforcex/api/*": {
                "origins": ["https://agentforge.services"]
            }
        },
        supports_credentials=True
    )

    # ✅ Register API routes with prefix
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/agentforcex/api")

    # 🔥 GRAPH SERVE ROUTE
    @app.route('/graphs/<process_key>/<file>')
    def serve_graph(process_key, file):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        graph_dir = os.path.join(PROJECT_ROOT, "graphs", process_key)

        print("SERVING FROM:", graph_dir)  # debug

        return send_from_directory(graph_dir, file)

    # ✅ After request headers (extra safety)
    @app.after_request
    def after_request(response):
        response.headers["Access-Control-Allow-Origin"] = "https://agentforge.services"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response

    return app