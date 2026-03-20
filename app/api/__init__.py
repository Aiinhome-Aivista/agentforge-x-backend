import os
from flask_cors import CORS
from flask import Flask, send_from_directory


def create_app():
    app = Flask(__name__)

    # ✅ Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})

    # ✅ Register API routes
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    # 🔥 GRAPH SERVE ROUTE
    @app.route('/graphs/<process_key>/<file>')
    def serve_graph(process_key, file):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app folder
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        graph_dir = os.path.join(PROJECT_ROOT, "graphs", process_key)

        print("SERVING FROM:", graph_dir)  # debug

        return send_from_directory(graph_dir, file)

    return app