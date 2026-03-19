"""Flask application factory."""

import logging
import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_SIZE_MB", 50)) * 1024 * 1024

    # ── CORS ──────────────────────────────────────────────────────────────────
    # FRONTEND_ORIGINS accepts a comma-separated list, e.g.:
    #   http://yourdomain.com,https://yourdomain.com,http://192.168.1.10:3000
    # Set to * to allow all origins (not recommended for production).
    raw_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://localhost:5173")
    if raw_origins.strip() == "*":
        origins = "*"
    else:
        origins = [o.strip().rstrip("/") for o in raw_origins.split(",") if o.strip()]

    CORS(
        app,
        origins=origins,
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )

    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app
