"""
Windows WSGI entrypoint using Waitress.
Use this instead of run.py when hosting on Windows VPS (IIS, NSSM, or direct).

Usage:
    python serve_windows.py

Or register as a Windows service with NSSM:
    nssm install ProcessAgentifier "C:\\Python312\\python.exe" "C:\\path\\to\\serve_windows.py"
    nssm set ProcessAgentifier AppDirectory "C:\\path\\to\\backend"
    nssm start ProcessAgentifier
"""

import os
import logging
from waitress import serve
from app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 3006))
    threads = int(os.getenv("WAITRESS_THREADS", 4))

    logger.info(f"Starting Waitress server on {host}:{port} with {threads} threads")
    serve(app, host=host, port=port, threads=threads)
