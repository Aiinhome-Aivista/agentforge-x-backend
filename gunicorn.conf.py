# gunicorn.conf.py
# Gunicorn configuration for Linux VPS deployment.
# Used by: gunicorn -c gunicorn.conf.py run:app

import os
import multiprocessing

# ── Binding ───────────────────────────────────────────────────────────────────
host = os.getenv("FLASK_HOST", "127.0.0.1")   # 127.0.0.1 when behind nginx
port = os.getenv("FLASK_PORT", "3006")
bind = f"{host}:{port}"

# ── Workers ───────────────────────────────────────────────────────────────────
# Rule of thumb: (2 x CPU cores) + 1
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
threads = int(os.getenv("GUNICORN_THREADS", 2))

# ── Timeouts ──────────────────────────────────────────────────────────────────
# LLM analysis can take 60-90s; set generous timeout
timeout = int(os.getenv("GUNICORN_TIMEOUT", 180))
graceful_timeout = 30
keepalive = 5

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")     # "-" = stdout
errorlog  = os.getenv("GUNICORN_ERROR_LOG",  "-")
loglevel  = os.getenv("LOG_LEVEL", "info").lower()

# ── Process naming ────────────────────────────────────────────────────────────
proc_name = "process-agentifier"

# ── Security ──────────────────────────────────────────────────────────────────
limit_request_line   = 8190
limit_request_fields = 100
