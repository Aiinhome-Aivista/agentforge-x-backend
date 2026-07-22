"""Settings service — manages LLM configuration and audit logs.

This service reads from and writes to the `system_settings` table.
It also tracks changes to LLM configurations in `llm_audit_logs`.
"""

import os
import logging
from typing import Dict, Any, List

from app.db.db_connection import get_mysql_connection

logger = logging.getLogger(__name__)

def _ensure_tables():
    """Ensure the system_settings and llm_audit_logs tables exist."""
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                setting_key VARCHAR(255) PRIMARY KEY,
                setting_value TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_audit_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                admin_id INT,
                provider VARCHAR(100),
                model_name VARCHAR(255),
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        conn.commit()
    except Exception as e:
        logger.error(f"Error ensuring settings tables: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def get_all_settings() -> Dict[str, str]:
    """Retrieve all system settings, falling back to environment variables if not in DB."""
    _ensure_tables()
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    db_settings = {}
    try:
        cur.execute("SELECT setting_key, setting_value FROM system_settings")
        rows = cur.fetchall()
        for row in rows:
            db_settings[row['setting_key']] = row['setting_value']
    except Exception as e:
        logger.error(f"Error reading settings: {e}")
    finally:
        cur.close()
        conn.close()

    # Map expected keys with fallback to OS env vars
    expected_keys = [
        "LLM_PROVIDER",
        "LLM_MODEL_NAME",
        "LLM_API_KEY",
        "LLM_BASE_URL"
    ]
    
    # Defaults mapping from old env vars
    defaults = {
        "LLM_PROVIDER": os.getenv("MISTRAL_MODE") or "Cloud",
        "LLM_MODEL_NAME": os.getenv("MODEL_NAME") or os.getenv("MISTRAL_LOCAL_MODEL") or "mistral-small-latest",
        "LLM_API_KEY": os.getenv("MISTRAL_API_KEY") or "",
        "LLM_BASE_URL": os.getenv("MISTRAL_LOCAL_URL") or ""
    }

    final_settings = {}
    for key in expected_keys:
        if key in db_settings:
            final_settings[key] = db_settings[key]
        else:
            final_settings[key] = defaults.get(key, "")

    # For UI, if it's mistral cloud we map it to "Mistral"
    if final_settings["LLM_PROVIDER"].lower() == "cloud":
        final_settings["LLM_PROVIDER"] = "Mistral"
        
    return final_settings

def update_settings(admin_id: int, updates: Dict[str, str]):
    """Update system settings and log the change if LLM provider/model changed."""
    _ensure_tables()
    
    current_settings = get_all_settings()
    
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        for key, value in updates.items():
            if value is None:
                value = ""
            cur.execute("""
                INSERT INTO system_settings (setting_key, setting_value)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
            """, (key, value))
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

    # Log change if provider or model name was updated
    new_provider = updates.get("LLM_PROVIDER")
    new_model = updates.get("LLM_MODEL_NAME")
    
    if (new_provider and new_provider != current_settings.get("LLM_PROVIDER")) or \
       (new_model and new_model != current_settings.get("LLM_MODEL_NAME")):
        _log_llm_change(
            admin_id=admin_id,
            provider=new_provider or current_settings.get("LLM_PROVIDER", ""),
            model_name=new_model or current_settings.get("LLM_MODEL_NAME", "")
        )

def _log_llm_change(admin_id: int, provider: str, model_name: str):
    conn = get_mysql_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO llm_audit_logs (admin_id, provider, model_name)
            VALUES (%s, %s, %s)
        """, (admin_id, provider, model_name))
        conn.commit()
    except Exception as e:
        logger.error(f"Error logging LLM change: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def get_llm_change_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve history of LLM configuration changes."""
    _ensure_tables()
    conn = get_mysql_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT l.id, l.provider, l.model_name, l.changed_at, u.name as admin_name
            FROM llm_audit_logs l
            LEFT JOIN users u ON l.admin_id = u.id
            ORDER BY l.changed_at DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall() or []
        for row in rows:
            if row.get("changed_at") and hasattr(row["changed_at"], "isoformat"):
                row["changed_at"] = row["changed_at"].isoformat()
        return rows
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return []
    finally:
        cur.close()
        conn.close()
