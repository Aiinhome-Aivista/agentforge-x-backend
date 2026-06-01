"""
app/services/reanalysis_service.py
─────────────────────────────────────────────────────────────────────────────
Captures additional process context provided by the user inside the chatbot
and triggers a re-analysis of the process with the enriched corpus.

Flow:
  1.  Chatbot detects a "context provision" turn  (handled in chatbot_service)
  2.  Frontend renders a "Re-analyze process" action button.
  3.  Clicking it POSTs to /api/chatbot/reanalyze with the accumulated
      additional context.
  4.  This service:
        • Appends the new context to the process document (so it survives
          page reloads and shows up in the next exports).
        • Re-runs the analysis pipeline on the enriched description.
        • Returns the refreshed process payload.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.db.arango import get_db, COLLECTIONS
from app.core.analysis_service import analysis_service

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Persist additional context on the process doc (audit trail)
# ─────────────────────────────────────────────────────────────────────────────
def _append_context_to_process(
    process_key: str,
    additional_context: str,
    source: str = "chatbot",
) -> Dict[str, Any]:
    """
    Persists the new context as a versioned entry on the process document
    so the audit trail survives re-analyses.
    """
    db = get_db()
    coll = db.collection(COLLECTIONS["documents"])

    process = coll.get(process_key) or {}
    history: List[Dict[str, Any]] = process.get("context_revisions") or []
    history.append({
        "added_at":   datetime.utcnow().isoformat() + "Z",
        "source":     source,
        "context":    additional_context,
    })

    # Build the enriched description: original + all revisions joined
    base_desc = process.get("original_description") or process.get("description") or ""
    if "original_description" not in process:
        process["original_description"] = base_desc

    revision_text = "\n\n".join(
        f"[Context update {i+1} via {h['source']} on {h['added_at']}]\n{h['context']}"
        for i, h in enumerate(history)
    )
    enriched = (base_desc + "\n\n" + revision_text).strip()

    process["description"]        = enriched
    process["context_revisions"]  = history
    process["last_context_update"] = datetime.utcnow().isoformat() + "Z"

    coll.update(process)
    logger.info(
        f"[reanalysis] appended context to process {process_key} "
        f"(now {len(history)} revisions)"
    )
    return process


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Trigger the actual re-analysis
# ─────────────────────────────────────────────────────────────────────────────
def reanalyze_process(
    process_key: str,
    additional_context: str,
    source: str = "chatbot",
) -> Dict[str, Any]:
    """
    Persists the new context, then re-runs the analysis pipeline on the
    enriched description and returns the refreshed process payload.

    Returns:
        {
          "status":           "ok" | "error",
          "process_key":      str,
          "revision_count":   int,
          "refreshed":        bool,
          "message":          str,
          "process":          {...},   # the refreshed process payload
        }
    """
    additional_context = (additional_context or "").strip()
    if not additional_context:
        return {
            "status":   "error",
            "message":  "additional_context is empty",
            "process_key": process_key,
        }
    if not process_key:
        return {
            "status":  "error",
            "message": "process_key is required",
        }

    # 1. Persist the new context (also enriches `description`)
    process = _append_context_to_process(process_key, additional_context, source)

    # 2. Re-run analysis on the enriched description.
    #    We reuse the existing analysis pipeline with no file uploads — the
    #    enriched description carries the new context.
    try:
        result = analysis_service.analyze(
            files=[],
            user_input=process.get("description") or "",
            session_id=process.get("session_id") or process_key,
            existing_process_key=process_key,        # so we update rather than create
        )
        refreshed = result.to_api() if hasattr(result, "to_api") else result
    except TypeError:
        # Fallback for analysis_service.analyze() signatures that don't accept
        # `existing_process_key` yet — just re-run, the duplicate detector on
        # the analysis layer will reconcile.
        try:
            result = analysis_service.analyze(
                files=[],
                user_input=process.get("description") or "",
                session_id=process.get("session_id") or process_key,
            )
            refreshed = result.to_api() if hasattr(result, "to_api") else result
        except Exception as e:
            logger.error(f"[reanalysis] analyze() failed: {e}", exc_info=True)
            return {
                "status":  "error",
                "message": f"Re-analysis failed: {e}",
                "process_key": process_key,
                "revision_count": len(process.get("context_revisions") or []),
                "refreshed": False,
            }
    except Exception as e:
        logger.error(f"[reanalysis] analyze() failed: {e}", exc_info=True)
        return {
            "status":  "error",
            "message": f"Re-analysis failed: {e}",
            "process_key": process_key,
            "revision_count": len(process.get("context_revisions") or []),
            "refreshed": False,
        }

    return {
        "status":          "ok",
        "process_key":     process_key,
        "revision_count":  len(process.get("context_revisions") or []),
        "refreshed":       True,
        "message":         "Process re-analyzed with the new context.",
        "process":         refreshed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Helper: peek at the staged context without re-analyzing
# ─────────────────────────────────────────────────────────────────────────────
def list_context_revisions(process_key: str) -> List[Dict[str, Any]]:
    db = get_db()
    coll = db.collection(COLLECTIONS["documents"])
    process = coll.get(process_key) or {}
    return process.get("context_revisions") or []
