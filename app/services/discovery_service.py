"""
app/services/discovery_service.py
─────────────────────────────────────────────────────────────────────────────
Process Discovery Assistant.

Given a processed file, this inspects the extracted process map (steps,
suggestions, ERP modules) and returns the highest-value follow-up QUESTIONS
that would refine the map and improve automation analysis — it deliberately
does NOT produce final recommendations.

Output shape (stable contract for the UI):
    {
      "process_key": "...",
      "summary": "...",
      "identified_gaps": ["...", ...],
      "follow_up_questions": [
          {"question": "...", "reason": "..."},
          ...
      ]
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.core.mistral_client import get_mistral_client
from app.prompts.prompts import build_discovery_prompt
# Reuse the single context loader already used by the chatbot so the two
# features always see an identical view of a process.
from app.services.chatbot_service import _load_process_context

logger = logging.getLogger(__name__)

MIN_QUESTIONS = 2
MAX_QUESTIONS = 4


def _empty_payload(process_key: str, summary: str) -> Dict[str, Any]:
    return {
        "process_key": process_key,
        "summary": summary,
        "identified_gaps": [],
        "follow_up_questions": [],
    }


def _coerce_questions(raw: Any) -> List[Dict[str, str]]:
    """Normalize the model's follow_up_questions into [{question, reason}]."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict):
            q = str(item.get("question", "")).strip()
            r = str(item.get("reason", "")).strip()
        elif isinstance(item, str):
            q, r = item.strip(), ""
        else:
            continue
        if q:
            out.append({"question": q, "reason": r})
    return out[:MAX_QUESTIONS]


def _coerce_gaps(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(g).strip() for g in raw if str(g).strip()]


def discover_process_gaps(process_key: str) -> Dict[str, Any]:
    """
    Analyze a process and return discovery JSON (summary, gaps, questions).

    Degrades gracefully:
      • unknown / empty process  -> empty payload with an explanatory summary
      • LLM failure / bad JSON    -> empty payload, error logged (never raises)
    """
    if not process_key:
        return _empty_payload("", "No process_key was provided.")

    ctx = _load_process_context(process_key)
    process = ctx.get("process")
    if not process:
        return _empty_payload(process_key, "Process not found.")

    steps = ctx.get("steps") or []
    suggestions = ctx.get("suggestions") or []
    erp_modules = ctx.get("erp_modules") or []

    if not steps:
        return _empty_payload(
            process_key,
            "No process steps have been extracted yet, so there is nothing to "
            "analyze for discovery questions.",
        )

    system_prompt, user_prompt = build_discovery_prompt(
        process_title=process.get("title", ""),
        process_description=process.get("description", ""),
        steps=steps,
        suggestions=suggestions,
        erp_modules=erp_modules,
    )

    try:
        llm = get_mistral_client()
        raw = llm._chat(system_prompt, user_prompt, temperature=0.2, force_json=True)
        parsed = llm._parse_json(raw)
    except Exception as e:  # noqa: BLE001 — never let discovery crash the request
        logger.error(f"[discovery] generation failed for {process_key}: {e}", exc_info=True)
        return _empty_payload(
            process_key,
            "Could not generate discovery questions at this time.",
        )

    if not isinstance(parsed, dict):
        logger.warning(f"[discovery] non-dict LLM output for {process_key}: {type(parsed).__name__}")
        return _empty_payload(process_key, "Discovery output was malformed.")

    questions = _coerce_questions(parsed.get("follow_up_questions"))
    gaps = _coerce_gaps(parsed.get("identified_gaps"))
    summary = str(parsed.get("summary", "")).strip()

    return {
        "process_key": process_key,
        "summary": summary,
        "identified_gaps": gaps,
        "follow_up_questions": questions,
    }
