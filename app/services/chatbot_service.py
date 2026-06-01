"""
app/services/chatbot_service.py
─────────────────────────────────────────────────────────────────────────────
AgentForgeX Chatbot Service — UPDATED for the Yes/No re-analyze flow.

WHAT CHANGED IN THIS VERSION
────────────────────────────
The original re-analyze flow returned a long markdown reply that the user
could not act on cleanly.  The new flow is much tighter:

  1. User sends a chat message.
  2. If we detect "context provision" keywords (re-analyze, add a step,
     extra step, actually we, etc.) → respond with the EXACT prompt:

         Do you want me to re-create the process map based on your
         suggested context "<the captured context>" ?
         Please acknowledge ( Yes / No ) ?

     and set `awaiting_confirmation = True` in the response payload.
     The frontend renders "Thinking…" while waiting AND renders Yes / No
     buttons under the bot bubble.

  3. The frontend remembers the most recent `captured_context` and the
     `awaiting_confirmation` flag.  When the user clicks Yes (or types
     "yes"/"y"/"sure"/"ok"), the chat calls /api/chatbot/reanalyze with
     the captured context.

  4. When the user clicks No (or types "no"/"n"/"cancel"), the bot
     acknowledges and drops the pending context.

This module also exposes `is_yes_word` / `is_no_word` helpers so the
chat route can interpret typed acknowledgments.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.core.mistral_client import get_mistral_client
from app.core.rag_service import rag_query
from app.db.arango import get_db, COLLECTIONS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OUT-OF-SCOPE MESSAGE (exact text from the spec)
# ─────────────────────────────────────────────────────────────────────────────
OUT_OF_SCOPE_MESSAGE = (
    "This question is not related to AgentForgeX workflow/process analysis. "
    "Please ask relevant process or automation questions."
)


# Allowed-topic vocabulary — generous; the LLM classifier handles edge cases.
_IN_SCOPE_KEYWORDS = {
    # process / workflow
    "process", "workflow", "swimlane", "step", "steps", "lane", "stage",
    "actor", "owner", "rac", "raci",
    # automation
    "automation", "automate", "automated", "agentic", "agent", "orchestrator",
    "ai", "rpa", "bot",
    # data & lineage
    "data source", "data target", "data lineage", "source system",
    "csv", "erp", "sap", "oracle", "mysql", "sql server", "salesforce",
    "servicenow", "workday", "adf",
    # technical design topics
    "architecture", "framework", "rag", "guardrail", "memory", "tool",
    "tool ecosystem", "tech stack",
    # the product itself
    "agentforgex", "agentforge", "blueprint", "suggestion", "recommendation",
    # context-edit verbs (so re-analyze requests are NEVER classified out-of-scope)
    "analyze", "analyse", "re-analyze", "re-analyse", "reanalyze", "reanalyse",
    "re-create", "recreate", "process map", "audit", "audit trail", "log",
}


_INTENT_PATTERNS = {
    "explain_process":    [r"\bexplain\s+(?:the\s+)?process\b", r"\boverview\b", r"\bsummary\b", r"\bwhat is this process\b"],
    "show_steps":         [r"\bshow\s+(?:the\s+)?(?:process\s+)?steps?\b", r"\bsteps\b", r"\bstep\s+breakdown\b", r"\blist steps\b"],
    "automation_details": [r"\bautomation\s+details?\b", r"\bautomation\s+suggestions?\b", r"\bagentic\s+suggestions?\b", r"\bautomate\b"],
    "actors":             [r"\bactors?\b", r"\bowners?\b", r"\bwho\s+does\b"],
    "source_target":      [r"\bdata\s+source\b", r"\bdata\s+target\b", r"\bdata\s+lineage\b", r"\bsource\s+system\b"],
    "kpis":               [r"\bkpis?\b", r"\bmetrics?\b", r"\btargets?\b"],
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT-PROVISION DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# These patterns are intentionally broad — any of them firing on a NON-question
# message makes the bot offer a re-analysis confirmation.

_CONTEXT_PROVISION_PATTERNS = [
    # Explicit re-analysis verbs
    r"\bre-?analy[sz]e\b",
    r"\b(?:can|could|please|kindly)\s+(?:you\s+)?(?:re-?)?analy[sz]e\b",
    r"\bre-?create\s+(?:the\s+)?process\b",
    r"\brecreate\s+(?:the\s+)?process\s+map\b",
    r"\bupdate\s+(?:the\s+)?(?:analysis|process|process map|workflow)\b",
    r"\brefresh\s+(?:the\s+)?(?:analysis|process|workflow)\b",
    r"\brerun\s+(?:the\s+)?analysis\b",
    # User wants to ADD content
    r"\bi\s+want\s+(?:to\s+)?add\b",
    r"\bi\s+(?:also\s+)?need(?:ed)?\s+to\s+(?:add|mention|include)\b",
    r"\b(?:to\s+)?(?:add|mention|include)\s+(?:a\s+|an\s+|the\s+)?(?:extra|new|additional|another)\b",
    r"\b(?:add|insert)\s+(?:a\s+|an\s+|one\s+)?(?:extra|new|additional|another)?\s*step\b",
    r"\bextra\s+(?:step|context|info|detail)\b",
    r"\badditional(?:ly)?\s+(?:step|context|info|note|detail)\b",
    r"\bone more\s+(?:thing|note|detail|step|point)\b",
    # Corrections / additions
    r"\bactually\b",
    r"\bwait[,]?\b",
    r"\bi\s+forgot\b",
    r"\bcorrection[:,]?\b",
    r"\b(?:let me\s+)?correct\b",
    r"\b(?:we|they)\s+(?:also\s+)?(?:do|don'?t|use|need|have|require)\b",
    r"\b(?:we|they)\s+(?:additionally|further|also)\b",
    r"\bthe\s+(?:real|actual)\s+(?:flow|process|step)\b",
    r"\bafter\s+(?:the\s+)?\w+\s+(?:step|log|update|action)\b",
]


# def _looks_like_context_provision(query: str) -> bool:
#     """Deterministic detector — fast, no LLM round-trip."""
#     q = (query or "").lower().strip()
#     if len(q) < 6:
#         return False

#     for pat in _CONTEXT_PROVISION_PATTERNS:
#         if re.search(pat, q):
#             # If the message ends with "?" treat it as a real question UNLESS
#             # it's a direct re-analyze ask.
#             if q.endswith("?"):
#                 if re.search(
#                 r"\bre-?analy[sz]e\b|"
#                 r"\brecreate\b|"
#                 r"\bre-?create\b|"
#                 r"\bupdate\s+(?:the\s+)?(?:analysis|process|workflow)\b|"
#                 r"\bmodify\b|"
#                 r"\bchange\b|"
#                 r"\bimprove\b|"
#                 r"\benhance\b|"
#                 r"\badd\b|"
#                 r"\binclude\b|"
#                 r"\bextra\s+step\b|"
#                 r"\badditional\s+step\b|"
#                 r"\bnew\s+step\b|"
#                 r"\bworkflow\s+change\b|"
#                 r"\bprocess\s+improvement\b|"
#                 r"\bdependency\b|"
#                 r"\broot\s+cause\b|"
#                 r"\bgraph\s+analysis\b|"
#                 r"\bvector\s+analysis\b",
#                 q
#             ):
#                     return True
#                 continue
#             return True

#     return False

def _looks_like_context_provision(query: str) -> bool:
    """Deterministic detector — fast, no LLM round-trip."""
    
    q = (query or "").lower().strip()

    if len(q) < 6:
        return False

    for pat in _CONTEXT_PROVISION_PATTERNS:
        if re.search(pat, q):

            # If message is a question mark sentence,
            # still allow explicit workflow modification intents
            if q.endswith("?"):

                if re.search(
                    r"\bre-?analy[sz]e\b|"
                    r"\brecreate\b|"
                    r"\bre-?create\b|"
                    r"\bupdate\s+(?:the\s+)?(?:analysis|process|workflow)\b|"
                    r"\bmodify\b|"
                    r"\bchange\b|"
                    r"\bimprove\b|"
                    r"\benhance\b|"
                    r"\badd\b|"
                    r"\binclude\b|"
                    r"\bextra\s+step\b|"
                    r"\badditional\s+step\b|"
                    r"\bnew\s+step\b|"
                    r"\bworkflow\s+change\b|"
                    r"\bprocess\s+improvement\b|"
                    r"\bdependency\b|"
                    r"\broot\s+cause\b|"
                    r"\bgraph\s+analysis\b|"
                    r"\bvector\s+analysis\b",
                    q
                ):
                    return True

                continue

            return True

    return False
_QUESTION_STARTERS = {
    "what", "why", "how", "when", "where", "who", "which", "is", "are",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "show", "explain", "list", "tell", "describe",
}


def _starts_like_question(q: str) -> bool:
    first = (q.split() or [""])[0].lower()
    return first in _QUESTION_STARTERS


def _llm_classify_context_provision(query: str) -> Optional[bool]:
    """LLM tie-breaker. Returns True/False/None."""
    try:
        llm = get_mistral_client()
        prompt = (
            "Classify the user's chat message as one of:\n"
            "  - QUESTION  (asking about something already known)\n"
            "  - CONTEXT   (adding NEW information / correcting / extending\n"
            "    a process they're being analyzed for)\n"
            "  - OTHER     (greeting, small talk, off-topic)\n\n"
            f"Message: \"\"\"{query}\"\"\"\n\n"
            "Respond with EXACTLY one word: QUESTION, CONTEXT, or OTHER."
        )
        resp = llm.client.chat.complete(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You classify chat messages. Respond with one word."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if "CONTEXT"  in raw: return True
        if "QUESTION" in raw: return False
        if "OTHER"    in raw: return False
        return None
    except Exception as e:
        logger.debug(f"[chatbot] LLM classification failed: {e}")
        return None


def _summarise_context_for_reanalysis(query: str) -> str:
    """
    Turn the user's chat message into a short, clean context note that we
    quote in the confirmation prompt AND feed into re-analysis.

    The output MUST stay short — ideally one sentence — because it goes
    inside the canonical confirmation string.  Aggressive guard rails on
    length so a chatty LLM can't blow it up.
    """
    base = (query or "").strip()
    # Fast path — short messages don't need an LLM round trip
    if len(base.split()) <= 18:
        return _light_normalize(base)

    try:
        llm = get_mistral_client()
        prompt = (
            "Rewrite the user's chat message as ONE short, neutral, third-person "
            "process-description sentence. RULES:\n"
            "  • Maximum 24 words.\n"
            "  • Strip filler ('actually', 'wait', 'I forgot', 'please re-analyze', "
            "    'I want to').\n"
            "  • Preserve actor names, system names, step names verbatim.\n"
            "  • Do NOT format as a list, headings, or markdown.\n"
            "  • Output ONLY the sentence — no preamble, no quotes.\n\n"
            f"Message: \"\"\"{query}\"\"\""
        )
        resp = llm.client.chat.complete(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You rewrite chat messages into short process notes."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        out = (resp.choices[0].message.content or "").strip()
        out = out.strip('"').strip("'").strip("•").strip("-").strip()
        # Hard guard: never exceed 40 words; fall back to the raw query
        if not out or len(out.split()) > 40:
            return _light_normalize(base)
        # Take only the first sentence
        first_sentence = re.split(r"(?<=[.!?])\s", out)[0].strip()
        return first_sentence or _light_normalize(base)
    except Exception:
        return _light_normalize(base)


def _light_normalize(text: str) -> str:
    """Strip filler verbs and collapse whitespace — no LLM."""
    t = (text or "").strip()
    t = re.sub(r"^(?:please\s+)?(?:can|could|would|kindly)\s+you\s+", "", t, flags=re.I)
    t = re.sub(r"^(?:i\s+want\s+(?:to\s+)?|i'?d\s+like\s+to\s+|let'?s\s+)", "", t, flags=re.I)
    t = re.sub(r"\bplease\s+re-?analy[sz]e\s+(?:the\s+)?process\.?$", "", t, flags=re.I)
    t = re.sub(r"\bre-?analy[sz]e\s+(?:the\s+)?process\.?$", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip().rstrip(".,;:")
    # Capitalize first letter
    return t[:1].upper() + t[1:] if t else text


def build_confirmation_prompt(captured_context: str) -> str:
    """The canonical prompt format requested by the spec."""
    safe = (captured_context or "").strip().replace('"', "'")
    return (
        f'Do you want me to re-create the process map based on your '
        f'suggested context "{safe}" ?\n'
        f'Please acknowledge ( Yes / No ) ?'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Yes/No interpretation — for users who type instead of clicking
# ─────────────────────────────────────────────────────────────────────────────
_YES_WORDS = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "okey",
    "confirm", "confirmed", "proceed", "do it", "go ahead", "go", "affirmative",
    "👍", "✅", "yes please", "ya",
}
_NO_WORDS = {
    "no", "n", "nope", "nah", "cancel", "stop", "skip", "drop", "abort",
    "negative", "don't", "not now", "later", "❌",
}


def is_yes_word(text: str) -> bool:
    t = (text or "").strip().lower().rstrip(".!?,;:")
    return t in _YES_WORDS or t.startswith("yes ") or t.startswith("yes,")


def is_no_word(text: str) -> bool:
    t = (text or "").strip().lower().rstrip(".!?,;:")
    return t in _NO_WORDS or t.startswith("no ") or t.startswith("no,")


# ─────────────────────────────────────────────────────────────────────────────
# Scope classifier
# ─────────────────────────────────────────────────────────────────────────────
def _keyword_in_scope(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in _IN_SCOPE_KEYWORDS)


def _llm_in_scope(query: str) -> Optional[bool]:
    try:
        llm = get_mistral_client()
    except Exception:
        return None

    try:
        prompt = (
            "Is this user message about a business process / workflow / "
            "automation / agentic-AI / data-source topic? Reply with EXACTLY "
            "one word: YES or NO.\n\n"
            f"Message: \"\"\"{query}\"\"\""
        )
        resp = llm.client.chat.complete(
            model=llm.model,
            messages=[
                {"role": "system", "content": "You classify messages. Reply YES or NO."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        raw = (resp.choices[0].message.content or "").strip().upper()
        if raw.startswith("YES"): return True
        if raw.startswith("NO"):  return False
        return None
    except Exception:
        return None


def is_in_scope(query: str) -> bool:
    """Cheap-first scope gate.  Defaults to in-scope to avoid false rejects on
    short messages that happen to lack our keyword list (e.g. 'yes', 'no')."""
    q = (query or "").strip().lower()
    # Short ack messages are always in-scope so the Yes/No flow keeps working
    if len(q) <= 4 or is_yes_word(q) or is_no_word(q):
        return True
    if _keyword_in_scope(q):
        return True
    # Conservative LLM tie-breaker
    decision = _llm_in_scope(q)
    if decision is True:
        return True
    if decision is False:
        return False
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Process context loader (used by intent handlers + RAG)
# ─────────────────────────────────────────────────────────────────────────────
def _load_process_context(process_key: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"process": None, "steps": [], "suggestions": [], "erp_modules": []}
    if not process_key:
        return ctx
    try:
        db = get_db()
        col = db.collection
        try:
            ctx["process"] = col(COLLECTIONS["documents"]).get(process_key)
        except Exception:
            pass
        try:
            ctx["steps"] = list(db.aql(
                "FOR s IN process_steps FILTER s.process_key == @k SORT s.step_number RETURN s",
                {"k": process_key},
            ))
        except Exception:
            pass
        try:
            ctx["suggestions"] = list(db.aql(
                "FOR s IN automation_suggestions FILTER s.process_key == @k RETURN s",
                {"k": process_key},
            ))
        except Exception:
            pass
        try:
            ctx["erp_modules"] = list(db.aql(
                "FOR m IN erp_modules FILTER m.process_key == @k RETURN m",
                {"k": process_key},
            ))
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[chatbot] context load failed: {e}")
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection + handlers
# ─────────────────────────────────────────────────────────────────────────────
def _detect_intent(query: str) -> Optional[str]:
    q = (query or "").lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return None


def _answer_explain_process(ctx: Dict[str, Any]) -> str:
    p = ctx.get("process") or {}
    steps = ctx.get("steps") or []
    title = p.get("title") or "this process"
    desc = (p.get("description") or "").strip()[:300]
    return (
        f"**{title}** is a {len(steps)}-step process.\n\n"
        f"{desc or 'No description available.'}\n\n"
        f"It involves {len({s.get('actor') for s in steps if s.get('actor')})} actors "
        f"and {len(ctx.get('suggestions') or [])} automation suggestions."
    )


def _answer_show_steps(ctx: Dict[str, Any]) -> str:
    steps = ctx.get("steps") or []
    if not steps:
        return "No steps are available for this process yet."
    lines = []
    for s in steps[:20]:
        lines.append(
            f"**Step {s.get('step_number', '?')}** — {s.get('title', '(untitled)')} "
            f"_(actor: {s.get('actor') or '—'}, automation potential: "
            f"{s.get('automation_potential', 0)}%)_"
        )
    return "Here are the process steps:\n\n" + "\n".join(lines)


def _answer_automation_details(ctx: Dict[str, Any]) -> str:
    sug = ctx.get("suggestions") or []
    if not sug:
        return "No automation suggestions have been generated yet."
    lines = []
    for s in sug[:10]:
        lines.append(
            f"• **{s.get('title') or 'Suggestion'}** — "
            f"{(s.get('description') or '')[:140]}"
        )
    return "Automation suggestions for this process:\n\n" + "\n".join(lines)


def _answer_actors(ctx: Dict[str, Any]) -> str:
    actors: Dict[str, int] = {}
    for s in (ctx.get("steps") or []):
        a = s.get("actor") or "Unknown"
        actors[a] = actors.get(a, 0) + 1
    if not actors:
        return "No actors are recorded for this process."
    lines = [f"• **{a}** — {n} step{'s' if n != 1 else ''}" for a, n in sorted(actors.items(), key=lambda x: -x[1])]
    return "Process actors:\n\n" + "\n".join(lines)


def _answer_source_target(ctx: Dict[str, Any]) -> str:
    p = ctx.get("process") or {}
    lineage = p.get("data_lineage") or {}
    src = (lineage.get("data_source") or {}).get("name", "—")
    tgt = (lineage.get("data_target") or {}).get("name", "—")
    fb = "  (ADF fallback applied)" if lineage.get("fallback_applied") else ""
    return f"**Source:** {src}{fb}\n**Target:** {tgt}"


def _answer_kpis(ctx: Dict[str, Any]) -> str:
    return (
        "Default KPI targets for this process:\n\n"
        "• Cycle time — target -50%\n"
        "• Touchless rate — target ≥70%\n"
        "• Exception backlog — target -40%\n"
        "• Decision auditability — target 100%\n"
        "• Rollback time — target <5 min"
    )


_INTENT_HANDLERS = {
    "explain_process":    _answer_explain_process,
    "show_steps":         _answer_show_steps,
    "automation_details": _answer_automation_details,
    "actors":             _answer_actors,
    "source_target":      _answer_source_target,
    "kpis":               _answer_kpis,
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: answer_query
# ─────────────────────────────────────────────────────────────────────────────
def answer_query(
    query: str,
    process_key: Optional[str] = None,
    pending_context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Args:
        query:           the user's message
        process_key:     current process id (optional)
        pending_context: if the previous bot turn asked for a Yes/No
                         confirmation, the frontend echoes back the
                         captured context here so a typed "yes" / "no"
                         can be interpreted.

    Returns dict including (when relevant):
        offer_reanalyze:        true     → render Yes/No buttons
        awaiting_confirmation:  true     → render "Thinking…" placeholder
        captured_context:       string   → store for the next turn
    """
    query = (query or "").strip()

    if not query:
        return {
            "answer":      "Please ask a question about your process or workflow.",
            "in_scope":    True,
            "intent":      None,
            "process_key": process_key,
        }

    # ────────────────────────────────────────────────────────────────────
    # PHASE A — typed Yes/No when a confirmation is already pending
    # ────────────────────────────────────────────────────────────────────
    if pending_context and is_yes_word(query):
        return {
            "answer":      "Great — re-creating the process map now…",
            "in_scope":    True,
            "intent":      "confirm_reanalyze_yes",
            "process_key": process_key,
            "confirmed":   True,
            "captured_context": pending_context,
        }
    if pending_context and is_no_word(query):
        return {
            "answer":      "Got it — I won't re-create the process map. Let me know if you change your mind.",
            "in_scope":    True,
            "intent":      "confirm_reanalyze_no",
            "process_key": process_key,
            "confirmed":   False,
        }

    # ────────────────────────────────────────────────────────────────────
    # 1. Scope gate
    # ────────────────────────────────────────────────────────────────────
    if not is_in_scope(query):
        return {
            "answer":      OUT_OF_SCOPE_MESSAGE,
            "in_scope":    False,
            "intent":      None,
            "process_key": process_key,
        }

    # ────────────────────────────────────────────────────────────────────
    # 2. Load process context
    # ────────────────────────────────────────────────────────────────────
    ctx = _load_process_context(process_key) if process_key else {}

    # ────────────────────────────────────────────────────────────────────
    # 3. PHASE B — detect "context provision" and ask for Yes/No
    # ────────────────────────────────────────────────────────────────────
    if process_key:
        deterministic = _looks_like_context_provision(query)
        is_context = deterministic
        if not deterministic and len(query.split()) >= 6 and not _starts_like_question(query.lower()):
            llm_says = _llm_classify_context_provision(query)
            if llm_says is True:
                is_context = True

        if is_context:
            captured = _summarise_context_for_reanalysis(query)
            confirmation = build_confirmation_prompt(captured)
            return {
                "answer":               confirmation,
                "in_scope":             True,
                "intent":               "provide_context",
                "process_key":          process_key,
                "offer_reanalyze":      True,
                "awaiting_confirmation": True,
                "captured_context":     captured,
            }

    # ────────────────────────────────────────────────────────────────────
    # 4. Intent fast-path
    # ────────────────────────────────────────────────────────────────────
    intent = _detect_intent(query)
    if intent and intent in _INTENT_HANDLERS:
        try:
            answer = _INTENT_HANDLERS[intent](ctx)
            return {
                "answer":      answer,
                "in_scope":    True,
                "intent":      intent,
                "process_key": process_key,
            }
        except Exception as e:
            logger.warning(f"[chatbot] intent handler {intent} failed: {e}")

    # ────────────────────────────────────────────────────────────────────
    # 5. RAG fallback
    # ────────────────────────────────────────────────────────────────────
    try:
        rag_answer = rag_query(query, process_key)
        if rag_answer:
            return {
                "answer":      rag_answer,
                "in_scope":    True,
                "intent":      "rag",
                "process_key": process_key,
            }
    except Exception as e:
        logger.warning(f"[chatbot] rag fallback failed: {e}")

    # ────────────────────────────────────────────────────────────────────
    # 6. Last-resort generic answer
    # ────────────────────────────────────────────────────────────────────
    return {
        "answer": (
            "I couldn't find a specific answer in the current workflow data. "
            "Try a more specific question like 'Show process steps', "
            "'Explain this process', or 'Automation details'."
        ),
        "in_scope":    True,
        "intent":      None,
        "process_key": process_key,
    }
