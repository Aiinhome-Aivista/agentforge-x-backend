"""
app/services/blueprint_builder_service.py
─────────────────────────────────────────────────────────────────────────────
Builds the FULL Process-Agentification Blueprint payload that the
/api/processes/<process_key>/blueprint-export endpoint returns.

The payload mirrors the reference `AgentForge_P2P_Agentic_Blueprint.docx`
EXACTLY in terms of:
  • section order and numbering (§0–§8 + Closing)
  • heading hierarchy (Heading1 for each section, Heading3 for each subsection)
  • the table column shapes (column headers are locked per subsection)

The CONTENT inside each block is generated dynamically from the actual
process data + an LLM enrichment pass.  When the LLM fails (or no LLM is
configured), a deterministic fallback fills every block — so the exported
documents never have empty sections.

Output schema (consumed by processPdfGenerator / processDocxGenerator /
                            processPptxGenerator on the frontend):

{
  "process_key": "<key>",
  "generated_at": "<ISO timestamp>",
  "llm_generated": true|false,
  "cover": {
    "brand_top":        "AGENTFORGE",
    "brand_subtitle":   "EXECUTION LAYER OF THE ENTERPRISE",
    "title":            "<process.title>",
    "subtitle":         "Process Agentification Blueprint",
    "tagline":          "<one-line anchor>",
    "deliverable_label":"Engagement deliverable",
    "footer_line":      "Prepared by AgentForge · Confidential · <year>"
  },
  "sections": [
    {
      "number": "0",
      "title":  "Executive Summary",
      "lead":   "<intro paragraph>",
      "blocks": [
        {"type":"heading3","text":"..."},
        {"type":"paragraph","text":"..."},
        {"type":"bullets","items":["..."]},
        {"type":"table","headers":[...],"rows":[[...]]},
        ...
      ]
    },
    ...
  ],
  "closing": {
    "title":  "Closing",
    "blocks": [...]
  }
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.db.arango import get_db, COLLECTIONS
from app.core.mistral_client import get_mistral_client

logger = logging.getLogger(__name__)

_LLM_MAX_TOKENS = 8192


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Context loader  (process + steps + suggestions + erp modules)
# ═════════════════════════════════════════════════════════════════════════════
def _load_process_context(process_key: str) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "process": None, "steps": [], "suggestions": [], "erp_modules": [],
    }
    db = get_db()
    col = db.collection
    try:
        ctx["process"] = col(COLLECTIONS["documents"]).get(process_key) or {}
    except Exception as e:
        logger.warning(f"[blueprint] process lookup failed: {e}")
        ctx["process"] = {}
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
    return ctx


# ═════════════════════════════════════════════════════════════════════════════
# 2.  LLM mega-prompt — asks for every dynamic field in one shot
# ═════════════════════════════════════════════════════════════════════════════
def _build_prompt(ctx: Dict[str, Any]) -> str:
    p     = ctx.get("process") or {}
    steps = ctx.get("steps") or []
    sugg  = ctx.get("suggestions") or []

    title       = p.get("title") or "this process"
    description = p.get("description") or ""
    erp         = p.get("erp") or "the target ERP"

    step_lines = "\n".join(
        f"  {s.get('step_number','?'):>2}. {s.get('title','')} "
        f"[actor: {s.get('actor','')}, automation: {s.get('automation_potential', 0)}%]"
        for s in steps[:30]
    ) or "  (no steps)"

    sugg_lines = "\n".join(
        f"  - {s.get('title') or 'Suggestion'}: "
        f"{(s.get('description') or '')[:140]}"
        for s in sugg[:15]
    ) or "  (no suggestions)"

    return f"""You are AgentForge's lead transformation architect.  Produce the
full **Process Agentification Blueprint** for the following process.

Return ONE valid JSON object only — no markdown, no fences, no commentary.

==================== CONTEXT ====================
Process:      {title}
ERP target:   {erp}
Description:  {description}

Process steps:
{step_lines}

Existing automation suggestions:
{sugg_lines}

==================== REQUIRED JSON SHAPE ====================
Every field must be tailored to THIS process.  Tables MUST have the exact
column headers shown — do not rename them.  Row counts shown are guidelines;
you may include more rows where useful, but stay within reason (max 12 per
table).

{{
  "cover": {{
    "tagline": "<one short sentence — constraint anchor or pilot focus>"
  }},

  "exec_summary": {{
    "lead":  "<one paragraph (3-5 sentences) opening the document>",
    "constraint_anchor":          "<3-4 sentence paragraph>",
    "future_state_operating_model":"<3-4 sentence paragraph>",
    "kpi_table_intro":            "<one sentence>",
    "kpi_targets": [
      {{"metric":"<m>","baseline":"<b>","target":"<t>","rationale":"<r>"}}
    ],
    "blueprint_to_prototype_path": "<3-4 sentence paragraph>",
    "pilot_delivery":              "<3-4 sentence paragraph>",
    "what_this_blueprint_is_not":  "<3-4 sentence paragraph>",
    "closing_tagline":             "<one short sentence>"
  }},

  "constraint_diagnosis": {{
    "bottleneck_named":   "<2-3 sentence paragraph>",
    "scope_paragraph":    "<2-3 sentence paragraph naming which steps are in pilot scope>",
    "scope_bullets":      ["<bullet 1>","<bullet 2>","<bullet 3>","<bullet 4>"],
    "step_selection_intro":"<one sentence>",
    "step_selection_rationale": [
      {{"no":"<step nos>","step":"<step name>","constraint":"<low/med/high/very high>",
        "volume":"<text>","repeatability":"<text>","exception_cost":"<text>",
        "reversibility":"<text>","decision":"<Select/Out of pilot>"}}
    ],
    "kpi_baselines_targets": [
      {{"metric":"<m>","today":"<t0>","pilot_30d":"<t30>","steady_90d":"<t90>"}}
    ],
    "why_one_step_not_enough": "<3-4 sentence paragraph>"
  }},

  "future_state_process": {{
    "lead":"<2-3 sentence paragraph>",
    "ownership_map": [
      {{"no":"<step no>","step":"<step name>","owner_today":"<t>",
        "owner_future":"<t>","notes":"<t>"}}
    ],
    "task_breakdown_intro":"<one sentence>",
    "task_breakdown": [
      {{"step":"<step no + name>","tasks":"<t>","future_owner":"<agent name>",
        "audit_evidence":"<t>"}}
    ],
    "autonomous_vs_human_intro":"<1-2 sentence paragraph>",
    "humans_remain_in_loop_intro":"<one sentence>",
    "humans_remain_bullets":["<b1>","<b2>","<b3>","<b4>"],
    "failure_modes_intro":"<one sentence>",
    "failure_modes_bullets":["<b1>","<b2>","<b3>","<b4>"],
    "cycle_time_paragraph":"<2-3 sentence paragraph>"
  }},

  "architecture": {{
    "lead":"<2-3 sentence paragraph>",
    "agentic_flow_paragraph":"<2-3 sentence paragraph>",
    "components": [
      {{"name":"<component name>","description":"<2-3 sentence description>"}}
    ],
    "layered_stack_paragraph":"<2-3 sentence paragraph>",
    "pilot_vs_production": [
      {{"layer":"<layer>","pilot":"<text>","production":"<text>"}}
    ],
    "connector_paragraph":"<one short paragraph about the connector interface>"
  }},

  "operating_model": {{
    "lead":"<2-3 sentence paragraph>",
    "decision_rights_paragraph":"<2-3 sentence paragraph>",
    "roles": [
      {{"role":"<r>","reports_to":"<r>","pilot_accountability":"<t>"}}
    ],
    "raci_steps": [
      {{"step":"<step>","clerk":"<R/A/C/I/->","manager":"<>","controller":"<>",
        "vendor_master":"<>","audit":"<>","agent":"<>"}}
    ],
    "escalation_matrix": [
      {{"trigger":"<t>","first_responder":"<r>","sla":"<t>","if_breached":"<t>"}}
    ],
    "day_in_the_life_paragraph":"<2-3 sentence paragraph>"
  }},

  "bill_of_materials": {{
    "lead":"<2-3 sentence paragraph>",
    "software_components": [
      {{"component":"<c>","pilot_tier":"<t>","production_tier":"<t>","license":"<l>","cost_band":"<usd/yr>"}}
    ],
    "engagement_people": [
      {{"role":"<r>","pilot_duration":"<t>","steady_state":"<t>"}}
    ],
    "client_tech_intake_intro":"<one sentence>",
    "client_tech_intake": [
      {{"topic":"<t>","question":"<q>","drives":"<t>"}}
    ],
    "not_in_bom_intro":"<one sentence>",
    "not_in_bom_bullets":["<b1>","<b2>","<b3>","<b4>"],
    "discipline_paragraph":"<2-3 sentence paragraph>"
  }},

  "deployment_plan": {{
    "lead":"<2-3 sentence paragraph>",
    "thirty_sixty_ninety_paragraph":"<2-3 sentence paragraph>",
    "pilot_plan": [
      {{"day":"<1-10>","focus":"<t>","exit_criteria":"<t>"}}
    ],
    "config_surface": [
      {{"file":"<path>","what_lives_here":"<t>","owner":"<role>"}}
    ],
    "graduation_gates_intro":"<one sentence>",
    "graduation_gates_bullets":["<b1>","<b2>","<b3>","<b4>","<b5>"],
    "graduation_paragraph":"<one sentence>"
  }},

  "governance": {{
    "lead":"<2-3 sentence paragraph>",
    "six_control_gates_paragraph":"<2-3 sentence paragraph>",
    "decision_provenance_paragraph":"<2-3 sentence paragraph>",
    "model_risk_controls": [
      {{"risk":"<r>","control":"<c>"}}
    ],
    "sox_touchpoints_intro":"<one sentence>",
    "sox_touchpoints_bullets":["<b1>","<b2>","<b3>","<b4>"],
    "rollback_paragraph":"<2-3 sentence paragraph>",
    "deliberately_not_done_intro":"<one sentence>",
    "deliberately_not_done_bullets":["<b1>","<b2>","<b3>"]
  }},

  "self_improvement": {{
    "lead":"<2-3 sentence paragraph>",
    "learning_loops_intro":"<one sentence>",
    "learning_loops":[
      "<Loop 1 — one paragraph>",
      "<Loop 2 — one paragraph>",
      "<Loop 3 — one paragraph>",
      "<Loop 4 — one paragraph>"
    ],
    "not_automated_paragraph":"<2-3 sentence paragraph>",
    "metrics_by_quarter":[
      {{"quarter":"<Q0/Q1/Q2/Q3>","override_rate":"<t>","touchless_rate":"<t>",
        "dpo_or_cycle":"<t>","exception_cycle":"<t>"}}
    ],
    "benefits_realisation_intro":"<2-3 sentence paragraph>",
    "benefits_realisation":[
      {{"kpi":"<k>","owner":"<r>","data_source":"<s>","baseline_method":"<m>","cadence":"<c>"}}
    ],
    "financial_impact_paragraph":"<2-3 sentence paragraph>",
    "projections_paragraph":"<2-3 sentence paragraph>"
  }},

  "closing": {{
    "primary_paragraph":"<2-3 sentence paragraph>",
    "tagline_one":"<one short sentence>",
    "tagline_two":"<one short sentence>"
  }}
}}

==================== HARD RULES ====================
- Output MUST be ONE valid JSON object. No comments, no trailing commas.
- Every paragraph must be tailored to the specific process — never generic.
- Tables: every row's values must be specific text (not "TBD" or "<value>").
- Use the actor / step / suggestion data above where applicable.
"""


def _call_llm(prompt: str) -> Dict[str, Any]:
    try:
        llm = get_mistral_client()
        resp = llm.client.chat.complete(
            model=llm.model,
            messages=[
                {"role": "system",
                 "content": "You are AgentForge's transformation architect. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=_LLM_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content
        parsed = llm._parse_json(raw)
        if isinstance(parsed, dict) and parsed:
            return parsed
        logger.warning("[blueprint] LLM returned non-dict — falling back")
    except Exception as e:
        logger.warning(f"[blueprint] LLM call failed: {e}")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Deterministic defaults  (used when LLM omits a field)
# ═════════════════════════════════════════════════════════════════════════════
def _defaults(ctx: Dict[str, Any]) -> Dict[str, Any]:
    p     = ctx.get("process") or {}
    steps = ctx.get("steps") or []
    erp   = p.get("erp") or "the target ERP"
    title = p.get("title") or "this process"

    n_steps = len(steps)
    n_actors = len({s.get("actor") for s in steps if s.get("actor")})
    avg_auto = (
        sum(int(s.get("automation_potential") or 0) for s in steps) // max(1, n_steps)
        if steps else 0
    )

    # First few step labels for the ownership map (real data, no LLM)
    ownership_map = []
    for s in steps[:14]:
        ownership_map.append({
            "no":           f"{s.get('step_number', '?'):02}" if isinstance(s.get('step_number'), int) else str(s.get('step_number','')),
            "step":         s.get("title") or "Step",
            "owner_today":  s.get("actor") or "Process owner",
            "owner_future": _suggest_future_owner(s),
            "notes":        _short_step_note(s),
        })

    return {
        "cover": {
            "tagline": f"Constraint-anchored pilot · {title} · Cycle-time uplift",
        },
        "exec_summary": {
            "lead": (
                f"AgentForge has been retained to operationalise outcomes inside the "
                f"{title} process by deploying a goal-driven agentic workflow. "
                f"The analysis surfaced {n_steps} process steps across {n_actors} actors "
                f"with an average automation potential of {avg_auto}%."
            ),
            "constraint_anchor": (
                f"Following AgentForge's Theory-of-Constraints methodology, the pilot does "
                f"not attempt to automate every step in parallel. It targets the highest-"
                f"value, highest-volume, highest-risk leg of {title} first — the leg where "
                f"cycle time compresses, where exceptions interleave with routine cases, "
                f"and where multiple systems must converge on a single decision."
            ),
            "future_state_operating_model": (
                "The redesigned process is a supervisor-orchestrated set of specialist "
                "agents that route every case through the right policy, escalate exceptions "
                "to the right humans, and emit auditable decisions on the way out."
            ),
            "kpi_table_intro": "Headline KPI targets the pilot is designed to deliver:",
            "kpi_targets": [
                {"metric": "Cycle time (mean)",       "baseline": "—", "target": "-50%",   "rationale": "Touchless flow eliminates handoffs."},
                {"metric": "Touchless rate",          "baseline": "—", "target": "≥70%",   "rationale": "Routine cases flow without humans."},
                {"metric": "Exception cycle time",    "baseline": "—", "target": "-60%",   "rationale": "Routed straight to the right reviewer with full context."},
                {"metric": "Decision auditability",   "baseline": "—", "target": "100%",   "rationale": "Every agent decision logged with prompts + tool calls."},
                {"metric": "Rollback time",           "baseline": "—", "target": "<5 min", "rationale": "Single config toggle to manual mode."},
            ],
            "blueprint_to_prototype_path": (
                "AgentForge moves from process intelligence to a runnable prototype in "
                "five phases: discover, diagnose, design, generate, pilot. Each phase has "
                "named exit criteria and a Steerco gate."
            ),
            "pilot_delivery": (
                "Ten working days from kickoff to first synthetic case processed end-to-end "
                "inside the AgentForge cockpit. Real workloads onboard at Steerco graduation, "
                "not before."
            ),
            "what_this_blueprint_is_not": (
                "It is not a feature catalogue, not a process-mining report, and not a "
                "consulting deck. It is an implementable plan, owned by named people, with "
                "an explicit rollback story and a single source of routing truth."
            ),
            "closing_tagline": "This enables goal-driven agentic workflows — not task automation.",
        },
        "constraint_diagnosis": {
            "bottleneck_named": (
                f"Across the {n_steps} process steps the AgentForge platform analysed for "
                f"{title}, automation potential is broadly distributed. The bottleneck does "
                f"not sit in any single step — it sits in the handoffs."
            ),
            "scope_paragraph": (
                "The pilot scope is the leg of the process where cash, time, or compliance "
                "risk concentrate, where exceptions interleave with routine cases, and "
                "where multiple upstream systems converge."
            ),
            "scope_bullets": [
                "High-stakes decisions interleave with routine cases, requiring per-case judgement that is today applied to every case.",
                "Time pressure compounds — due dates and SLAs drive downstream risk.",
                "Multiple systems converge into the same workflow, each with its own data model.",
                "Errors here are materially expensive and difficult to reverse downstream.",
            ],
            "step_selection_intro": "To make the choice of steps transparent, each step is scored against five criteria.",
            "step_selection_rationale": _default_step_selection(steps),
            "kpi_baselines_targets": [
                {"metric": "Cycle time (mean)",       "today": "—", "pilot_30d": "-25%", "steady_90d": "-50%"},
                {"metric": "Touchless rate",          "today": "—", "pilot_30d": "50%",  "steady_90d": "70%"},
                {"metric": "Average exception cycle", "today": "—", "pilot_30d": "5 d",  "steady_90d": "3 d"},
                {"metric": "Duplicate / error rate",  "today": "—", "pilot_30d": "<0.3%","steady_90d": "<0.1%"},
                {"metric": "Decision audit coverage", "today": "manual", "pilot_30d": "100%", "steady_90d": "100%"},
            ],
            "why_one_step_not_enough": (
                "Automating a single high-potential step in isolation is correct but "
                "misleading. The cycle-time gain from any one step is lost in the upstream "
                "handoff and the downstream exception queue. The pilot therefore targets a "
                "contiguous slice that owns the constraint end-to-end."
            ),
        },
        "future_state_process": {
            "lead": (
                f"The future-state {title} keeps the same workflow steps but reassigns who "
                f"owns each. Routine cases inside the policy envelope flow autonomously; "
                f"every exception is routed to the right human with the agent's reasoning attached."
            ),
            "ownership_map": ownership_map or [{"no":"01","step":"(no steps available)","owner_today":"-","owner_future":"-","notes":"-"}],
            "task_breakdown_intro": "Each in-scope step decomposes into a small set of tasks with explicit owners and audit evidence.",
            "task_breakdown": _default_task_breakdown(steps),
            "autonomous_vs_human_intro": (
                "A clean case — all validations pass, no risk signals, within policy thresholds "
                "— flows from intake to outcome autonomously."
            ),
            "humans_remain_in_loop_intro": "Humans remain in the loop for:",
            "humans_remain_bullets": [
                "Risk-flagged cases above the rejection threshold — escalated with full agent reasoning attached.",
                "Approval-required cases above the auto-approval cap — routed per the RACI matrix.",
                "Validation failures — routed to the relevant specialist with the specific delta attached.",
                "Master-data deltas (new entity, attribute change) — routed to the steward.",
            ],
            "failure_modes_intro": "Failure modes the design absorbs:",
            "failure_modes_bullets": [
                f"{erp} outage — cases buffer in the orchestrator's queue; nothing is lost.",
                "LLM unavailability — deterministic heuristic fallback ships in the codebase; degradation, not outage.",
                "Agent disagreement — the orchestrator is the single source of routing truth; agents return judgments, not decisions.",
                "Human override timeout — escalations carry an SLA; breach reassigns to the manager's manager.",
            ],
            "cycle_time_paragraph": (
                "Today, the mean end-to-end cycle is dominated by handoff and queueing time. "
                "After the pilot, the same case flows through the orchestrator in minutes for "
                "the routine path and hours (not days) for the exception path."
            ),
        },
        "architecture": {
            "lead": (
                f"The architecture mirrors the AgentForge platform's published reference. "
                f"The pilot stack and the production stack share the same shape — only the "
                f"data-plane targets differ. Connector to {erp} is the only client-specific piece."
            ),
            "agentic_flow_paragraph": (
                "The agentic orchestrator routes each case through specialist agents, applies "
                "policy on agent outputs, escalates exceptions to humans, and emits an audit "
                "record on every transition."
            ),
            "components": _default_components(erp),
            "layered_stack_paragraph": (
                "The stack is layered: ingress, orchestration, agent services, data and "
                "infrastructure, and controlled outputs. Pilot defaults are local-friendly; "
                "production defaults are HA-friendly."
            ),
            "pilot_vs_production": [
                {"layer": "Durable state + audit",   "pilot": "Postgres (single)",   "production": "Postgres (managed multi-AZ)"},
                {"layer": "Short-term memory",       "pilot": "Redis (single)",      "production": "Redis cluster"},
                {"layer": "Event stream",            "pilot": "Redis Streams",       "production": "Kafka"},
                {"layer": "Compute",                 "pilot": "docker-compose VM",   "production": "Kubernetes (Helm)"},
                {"layer": f"{erp} integration",      "pilot": "mock service",         "production": "real gateway, same interface"},
                {"layer": "Observability",           "pilot": "OTEL console",         "production": "Hosted backend"},
            ],
            "connector_paragraph": (
                "The connector interface is intentionally identical between pilot and "
                "production. Repointing means changing one env var — no agent code changes."
            ),
        },
        "operating_model": {
            "lead": (
                "Without an explicit operating model, the platform is a science project. "
                "The principle: agents make routine decisions; humans own policy, "
                "exceptions, and anything above auto-approval thresholds."
            ),
            "decision_rights_paragraph": (
                "Decision rights map to risk and value tiers. Low-value, fully-matched, "
                "low-risk cases flow autonomously. High-value or high-risk cases require an "
                "explicit human signature."
            ),
            "roles": _default_roles(),
            "raci_steps": _default_raci(steps),
            "escalation_matrix": [
                {"trigger": "Risk score ≥ 0.75",       "first_responder": "Process Manager",  "sla": "4 business hours", "if_breached": "Reassign to Controller"},
                {"trigger": "Risk score 0.4 – 0.75",   "first_responder": "Process Clerk",    "sla": "Same day",         "if_breached": "Reassign to Process Manager"},
                {"trigger": "Validation mismatch",     "first_responder": "Process Clerk",    "sla": "Same day",         "if_breached": "Reassign to Process Manager"},
                {"trigger": "Missing reference data",  "first_responder": "Process Clerk",    "sla": "Next business day", "if_breached": "Reassign to Data Steward"},
                {"trigger": "New-entity case",         "first_responder": "Master-data Steward","sla": "Next business day", "if_breached": "Reassign to Director"},
                {"trigger": "Critical failure (P1)",   "first_responder": "On-call engineer", "sla": "1 hour",           "if_breached": "Reassign to platform owner"},
            ],
            "day_in_the_life_paragraph": (
                "Today: most of the day on routine handling, the remainder on exceptions. "
                "After the pilot: a handful of escalation items in the morning queue, the "
                "rest of the day on vendor management, policy tuning, and audit support."
            ),
        },
        "bill_of_materials": {
            "lead": "Every component required to ship pilot and graduate to production — license, cost band, and which tier each lives in.",
            "software_components": [
                {"component": "Python 3.11 + FastAPI + SQLAlchemy", "pilot_tier": "✓", "production_tier": "✓", "license": "OSS", "cost_band": "$0"},
                {"component": "PostgreSQL 15",                      "pilot_tier": "single", "production_tier": "managed multi-AZ", "license": "OSS", "cost_band": "$0 / $8-24k"},
                {"component": "Redis 7",                            "pilot_tier": "single", "production_tier": "managed cluster",  "license": "OSS", "cost_band": "$0 / $4-10k"},
                {"component": "Kafka (or Redpanda)",                "pilot_tier": "—",      "production_tier": "required at scale", "license": "Apache 2.0", "cost_band": "$0-30k"},
                {"component": "Anthropic / Mistral LLM API",        "pilot_tier": "optional", "production_tier": "required",        "license": "Commercial", "cost_band": "$5-20k / $50-250k"},
                {"component": "ArangoDB (graph store)",             "pilot_tier": "✓",       "production_tier": "✓",                 "license": "Apache 2.0", "cost_band": "$0"},
                {"component": "ChromaDB (vector store)",            "pilot_tier": "✓",       "production_tier": "✓",                 "license": "Apache 2.0", "cost_band": "$0"},
                {"component": "OpenTelemetry stack",                "pilot_tier": "console", "production_tier": "hosted backend",   "license": "Apache 2.0", "cost_band": "$0 / $5-20k"},
                {"component": f"{erp} connector",                   "pilot_tier": "mock",    "production_tier": "real",             "license": "Custom",     "cost_band": "—"},
            ],
            "engagement_people": [
                {"role": "AgentForge Partner (engagement lead)",   "pilot_duration": "0.2 FTE × 2 weeks", "steady_state": "0.05 FTE quarterly"},
                {"role": "AgentForge Principal Engineer",          "pilot_duration": "1.0 FTE × 2 weeks", "steady_state": "0.2 FTE"},
                {"role": "AgentForge ML / Prompt Engineer",        "pilot_duration": "0.5 FTE × 2 weeks", "steady_state": "0.1 FTE"},
                {"role": "Client Pilot Sponsor (Controller)",      "pilot_duration": "0.2 FTE × 2 weeks", "steady_state": "0.05 FTE"},
                {"role": "Client Process Manager",                  "pilot_duration": "0.5 FTE × 2 weeks", "steady_state": "0.1 FTE"},
                {"role": "Client Data Steward (master-data)",      "pilot_duration": "0.2 FTE × 2 weeks", "steady_state": "0.05 FTE"},
            ],
            "client_tech_intake_intro": "Before generating the final BOM, AgentForge captures the client's technology preferences.",
            "client_tech_intake": [
                {"topic": "ERP version",         "question": f"Which {erp} version?",                 "drives": "Connector method, posting strategy"},
                {"topic": "Integration method",  "question": "BAPI, OData, IDoc, or API gateway?",   "drives": "Connector implementation"},
                {"topic": "Cloud provider",      "question": "AWS, Azure, GCP, or private cloud?",   "drives": "Compute, network, managed services"},
                {"topic": "Container platform",  "question": "Kubernetes flavour, ECS, or other?",   "drives": "Deployment manifests"},
                {"topic": "Secrets management",  "question": "Vault, AWS SM, Azure KV?",             "drives": "Secrets injection"},
                {"topic": "Identity provider",   "question": "Okta, Azure AD, other OIDC?",          "drives": "Cockpit + service auth"},
                {"topic": "Observability stack", "question": "Datadog, New Relic, Grafana, other?",  "drives": "Telemetry export targets"},
            ],
            "not_in_bom_intro": "What is deliberately not in the BOM:",
            "not_in_bom_bullets": [
                "A new master-data system. Reuse the existing source of truth.",
                "A new identity provider. Reuse the client IdP.",
                "A new RPA platform. The agents replace, not augment, RPA tasks in scope.",
                "A new BI tool. The cockpit ships its own; downstream BI can pull from the decision log directly.",
            ],
            "discipline_paragraph": (
                "The discipline of not adding tools is part of the value AgentForge offers. "
                "Most enterprise AI programs fail because the new platform becomes the project."
            ),
        },
        "deployment_plan": {
            "lead": "Ten working days from a fresh cloud account to the first case processed end-to-end.",
            "thirty_sixty_ninety_paragraph": (
                "The pilot begins on day 1 and ends with a Steerco graduation decision on "
                "day 10. Beyond the pilot, the staged 30-60-90 day path scales the rollout "
                "without changing the agent code."
            ),
            "pilot_plan": [
                {"day": "1",  "focus": "Kickoff & environment provisioning",       "exit_criteria": "All containers up, healthchecks green."},
                {"day": "2",  "focus": "Synthetic seed & cockpit walkthrough",     "exit_criteria": "Pilot Sponsor signs off cockpit UX."},
                {"day": "3",  "focus": f"{erp} connector wiring",                  "exit_criteria": "Test case flows through all agents end-to-end."},
                {"day": "4",  "focus": "Policy file customisation",                "exit_criteria": "Controller signs off the policy file."},
                {"day": "5",  "focus": "Replay against historical data",           "exit_criteria": "Disagreement rate <10% on trailing 30 days."},
                {"day": "6",  "focus": "Exception path drills",                    "exit_criteria": "Every escalation path exercised once."},
                {"day": "7",  "focus": "Observability + decision-log audit",       "exit_criteria": "Audit pack exports cleanly."},
                {"day": "8",  "focus": "Operating-model dry-run with humans",      "exit_criteria": "SLAs met for every drill."},
                {"day": "9",  "focus": "Steerco preparation + KPI rehearsal",      "exit_criteria": "Dashboard signed off."},
                {"day": "10", "focus": "Steerco graduation decision",              "exit_criteria": "Go / no-go from Pilot Sponsor."},
            ],
            "config_surface": [
                {"file": ".env",                                       "what_lives_here": "API keys, DB URLs, mode flags",     "owner": "Pilot Delivery Lead"},
                {"file": "services/orchestrator/graph.py",             "what_lives_here": "Routing rules, thresholds",          "owner": "AgentForge engineering"},
                {"file": "services/agents/approval/policy.py",         "what_lives_here": "Approval thresholds and tiers",      "owner": "Controller"},
                {"file": "services/agents/fraud/heuristics.py",        "what_lives_here": "Risk tolerances",                    "owner": "Process Manager + AgentForge"},
                {"file": "services/agents/payment/three_way_match.py", "what_lives_here": "Validation tolerance bands",         "owner": "Process Manager + AgentForge"},
                {"file": "config/raci.yaml",                           "what_lives_here": "RACI by step + escalation matrix",   "owner": "Process Owner"},
            ],
            "graduation_gates_intro": "Gates that must clear before production graduation:",
            "graduation_gates_bullets": [
                "Replace docker-compose with the production orchestrator (Helm chart on Kubernetes).",
                "Replace Redis Streams with Kafka.",
                "Replace single Postgres with managed multi-AZ.",
                "Replace OTEL console exporter with the hosted observability backend.",
                "Replace the mock connector with the real gateway.",
                "Wire IdP via OIDC, and the secrets manager.",
            ],
            "graduation_paragraph": "Each is a single-day task once the pilot is accepted — none changes the agent code.",
        },
        "governance": {
            "lead": (
                "The system must be defensible to internal audit, external auditors, and "
                "the SOX programme on day one. This section is the audit story, end to end."
            ),
            "six_control_gates_paragraph": (
                "Six gates wrap every agent decision: policy, data, risk, human approval, "
                "execution, and audit. Each gate is implemented in code with an explicit "
                "name in the source tree."
            ),
            "decision_provenance_paragraph": (
                "Every agent action writes one row to decision_log with run_id, case_id, "
                "agent name, action, full input hash, structured output, and the prompt + "
                "tool calls used to produce it. Provenance is an artefact, not a feature."
            ),
            "model_risk_controls": [
                {"risk": "Model hallucination on a critical value",  "control": "Decision is policy-deterministic; LLM produces narrative only."},
                {"risk": "Model unavailability",                      "control": "Heuristic fallback path; degradation, not outage."},
                {"risk": "Prompt injection from upstream content",    "control": "Free-text fields sanitised before being added to prompts; tools cannot be invoked from prompted content."},
                {"risk": "Distribution shift over time",              "control": "Weekly KPI watch + override-disagreement review; prompt updates require Controller sign-off."},
                {"risk": "LLM vendor pricing or behaviour change",    "control": "Provider abstraction in app/llm.py; a second provider can be swapped behind the same interface."},
                {"risk": "PII leakage through prompts",                "control": "PII scrubber in the prompt pipeline; redaction tested in the eval suite."},
            ],
            "sox_touchpoints_intro": "SOX touchpoints handled at the platform level:",
            "sox_touchpoints_bullets": [
                "Segregation of duties — agents do not initiate critical actions without an explicit human approval record where the policy requires one.",
                "Change management — all prompt and policy changes go through PR review with the Controller as a required approver.",
                "Access controls — role-based grants on the durable store; cockpit access via the client IdP.",
                "Audit pack export — scripted export pulls full decision history into a portable evidence file.",
            ],
            "rollback_paragraph": (
                "Rollback is a single config flip — the manual path stays warm throughout "
                "the pilot. Reversal hooks for downstream side-effects are documented "
                "per-agent and tested in the eval suite."
            ),
            "deliberately_not_done_intro": "What this design deliberately does NOT do:",
            "deliberately_not_done_bullets": [
                "It does not pretend the agents will be right 100% of the time. The override queue is a first-class component.",
                "It does not run the LLM on critical-path arithmetic. Money/inventory never moves on a model output that hasn't been policy-validated.",
                "It does not write to systems of record without an idempotency key, ensuring retries are safe.",
            ],
        },
        "self_improvement": {
            "lead": (
                "AgentForge promises self-improving execution systems, not static deployments. "
                "That promise is operationalised through a small number of structured "
                "feedback loops with explicit owners and cadences."
            ),
            "learning_loops_intro": "Four learning loops operate continuously:",
            "learning_loops": [
                "Loop 1 — Override-disagreement review (weekly). Pilot Lead pulls all human-override rows from the past week and classifies the delta with the AP Manager.",
                "Loop 2 — Replay regression (per change). Before a prompt or policy change deploys, it is replayed against the trailing 30 days; disagreements above threshold block the change.",
                "Loop 3 — Threshold tuning (monthly). Three numeric levers — risk threshold, tolerance bands, and per-tier auto-approval caps — are reviewed with the Controller.",
                "Loop 4 — Vendor / entity segmentation (quarterly). The system surfaces entities whose behaviour suggests a tier mismatch; the steward reclassifies.",
            ],
            "not_automated_paragraph": (
                "Two things stay manual on purpose. First, policy itself: auto-tuning a "
                "policy file from data is appealing and dangerous. Second, the override "
                "queue: it is the system's eyes on its own blind spots."
            ),
            "metrics_by_quarter": [
                {"quarter": "Q0 (pilot)", "override_rate": "n/a → 18%", "touchless_rate": "—  → 60%", "dpo_or_cycle": "— → -25%", "exception_cycle": "—  → -50%"},
                {"quarter": "Q1",         "override_rate": "18% → 12%", "touchless_rate": "60% → 70%", "dpo_or_cycle": "-25% → -35%", "exception_cycle": "-50% → -55%"},
                {"quarter": "Q2",         "override_rate": "12% → 8%",  "touchless_rate": "70% → 78%", "dpo_or_cycle": "-35% → -45%", "exception_cycle": "-55% → -60%"},
                {"quarter": "Q3",         "override_rate": "8% → 6%",   "touchless_rate": "78% → 82%", "dpo_or_cycle": "-50% (steady)", "exception_cycle": "-60% (steady)"},
            ],
            "benefits_realisation_intro": (
                "Every KPI target above has a named owner, a baseline measurement method, "
                "a data source, and a measurement cadence."
            ),
            "benefits_realisation": [
                {"kpi": "Cycle time (mean)",         "owner": "Process Manager",   "data_source": "decision_log state transitions", "baseline_method": "Trailing 90-day average", "cadence": "Weekly"},
                {"kpi": "Touchless rate",            "owner": "Process Manager",   "data_source": "decision_log",                   "baseline_method": "Trailing 90-day count",  "cadence": "Daily"},
                {"kpi": "Exception cycle time",      "owner": "Process Manager",   "data_source": "decision_log state transitions", "baseline_method": "Mean + P75 over 90 days","cadence": "Weekly"},
                {"kpi": "Decision auditability",     "owner": "Controller + Audit","data_source": "decision_log + audit reviews",   "baseline_method": "Manual sample audit",    "cadence": "Monthly"},
                {"kpi": "Capacity reclaimed",        "owner": "Process Manager + HRBP","data_source": "Time-tracking sample + volumes", "baseline_method": "Two-week time-and-motion","cadence": "Quarterly"},
            ],
            "financial_impact_paragraph": (
                "Financial impact is translated to currency via a defensible model the "
                "Controller maintains: capacity release from cycle-time reduction, error "
                "avoidance from automation, and any direct margin from price-/discount-"
                "capture opportunities the pilot exposes."
            ),
            "projections_paragraph": (
                "These are projections, not promises. The point is that they are traceable "
                "— every quarter's number can be tied back to the decision_log."
            ),
        },
        "deploying_in_your_environment": {
            "lead": (
                "This section translates the agentic architecture into a concrete, "
                "client-side deployment. It is written so your platform, integration, "
                "audit, and operations teams can read it together and know exactly "
                "what to provision, integrate, test, and operate."
            ),
            "deployment_approach_paragraph": (
                "Seven stages take the blueprint output from a packaged build to live, "
                "scaled execution. Every transition is a gate with explicit entry and "
                "exit criteria, owners, and exit evidence. No stage advances on opinion."
            ),
            # 9.1 Solution components
            "solution_components_intro": (
                "The deployable unit is a set of containerised services plus their "
                "configuration. Each component has a single responsibility and a "
                "stable interface."
            ),
            "solution_components": [
                {"component": "api-gateway",         "role": "Ingress, schema validation, auth", "deploys_as": "Container",            "owner": "Platform team"},
                {"component": "orchestrator",        "role": "Agent graph + policy enforcement",  "deploys_as": "Container",            "owner": "AgentForge engineering"},
                {"component": "agent services",     "role": "Specialist agents (intake/validation/risk/approval/execution/tracker)", "deploys_as": "Containers (one per agent)", "owner": "AgentForge engineering"},
                {"component": "policy store",       "role": "Versioned policy files",            "deploys_as": "Git repo + mounted ConfigMap", "owner": "Controller"},
                {"component": "data services",      "role": "Postgres, Redis, message stream",   "deploys_as": "Managed (prod) / single host (pilot)", "owner": "Platform team"},
                {"component": "ERP connector",      "role": "Stable interface to systems of record", "deploys_as": "Container",        "owner": "Integration team"},
                {"component": "cockpit UI",         "role": "Operator-facing dashboard",         "deploys_as": "Container + reverse proxy", "owner": "AgentForge engineering"},
                {"component": "observability stack","role": "Logs, metrics, traces, audit log",  "deploys_as": "Hosted (prod) / console (pilot)", "owner": "Platform team"},
            ],
            # 9.2 Bill of materials — pilot vs production
            "bom_summary_paragraph": (
                "The full BOM is in §5. In summary, the pilot runs on a single host "
                "with open-source data services and an optional LLM key; production "
                "graduates to managed multi-AZ data services and a hosted observability "
                "backend without changing any agent code."
            ),
            "bom_pilot_vs_production": [
                {"layer": "Compute",                     "pilot": "Single VM / docker-compose",    "production": "Kubernetes (Helm)"},
                {"layer": "Durable state",               "pilot": "Postgres (single)",             "production": "Postgres (managed multi-AZ)"},
                {"layer": "Short-term state + locks",    "pilot": "Redis (single)",                "production": "Redis cluster"},
                {"layer": "Event stream",                "pilot": "Redis Streams",                 "production": "Kafka"},
                {"layer": "Observability",               "pilot": "OTEL console exporter",         "production": "Hosted backend (Datadog / New Relic / Grafana)"},
                {"layer": "Secrets",                     "pilot": ".env",                           "production": "Vault / AWS SM / Azure KV"},
                {"layer": "ERP integration",             "pilot": "mock connector",                 "production": "real ERP gateway, same interface"},
            ],
            # 9.3 Step-by-step deployment
            "step_by_step_intro": (
                "The deployment runs as a numbered runbook. Steps 1–5 require no client "
                "production access and can run in a sandbox; steps 6–9 bring the system "
                "into the client's environment with progressively wider scope."
            ),
            "step_by_step_bullets": [
                "Step 1 — Provision. Stand up the cloud account, network, and a Kubernetes namespace (or a single VM for pilot) and apply the IaC templates. Exit: namespaces and base IAM in place.",
                "Step 2 — Deploy services. Pull the container images, apply the manifests or compose file, and bring up the data services. Exit: all healthchecks green.",
                "Step 3 — Ground the agents (see 9.4). Load policy files, the process ontology, vendor / entity history, and few-shot examples. Exit: agents start on grounded context.",
                "Step 4 — Wire integrations (see 9.5). Connect the ERP connector, IdP, and data feeds in read-only / sandbox mode. Exit: end-to-end smoke test passes.",
                "Step 5 — Test (see 9.6). Run unit, integration, shadow-replay, and UAT. Exit: shadow disagreement under 10%; UAT signed.",
                "Step 6 — Arm guardrails (see 9.7). Enable the six control gates and configure thresholds and escalation SLAs. Exit: Internal Audit sign-off.",
                "Step 7 — Go-live, limited scope. Open one company code or entity segment with writeback still gated. Exit: 24 hours live, zero P1 incidents.",
                "Step 8 — Rollback rehearsal. Execute the documented reversal path on a live-like case. Exit: zero-data-loss recovery demonstrated.",
                "Step 9 — Scale. Follow the 30-60-90 path, expand segments, and graduate the data plane to the production tier. Exit: KPI-evidenced run-rate.",
            ],
            # 9.4 Grounding
            "grounding_paragraph": (
                "Grounding is how the agents acquire the context to make correct, "
                "client-specific decisions. AgentForge grounds the system in four "
                "layers, captured during the client technology intake and the policy "
                "workshop. They are owned by the Controller and Process Owner."
            ),
            "grounding_bullets": [
                "Policy grounding. Versioned policy files encode approval thresholds, entity tiers, and tolerance bands. They are the source of truth for routine decisions.",
                "Master-data grounding. The ERP connector supplies live records so agents reason over real, current data — not on stale extracts.",
                "Historical grounding. Past exceptions and outcomes are indexed so the risk-agent can score similar cases against precedent.",
                "Example grounding. A small, curated set of resolved exceptions is attached to agent prompts so reasoning matches house style.",
            ],
            # 9.5 Integration
            "integration_intro": (
                "The system integrates at a small number of boundaries. Each uses a "
                "stable interface so the agent code is unaware of the specific product "
                "behind it — switching pilot mock to real ERP changes only configuration."
            ),
            "integration_table": [
                {"system": "ERP / system of record", "interface": "REST / BAPI / OData",   "direction": "Bi-directional", "pilot": "mock",              "production": "real gateway"},
                {"system": "Identity provider",     "interface": "OIDC",                   "direction": "Inbound",        "pilot": "local accounts",    "production": "Client IdP"},
                {"system": "Document store",        "interface": "S3-compatible",          "direction": "Bi-directional", "pilot": "MinIO",             "production": "S3 / Azure Blob"},
                {"system": "Notification channel",  "interface": "Webhook / Email / Slack","direction": "Outbound",       "pilot": "console",           "production": "Real channel"},
                {"system": "Observability backend", "interface": "OTLP",                   "direction": "Outbound",       "pilot": "console exporter",  "production": "Hosted backend"},
            ],
            # 9.6 Test approach
            "test_intro": (
                "Testing is layered and moves progressively closer to production reality. "
                "Each layer has an owner and an exit gate."
            ),
            "test_layers": [
                {"layer": "Unit",            "proves": "Each agent function returns correctly", "owner": "AgentForge engineering", "gate": "100% pass on CI"},
                {"layer": "Integration",     "proves": "Cross-agent + tool calls behave",       "owner": "AgentForge engineering", "gate": "End-to-end smoke green"},
                {"layer": "Shadow replay",   "proves": "Behaviour matches the trailing-30-day baseline", "owner": "AgentForge + Process Owner", "gate": "Disagreement < 10%"},
                {"layer": "UAT",             "proves": "Operator workflows are usable",         "owner": "Process Owner",          "gate": "Sign-off"},
                {"layer": "Performance",     "proves": "Latency + throughput within target",    "owner": "Platform team",          "gate": "P95 within target"},
                {"layer": "Security",        "proves": "AuthZ + tenancy + secrets safe",        "owner": "Security",               "gate": "Findings closed"},
                {"layer": "Resilience",      "proves": "Failure modes degrade, not break",      "owner": "Platform + AgentForge",   "gate": "Chaos drill passes"},
            ],
            # 9.7 Guardrails
            "guardrails_intro": (
                "Guardrails wrap every agent decision. They are enforceable checks in "
                "code and configuration, not documentation. The six control gates each "
                "block, route, or escalate the decision when a rule is violated."
            ),
            "guardrails_bullets": [
                "Policy gate. Decisions outside the versioned policy are blocked, not best-guessed. The LLM produces narrative; policy produces decisions.",
                "Data gate. Agents act only on validated, master-data-backed records; missing reference data routes to a human.",
                "Risk gate. Risk score, amount, and tier set the autonomy ceiling; above it, the case escalates.",
                "Human-in-the-loop gate. Every escalation carries an SLA and reassigns on breach; nothing sits silently in a queue.",
                "Execution gate. Writeback is disabled until controls, approvals, and audit evidence are signed; idempotency keys make retries safe.",
                "Audit gate. Every decision and override writes an immutable decision_log row; the audit pack reconstructs any case in a single command.",
            ],
            "guardrails_closing_paragraph": (
                "Two hard rules sit above the gates: the model never performs "
                "critical-path arithmetic, and no irreversible action runs on an "
                "unvalidated model output. The net effect: your teams deploy a system "
                "they can see into, govern, and roll back — one that automates the "
                "routine 70% and routes the rest to the people who should always "
                "have seen it."
            ),
        },

        "closing": {
            "primary_paragraph": (
                f"This blueprint is opinionated by design. It names a single constraint "
                f"inside {title}, names the agents that operate on it, names the humans "
                f"who supervise them, and names the controls that defend them."
            ),
            "tagline_one": "We do not sell AI. We do not sell consulting. We operationalise outcomes.",
            "tagline_two": "AgentForge · Execution Layer of the Enterprise · Confidential",
        },
    }


# ─── helpers used by the defaults ───────────────────────────────────────────
def _suggest_future_owner(step: Dict[str, Any]) -> str:
    pot = int(step.get("automation_potential") or 0)
    actor = (step.get("actor") or "").lower()
    if pot >= 70:
        return "agentic-orchestrator"
    if pot >= 40:
        return f"{actor or 'human'} + agent"
    return actor or "Process owner"


def _short_step_note(step: Dict[str, Any]) -> str:
    pot = int(step.get("automation_potential") or 0)
    if pot >= 80: return "High-confidence autonomous execution."
    if pot >= 50: return "Agent-assisted; human approves above threshold."
    if pot >= 30: return "Mostly human; agent supports validation and audit."
    return "Out of pilot scope — minimal change."


def _default_step_selection(steps: list) -> list:
    if not steps:
        return [{"no": "01–N", "step": "(no steps available)", "constraint": "—",
                 "volume": "—", "repeatability": "—", "exception_cost": "—",
                 "reversibility": "—", "decision": "Out of pilot"}]
    rows = []
    for s in steps[:8]:
        pot = int(s.get("automation_potential") or 0)
        cons = "Very high" if pot >= 80 else "High" if pot >= 60 else "Medium" if pot >= 40 else "Low"
        repeat = "High" if pot >= 50 else "Mixed"
        decision = "Select" if pot >= 50 else "Out of pilot"
        rows.append({
            "no":             f"{s.get('step_number','?')}",
            "step":           s.get("title") or "Step",
            "constraint":     cons,
            "volume":         "High" if pot >= 60 else "Mixed",
            "repeatability":  repeat,
            "exception_cost": "High" if pot >= 70 else "Medium",
            "reversibility":  "High" if pot < 70 else "Medium",
            "decision":       decision,
        })
    return rows


def _default_task_breakdown(steps: list) -> list:
    rows = []
    for s in steps[:6]:
        pot = int(s.get("automation_potential") or 0)
        if pot < 50:
            continue
        rows.append({
            "step":           f"{s.get('step_number','?')} {s.get('title','')}",
            "tasks":          (s.get("description") or "Validate, classify, route, persist, emit event.")[:200],
            "future_owner":   _suggest_future_owner(s),
            "audit_evidence": "Input hash, classification, routing decision, persisted-row hash.",
        })
    if not rows:
        rows = [{"step": "(no in-scope steps)", "tasks": "—", "future_owner": "—", "audit_evidence": "—"}]
    return rows


def _default_roles() -> list:
    return [
        {"role": "Process Clerk",         "reports_to": "Process Manager",     "pilot_accountability": "Resolves validation exceptions and missing-reference cases. SLA: same-day."},
        {"role": "Process Manager",        "reports_to": "Controller",          "pilot_accountability": "Approves agent escalations above auto-approval cap; resolves risk-flagged cases (HIGH severity). SLA: 4 business hours."},
        {"role": "Controller",             "reports_to": "CFO / Process Owner", "pilot_accountability": "Owns the policy file. Signs off agent prompts and policy changes. Reads weekly KPI report."},
        {"role": "Master-data Steward",    "reports_to": "Director",            "pilot_accountability": "Resolves master-data discrepancies. SLA: next business day."},
        {"role": "Internal Audit Liaison", "reports_to": "Chief Audit Exec",    "pilot_accountability": "Read-only access to decision_log and override audit. Quarterly attestation."},
        {"role": "AgentForge Engineering", "reports_to": "Pilot Sponsor",       "pilot_accountability": "Owns the agent code and orchestrator graph. SLA: same-day for incidents."},
    ]


def _default_raci(steps: list) -> list:
    rows = []
    for s in (steps or [])[:8]:
        pot = int(s.get("automation_potential") or 0)
        agent = "R" if pot >= 50 else "—"
        clerk = "C" if pot < 50 else "I"
        manager = "A" if pot >= 30 else "I"
        rows.append({
            "step": (s.get("title") or "Step")[:60],
            "clerk":          clerk,
            "manager":        manager,
            "controller":     "C" if pot >= 70 else "—",
            "vendor_master":  "—",
            "audit":          "I",
            "agent":          agent,
        })
    if not rows:
        rows = [{"step": "(no steps)", "clerk": "—", "manager": "—", "controller": "—",
                 "vendor_master": "—", "audit": "—", "agent": "—"}]
    return rows


def _default_components(erp: str) -> list:
    return [
        {"name": "api-gateway",   "description": "Accepts inbound cases from upstream sources. Validates schema and forwards to the orchestrator. No business logic."},
        {"name": "orchestrator",  "description": "Holds the agent graph; routes cases through specialist agents; applies policy on agent outputs; emits decision-log rows on every transition."},
        {"name": "intake-agent",  "description": "Normalises inbound payloads, performs idempotency checks, persists raw state, emits the workflow-start event."},
        {"name": "validation-agent","description": "Applies cross-system reference checks (match against systems of record, tolerance bands). Returns a structured judgment."},
        {"name": "risk-agent",    "description": "Detects duplicate, anomaly, and policy-violation patterns. Scores each case on the configured risk model."},
        {"name": "approval-agent","description": "Applies a versioned policy file. Per-entity-tier thresholds. Above-cap cases are routed to the human owner."},
        {"name": "execution-agent","description": f"Writes the committed action into {erp} via the connector. Idempotency-key enforced; never writes twice."},
        {"name": "tracker-agent", "description": "Updates case status, writes the run's decisions to decision_log, emits notifications, produces the audit trail."},
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Merge LLM output with defaults (LLM overrides defaults field-by-field)
# ═════════════════════════════════════════════════════════════════════════════
def _deep_merge_dict(default: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Field-level merge. Overlay wins for non-empty values; defaults fill gaps."""
    out: Dict[str, Any] = dict(default)
    if not isinstance(overlay, dict):
        return out
    for k, v in overlay.items():
        if v is None: continue
        if isinstance(v, str) and not v.strip(): continue
        if isinstance(v, list) and len(v) == 0: continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Assemble the wire payload  (cover + sections[] + closing)
# ═════════════════════════════════════════════════════════════════════════════
def _assemble_payload(process_key: str, ctx: Dict[str, Any], data: Dict[str, Any], llm_used: bool) -> Dict[str, Any]:
    p     = ctx.get("process") or {}
    title = p.get("title") or "Process Agentification Blueprint"
    now   = datetime.utcnow()

    es = data["exec_summary"]
    cd = data["constraint_diagnosis"]
    fs = data["future_state_process"]
    ar = data["architecture"]
    om = data["operating_model"]
    bo = data["bill_of_materials"]
    dp = data["deployment_plan"]
    gv = data["governance"]
    si = data["self_improvement"]
    cl = data["closing"]

    def H3(t): return {"type": "heading3", "text": t}
    def P(t):  return {"type": "paragraph", "text": t}
    def B(items): return {"type": "bullets", "items": items}
    def TBL(headers, rows): return {"type": "table", "headers": headers, "rows": rows}

    sections: List[Dict[str, Any]] = []

    # ─── §0 — Executive Summary ────────────────────────────────────────
    sections.append({
        "number": "0",
        "title":  "Executive Summary",
        "lead":   es["lead"],
        "blocks": [
            H3("Constraint anchor"),
            P(es["constraint_anchor"]),
            H3("Future-state operating model"),
            P(es["future_state_operating_model"]),
            H3("Headline KPI targets (90 days)"),
            P(es["kpi_table_intro"]),
            TBL(
                ["Metric", "Baseline", "Target", "Rationale"],
                [[r["metric"], r["baseline"], r["target"], r["rationale"]] for r in es["kpi_targets"]],
            ),
            H3("Blueprint-to-prototype path"),
            P(es["blueprint_to_prototype_path"]),
            H3("Pilot delivery"),
            P(es["pilot_delivery"]),
            H3("What this blueprint is not"),
            P(es["what_this_blueprint_is_not"]),
            P(es["closing_tagline"]),
        ],
    })

    # ─── §1 — Constraint Diagnosis ─────────────────────────────────────
    sections.append({
        "number": "1",
        "title":  f"Constraint Diagnosis: {title}",
        "lead":   "",
        "blocks": [
            H3("The bottleneck named"),
            P(cd["bottleneck_named"]),
            H3("End-to-end process scope"),
            P(cd["scope_paragraph"]),
            B(cd["scope_bullets"]),
            H3("Step-selection rationale"),
            P(cd["step_selection_intro"]),
            TBL(
                ["No.", "Step", "Constraint", "Volume", "Repeatability", "Exception cost", "Reversibility", "Decision"],
                [[r["no"], r["step"], r["constraint"], r["volume"], r["repeatability"], r["exception_cost"], r["reversibility"], r["decision"]] for r in cd["step_selection_rationale"]],
            ),
            H3("KPI baselines vs targets"),
            TBL(
                ["Metric", "Today", "30-day pilot", "90-day steady"],
                [[r["metric"], r["today"], r["pilot_30d"], r["steady_90d"]] for r in cd["kpi_baselines_targets"]],
            ),
            H3("Why one step alone is not enough"),
            P(cd["why_one_step_not_enough"]),
        ],
    })

    # ─── §2 — Future-State Process ─────────────────────────────────────
    sections.append({
        "number": "2",
        "title":  "Future-State Process",
        "lead":   fs["lead"],
        "blocks": [
            H3("Step-by-step ownership map"),
            TBL(
                ["No.", "Step", "Owner today", "Owner future", "Notes"],
                [[r["no"], r["step"], r["owner_today"], r["owner_future"], r["notes"]] for r in fs["ownership_map"]],
            ),
            H3("Task-level breakdown for in-scope steps"),
            P(fs["task_breakdown_intro"]),
            TBL(
                ["Step", "Key tasks", "Future owner", "Audit evidence"],
                [[r["step"], r["tasks"], r["future_owner"], r["audit_evidence"]] for r in fs["task_breakdown"]],
            ),
            H3("What flows are autonomous, and what stays human"),
            P(fs["autonomous_vs_human_intro"]),
            P(fs["humans_remain_in_loop_intro"]),
            B(fs["humans_remain_bullets"]),
            H3("Failure modes the design absorbs"),
            P(fs["failure_modes_intro"]),
            B(fs["failure_modes_bullets"]),
            H3("Cycle-time impact"),
            P(fs["cycle_time_paragraph"]),
        ],
    })

    # ─── §3 — Architecture ─────────────────────────────────────────────
    sections.append({
        "number": "3",
        "title":  "Architecture",
        "lead":   ar["lead"],
        "blocks": [
            H3("Future-state agentic flow"),
            P(ar["agentic_flow_paragraph"]),
            H3("Component-level annotation"),
            *[P(f"**{c['name']}**. {c['description']}") for c in ar["components"]],
            H3("Layered technical stack"),
            P(ar["layered_stack_paragraph"]),
            H3("Pilot vs production data plane"),
            TBL(
                ["Layer", "Pilot", "Production"],
                [[r["layer"], r["pilot"], r["production"]] for r in ar["pilot_vs_production"]],
            ),
            P(ar["connector_paragraph"]),
        ],
    })

    # ─── §4 — Human–Agent Operating Model ──────────────────────────────
    sections.append({
        "number": "4",
        "title":  "Human–Agent Operating Model",
        "lead":   om["lead"],
        "blocks": [
            H3("Decision rights by risk and value"),
            P(om["decision_rights_paragraph"]),
            H3("Roles"),
            TBL(
                ["Role", "Reports to", "Pilot accountability"],
                [[r["role"], r["reports_to"], r["pilot_accountability"]] for r in om["roles"]],
            ),
            H3("RACI by future-state step"),
            TBL(
                ["Step", "Clerk", "Mgr", "Controller", "Vendor Master", "Audit", "Agent"],
                [[r["step"], r["clerk"], r["manager"], r["controller"], r["vendor_master"], r["audit"], r["agent"]] for r in om["raci_steps"]],
            ),
            H3("Escalation matrix"),
            TBL(
                ["Trigger", "First responder", "SLA", "If breached"],
                [[r["trigger"], r["first_responder"], r["sla"], r["if_breached"]] for r in om["escalation_matrix"]],
            ),
            H3("Day-in-the-life — after pilot"),
            P(om["day_in_the_life_paragraph"]),
        ],
    })

    # ─── §5 — Bill of Materials ────────────────────────────────────────
    sections.append({
        "number": "5",
        "title":  "Bill of Materials",
        "lead":   bo["lead"],
        "blocks": [
            H3("Software components"),
            TBL(
                ["Component", "Pilot tier", "Production tier", "License", "Cost band (USD/yr)"],
                [[r["component"], r["pilot_tier"], r["production_tier"], r["license"], r["cost_band"]] for r in bo["software_components"]],
            ),
            H3("Engagement & people"),
            TBL(
                ["Role", "Pilot duration", "Steady-state"],
                [[r["role"], r["pilot_duration"], r["steady_state"]] for r in bo["engagement_people"]],
            ),
            H3("Client technology intake"),
            P(bo["client_tech_intake_intro"]),
            TBL(
                ["Topic", "Question", "Drives"],
                [[r["topic"], r["question"], r["drives"]] for r in bo["client_tech_intake"]],
            ),
            H3("What is not in the BOM"),
            P(bo["not_in_bom_intro"]),
            B(bo["not_in_bom_bullets"]),
            P(bo["discipline_paragraph"]),
        ],
    })

    # ─── §6 — Configuration & Deployment Plan ──────────────────────────
    sections.append({
        "number": "6",
        "title":  "Configuration & Deployment Plan",
        "lead":   dp["lead"],
        "blocks": [
            H3("30-60-90 day path from blueprint to value evidence"),
            P(dp["thirty_sixty_ninety_paragraph"]),
            H3("Ten-day pilot plan"),
            TBL(
                ["Day", "Focus", "Exit criteria"],
                [
                    [
                        r.get("day", ""),
                        r.get("focus", ""),
                        r.get("exit_criteria", "Pending validation")
                    ]
                    for r in dp.get("pilot_plan", [])
                ],
            ),
            H3("Configuration surface — what the team will actually edit"),
            TBL(
                ["File", "What lives here", "Owner"],
                [[r["file"], r["what_lives_here"], r["owner"]] for r in dp["config_surface"]],
            ),
            H3("Gates that block production graduation"),
            P(dp["graduation_gates_intro"]),
            B(dp["graduation_gates_bullets"]),
            P(dp["graduation_paragraph"]),
        ],
    })

    # ─── §7 — Governance, Safety, and Audit ────────────────────────────
    sections.append({
        "number": "7",
        "title":  "Governance, Safety, and Audit",
        "lead":   gv["lead"],
        "blocks": [
            H3("Six control gates"),
            P(gv["six_control_gates_paragraph"]),
            H3("Decision provenance"),
            P(gv["decision_provenance_paragraph"]),
            H3("Model-risk controls"),
            TBL(
                ["Risk", "Control"],
                [[r["risk"], r["control"]] for r in gv["model_risk_controls"]],
            ),
            H3("SOX touchpoints"),
            P(gv["sox_touchpoints_intro"]),
            B(gv["sox_touchpoints_bullets"]),
            H3("Rollback story"),
            P(gv["rollback_paragraph"]),
            H3("What this design deliberately does not do"),
            P(gv["deliberately_not_done_intro"]),
            B(gv["deliberately_not_done_bullets"]),
        ],
    })

    # ─── §8 — Self-Improvement Loop and Benefits Realisation ───────────
    sections.append({
        "number": "8",
        "title":  "Self-Improvement Loop and Benefits Realisation",
        "lead":   si["lead"],
        "blocks": [
            H3("The four learning loops"),
            P(si["learning_loops_intro"]),
            *[P(loop) for loop in si["learning_loops"]],
            H3("What is deliberately not automated in the loop"),
            P(si["not_automated_paragraph"]),
            H3("What the metrics will show by quarter"),
            TBL(
                ["Quarter", "Override rate", "Touchless rate", "DPO / cycle", "Exception cycle"],
                [[r["quarter"], r["override_rate"], r["touchless_rate"], r["dpo_or_cycle"], r["exception_cycle"]] for r in si["metrics_by_quarter"]],
            ),
            H3("Benefits realisation"),
            P(si["benefits_realisation_intro"]),
            TBL(
                ["KPI", "Owner", "Data source", "Baseline method", "Cadence"],
                [[r["kpi"], r["owner"], r["data_source"], r["baseline_method"], r["cadence"]] for r in si["benefits_realisation"]],
            ),
            P(si["financial_impact_paragraph"]),
            P(si["projections_paragraph"]),
        ],
    })

    # ── §9 — Deploying in Your Environment (NEW per v2 reference) ────────────
    dy = data["deploying_in_your_environment"]
    sections.append({
        "number": "9",
        "title":  "Deploying in Your Environment",
        "lead":   dy["lead"],
        "blocks": [
            H3("Deployment approach at a glance"),
            P(dy["deployment_approach_paragraph"]),

            H3("9.1 Solution components — what gets deployed"),
            P(dy["solution_components_intro"]),
            TBL(
                ["Component", "Role", "Deploys as", "Owner"],
                [[r["component"], r["role"], r["deploys_as"], r["owner"]] for r in dy["solution_components"]],
            ),

            H3("9.2 Bill of materials — pilot vs production"),
            P(dy["bom_summary_paragraph"]),
            TBL(
                ["Layer", "Pilot", "Production"],
                [[r["layer"], r["pilot"], r["production"]] for r in dy["bom_pilot_vs_production"]],
            ),

            H3("9.3 Step-by-step deployment"),
            P(dy["step_by_step_intro"]),
            B(dy["step_by_step_bullets"]),

            H3("9.4 How the solution is grounded"),
            P(dy["grounding_paragraph"]),
            B(dy["grounding_bullets"]),

            H3("9.5 Integration with data and other systems"),
            P(dy["integration_intro"]),
            TBL(
                ["System", "Interface", "Direction", "Pilot", "Production"],
                [[r["system"], r["interface"], r["direction"], r["pilot"], r["production"]] for r in dy["integration_table"]],
            ),

            H3("9.6 Test approach"),
            P(dy["test_intro"]),
            TBL(
                ["Layer", "What it proves", "Owner", "Gate"],
                [[r["layer"], r["proves"], r["owner"], r["gate"]] for r in dy["test_layers"]],
            ),

            H3("9.7 Guardrails"),
            P(dy["guardrails_intro"]),
            B(dy["guardrails_bullets"]),
            P(dy["guardrails_closing_paragraph"]),
        ],
    })

    return {
        "process_key":   process_key,
        "generated_at":  now.isoformat() + "Z",
        "llm_generated": llm_used,
        "cover": {
            "brand_top":         "AGENTFORGE",
            "brand_subtitle":    "EXECUTION LAYER OF THE ENTERPRISE",
            "title":             title,
            "subtitle":          "Process Agentification Blueprint",
            "tagline":           data["cover"]["tagline"],
            "deliverable_label": "Engagement deliverable",
            "footer_line":       f"Prepared by AgentForge · Confidential · {now.year}",
        },
        "sections": sections,
        "closing": {
            "title":  "Closing",
            "blocks": [
                {"type": "paragraph", "text": cl["primary_paragraph"]},
                {"type": "paragraph", "text": cl["tagline_one"]},
                {"type": "paragraph", "text": cl["tagline_two"]},
            ],
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Public entry point
# ═════════════════════════════════════════════════════════════════════════════
def build_blueprint(process_key: str) -> Dict[str, Any]:
    """
    Returns the full blueprint payload for the given process_key.  Never
    raises — always returns a well-formed document (with deterministic
    defaults when LLM/data is missing).
    """
    ctx = _load_process_context(process_key)

    # Always start from deterministic defaults
    defaults = _defaults(ctx)

    # Try the LLM — merge over the defaults
    dyn = _call_llm(_build_prompt(ctx))
    llm_used = bool(dyn)

    merged = _deep_merge_dict(defaults, dyn) if dyn else defaults

    # Ensure every required key is present (LLM may have omitted whole sections)
    for required_key in (
        "cover", "exec_summary", "constraint_diagnosis", "future_state_process",
        "architecture", "operating_model", "bill_of_materials", "deployment_plan",
        "governance", "self_improvement", "deploying_in_your_environment", "closing",
    ):
        if required_key not in merged or not isinstance(merged[required_key], dict):
            merged[required_key] = defaults[required_key]
        else:
            merged[required_key] = _deep_merge_dict(defaults[required_key], merged[required_key])

    payload = _assemble_payload(process_key, ctx, merged, llm_used)
    return payload



# ═════════════════════════════════════════════════════════════════════════════
# 7.  Suggestion-focused entry point (NEW per spec)
# ═════════════════════════════════════════════════════════════════════════════
def _resolve_suggestion_to_process(suggestion_id: str):
    """
    Look up a suggestion by _key.  Returns (process_key, suggestion_dict, step_dict).
    The suggestion's process_key is the anchor; the step is whichever step the
    suggestion targets (or the highest-automation step if the suggestion does
    not name one).
    """
    db = get_db()
    col = db.collection
    sug = None
    try:
        sug = col(COLLECTIONS["suggestions"]).get(suggestion_id)
    except Exception:
        sug = None
    if not sug:
        # Caller will fall back to defaults; never raise.
        return None, None, None

    process_key = sug.get("process_key")
    step = None

    # Prefer an explicit step_key/step_id linkage if the suggestion carries one
    step_key = sug.get("step_key") or sug.get("step_id") or sug.get("anchor_step_key")
    if step_key:
        try:
            step = col("process_steps").get(step_key)
        except Exception:
            step = None

    # Otherwise, pick the highest-intervention step in the process — that is
    # the natural focus for a "higher agentic intervention" blueprint.
    if not step and process_key:
        try:
            steps = list(db.aql(
                "FOR s IN process_steps FILTER s.process_key == @k SORT s.automation_potential DESC RETURN s",
                {"k": process_key},
            ))
            step = steps[0] if steps else None
        except Exception:
            step = None

    return process_key, sug, step


def build_blueprint_for_suggestion(suggestion_id: str) -> Dict[str, Any]:
    """
    NEW per spec — build a blueprint payload FOCUSED ON the chosen suggestion.

    The shape is identical to build_blueprint(process_key) so the same frontend
    generators render it.  The difference is content focus:

      • title / tagline / cover are anchored to the suggestion + its step
      • the LLM prompt is told which step + suggestion to centre on
      • defaults pre-fill the suggestion's anchor step into ownership / RACI /
        task-breakdown tables so even the deterministic fallback is on-topic

    Never raises — falls back to defaults when data or LLM is missing.
    """
    process_key, suggestion, step = _resolve_suggestion_to_process(suggestion_id)

    if not process_key:
        # No resolvable process — return a minimal payload built off defaults so
        # the exported document is not empty.  We synthesise a placeholder ctx.
        ctx = {"process": {"title": "Selected Suggestion"}, "steps": [], "suggestions": [], "erp_modules": []}
        defaults = _defaults(ctx)
        payload = _assemble_payload(suggestion_id, ctx, defaults, llm_used=False)
        payload["focus"] = {"suggestion_id": suggestion_id, "resolved": False}
        return payload

    # Load the full process context, then attach focus metadata
    ctx = _load_process_context(process_key)
    ctx["focus_suggestion"] = suggestion or {}
    ctx["focus_step"]       = step or {}

    # Defaults built off the process — then biased toward the chosen step.
    defaults = _defaults(ctx)
    _bias_defaults_to_focus(defaults, suggestion or {}, step or {})

    # LLM prompt — append a focus directive so the model centres on the step.
    prompt = _build_prompt(ctx)
    if suggestion or step:
        prompt += _build_focus_addendum(suggestion or {}, step or {})

    dyn = _call_llm(prompt)
    llm_used = bool(dyn)
    merged = _deep_merge_dict(defaults, dyn) if dyn else defaults

    for required_key in (
        "cover", "exec_summary", "constraint_diagnosis", "future_state_process",
        "architecture", "operating_model", "bill_of_materials", "deployment_plan",
        "governance", "self_improvement", "deploying_in_your_environment", "closing",
    ):
        if required_key not in merged or not isinstance(merged[required_key], dict):
            merged[required_key] = defaults[required_key]
        else:
            merged[required_key] = _deep_merge_dict(defaults[required_key], merged[required_key])

    payload = _assemble_payload(process_key, ctx, merged, llm_used)

    # Stamp focus metadata on the payload so the FE can render the suggestion
    # title and the step automation potential on the cover.
    payload["focus"] = {
        "suggestion_id":         suggestion_id,
        "resolved":              True,
        "suggestion_title":      (suggestion or {}).get("title"),
        "suggestion_summary":    (suggestion or {}).get("description"),
        "step_number":           (step or {}).get("step_number"),
        "step_title":            (step or {}).get("title"),
        "automation_potential":  (step or {}).get("automation_potential"),
        "is_higher_intervention": int((step or {}).get("automation_potential") or 0) >= 70,
    }

    # Cover tagline / subtitle re-anchored to the chosen suggestion + step
    if suggestion:
        focus_tag = (suggestion.get("title") or "").strip()
        if focus_tag:
            payload["cover"]["subtitle"] = f"Agentic Blueprint — {focus_tag}"
    if step:
        pot = int(step.get("automation_potential") or 0)
        step_no = step.get("step_number")
        step_title = step.get("title") or ""
        if step_title:
            payload["cover"]["tagline"] = (
                f"Focused on Step {step_no} · {step_title} · "
                f"Automation Potential {pot}%"
            )

    return payload


def _bias_defaults_to_focus(defaults: Dict[str, Any], suggestion: Dict[str, Any], step: Dict[str, Any]) -> None:
    """Tweak the deterministic defaults so they read on-topic for the chosen
    suggestion + step.  Pure local edits — no LLM cost."""
    step_no    = step.get("step_number")
    step_title = (step.get("title") or "").strip()
    pot        = int(step.get("automation_potential") or 0)
    sug_title  = (suggestion.get("title") or "").strip()
    sug_desc   = (suggestion.get("description") or "").strip()

    if step_title and "exec_summary" in defaults:
        defaults["exec_summary"]["lead"] = (
            f"This blueprint is focused on Step {step_no} — {step_title}, "
            f"which the analysis flagged with the higher agentic intervention "
            f"({pot}% automation potential). The pilot operationalises the "
            f"redesign of this step end-to-end while keeping upstream and "
            f"downstream steps unchanged."
        )

    if sug_title and "exec_summary" in defaults:
        defaults["exec_summary"]["constraint_anchor"] = (
            f"The constraint anchor is the chosen suggestion: \"{sug_title}\". "
            f"{sug_desc[:240]}"
        )

    # Bias the step-selection table so the focused step is the first SELECTED row
    if step_title and "constraint_diagnosis" in defaults:
        rows = defaults["constraint_diagnosis"].get("step_selection_rationale") or []
        focused_row = {
            "no":             str(step_no) if step_no is not None else "—",
            "step":           step_title,
            "constraint":     "Very high" if pot >= 80 else "High",
            "volume":         "High",
            "repeatability":  "High" if pot >= 60 else "Mixed",
            "exception_cost": "High",
            "reversibility":  "High",
            "decision":       "Select",
        }
        defaults["constraint_diagnosis"]["step_selection_rationale"] = (
            [focused_row] + [r for r in rows if r.get("step") != step_title]
        )[:8]


def _build_focus_addendum(suggestion: Dict[str, Any], step: Dict[str, Any]) -> str:
    """Append a focus directive to the LLM prompt so the model centres on the
    chosen suggestion + its anchor step (the higher-intervention one)."""
    return f"""

==================== FOCUS (suggestion-level blueprint) ====================
The blueprint MUST be focused on this specific suggestion + its anchor step:

  Suggestion title:        {suggestion.get('title', '')!r}
  Suggestion description:  {(suggestion.get('description') or '')[:600]!r}
  Anchor step number:      {step.get('step_number', '?')}
  Anchor step title:       {step.get('title', '')!r}
  Anchor step description: {(step.get('description') or '')[:400]!r}
  Anchor step actor:       {step.get('actor', '')!r}
  Automation potential:    {step.get('automation_potential', 0)}%

Write every section so the reader can tell, without checking the cover,
that this is the blueprint for THAT suggestion on THAT step:
  - exec summary leads with the step name
  - constraint diagnosis names the step as the bottleneck
  - future-state process tables put this step in the first row
  - architecture's component annotations call out which agent owns this step
  - operating-model RACI lists this step explicitly
  - deployment plan's pilot scope is this step
  - benefits realisation table cites the metrics this step moves
  - §9 (Deploying in Your Environment) writes integration + test layers
    for the systems this step touches
"""

