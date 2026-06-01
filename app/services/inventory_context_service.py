"""
app/services/inventory_context_service.py
─────────────────────────────────────────────────────────────────────────────
System & Module Inventory — Dynamic Context Generator

The "System & Module Inventory" section in the downloadable PDF/PPTX/DOCX
was previously empty (or missing) for many uploads.  This module produces
a *dynamic, contextual* inventory that is always populated, derived from:

  • the uploaded source files (CSV schemas, doc filenames)
  • the analysis pipeline output (extracted ERP modules, process steps)
  • a deterministic LLM enrichment pass

Even when the analysis pipeline yields ZERO `erp_modules` rows, this
service synthesises a meaningful inventory from process steps + actor
information so the export never shows an empty section.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.mistral_client import get_mistral_client

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic seed — common ERP module families we can fall back to when the
# pipeline produced nothing.  Keyed by "tag" used in step text.
# ─────────────────────────────────────────────────────────────────────────────
_MODULE_SEEDS: Dict[str, Dict[str, Any]] = {
    "procurement": {
        "module_name":       "Procurement / Materials Management",
        "description":       "Handles purchase requisitions, vendor master, and PO lifecycle.",
        "entities":          ["EKKO", "EKPO", "LFA1", "T024E"],
        "typical_source":    "SAP MM",
    },
    "finance": {
        "module_name":       "Financial Accounting (FI)",
        "description":       "Captures invoice verification, journal entries, payment posting.",
        "entities":          ["BKPF", "BSEG", "RBKP", "RSEG", "T001"],
        "typical_source":    "SAP FI",
    },
    "sales": {
        "module_name":       "Sales & Distribution (SD)",
        "description":       "Order capture, pricing, shipping, billing, and credit checks.",
        "entities":          ["VBAK", "VBAP", "VBRK", "VBRP"],
        "typical_source":    "SAP SD",
    },
    "inventory": {
        "module_name":       "Inventory & Warehouse Management",
        "description":       "Goods receipt, stock postings, and warehouse movements.",
        "entities":          ["MARD", "MSEG", "MARA"],
        "typical_source":    "SAP MM / WM",
    },
    "hr": {
        "module_name":       "Human Capital Management",
        "description":       "Employee master, organisational assignment, and time data.",
        "entities":          ["PA0001", "PA0002", "HRP1001"],
        "typical_source":    "SAP HCM / SuccessFactors",
    },
    "logistics": {
        "module_name":       "Logistics Execution",
        "description":       "Delivery processing, transportation, and shipment tracking.",
        "entities":          ["LIKP", "LIPS", "VTTK"],
        "typical_source":    "SAP LE",
    },
    "crm": {
        "module_name":       "Customer Relationship Management",
        "description":       "Account, opportunity, and case lifecycle management.",
        "entities":          ["Account", "Opportunity", "Case"],
        "typical_source":    "Salesforce / SAP CRM",
    },
    "default": {
        "module_name":       "Enterprise Workflow Layer",
        "description":       "Cross-functional orchestration of business process steps.",
        "entities":          ["Process Steps", "Audit Log", "Workflow State"],
        "typical_source":    "ADF (Azure Data Factory)",
    },
}


_TAG_KEYWORDS = {
    "procurement": ["procure", "purchase", "vendor", "supplier", "po ", "requisition"],
    "finance":     ["invoice", "payment", "ledger", "accounting", "credit", "ap ", "ar "],
    "sales":       ["sales order", "sales", "quote", "customer order", "billing"],
    "inventory":   ["inventory", "stock", "warehouse", "goods receipt", "gr "],
    "hr":          ["employee", "payroll", "onboard", "leave", "appraisal"],
    "logistics":   ["ship", "logistic", "delivery", "dispatch", "carrier"],
    "crm":         ["customer", "lead", "opportunity", "case", "ticket"],
}


def _tag_for_text(blob: str) -> str:
    low = (blob or "").lower()
    scores = {tag: sum(low.count(k) for k in kws) for tag, kws in _TAG_KEYWORDS.items()}
    best = max(scores.items(), key=lambda kv: kv[1]) if scores else ("default", 0)
    return best[0] if best[1] > 0 else "default"


def _seed_inventory_from_steps(
    process_title: str,
    process_description: str,
    steps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate inventory entries from step actors + step content."""
    blob = f"{process_title} {process_description} " + " ".join(
        f"{s.get('title','')} {s.get('description','')}"
        for s in (steps or [])
    )
    primary_tag = _tag_for_text(blob)

    # Always include the primary inferred module
    inventory: List[Dict[str, Any]] = [
        {
            "module_name":     _MODULE_SEEDS[primary_tag]["module_name"],
            "description":     _MODULE_SEEDS[primary_tag]["description"],
            "entities":        list(_MODULE_SEEDS[primary_tag]["entities"]),
            "source_system":   _MODULE_SEEDS[primary_tag]["typical_source"],
            "context_origin":  "Inferred from process content keywords",
        }
    ]

    # Add a secondary module if multiple actor families are present
    actor_set = {s.get("actor", "").lower() for s in (steps or []) if s.get("actor")}
    secondary_candidates: List[str] = []
    for tag in _TAG_KEYWORDS:
        if tag == primary_tag:
            continue
        for kw in _TAG_KEYWORDS[tag]:
            if any(kw in a for a in actor_set):
                secondary_candidates.append(tag)
                break

    seen = {primary_tag}
    for tag in secondary_candidates[:2]:
        if tag in seen:
            continue
        seen.add(tag)
        seed = _MODULE_SEEDS[tag]
        inventory.append({
            "module_name":     seed["module_name"],
            "description":     seed["description"],
            "entities":        list(seed["entities"]),
            "source_system":   seed["typical_source"],
            "context_origin":  f"Inferred from actor signatures ({tag})",
        })

    return inventory


def _normalize_pipeline_erp_modules(modules: List[Any]) -> List[Dict[str, Any]]:
    """Convert pipeline ERP module objects/dicts into the export shape."""
    out: List[Dict[str, Any]] = []
    for m in (modules or []):
        if hasattr(m, "to_doc"):
            try:
                m = m.to_doc()
            except Exception:
                pass
        if not isinstance(m, dict):
            continue
        entities = m.get("tables_identified") or m.get("entities") or []
        if isinstance(entities, str):
            entities = [e.strip() for e in entities.split(",") if e.strip()]

        out.append({
            "module_name":    m.get("module_name") or m.get("name") or "Module",
            "description":    m.get("description") or "Identified module from analysis pipeline.",
            "entities":       entities,
            "source_system":  m.get("source_file") or m.get("source_system") or "Detected source",
            "context_origin": "Extracted by analysis pipeline",
        })
    return out


def _llm_enrich_inventory(
    process_title: str,
    process_description: str,
    inventory_seed: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Single LLM pass that fleshes out the inventory with two extra fields per
    module: `responsibilities` (list[str]) and `data_flow_notes` (string).
    Returns None on any failure — callers must handle.
    """
    if not inventory_seed:
        return None
    try:
        llm = get_mistral_client()
    except Exception as e:
        logger.warning(f"[inventory] LLM unavailable: {e}")
        return None

    step_titles = [s.get("title", "") for s in (steps or [])[:15] if s.get("title")]
    step_block = "\n".join(f"  - {t}" for t in step_titles) or "  (none)"

    seed_block = "\n".join(
        f"  - {m['module_name']} (source: {m['source_system']})"
        for m in inventory_seed
    )

    prompt = f"""You are enriching a System & Module Inventory section for an
enterprise technical design document.

Process:        {process_title}
Description:    {process_description or '(none)'}

Process steps:
{step_block}

Seed modules:
{seed_block}

For EACH seed module above, return an enriched description, 3-5
`responsibilities`, and one `data_flow_notes` sentence describing how it
participates in the process.

Return ONLY a JSON array — no markdown, no commentary:
[
  {{
    "module_name":       "<same as seed>",
    "description":       "<2-sentence enriched description>",
    "entities":          ["<entity-1>","<entity-2>", ...],
    "source_system":     "<carry through from seed if not detected>",
    "responsibilities":  ["<r1>","<r2>","<r3>"],
    "data_flow_notes":   "<one sentence>",
    "context_origin":    "AI-enriched from uploaded content"
  }}
]
""".strip()

    try:
        raw = llm._chat(
            "You are a senior enterprise-architecture analyst. Return ONLY valid JSON.",
            prompt,
            temperature=0.2,
        )
        parsed = llm._parse_json(raw)
        if isinstance(parsed, list) and parsed:
            cleaned = []
            for m in parsed:
                if not isinstance(m, dict):
                    continue
                cleaned.append({
                    "module_name":     m.get("module_name", "Module"),
                    "description":     m.get("description", ""),
                    "entities":        m.get("entities") or [],
                    "source_system":   m.get("source_system") or "Detected source",
                    "responsibilities": m.get("responsibilities") or [],
                    "data_flow_notes":  m.get("data_flow_notes") or "",
                    "context_origin":   m.get("context_origin", "AI-enriched"),
                })
            return cleaned
    except Exception as e:
        logger.warning(f"[inventory] LLM enrichment failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def build_system_module_inventory(
    process: Dict[str, Any],
    steps: List[Dict[str, Any]],
    erp_modules: List[Any],
) -> List[Dict[str, Any]]:
    """
    Returns a non-empty list of inventory rows for the export.

    Priority:
      1. Use whatever the pipeline already identified (erp_modules).
      2. Fall back to keyword-seeded synthesis from steps/actors.
      3. LLM enrichment pass to add responsibilities + data_flow_notes.
    """
    process = process or {}
    title = process.get("title") or process.get("name") or "Business Process"
    desc  = process.get("description") or ""

    inventory = _normalize_pipeline_erp_modules(erp_modules)
    if not inventory:
        inventory = _seed_inventory_from_steps(title, desc, steps)

    enriched = _llm_enrich_inventory(title, desc, inventory, steps)
    if enriched:
        # Merge enriched fields onto the seed list, preserving seed source_system
        by_name = {m.get("module_name", "").lower(): m for m in inventory}
        for em in enriched:
            key = (em.get("module_name") or "").lower()
            base = by_name.get(key)
            if base:
                base.update({
                    "description":      em.get("description") or base.get("description"),
                    "entities":         em.get("entities") or base.get("entities") or [],
                    "responsibilities": em.get("responsibilities") or [],
                    "data_flow_notes":  em.get("data_flow_notes") or "",
                    "context_origin":   em.get("context_origin") or base.get("context_origin"),
                })
            else:
                inventory.append(em)

    # Final safety net — guarantee at least one row
    if not inventory:
        seed = _MODULE_SEEDS["default"]
        inventory = [{
            "module_name":      seed["module_name"],
            "description":      seed["description"],
            "entities":         list(seed["entities"]),
            "source_system":    seed["typical_source"],
            "responsibilities": [],
            "data_flow_notes":  "Acts as the orchestration backbone for the workflow.",
            "context_origin":   "Default fallback (no signals available)",
        }]

    return inventory
