"""
app/services/semantic_kg_service.py
─────────────────────────────────────────────────────────────────────────────
Data-agnostic base Knowledge Graph builder + SME enrichment.

Pipeline (matches the new SME-driven workflow):

    Step 1  ingest_to_base_graph(files, user_input, session_id)
              • parse any uploaded data (CSV / Excel / PDF / ERP log / SOP /
                free text / web content)  → text + metadata
              • store the RAW data as JSON (with metadata) in `kg_raw_data`
              • generate the semantic layer + ER model via the generic
                BASE_GRAPH_SYSTEM_PROMPT (entities, attributes, relationships,
                metrics, lineage)
              • map entities → NODES (kg_nodes) and relationships → EDGES
                (kg_edges), with properties + metrics + source lineage
              • persist the semantic layer in `kg_semantic`

    Step 2  sme_chat(session_id, query, history)
              • answer the user's question grounded in the base graph
                (what / where / why philosophy)

    Step 3  enrich_from_sme(session_id, transcript)
              • extract NEW business knowledge from the conversation and fold
                it into EXISTING nodes/edges (no duplicate nodes), preserving
                graph consistency

The final analysis (Step 5/6 of the spec) is produced by the existing
analysis pipeline, which receives the original data plus the SME context.

Everything degrades gracefully: if the LLM or DB is unavailable the functions
return safe, empty-ish payloads so the API never 500s the user flow.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.db.arango import get_db
from app.core.mistral_client import get_mistral_client
from app.parsers.file_parser import parse_file, detect_source_type
from app.prompts.kg_prompts import (
    BASE_GRAPH_SYSTEM_PROMPT,
    build_base_graph_user_prompt,
    SME_CHAT_SYSTEM_PROMPT,
    build_sme_chat_user_prompt,
    SME_OPENING_SYSTEM_PROMPT,
    build_sme_opening_user_prompt,
    SME_ENRICHMENT_SYSTEM_PROMPT,
    build_sme_enrichment_user_prompt,
    SUGGESTED_QUESTIONS_SYSTEM_PROMPT,
    build_suggested_questions_user_prompt,
)

logger = logging.getLogger(__name__)

# ── Collections (idempotently created) ───────────────────────────────────────
RAW_COLL      = "kg_raw_data"     # raw uploaded data stored as JSON + metadata
NODE_COLL     = "kg_nodes"        # base-graph entities
EDGE_COLL     = "kg_edges"        # base-graph relationships  (edge collection)
SEMANTIC_COLL = "kg_semantic"     # semantic layer per session

_MAX_CONTENT_CHARS = 100000       # cap LLM input size
_MAX_JSON_RECORDS  = 40           # sample records embedded in the prompt

# ── Discovery phase ordering (What → Where → Why/How) ────────────────────────
# The engagement gathers information in a strict order. We advance to the next
# phase only once the current phase is sufficiently understood, so the chat can
# never jump ahead to a later phase.
PHASE_ORDER = ("what", "where", "why")
PHASE_ADVANCE_THRESHOLD = 80      # % coverage at which a phase is "done enough"


def _resolve_phase(coverage_state: Dict[str, Any]) -> str:
    """Return the active discovery phase as the LOWEST phase not yet complete.

    This enforces the What → Where → Why/How order: we stay on WHAT until
    what_coverage clears the threshold, then WHERE until where_coverage clears
    it, then WHY/HOW. Because the chat is constrained to the active phase, a
    later phase's coverage cannot rise (and the conversation cannot skip a
    phase) before it is actually reached.
    """
    try:
        what  = float(coverage_state.get("what_coverage")  or 0)
        where = float(coverage_state.get("where_coverage") or 0)
    except (TypeError, ValueError):
        what = where = 0.0
    if what < PHASE_ADVANCE_THRESHOLD:
        return "what"
    if where < PHASE_ADVANCE_THRESHOLD:
        return "where"
    return "why"


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def ensure_collections() -> None:
    db = get_db().db
    for name in (RAW_COLL, NODE_COLL, SEMANTIC_COLL):
        if not db.has_collection(name):
            db.create_collection(name)
            logger.info(f"[skg] created collection {name}")
    if not db.has_collection(EDGE_COLL):
        db.create_collection(EDGE_COLL, edge=True)
        logger.info(f"[skg] created edge collection {EDGE_COLL}")


def _node_key(session_id: str, entity_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in entity_id)[:120]
    return f"{session_id}__{safe}"


# ─────────────────────────────────────────────────────────────────────────────
# Attribute / metric merge helpers (entity-attribute-metric correctness)
# Used by BOTH the live SME chat and the finalize enrichment so the graph records
# entities, their attributes and their metrics consistently, without duplicates.
# ─────────────────────────────────────────────────────────────────────────────
def _merge_attributes(existing: List[Dict[str, Any]],
                      additions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge SME-revealed attributes into a node's attribute list, keyed by
    (lowercased) name so the same field is never recorded twice. Newly seen
    example values are folded into the existing attribute's examples."""
    merged = list(existing or [])
    index = {(a.get("name") or "").strip().lower(): a
             for a in merged if a.get("name")}
    for a in (additions or []):
        name = (a.get("name") or "").strip()
        if not name:
            continue
        value = a.get("value")
        key = name.lower()
        if key in index:
            # Attribute already known — enrich its examples / description only.
            attr = index[key]
            if value:
                examples = attr.get("examples") or []
                if value not in examples:
                    examples.append(value)
                attr["examples"] = examples
            if a.get("description") and not attr.get("description"):
                attr["description"] = a["description"]
            continue
        attr = {
            "name": name,
            "data_type": a.get("data_type") or "string",
            "description": a.get("description") or a.get("value") or "",
            "is_key": False, "is_foreign_key": False, "is_required": False,
            "examples": [value] if value else [],
            "origin": "sme",
        }
        merged.append(attr)
        index[key] = attr
    return merged


def _merge_metrics(existing: List[Dict[str, Any]],
                   additions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge metrics into a node/edge metric list, de-duplicated by metric name
    (case-insensitive). Prevents the same KPI being appended on every turn."""
    merged = list(existing or [])
    seen = {(m.get("name") or "").strip().lower()
            for m in merged if isinstance(m, dict) and m.get("name")}
    for m in (additions or []):
        if not isinstance(m, dict):
            continue
        name = (m.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        merged.append({
            "name": name,
            "definition": m.get("definition") or m.get("description") or "",
            "unit": m.get("unit") or "",
            "origin": "sme",
        })
        seen.add(name.lower())
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Tabular → JSON records (for the prompt + raw storage)
# ─────────────────────────────────────────────────────────────────────────────
def _tabular_records(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Best-effort conversion of a CSV/XLSX file to a small list of JSON rows."""
    try:
        import pandas as pd  # already a backend dependency
        lower = filename.lower()
        if lower.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), nrows=_MAX_JSON_RECORDS)
            except Exception:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1",
                                 nrows=_MAX_JSON_RECORDS)
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes), nrows=_MAX_JSON_RECORDS)
        else:
            return []
        df = df.where(df.notna(), None)
        return json.loads(df.to_json(orient="records"))
    except Exception as e:
        logger.debug(f"[skg] tabular->json failed for {filename}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Build the base graph from any uploaded data
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_semantic_dict(value: Any) -> Dict[str, Any]:
    """Normalize a stored/produced semantic_layer to a dict.

    Local models sometimes emit it as a list (or other shape) rather than the
    expected object, which previously crashed every `.get(...)` call with
    'list' object has no attribute 'get'.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((x for x in value if isinstance(x, dict)), {})
    return {}


def ingest_to_base_graph(
    files: List[Tuple[bytes, str]],
    user_input: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Parse the data, store it as JSON, extract the ER/semantic model with the
    generic prompt, and persist nodes + edges into the base knowledge graph.

    Returns a summary payload for the frontend (counts + samples + suggested
    questions). Never raises — returns {"status": "error", ...} on failure.
    """
    ensure_collections()
    db = get_db()

    text_parts: List[str] = []
    metadata: Dict[str, Any] = {}
    json_records: Dict[str, List[Dict[str, Any]]] = {}
    data_kinds: List[str] = []

    for file_bytes, filename in (files or []):
        try:
            text, meta = parse_file(file_bytes, filename)
        except Exception as e:
            logger.warning(f"[skg] parse failed {filename}: {e}")
            text, meta = "", {"error": str(e)}
        text_parts.append(f"=== File: {filename} ===\n{text}")
        metadata[filename] = meta
        data_kinds.append(detect_source_type(filename))
        recs = _tabular_records(file_bytes, filename)
        if recs:
            json_records[filename] = recs

    if user_input:
        text_parts.append(f"=== USER INPUT ===\n{user_input}")
        data_kinds.append("conversation")

    combined_text = "\n\n".join(text_parts).strip()
    if not combined_text:
        return {"status": "error", "message": "No analyzable input provided.",
                "session_id": session_id}

    # 🔥 NEW VALIDATION: Reject irrelevant files early before graph building
    llm = get_mistral_client()
    relevance = llm.validate_relevance(user_input, combined_text)
    if not relevance.get("is_relevant", True):
        error_msg = relevance.get("error", "Irrelevant file.")
        solution = relevance.get("recommended_solution", "Upload a relevant file.")
        return {
            "status": "error",
            "error_type": "IRRELEVANT_FILE",
            "message": error_msg,
            "recommended_solution": solution,
            "session_id": session_id
        }

    # Pick a representative data_kind for the prompt
    if "erp_dump" in data_kinds or "csv" in data_kinds:
        data_kind = "dataset"
    elif data_kinds and all(k in ("conversation",) for k in data_kinds):
        data_kind = "conversation"
    else:
        data_kind = "mixed" if len(set(data_kinds)) > 1 else (data_kinds[0] if data_kinds else "document")

    # ── Persist the raw data as JSON (with metadata + lineage) ──────────────
    try:
        db.collection(RAW_COLL).insert({
            "_key": f"raw_{session_id}",
            "session_id": session_id,
            "data_kind": data_kind,
            "metadata": metadata,
            "json_records": json_records,
            "content_preview": combined_text[:5000],
            "stored_at": datetime.utcnow().isoformat() + "Z",
        }, overwrite=True)
    except Exception as e:
        logger.warning(f"[skg] raw store failed: {e}")

    # ── LLM: extract ER model + semantic layer ──────────────────────────────
    sample_json = json.dumps(
        {k: v[:8] for k, v in json_records.items()}, default=str
    )[:6000] if json_records else "{}"

    user_prompt = build_base_graph_user_prompt(
        data_kind=data_kind,
        source_summary=combined_text[:_MAX_CONTENT_CHARS],
        metadata_json=json.dumps(metadata, default=str)[:6000],
        sample_data=sample_json,
    )

    model: Dict[str, Any] = {}
    try:
        llm = get_mistral_client()
        model = llm._chat_json(BASE_GRAPH_SYSTEM_PROMPT, user_prompt, expect="object") or {}
        if not isinstance(model, dict):
            model = {}
    except Exception as e:
        logger.error(f"[skg] base-graph LLM failed: {e}", exc_info=True)
        model = {}

    semantic_layer = model.get("semantic_layer") or {
        "domain": "Generic",
        "summary": "Base graph extracted from the uploaded data.",
        "business_context": "",
        "data_kind": data_kind,
        "process_domains": [],
        "ontology_mappings": [],
        "relationship_insights": [],
        "glossary": [],
    }
    # Local models sometimes return semantic_layer as a list (or other shape)
    # instead of an object. Normalize to a dict so downstream .get() is safe.
    if isinstance(semantic_layer, list):
        semantic_layer = next(
            (x for x in semantic_layer if isinstance(x, dict)), {}
        )
    if not isinstance(semantic_layer, dict) or not semantic_layer:
        semantic_layer = {
            "domain": "Generic",
            "summary": "Base graph extracted from the uploaded data.",
            "business_context": "",
            "data_kind": data_kind,
            "process_domains": [],
            "ontology_mappings": [],
            "relationship_insights": [],
            "glossary": [],
        }
    entities      = model.get("entities") or []
    relationships = model.get("relationships") or []

    # ── Persist semantic layer ──────────────────────────────────────────────
    try:
        db.collection(SEMANTIC_COLL).insert({
            "_key": f"sem_{session_id}",
            "session_id": session_id,
            "semantic_layer": semantic_layer,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }, overwrite=True)
    except Exception as e:
        logger.warning(f"[skg] semantic store failed: {e}")

    # ── Map entities → nodes (merge by stable id) ───────────────────────────
    node_coll = db.collection(NODE_COLL)
    persisted_nodes = 0
    for ent in entities:
        eid = (ent.get("id") or ent.get("name") or "").strip()
        if not eid:
            continue
        key = _node_key(session_id, eid)
        doc = {
            "_key": key,
            "session_id": session_id,
            "entity_id": eid,
            "name": ent.get("name") or eid,
            "entity_type": ent.get("entity_type") or "strong",
            "description": ent.get("description") or "",
            "meaning": ent.get("meaning") or "",
            "instances": ent.get("instances") or [],
            "attributes": ent.get("attributes") or [],
            "metrics": ent.get("metrics") or [],
            "source": ent.get("source") or {},
            "confidence": ent.get("confidence", 0.6),
            "tags": [],
            "business_rules": [],
            "origin": "base_graph",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            node_coll.insert(doc, overwrite=True)
            persisted_nodes += 1
        except Exception as e:
            logger.debug(f"[skg] node insert failed {key}: {e}")

    # ── Map relationships → edges ───────────────────────────────────────────
    edge_coll = db.collection(EDGE_COLL)
    persisted_edges = 0
    for rel in relationships:
        frm = (rel.get("from") or "").strip()
        to  = (rel.get("to") or "").strip()
        if not frm or not to:
            continue
        from_id = f"{NODE_COLL}/{_node_key(session_id, frm)}"
        to_id   = f"{NODE_COLL}/{_node_key(session_id, to)}"
        rid = (rel.get("id") or f"{frm}_{rel.get('name','rel')}_{to}").strip()
        edoc = {
            "_key": _node_key(session_id, rid),
            "_from": from_id,
            "_to": to_id,
            "session_id": session_id,
            "rel_id": rid,
            "name": rel.get("name") or "related_to",
            "cardinality": rel.get("cardinality") or "1:N",
            "relationship_type": rel.get("relationship_type") or "association",
            "description": rel.get("description") or "",
            "meaning": rel.get("meaning") or "",
            "properties": rel.get("properties") or {},
            "metrics": rel.get("metrics") or [],
            "source": rel.get("source") or {},
            "confidence": rel.get("confidence", 0.6),
            "origin": "base_graph",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            edge_coll.insert(edoc, overwrite=True)
            persisted_edges += 1
        except Exception as e:
            logger.debug(f"[skg] edge insert failed {rid}: {e}")

    logger.info(
        f"[skg] base graph for {session_id}: {persisted_nodes} nodes, "
        f"{persisted_edges} edges, domain={semantic_layer.get('domain')}"
    )

    suggested = generate_suggested_questions(session_id)

    return {
        "status": "ok",
        "session_id": session_id,
        "domain": semantic_layer.get("domain", "Generic"),
        "summary": semantic_layer.get("summary", ""),
        "business_context": semantic_layer.get("business_context", ""),
        "data_kind": data_kind,
        "node_count": persisted_nodes,
        "edge_count": persisted_edges,
        "entities": [
            {"id": e.get("id"), "name": e.get("name"),
             "meaning": e.get("meaning", "")}
            for e in entities[:30]
        ],
        "relationships": [
            {"id": r.get("id") or f"{r.get('from')}_{r.get('name')}_{r.get('to')}", 
             "source": r.get("from"), "target": r.get("to"), 
             "name": r.get("name"), "meaning": r.get("meaning", "")}
            for r in relationships[:50]
        ],
        "edges": [
            {"id": r.get("id") or f"{r.get('from')}_{r.get('name')}_{r.get('to')}", 
             "source": r.get("from"), "target": r.get("to"), "label": r.get("name")}
            for r in relationships[:50]
        ],
        "glossary": semantic_layer.get("glossary", [])[:20],
        "process_domains": semantic_layer.get("process_domains", []),
        "ontology_mappings": semantic_layer.get("ontology_mappings", [])[:20],
        "relationship_insights": semantic_layer.get("relationship_insights", [])[:10],
        "suggested_questions": suggested,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build a readable context block for the chat / enrichment LLM calls
# ─────────────────────────────────────────────────────────────────────────────
def build_base_graph_context(session_id: str, *, limit: int = 150) -> str:
    db = get_db()
    lines: List[str] = []

    # Semantic header
    try:
        sem = db.collection(SEMANTIC_COLL).get(f"sem_{session_id}") or {}
        sl = _ensure_semantic_dict(sem.get("semantic_layer"))
        if sl:
            lines.append(f"DOMAIN: {sl.get('domain','Generic')}")
            if sl.get("summary"):
                lines.append(f"SUMMARY: {sl['summary']}")
            if sl.get("business_context"):
                lines.append(f"BUSINESS CONTEXT: {sl['business_context']}")
            glossary = sl.get("glossary") or []
            if glossary:
                lines.append("GLOSSARY: " + "; ".join(
                    f"{g.get('term')}={g.get('meaning')}" for g in glossary[:12]))
            if sl.get("process_domains"):
                lines.append("PROCESS DOMAINS: " + ", ".join(sl["process_domains"][:8]))
            if sl.get("ontology_mappings"):
                lines.append("ONTOLOGY MAPPINGS: " + "; ".join(
                    f"{m.get('concept')}→{m.get('maps_to')}"
                    for m in sl["ontology_mappings"][:10]))
            if sl.get("relationship_insights"):
                lines.append("RELATIONSHIP INSIGHTS:\n" + "\n".join(
                    f"- {i}" for i in sl["relationship_insights"][:8]))

        prof = sem.get("sme_profile") or {}
        if any(prof.get(k) for k in ("role", "expertise", "business_area")):
            lines.append("SME PROFILE: " + "; ".join(
                f"{k}={v}" for k, v in prof.items() if v))
        
        global_rules = sem.get("business_rules") or []
        if global_rules:
            lines.append("GLOBAL BUSINESS RULES:\n" + "\n".join(f"- {r}" for r in global_rules))
    except Exception:
        pass

    # Nodes
    try:
        nodes = list(db.aql(
            "FOR n IN @@c FILTER n.session_id == @s LIMIT @l RETURN n",
            {"@c": NODE_COLL, "s": session_id, "l": limit},
        ))
        if nodes:
            lines.append("\nENTITIES (NODES):")
            for n in nodes:
                attrs = ", ".join(a.get("name", "") for a in (n.get("attributes") or [])[:8])
                src = (n.get("source") or {}).get("file") or "n/a"
                tags = ", ".join(n.get("tags") or [])
                rules = "; ".join(n.get("business_rules") or [])
                instances = ", ".join(n.get("instances") or [])
                metrics = ", ".join(
                    f"{m.get('name')}{(' [' + m.get('unit') + ']') if m.get('unit') else ''}"
                    for m in (n.get("metrics") or [])[:8] if isinstance(m, dict) and m.get("name"))
                line = (f"- [{n.get('entity_id')}] {n.get('name')} "
                        f"({n.get('entity_type')}): {n.get('meaning') or n.get('description')} ")
                if instances:
                    line += f"| instances: {instances} "
                line += f"| attrs: {attrs} | source: {src}"
                if metrics:
                    line += f" | metrics: {metrics}"
                if tags:
                    line += f" | tags: {tags}"
                if rules:
                    line += f" | rules: {rules}"
                lines.append(line)
    except Exception as e:
        logger.debug(f"[skg] context nodes failed: {e}")

    # Edges
    try:
        edges = list(db.aql(
            "FOR e IN @@c FILTER e.session_id == @s LIMIT @l RETURN e",
            {"@c": EDGE_COLL, "s": session_id, "l": limit},
        ))
        if edges:
            lines.append("\nRELATIONSHIPS (EDGES):")
            for e in edges:
                frm = (e.get("_from") or "").split("__")[-1]
                to  = (e.get("_to") or "").split("__")[-1]
                lines.append(
                    f"- {frm} --{e.get('name')} [{e.get('cardinality')}]--> {to}: "
                    f"{e.get('meaning') or e.get('description')}"
                )
    except Exception as e:
        logger.debug(f"[skg] context edges failed: {e}")

    return "\n".join(lines) if lines else "(no base graph found for this session)"


def _nodes_summary(session_id: str) -> str:
    db = get_db()
    try:
        nodes = list(db.aql(
            "FOR n IN @@c FILTER n.session_id == @s RETURN n",
            {"@c": NODE_COLL, "s": session_id},
        ))
    except Exception:
        nodes = []
    out = []
    for n in nodes:
        attrs = ", ".join(a.get("name", "") for a in (n.get("attributes") or []))
        out.append(f"- {n.get('entity_id')} | {n.get('name')} | attrs: {attrs}")
    return "\n".join(out) if out else "(no nodes)"


# ─────────────────────────────────────────────────────────────────────────────
# Suggested starter questions (what / where / why)
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_QUESTIONS = [
    "What are the main entities in my data and what does each represent?",
    "Where does each key field originate from in the source files?",
    "Why are these entities related the way they are?",
    "Which business rules or exceptions should the analysis know about?",
]


def generate_suggested_questions(session_id: str) -> List[str]:
    ctx = build_base_graph_context(session_id, limit=40)
    if ctx.startswith("(no base graph"):
        return _FALLBACK_QUESTIONS
    try:
        llm = get_mistral_client()
        raw = llm._chat(
            SUGGESTED_QUESTIONS_SYSTEM_PROMPT,
            build_suggested_questions_user_prompt(ctx),
        )
        parsed = llm._parse_json(raw) or {}
        qs = parsed.get("questions") or []
        qs = [q for q in qs if isinstance(q, str) and q.strip()]
        return qs[:6] if qs else _FALLBACK_QUESTIONS
    except Exception as e:
        logger.debug(f"[skg] suggested questions failed: {e}")
        return _FALLBACK_QUESTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — SME chat grounded in the base graph
# ─────────────────────────────────────────────────────────────────────────────
def _progressive_graph_update(session_id: str, db, entities: list, relationships: list, business_rules: list):
    ensure_collections()
    
    # Save business rules globally so they aren't lost if no entities are found
    if business_rules:
        try:
            sem_coll = db.collection(SEMANTIC_COLL)
            sem_key = f"sem_{session_id}"
            sem_doc = sem_coll.get(sem_key)
            if not sem_doc:
                sem_doc = {"_key": sem_key, "session_id": session_id}
            
            existing_rules = sem_doc.get("business_rules") or []
            sem_doc["business_rules"] = list(dict.fromkeys(existing_rules + business_rules))
            sem_coll.insert(sem_doc, overwrite=True)
        except Exception as e:
            logger.debug(f"[skg] failed to save global business rules: {e}")

    node_coll = db.collection(NODE_COLL)
    edge_coll = db.collection(EDGE_COLL)
    
    for ent in entities:
        eid = (ent.get("id") or "").strip()
        if not eid:
            continue
        key = _node_key(session_id, eid)
        try:
            doc = node_coll.get(key)
        except Exception:
            doc = None
            
        if not doc:
            doc = {
                "_key": key, "session_id": session_id, "entity_id": eid,
                "name": ent.get("name") or eid,
                "entity_type": ent.get("type") or "strong",
                "description": ent.get("description") or "",
                "meaning": "", "attributes": [], "metrics": [],
                "source": {"file": None, "evidence": ""},
                "confidence": 0.5, "tags": [], "business_rules": [],
                "origin": "sme_progressive",
            }
        else:
            # Enrich the existing node instead of duplicating it:
            # extend the description with new semantic context and
            # strengthen confidence since the SME confirmed the entity.
            new_desc = (ent.get("description") or "").strip()
            if new_desc and new_desc not in (doc.get("description") or ""):
                doc["description"] = ((doc.get("description") or "").strip()
                                      + (" | " if doc.get("description") else "")
                                      + new_desc)
            doc["confidence"] = min(0.95, float(doc.get("confidence") or 0.5) + 0.05)

        # Attach business rules revealed in this turn to the node as well.
        if business_rules:
            doc["business_rules"] = list(dict.fromkeys(
                (doc.get("business_rules") or []) + list(business_rules)))

        # Record structured attributes and metrics the SME revealed this turn
        # so the graph stays precise mid-conversation (not only at finalize).
        added_attrs = ent.get("added_attributes") or ent.get("attributes") or []
        if added_attrs:
            doc["attributes"] = _merge_attributes(doc.get("attributes"), added_attrs)
        added_metrics = ent.get("added_metrics") or ent.get("metrics") or []
        if added_metrics:
            doc["metrics"] = _merge_metrics(doc.get("metrics"), added_metrics)

        doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
        try:
            node_coll.insert(doc, overwrite=True)
        except Exception:
            pass
            
    for rel in relationships:
        frm = (rel.get("from") or "").strip()
        to = (rel.get("to") or "").strip()
        if not frm or not to:
            continue
        rid = (rel.get("id") or f"{frm}_{rel.get('name', 'rel')}_{to}").strip()
        ekey = _node_key(session_id, rid)
        try:
            existing_edge = edge_coll.get(ekey)
        except Exception:
            existing_edge = None

        if existing_edge:
            # Strengthen the existing relationship instead of overwriting it.
            new_desc = (rel.get("description") or "").strip()
            if new_desc and new_desc not in (existing_edge.get("description") or ""):
                existing_edge["description"] = ((existing_edge.get("description") or "").strip()
                                                + (" | " if existing_edge.get("description") else "")
                                                + new_desc)
            existing_edge["confidence"] = min(
                0.95, float(existing_edge.get("confidence") or 0.5) + 0.05)
            rel_metrics = rel.get("metrics") or []
            if rel_metrics:
                existing_edge["metrics"] = _merge_metrics(
                    existing_edge.get("metrics"), rel_metrics)
            existing_edge["updated_at"] = datetime.utcnow().isoformat() + "Z"
            try:
                edge_coll.insert(existing_edge, overwrite=True)
            except Exception:
                pass
            continue

        edoc = {
            "_key": ekey,
            "_from": f"{NODE_COLL}/{_node_key(session_id, frm)}",
            "_to": f"{NODE_COLL}/{_node_key(session_id, to)}",
            "session_id": session_id, "rel_id": rid,
            "name": rel.get("name") or "related_to",
            "cardinality": "1:N",
            "relationship_type": "association",
            "description": rel.get("description") or "",
            "meaning": "", "properties": {},
            "metrics": _merge_metrics([], rel.get("metrics") or []),
            "source": {"evidence": ""}, "confidence": 0.5,
            "origin": "sme_progressive",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            edge_coll.insert(edoc, overwrite=True)
        except Exception:
            pass


def sme_chat(
    session_id: str,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"answer": "Please ask a question about your data.",
                "status": "in_progress", "what_coverage": 0, "where_coverage": 0,
                "why_coverage": 0, "analysis_ready": False,
                "grounded": False, "followup_question": ""}

    ctx = build_base_graph_context(session_id)
    hist_text = ""
    if history:
        hist_text = "\n".join(
            f"{m.get('role','user').upper()}: {m.get('content','')}"
            for m in history[-10:]
        )

    db = get_db()
    try:
        sem = db.collection(SEMANTIC_COLL).get(f"sem_{session_id}") or {}
        coverage_state = sem.get("coverage_state", {
            "what_coverage": 0, "where_coverage": 0, "why_coverage": 0,
            "semantic_confidence": 0, "consecutive_no_new_info": 0,
            "current_phase": "what"
        })
        coverage_state.setdefault("semantic_confidence", 0)
        coverage_state.setdefault("current_phase", "what")
        # Pass any known SME profile so questions can be personalised.
        if sem.get("sme_profile"):
            coverage_state["sme_profile"] = sem["sme_profile"]
    except Exception:
        sem = {}
        coverage_state = {
            "what_coverage": 0, "where_coverage": 0, "why_coverage": 0,
            "semantic_confidence": 0, "consecutive_no_new_info": 0,
            "current_phase": "what"
        }

    # Resolve the active phase from current coverage BEFORE asking, so the
    # question this turn is locked to the correct phase (What → Where → Why/How).
    coverage_state["current_phase"] = _resolve_phase(coverage_state)

    try:
        llm = get_mistral_client()
        raw = llm._chat(
            SME_CHAT_SYSTEM_PROMPT,
            build_sme_chat_user_prompt(graph_context=ctx, history=hist_text, query=query, coverage_state=coverage_state),
        )
        parsed = llm._parse_json(raw) or {}
        if not parsed.get("answer"):
            parsed = {"answer": raw.strip() if isinstance(raw, str) else "", "grounded": False}

        new_entities = parsed.get("entities") or []
        new_rels = parsed.get("relationships") or []
        new_rules = parsed.get("business_rules") or []
        
        if not new_entities and not new_rels and not new_rules:
            coverage_state["consecutive_no_new_info"] += 1
        else:
            coverage_state["consecutive_no_new_info"] = 0
            
        # Only the active phase (and already-completed earlier phases) may have
        # their coverage updated. A later, not-yet-active phase is frozen so the
        # model cannot inflate it and skip ahead — this hard-enforces the
        # What → Where → Why/How order regardless of what the LLM returns.
        active_phase = coverage_state.get("current_phase", "what")
        active_idx = PHASE_ORDER.index(active_phase) if active_phase in PHASE_ORDER else 0
        for idx, pkey in enumerate(("what_coverage", "where_coverage", "why_coverage")):
            if idx > active_idx:
                continue  # later phase not reached yet — keep it frozen
            coverage_state[pkey] = parsed.get(pkey, coverage_state.get(pkey, 0))

        coverage_state["semantic_confidence"] = parsed.get(
            "semantic_confidence", coverage_state.get("semantic_confidence", 0))

        # Recompute the active phase for the NEXT turn from the new coverage.
        coverage_state["current_phase"] = _resolve_phase(coverage_state)
        coverage_state.pop("sme_profile", None)  # stored on the semantic doc, not the state
        
        analysis_ready = parsed.get("analysis_ready", False)
        if coverage_state["consecutive_no_new_info"] >= 3:
            analysis_ready = True
            
        status = "knowledge_collection_complete" if analysis_ready else "in_progress"
        
        try:
            if not sem:
                sem = {"_key": f"sem_{session_id}", "session_id": session_id}
            sem["coverage_state"] = coverage_state
            # Persist / merge the SME profile as soon as the chat reveals it
            # (role-aware questioning per the spec).
            prof = parsed.get("sme_profile") or {}
            if any(prof.get(k) for k in ("role", "expertise", "business_area")):
                merged = dict(sem.get("sme_profile") or {})
                merged.update({k: v for k, v in prof.items() if v})
                sem["sme_profile"] = merged
            db.collection(SEMANTIC_COLL).insert(sem, overwrite=True)
        except Exception as e:
            logger.warning(f"[skg] failed to save coverage state: {e}")
            
        if new_entities or new_rels or new_rules:
            _progressive_graph_update(session_id, db, new_entities, new_rels, new_rules)

        return {
            "answer": parsed.get("answer", ""),
            "what_coverage": coverage_state["what_coverage"],
            "where_coverage": coverage_state["where_coverage"],
            "why_coverage": coverage_state["why_coverage"],
            "semantic_confidence": coverage_state["semantic_confidence"],
            "current_phase": coverage_state.get("current_phase", "what"),
            "sme_profile": sem.get("sme_profile", {}) if isinstance(sem, dict) else {},
            "analysis_ready": analysis_ready,
            "status": status,
            "grounded": bool(parsed.get("grounded", False)),
            "followup_question": parsed.get("followup_question", "") or "",
            "referenced_entities": parsed.get("referenced_entities", []) or [],
        }
    except Exception as e:
        logger.error(f"[skg] sme_chat failed: {e}", exc_info=True)
        return {
            "answer": "I couldn't process that right now, but your note is captured.",
            "what_coverage": coverage_state.get("what_coverage", 0),
            "where_coverage": coverage_state.get("where_coverage", 0),
            "why_coverage": coverage_state.get("why_coverage", 0),
            "analysis_ready": False,
            "status": "in_progress",
            "grounded": False, "followup_question": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Enrich the graph from the SME conversation (no duplicate nodes)
# ─────────────────────────────────────────────────────────────────────────────
def enrich_from_sme(session_id: str, transcript: str) -> Dict[str, Any]:
    transcript = (transcript or "").strip()
    if not transcript:
        return {"status": "skipped", "enriched_nodes": 0, "new_relationships": 0}

    ensure_collections()
    db = get_db()
    existing = _nodes_summary(session_id)
    try:
        edge_rows = list(db.aql(
            "FOR e IN @@c FILTER e.session_id == @s "
            "RETURN {rel_id: e.rel_id, name: e.name, frm: e._from, to: e._to}",
            {"@c": EDGE_COLL, "s": session_id}))
        existing_edges = "\n".join(
            f"- {r.get('rel_id')} · {r.get('name')} · "
            f"{(r.get('frm') or '').split('__')[-1]} → {(r.get('to') or '').split('__')[-1]}"
            for r in edge_rows[:150])
    except Exception:
        existing_edges = ""

    try:
        llm = get_mistral_client()
        raw = llm._chat(
            SME_ENRICHMENT_SYSTEM_PROMPT,
            build_sme_enrichment_user_prompt(existing_nodes=existing,
                                             transcript=transcript,
                                             existing_edges=existing_edges),
        )
        parsed = llm._parse_json(raw) or {}
    except Exception as e:
        logger.error(f"[skg] enrichment LLM failed: {e}", exc_info=True)
        return {"status": "error", "enriched_nodes": 0, "new_relationships": 0,
                "message": str(e)}

    node_coll = db.collection(NODE_COLL)
    enriched = 0
    for enr in (parsed.get("node_enrichments") or []):
        eid = (enr.get("entity_id") or "").strip()
        if not eid:
            continue
        key = _node_key(session_id, eid)
        try:
            doc = node_coll.get(key)
        except Exception:
            doc = None

        if not doc:
            if not enr.get("is_new"):
                continue
            doc = {
                "_key": key, "session_id": session_id, "entity_id": eid,
                "name": enr.get("name") or eid, "entity_type": "strong",
                "description": "", "meaning": "", "attributes": [],
                "metrics": [], "source": {"file": None, "evidence": enr.get("evidence", "")},
                "confidence": 0.5, "tags": [], "business_rules": [],
                "origin": "sme",
            }

        # Merge attributes (de-duplicated by name, examples folded in)
        doc["attributes"] = _merge_attributes(
            doc.get("attributes"), enr.get("added_attributes") or [])

        # Merge tags / rules / metrics / semantic context
        doc["tags"] = sorted(set((doc.get("tags") or []) + (enr.get("added_tags") or [])))
        doc["business_rules"] = list(dict.fromkeys(
            (doc.get("business_rules") or []) + (enr.get("business_rules") or [])))
        doc["metrics"] = _merge_metrics(
            doc.get("metrics"), enr.get("added_metrics") or [])
        sem_ctx = (enr.get("semantic_context") or "").strip()
        if sem_ctx and sem_ctx not in (doc.get("meaning") or ""):
            doc["meaning"] = ((doc.get("meaning") or "").strip()
                              + (" | " if doc.get("meaning") else "") + sem_ctx)
        # SME confirmation strengthens our confidence in the node.
        doc["confidence"] = min(0.95, float(doc.get("confidence") or 0.5) + 0.1)
        doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
        doc.setdefault("origin", "sme")

        try:
            node_coll.insert(doc, overwrite=True)
            enriched += 1
        except Exception as e:
            logger.debug(f"[skg] enrich node failed {key}: {e}")

    # New relationships
    edge_coll = db.collection(EDGE_COLL)
    new_rels = 0
    for rel in (parsed.get("new_relationships") or []):
        frm = (rel.get("from") or "").strip()
        to  = (rel.get("to") or "").strip()
        if not frm or not to:
            continue
        rid = (rel.get("id") or f"{frm}_{rel.get('name','rel')}_{to}").strip()
        edoc = {
            "_key": _node_key(session_id, rid),
            "_from": f"{NODE_COLL}/{_node_key(session_id, frm)}",
            "_to": f"{NODE_COLL}/{_node_key(session_id, to)}",
            "session_id": session_id, "rel_id": rid,
            "name": rel.get("name") or "related_to",
            "cardinality": rel.get("cardinality") or "1:N",
            "relationship_type": rel.get("relationship_type") or "association",
            "description": "", "meaning": rel.get("meaning") or "",
            "properties": {}, "metrics": [], "source": {"evidence": rel.get("evidence", "")},
            "confidence": 0.5, "origin": "sme",
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        try:
            edge_coll.insert(edoc, overwrite=True)
            new_rels += 1
        except Exception as e:
            logger.debug(f"[skg] enrich edge failed {rid}: {e}")

    # Relationship strength / confidence updates (spec: "Update relationship
    # strength/confidence" when the SME confirms or clarifies an edge).
    updated_rels = 0
    for upd in (parsed.get("relationship_updates") or []):
        rid = (upd.get("relationship_id") or "").strip()
        if not rid:
            continue
        ekey = _node_key(session_id, rid)
        try:
            edoc = edge_coll.get(ekey)
        except Exception:
            edoc = None
        if not edoc:
            continue
        try:
            delta = float(upd.get("confidence_delta") or 0.1)
        except (TypeError, ValueError):
            delta = 0.1
        delta = max(0.0, min(0.25, delta))
        edoc["confidence"] = min(0.95, float(edoc.get("confidence") or 0.5) + delta)
        ctx = (upd.get("added_semantic_context") or "").strip()
        if ctx and ctx not in (edoc.get("meaning") or ""):
            edoc["meaning"] = ((edoc.get("meaning") or "").strip()
                               + (" | " if edoc.get("meaning") else "") + ctx)
        rules = upd.get("business_rules") or []
        if rules:
            props = edoc.get("properties") or {}
            props["business_rules"] = list(dict.fromkeys(
                (props.get("business_rules") or []) + rules))
            edoc["properties"] = props
        if upd.get("evidence"):
            src = edoc.get("source") or {}
            src["sme_evidence"] = upd["evidence"]
            edoc["source"] = src
        edoc["updated_at"] = datetime.utcnow().isoformat() + "Z"
        try:
            edge_coll.insert(edoc, overwrite=True)
            updated_rels += 1
        except Exception as e:
            logger.debug(f"[skg] relationship update failed {rid}: {e}")

    # Persist SME profile on the semantic doc (best effort)
    try:
        prof = parsed.get("sme_profile") or {}
        if any(prof.get(k) for k in ("role", "expertise", "business_area")):
            sem = db.collection(SEMANTIC_COLL).get(f"sem_{session_id}") or {
                "_key": f"sem_{session_id}", "session_id": session_id}
            merged = dict(sem.get("sme_profile") or {})
            merged.update({k: v for k, v in prof.items() if v})
            sem["sme_profile"] = merged
            db.collection(SEMANTIC_COLL).insert(sem, overwrite=True)
    except Exception:
        pass

    logger.info(f"[skg] SME enrichment {session_id}: {enriched} nodes, "
                f"{new_rels} new rels, {updated_rels} strengthened rels")
    return {
        "status": "ok",
        "enriched_nodes": enriched,
        "new_relationships": new_rels,
        "updated_relationships": updated_rels,
        "sme_profile": parsed.get("sme_profile") or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Read helpers (for the frontend / final analysis)
# ─────────────────────────────────────────────────────────────────────────────
def get_base_graph(session_id: str) -> Dict[str, Any]:
    db = get_db()
    try:
        nodes = list(db.aql(
            "FOR n IN @@c FILTER n.session_id == @s RETURN n",
            {"@c": NODE_COLL, "s": session_id}))
    except Exception:
        nodes = []
    try:
        edges = list(db.aql(
            "FOR e IN @@c FILTER e.session_id == @s RETURN e",
            {"@c": EDGE_COLL, "s": session_id}))
    except Exception:
        edges = []
        
    mapped_nodes = []
    for n in nodes:
        n_copy = dict(n)
        n_copy["id"] = n_copy.get("_key") or n_copy.get("entity_id")
        n_copy["label"] = n_copy.get("name") or n_copy.get("entity_id")
        mapped_nodes.append(n_copy)

    mapped_edges = []
    for e in edges:
        e_copy = dict(e)
        from_raw = e_copy.get("_from", "")
        to_raw = e_copy.get("_to", "")
        e_copy["id"] = e_copy.get("_key")
        e_copy["source"] = from_raw.split("/")[-1] if "/" in from_raw else from_raw
        e_copy["target"] = to_raw.split("/")[-1] if "/" in to_raw else to_raw
        e_copy["label"] = e_copy.get("name")
        mapped_edges.append(e_copy)

    try:
        sem = db.collection(SEMANTIC_COLL).get(f"sem_{session_id}") or {}
    except Exception:
        sem = {}
    return {
        "session_id": session_id,
        "semantic_layer": _ensure_semantic_dict(sem.get("semantic_layer")),
        "sme_profile": sem.get("sme_profile", {}),
        "nodes": mapped_nodes,
        "edges": mapped_edges,
        "node_count": len(mapped_nodes),
        "edge_count": len(mapped_edges),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Final-analysis gating (spec: analysis only after sufficient semantic
# confidence, understood entities, clarified relationships, SME context)
# ─────────────────────────────────────────────────────────────────────────────
READINESS_COVERAGE_THRESHOLD   = 90   # WHAT/WHERE/WHY-or-HOW coverage %
READINESS_CONFIDENCE_THRESHOLD = 70   # overall semantic confidence %


def get_readiness(session_id: str) -> Dict[str, Any]:
    """Return whether the session has enough contextual understanding for the
    final process analysis, plus the signals behind the decision."""
    db = get_db()
    try:
        sem = db.collection(SEMANTIC_COLL).get(f"sem_{session_id}") or {}
    except Exception:
        sem = {}

    state = sem.get("coverage_state") or {}
    what  = float(state.get("what_coverage") or 0)
    where = float(state.get("where_coverage") or 0)
    why   = float(state.get("why_coverage") or 0)
    conf  = float(state.get("semantic_confidence") or 0)
    stalled = int(state.get("consecutive_no_new_info") or 0) >= 3

    try:
        node_count = len(list(db.aql(
            "FOR n IN @@c FILTER n.session_id == @s RETURN 1",
            {"@c": NODE_COLL, "s": session_id})))
        edge_count = len(list(db.aql(
            "FOR e IN @@c FILTER e.session_id == @s RETURN 1",
            {"@c": EDGE_COLL, "s": session_id})))
    except Exception:
        node_count = edge_count = 0

    coverage_ok = (what >= READINESS_COVERAGE_THRESHOLD and
                   where >= READINESS_COVERAGE_THRESHOLD and
                   why >= READINESS_COVERAGE_THRESHOLD)
    confidence_ok = conf >= READINESS_CONFIDENCE_THRESHOLD
    analysis_ready = bool((coverage_ok and confidence_ok) or stalled)

    gaps = []
    if what < READINESS_COVERAGE_THRESHOLD:
        gaps.append("WHAT understanding incomplete (purpose, outputs, actors, workflows)")
    if where < READINESS_COVERAGE_THRESHOLD:
        gaps.append("WHERE understanding incomplete (data origins, storage, integrations)")
    if why < READINESS_COVERAGE_THRESHOLD:
        gaps.append("WHY/HOW understanding incomplete (reasoning, rules, mechanisms)")
    if not confidence_ok:
        gaps.append("Overall semantic confidence below threshold")

    return {
        "session_id": session_id,
        "analysis_ready": analysis_ready,
        "what_coverage": what,
        "where_coverage": where,
        "why_coverage": why,
        "semantic_confidence": conf,
        "stalled": stalled,
        "node_count": node_count,
        "edge_count": edge_count,
        "sme_profile": sem.get("sme_profile", {}),
        "business_rules": sem.get("business_rules", []),
        "remaining_gaps": gaps if not analysis_ready else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interview opening — the system speaks first and leads the conversation
# ─────────────────────────────────────────────────────────────────────────────
def start_sme_interview(session_id: str) -> Dict[str, Any]:
    """Generate the system's opening message for the SME interview: a warm,
    personalised greeting that shows what was understood from the upload and
    asks the first question (the SME's role, if unknown)."""
    ctx = build_base_graph_context(session_id)

    db = get_db()
    try:
        sem = db.collection(SEMANTIC_COLL).get(f"sem_{session_id}") or {}
    except Exception:
        sem = {}
    profile = sem.get("sme_profile") or {}
    sl = _ensure_semantic_dict(sem.get("semantic_layer"))

    # Friendly fallback built from the semantic layer (used if the LLM fails)
    domain = (sl.get("domain") or "").strip()
    fallback_q = "To start, I'd love to know where you fit into all this — what's your role here?"
    fallback_msg = (
        (f"Thanks so much for sharing this with me — it looks like this is all about "
         f"{domain.lower()}, and there's a lot here I'm genuinely curious about. "
         if domain and domain.lower() != "generic"
         else "Thanks so much for sharing this with me — I've had a really good look "
              "through it and there's a lot here I'm curious about. ")
        + "I'd love to understand how it actually works for you, in your own words — "
          "there are no wrong answers. " + fallback_q
    )

    try:
        llm = get_mistral_client()
        raw = llm._chat(
            SME_OPENING_SYSTEM_PROMPT,
            build_sme_opening_user_prompt(ctx, profile),
        )
        parsed = llm._parse_json(raw) or {}
        message = (parsed.get("message") or "").strip() or fallback_msg
        question = (parsed.get("question") or "").strip() or fallback_q
    except Exception as e:
        logger.error(f"[skg] start_sme_interview failed: {e}", exc_info=True)
        message, question = fallback_msg, fallback_q

    # Remember the open question so the chat never repeats itself
    try:
        if not sem:
            sem = {"_key": f"sem_{session_id}", "session_id": session_id}
        sem["interview_opened"] = True
        sem["last_question"] = question
        db.collection(SEMANTIC_COLL).insert(sem, overwrite=True)
    except Exception:
        pass

    return {"message": message, "question": question}
