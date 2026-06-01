"""
app/api/technical_design_routes.py
─────────────────────────────────────────────────────────────────────────────
UPDATED (drop-in replacement)

Changes vs previous version:
  1. **System & Module Inventory** is now always populated via
     `inventory_context_service.build_system_module_inventory(...)`.
  2. **CSV Source Detection** + **Document Data Lineage** (with ADF
     fallback) are pulled in via `source_detector_service` and added to the
     response payload so the PDF/PPTX/DOCX exporters can render them.
  3. **Workflow Graph** (the lane-based React Flow data with explicit
     Start and End nodes) is included on the payload so all three exporters
     render the SAME graph that's shown in the UI.
  4. **Per-suggestion blueprint section** — each suggestion contributes its
     own dynamic content block (workflow summary, architecture summary,
     automation logic, recommendations) following the
     "AgentForge_P2P_Agentic_Blueprint.docx" structure.

The previous mega-prompt + LLM merge logic is preserved; only additive
enrichment was added.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from flask import Blueprint, jsonify

from app.db.arango import get_db, COLLECTIONS
from app.core.mistral_client import get_mistral_client
from app.services.inventory_context_service import build_system_module_inventory
from app.services.source_detector_service import detect_document_data_lineage

logger = logging.getLogger(__name__)

technical_design_bp = Blueprint("technical_design", __name__)

_LLM_MAX_TOKENS = 8192


# ═════════════════════════════════════════════════════════════════════════════
# 1.  CONTEXT RESOLVER
# ═════════════════════════════════════════════════════════════════════════════
def _resolve_doc_context(suggestion_key: str) -> dict:
    now = datetime.utcnow()
    ctx = {
        "suggestion_key": suggestion_key,
        "suggestion": None, "step": None, "process": None,
        "process_steps": [], "erp_modules": [], "erp_module": None,
        "all_suggestions": [],
        "date": now.strftime("%B %Y"),
        "year": now.year,
        "organization": "AgentForge",
        "found": False,
    }
    try:
        db = get_db()
        col = db.collection

        suggestion = col(COLLECTIONS["suggestions"]).get(suggestion_key)
        if not suggestion:
            return ctx
        ctx["suggestion"] = suggestion
        ctx["found"] = True

        step_key = suggestion.get("step_key")
        if step_key:
            try:
                ctx["step"] = col(COLLECTIONS["steps"]).get(step_key)
            except Exception:
                pass

        process_key = suggestion.get("process_key")
        if process_key:
            try:
                ctx["process"] = col(COLLECTIONS["documents"]).get(process_key)
            except Exception:
                pass
            try:
                ctx["process_steps"] = list(db.aql(
                    "FOR s IN process_steps FILTER s.process_key == @key "
                    "SORT s.step_number RETURN s",
                    {"key": process_key},
                ))
            except Exception:
                pass
            try:
                ctx["erp_modules"] = list(db.aql(
                    "FOR m IN erp_modules FILTER m.process_key == @key RETURN m",
                    {"key": process_key},
                ))
                if ctx["erp_modules"]:
                    ctx["erp_module"] = ctx["erp_modules"][0]
            except Exception:
                pass
            try:
                ctx["all_suggestions"] = list(db.aql(
                    "FOR s IN automation_suggestions FILTER s.process_key == @key RETURN s",
                    {"key": process_key},
                ))
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[technical-design] context resolve failed: {e}")
    return ctx


# ═════════════════════════════════════════════════════════════════════════════
# 2.  HEADER FIELDS
# ═════════════════════════════════════════════════════════════════════════════
def _derive_header_fields(ctx: dict) -> dict:
    suggestion = ctx.get("suggestion") or {}
    process = ctx.get("process") or {}
    erp = ctx.get("erp_module") or {}

    suggestion_title = (
        suggestion.get("title") or suggestion.get("name") or "Agentic AI Solution"
    )
    process_title = process.get("title") or process.get("name")
    erp_name = erp.get("name") or process.get("erp")

    parts = ["Agentic AI"]
    if process_title:
        parts.append(process_title)
    parts.append(suggestion_title)
    parts.append("Technical Design")
    doc_title = " – ".join(parts)

    return {
        "doc_title": doc_title,
        "subtitle": suggestion_title,
        "process_title": process_title or "Business Process",
        "suggestion_title": suggestion_title,
        "erp_name": erp_name,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MEGA-PROMPT
# ═════════════════════════════════════════════════════════════════════════════
def _build_full_prompt(ctx: dict, header: dict) -> str:
    suggestion = ctx.get("suggestion") or {}
    step = ctx.get("step") or {}
    process_steps = ctx.get("process_steps") or []

    suggestion_desc = suggestion.get("description") or suggestion.get("summary") or ""
    step_title = step.get("title") or ""
    step_desc = step.get("description") or ""

    step_lines = []
    for s in process_steps[:25]:
        step_lines.append(f"  - Step {s.get('step_number','?')}: {s.get('title','')}")
    steps_block = "\n".join(step_lines) or "  (no steps available)"

    return f"""You are a principal AI platform architect generating a FULL Technical Design
Document for a specific Agentic AI solution. Return ONLY a valid JSON object —
no markdown, no fences, no commentary.

==================== CONTEXT ====================
Process:               {header['process_title']}
Suggestion / Use-Case: {header['suggestion_title']}
ERP / Platform:        {header['erp_name'] or 'N/A'}
Step focus:            {step_title or 'N/A'}
Step description:      {step_desc or 'N/A'}
Suggestion notes:      {suggestion_desc or 'N/A'}

Process Steps:
{steps_block}

==================== REQUIRED JSON SHAPE ====================
Every field below MUST be present and tailored to the CONTEXT.

{{
  "domain": "<short domain tag>",
  "exec_summary": {{
    "purpose": "<2 sentence purpose>",
    "problem_statement": "<problem>",
    "primary_goals": ["<g1>","<g2>","<g3>","<g4>","<g5>"],
    "design_philosophy": "<sentence>"
  }},
  "design_principle_apps": {{
    "Context is King": "<how applied>",
    "System Prompts are Architecture": "<how applied>",
    "Agent Loop as Control System": "<how applied>",
    "Plan-and-Execute Reasoning": "<how applied>",
    "Design for Multi-Agent from Day One": "<how applied>",
    "Guardrails are Load-Bearing Walls": "<how applied>",
    "Evals are the Test Suite": "<how applied>"
  }},
  "agent_categories": [
    {{"type": "Advisory Agent",       "description": "<dom-specific>"}},
    {{"type": "Conversational Agent", "description": "<dom-specific>"}}
  ],
  "presentation_components": [
    {{"component_name": "<n>", "type": "<t>", "features": ["<f1>","<f2>","<f3>"]}}
  ],
  "presentation_supported_formats": ["PDF","DOCX","XLSX","CSV","TXT"],
  "frontend_stack": {{
    "framework": "<f>", "language": "<l>",
    "state_management": ["<sm1>","<sm2>"], "styling": ["<s1>","<s2>"],
    "chat_ui": ["<c1>","<c2>"], "document_viewer": "<v>",
    "api_communication": ["<a1>","<a2>","<a3>"]
  }},
  "api_gateway_components": [
    {{"component_name":"API Gateway","technologies":["<t1>","<t2>"],"responsibilities":["<r1>","<r2>","<r3>"]}},
    {{"component_name":"Session Manager","responsibilities":["<r1>","<r2>"]}},
    {{"component_name":"Request Router","responsibilities":["<r1>","<r2>"]}}
  ],
  "backend_server": {{
    "runtime":["<r1>","<r2>"],"language":"Python",
    "features":["<f1>","<f2>","<f3>","<f4>"]
  }},
  "orchestration_pattern": {{"type":"<p>","workflow":"<w>"}},
  "agents": [
    {{"agent_id":1,"name":"Orchestrator Agent","role":"Central coordinator",
      "reasoning_framework":"Plan-and-Execute","model_tier":"<m>",
      "responsibilities":["<r1>","<r2>","<r3>","<r4>"]}}
  ],
  "analysis_dimensions":["<d1>","<d2>","<d3>","<d4>","<d5>","<d6>"],
  "rag_pipeline":[
    {{"stage":1,"name":"<n>","components":["<c1>","<c2>"]}}
  ],
  "knowledge_stores":[
    {{"store_type":"Vector Database","technologies":["<t1>","<t2>"]}},
    {{"store_type":"Knowledge Graph","technologies":["<t1>"]}},
    {{"store_type":"Document Store","technologies":["<t1>"]}},
    {{"store_type":"Metadata Store","technologies":["<t1>"]}}
  ],
  "chunking_strategies":[{{"type":"<t>","use_case":"<u>"}}],
  "frameworks":{{
    "orchestration":[{{"name":"<n>","role":"<r>","features":["<f1>","<f2>"]}}],
    "rag_frameworks":[{{"name":"<n>","role":"<r>"}}],
    "protocols":[{{"name":"<n>","full_form":"<ff>","purpose":"<p>"}}],
    "guardrails":[{{"name":"<n>"}}],
    "evaluation_tools":[{{"name":"<n>"}}]
  }},
  "tools":[{{"tool_name":"<n>","invoked_by":"<a>","purpose":"<p>"}}],
  "guardrails":[
    {{"rail_type":"Input Rails","functions":["<f1>","<f2>"]}},
    {{"rail_type":"Dialog Rails","functions":["<f1>","<f2>"]}},
    {{"rail_type":"Retrieval Rails","functions":["<f1>","<f2>"]}},
    {{"rail_type":"Execution Rails","functions":["<f1>","<f2>"]}},
    {{"rail_type":"Output Rails","functions":["<f1>","<f2>"]}}
  ],
  "observability":{{"structured_logging":true,"distributed_tracing":true,
    "metrics_dashboard":true,"anomaly_detection":true}},
  "governance":{{"prompt_registry":true,"eval_pipeline":true,
    "incident_response":true,"audit_trail":true}},
  "report_structure":["Cover Page","Executive Summary","<s3>","<s4>","<s5>","<s6>","Recommended Actions","Appendices"],
  "workflows":[{{"workflow_name":"<wf>"}},{{"workflow_name":"<wf2>"}}],
  "memory_architecture":[
    {{"memory_type":"Episodic Memory","contents":["<c1>","<c2>","<c3>"],"storage":["<s1>","<s2>"]}},
    {{"memory_type":"Semantic Memory","contents":["<c1>","<c2>","<c3>"],"storage":["<s1>","<s2>"]}},
    {{"memory_type":"Procedural Memory","contents":["<c1>","<c2>","<c3>"],"storage":["<s1>","<s2>"]}}
  ],
  "memory_critical_practices":["<p1>","<p2>","<p3>","<p4>"],
  "tech_stack":{{
    "frontend":["<t1>","<t2>","<t3>","<t4>"],"api_gateway":["<t1>","<t2>"],
    "backend":["<t1>","<t2>","<t3>"],"llm_models":["<m1>","<m2>","<m3>"],
    "vector_db":["<v1>"],"knowledge_graph":["<k1>"],
    "storage":["<s1>","<s2>"],"session_db":["<s1>"],
    "guardrails":["<g1>","<g2>"],"observability":["<o1>","<o2>"],
    "containerization":["<c1>","<c2>"]
  }},
  "eval_metrics":[{{"metric":"<n>","target":"<t>"}}],
  "chatbot":{{
    "termination_conditions":["<c1>","<c2>","<c3>","<c4>","<c5>"],
    "loop_prevention":["<p1>","<p2>","<p3>"],
    "recovery_strategies":["<r1>","<r2>","<r3>","<r4>"],
    "compression_strategies":["<s1>","<s2>","<s3>","<s4>","<s5>"],
    "budget_allocation":{{
      "system_prompt":"<%>","domain_knowledge":"<%>",
      "conversation_summary":"<%>","recent_turns":"<%>",
      "retrieved_chunks":"<%>","tool_outputs":"<%>","safety_buffer":"<%>"
    }},
    "overflow_prevention":["<o1>","<o2>","<o3>","<o4>","<o5>"]
  }},
  "automation_logic":{{
    "trigger":"<trigger>","preconditions":["<pc1>","<pc2>"],
    "core_loop":["<step1>","<step2>","<step3>","<step4>"],
    "decision_branches":["<b1>","<b2>","<b3>"],
    "failure_modes":["<fm1>","<fm2>","<fm3>"]
  }},
  "recommendations":[
    {{"title":"<rec>","rationale":"<why>","priority":"high|medium|low"}}
  ]
}}

==================== HARD RULES ====================
- Output MUST be ONE valid JSON object.
- 5 presentation_components, 5–7 agents, 6–10 tools, 5–7 rag_pipeline stages.
- 5–7 eval_metrics.  budget_allocation must sum to 100%.
- automation_logic.core_loop must contain at least 4 steps.
- TAILOR every string to the CONTEXT.
""".strip()


# ═════════════════════════════════════════════════════════════════════════════
# 4.  LLM CALL
# ═════════════════════════════════════════════════════════════════════════════
def _call_llm_full(prompt: str) -> dict:
    try:
        llm = get_mistral_client()
        response = llm.client.chat.complete(
            model=llm.model,
            messages=[
                {"role": "system",
                 "content": "You are an expert agentic-AI platform architect. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=_LLM_MAX_TOKENS,
        )
        raw = response.choices[0].message.content
        parsed = llm._parse_json(raw)
        if isinstance(parsed, dict) and parsed:
            return parsed
    except Exception as e:
        logger.warning(f"[technical-design] LLM call failed: {e}")
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# 5.  CANONICAL DEFAULTS (subset)
# ═════════════════════════════════════════════════════════════════════════════
_DEF_EXEC_SUMMARY = {
    "purpose": "Agentic AI application that automates process steps with high accuracy.",
    "problem_statement": "Manual process steps consume time and are error-prone.",
    "primary_goals": [
        "Automate routine decisions", "Reduce cycle time",
        "Improve accuracy", "Provide auditability", "Enable scalability",
    ],
    "design_philosophy": (
        "Agentic AI orchestrators decompose work, specialist agents execute, "
        "guardrails enforce policy, and humans handle exceptions."
    ),
}
_DEF_PRINCIPLE_APPS = {
    "Context is King": "Context engineering across uploaded documents and process knowledge",
    "System Prompts are Architecture": "Production system prompts with safety guardrails",
    "Agent Loop as Control System": "Observe → Reason → Plan → Act → Evaluate → Update Memory",
    "Plan-and-Execute Reasoning": "Structured multi-step workflow orchestration",
    "Design for Multi-Agent from Day One": "Specialist sub-agents per capability",
    "Guardrails are Load-Bearing Walls": "Enterprise-grade safety guardrails",
    "Evals are the Test Suite": "Eval-driven development with RAGAS",
}
_DEF_AGENT_CATEGORIES = [
    {"type": "Advisory Agent",       "description": "Proactive risk and opportunity identification"},
    {"type": "Conversational Agent", "description": "Chatbot for in-platform support"},
]
_DEF_PRESENTATION_COMPONENTS = [
    {"component_name": "Application Selector Panel", "type": "Dropdown Interface",
     "features": ["Application type dropdown", "Dynamic instance loading", "Metadata-driven population"]},
    {"component_name": "Document Upload Module", "type": "File Ingestion",
     "features": ["Drag-and-drop", "Connector selection", "Progress indicators", "Validation"]},
    {"component_name": "Report Generation Dashboard", "type": "Workflow Dashboard",
     "features": ["One-click generation", "Real-time progress", "Downloadable output"]},
    {"component_name": "Embedded Chatbot Panel", "type": "Conversational UI",
     "features": ["Persistent window", "History", "Context awareness", "Follow-up support"]},
    {"component_name": "Risk Summary View", "type": "Visualization Dashboard",
     "features": ["Heat map", "Key findings cards", "Severity indicators"]},
]
_DEF_FRONTEND_STACK = {
    "framework": "React JS", "language": "TypeScript",
    "state_management": ["Redux Toolkit", "Zustand"],
    "styling": ["Tailwind CSS", "Ant Design"],
    "chat_ui": ["Chatscope", "Custom Widget"],
    "document_viewer": "React-PDF",
    "api_communication": ["Axios", "React Query", "WebSocket"],
}
_DEF_API_GATEWAY_COMPONENTS = [
    {"component_name": "API Gateway", "technologies": ["Kong", "AWS API Gateway"],
     "responsibilities": ["Route management", "Authentication", "Authorization", "Rate limiting", "Logging", "CORS"]},
    {"component_name": "Session Manager",
     "responsibilities": ["User session tracking", "Context mapping", "Session cleanup"]},
    {"component_name": "Request Router",
     "responsibilities": ["Intent classification", "Pipeline routing"]},
]
_DEF_BACKEND_SERVER = {
    "runtime": ["FastAPI", "Flask"], "language": "Python",
    "features": ["Async processing", "WebSocket streaming", "Celery queues", "Redis integration"],
}
_DEF_ORCHESTRATION_PATTERN = {"type": "Hierarchical Orchestrator", "workflow": "Sequential Sub-Pipelines"}
_DEF_AGENTS = [
    {"agent_id": 1, "name": "Orchestrator Agent", "role": "Central coordinator",
     "reasoning_framework": "Plan-and-Execute", "model_tier": "GPT-4o / Claude Sonnet",
     "responsibilities": ["Task decomposition", "Sub-agent delegation", "Error handling", "Termination management"]},
    {"agent_id": 2, "name": "Document Intake & Process Agent", "role": "Document ingestion and parsing",
     "model_tier": "GPT-4o-mini"},
    {"agent_id": 3, "name": "Knowledge Base Builder Agent", "role": "Build vector store and knowledge graph",
     "responsibilities": ["Generate embeddings", "Index vector DB", "Build knowledge graph"]},
    {"agent_id": 4, "name": "Risk Analysis Agent", "role": "Core analytical intelligence",
     "reasoning_framework": "Reflexion / Self-Refine"},
    {"agent_id": 5, "name": "Report Generator Agent", "role": "Generate formatted Word document",
     "tool": "python-docx"},
    {"agent_id": 6, "name": "Chat Bot Agent", "role": "Conversational assistant",
     "reasoning_framework": "ReAct"},
]
_DEF_ANALYSIS_DIMENSIONS = [
    "Coverage gaps", "Historical pain points", "SLA risk flags",
    "Knowledge dependency risks", "Severity classification", "Mitigation recommendations",
]
_DEF_RAG_PIPELINE = [
    {"stage": 1, "name": "Query Analysis",     "components": ["Intent Detector", "Entity Extractor"]},
    {"stage": 2, "name": "Query Rewriting",    "components": ["Query Reformulator"]},
    {"stage": 3, "name": "Hybrid Retrieval",   "components": ["Dense Retrieval", "Sparse Retrieval", "RRF"]},
    {"stage": 4, "name": "Metadata Filtering", "components": ["Filter Engine"]},
    {"stage": 5, "name": "Re-Ranking",         "components": ["Cross-Encoder Re-Ranker"]},
    {"stage": 6, "name": "GraphRAG",           "components": ["Knowledge Graph Traversal"]},
    {"stage": 7, "name": "Retrieval Rails",    "components": ["Safety Filter", "Freshness Validation"]},
]
_DEF_KNOWLEDGE_STORES = [
    {"store_type": "Vector Database", "technologies": ["Pinecone", "Weaviate", "ChromaDB"]},
    {"store_type": "Knowledge Graph", "technologies": ["Neo4j", "Amazon Neptune"]},
    {"store_type": "Document Store",  "technologies": ["AWS S3", "Azure Blob"]},
    {"store_type": "Metadata Store",  "technologies": ["PostgreSQL"]},
]
_DEF_CHUNKING = [
    {"type": "Semantic Chunking",     "use_case": "Process knowledge documents"},
    {"type": "Parent-Child Chunking", "use_case": "Ticket logs"},
    {"type": "Fixed-Size Chunking",   "use_case": "CSV/XLSX exports"},
]
_DEF_FRAMEWORKS = {
    "orchestration": [
        {"name": "LangGraph", "role": "Primary orchestration engine",
         "features": ["Stateful workflows", "Cyclic graphs", "Human-in-the-loop", "Streaming"]},
        {"name": "LangChain", "role": "LLM abstraction and tooling",
         "features": ["Prompt templates", "Tool calling", "Document loaders", "Output parsing"]},
    ],
    "rag_frameworks": [
        {"name": "LlamaIndex",           "role": "Advanced RAG pipeline"},
        {"name": "LangChain Retrievers", "role": "Lightweight retrieval"},
    ],
    "protocols": [
        {"name": "MCP", "full_form": "Model Context Protocol",  "purpose": "Tool connectivity"},
        {"name": "A2A", "full_form": "Agent-to-Agent Protocol", "purpose": "Cross-agent communication"},
    ],
    "guardrails":       [{"name": "NVIDIA NeMo Guardrails"}, {"name": "Guardrails AI"}, {"name": "Llama Guard"}],
    "evaluation_tools": [{"name": "RAGAS"}, {"name": "LangSmith"}, {"name": "OpenTelemetry"}],
}
_DEF_TOOLS = [
    {"tool_name": "Document Parser Tool",              "invoked_by": "Document Intake Agent",  "purpose": "Parse structured and unstructured documents"},
    {"tool_name": "Embedding Generator Tool",          "invoked_by": "Knowledge Base Builder", "purpose": "Generate embeddings"},
    {"tool_name": "Vector DB CRUD Tool",               "invoked_by": "RAG Pipeline",           "purpose": "Manage vector operations"},
    {"tool_name": "Knowledge Graph Tool",              "invoked_by": "Risk Analysis Agent",    "purpose": "Entity relationship operations"},
    {"tool_name": "Word Document Generator Tool",      "invoked_by": "Report Generator Agent", "purpose": "Generate DOCX"},
    {"tool_name": "Web Search Tool",                   "invoked_by": "Risk Analysis Agent",    "purpose": "Fetch external advisories"},
]
_DEF_GUARDRAILS = [
    {"rail_type": "Input Rails",     "functions": ["PII detection", "Injection detection", "Topic filtering"]},
    {"rail_type": "Dialog Rails",    "functions": ["Conversation control", "Behavior enforcement"]},
    {"rail_type": "Retrieval Rails", "functions": ["Relevance filtering", "Freshness validation"]},
    {"rail_type": "Execution Rails", "functions": ["Tool parameter validation", "Permission checks"]},
    {"rail_type": "Output Rails",    "functions": ["Hallucination detection", "PII scrubbing", "Compliance checks"]},
]
_DEF_OBSERVABILITY = {"structured_logging": True, "distributed_tracing": True,
                       "metrics_dashboard": True, "anomaly_detection": True}
_DEF_GOVERNANCE    = {"prompt_registry": True, "eval_pipeline": True,
                       "incident_response": True, "audit_trail": True}
_DEF_REPORT_STRUCTURE = [
    "Cover Page", "Executive Summary", "Risk Register Table",
    "Application-Wise Findings", "KT Coverage Gap Analysis", "SLA Risk Flags",
    "Recommended Actions", "Appendices",
]
_DEF_WORKFLOWS = [
    {"workflow_name": "Report Generation Flow"},
    {"workflow_name": "ChatBot Query Flow"},
]
_DEF_MEMORY_ARCH = [
    {"memory_type": "Episodic Memory",   "contents": ["Conversation history", "Past sessions", "Interaction logs"], "storage": ["Redis", "PostgreSQL"]},
    {"memory_type": "Semantic Memory",   "contents": ["Knowledge base", "SLA libraries", "Risk patterns"],          "storage": ["Vector DB", "Knowledge Graph"]},
    {"memory_type": "Procedural Memory", "contents": ["Learned workflows", "Response templates", "Retrieval strategies"], "storage": ["Prompt configurations", "Fine-tuning store"]},
]
_DEF_MEMORY_PRACTICES = ["TTL on memories", "Relevance scoring", "Memory poisoning defense", "Tenant isolation"]
_DEF_TECH_STACK = {
    "frontend":         ["React JS", "TypeScript", "Redux", "Tailwind CSS"],
    "api_gateway":      ["Kong", "AWS API Gateway"],
    "backend":          ["FastAPI", "Celery", "Redis"],
    "llm_models":       ["GPT-4o", "Claude Sonnet", "GPT-4o-mini"],
    "vector_db":        ["Pinecone", "Weaviate", "ChromaDB"],
    "knowledge_graph":  ["Neo4j", "Amazon Neptune"],
    "storage":          ["AWS S3", "Azure Blob"],
    "session_db":       ["PostgreSQL"],
    "guardrails":       ["NeMo Guardrails", "Guardrails AI"],
    "observability":    ["OpenTelemetry", "Datadog", "Jaeger"],
    "containerization": ["Docker", "Kubernetes"],
}
_DEF_EVAL_METRICS = [
    {"metric": "Risk Identification Accuracy", "target": ">= 85%"},
    {"metric": "Faithfulness (RAGAS)",         "target": ">= 0.90"},
    {"metric": "Context Precision",            "target": ">= 0.85"},
    {"metric": "Report Generation Time",       "target": "< 3 minutes"},
    {"metric": "Chatbot Answer Relevancy",     "target": ">= 0.85"},
    {"metric": "User Satisfaction",            "target": ">= 4.0 / 5.0"},
    {"metric": "Hallucination Rate",           "target": "< 5%"},
]
_DEF_CHATBOT = {
    "termination_conditions": ["Maximum iteration limit", "Confidence threshold", "Task completion signal",
                               "User-initiated stop", "Timeout threshold", "Token budget exhaustion"],
    "loop_prevention":    ["Repetition detector", "Progress tracker", "Observation anomaly monitor"],
    "recovery_strategies":["Retry with reformulation", "Fallback retrieval", "Human handoff", "Manus principle"],
    "compression_strategies": ["Sliding window", "Summarization chains", "Note-taking pattern",
                               "Self-baking context", "Stable prefix/dynamic suffix"],
    "budget_allocation": {"system_prompt": "15%", "domain_knowledge": "10%",
                          "conversation_summary": "10%", "recent_turns": "20%",
                          "retrieved_chunks": "30%", "tool_outputs": "10%", "safety_buffer": "5%"},
    "overflow_prevention": ["Pre-assembly token counting", "Aggressive summarization",
                            "Retrieval chunk limiting", "Tool output truncation", "Circuit breaker"],
}
_DEF_AUTOMATION_LOGIC = {
    "trigger":           "Inbound event from upstream process step.",
    "preconditions":     ["Required input fields present", "Authorisation validated"],
    "core_loop":         ["Validate input", "Run agent reasoning", "Execute tool calls", "Update process state"],
    "decision_branches": ["Auto-approve", "Human review", "Reject and escalate"],
    "failure_modes":     ["Tool timeout", "LLM unavailability", "Policy violation"],
}
_DEF_RECOMMENDATIONS = [
    {"title": "Pilot in a sandboxed environment first", "rationale": "De-risks the rollout.", "priority": "high"},
    {"title": "Tag every agent decision in audit log",  "rationale": "Required for SOX/GDPR.", "priority": "high"},
    {"title": "Run weekly evals on hallucination rate", "rationale": "Catches model drift early.", "priority": "medium"},
]


def _pick(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict, str)) and not value:
        return default
    return value


def _merge_dict(value, default):
    if not isinstance(value, dict) or not value:
        return default
    out = dict(default)
    for k, v in value.items():
        if v is None or (isinstance(v, (list, dict, str)) and not v):
            continue
        out[k] = v
    return out


def _slug(s: str) -> str:
    s = (s or "lane").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "lane"


# ═════════════════════════════════════════════════════════════════════════════
# 5b.  LANE-BASED WORKFLOW GRAPH (with Start + End)
# ═════════════════════════════════════════════════════════════════════════════
def _build_lane_workflow_graph(ctx: dict) -> dict:
    """
    Construct a lane-based workflow graph in the SAME shape that
    `getProcessFlow` returns to the React UI, GUARANTEED to include
    Start and End nodes.  Consumed identically by all three exporters.
    """
    steps = ctx.get("process_steps") or []
    process = ctx.get("process") or {}

    if not steps:
        return {
            "title": process.get("title") or "Workflow",
            "lanes": [{
                "id": "lane-default", "label": "Workflow",
                "nodes": [
                    {"id": "start", "type": "start", "label": "Start", "column": 1},
                    {"id": "end",   "type": "end",   "label": "End",   "column": 2},
                ],
            }],
            "flow": [{"from": "start", "to": "end"}],
        }

    sorted_steps = sorted(steps, key=lambda x: x.get("step_number", 0))
    first_step_id = f"step-{sorted_steps[0].get('step_number')}"
    last_step_id  = f"step-{sorted_steps[-1].get('step_number')}"

    # Group by actor
    actor_groups: dict = {}
    actor_order: list = []
    for s in sorted_steps:
        actor = s.get("actor") or "Process"
        if actor not in actor_groups:
            actor_groups[actor] = []
            actor_order.append(actor)
        actor_groups[actor].append(s)

    # Column placement: column 1 = Start, then unique col per step, last = End
    step_id_to_col: dict = {}
    next_col = 2
    for s in sorted_steps:
        step_id_to_col[f"step-{s.get('step_number')}"] = next_col
        next_col += 1
    end_col = next_col

    lanes: list = [
        {
            "id": "lane-start", "label": "Start",
            "nodes": [{"id": "start", "type": "start", "label": "Start", "column": 1}],
        }
    ]
    for actor in actor_order:
        lane_nodes = []
        for s in actor_groups[actor]:
            sid = f"step-{s.get('step_number')}"
            lane_nodes.append({
                "id":     sid,
                "type":   "process",
                "label":  s.get("title") or f"Step {s.get('step_number')}",
                "column": step_id_to_col[sid],
                "actor":  actor,
                "automation_potential": int(s.get("automation_potential") or 0),
            })
        lanes.append({
            "id":    f"lane-{_slug(actor)}",
            "label": actor,
            "nodes": lane_nodes,
        })
    lanes.append({
        "id": "lane-end", "label": "End",
        "nodes": [{"id": "end", "type": "end", "label": "End", "column": end_col}],
    })

    flow = [{"from": "start", "to": first_step_id, "label": "begin"}]
    for i in range(len(sorted_steps) - 1):
        flow.append({
            "from":  f"step-{sorted_steps[i].get('step_number')}",
            "to":    f"step-{sorted_steps[i+1].get('step_number')}",
            "label": "next",
        })
    flow.append({"from": last_step_id, "to": "end", "label": "complete"})

    return {
        "title": process.get("title") or "Workflow",
        "lanes": lanes,
        "flow":  flow,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5c.  PER-SUGGESTION BLUEPRINT BUILDER (blueprint-format section)
# ═════════════════════════════════════════════════════════════════════════════
def _build_suggestion_section(
    suggestion: dict, process: dict, steps: list,
    automation_logic: dict, recommendations: list,
    workflow_graph: dict, agents: list,
) -> dict:
    title = suggestion.get("title") or "Agentic Suggestion"
    desc  = suggestion.get("description") or suggestion.get("summary") or ""
    return {
        "suggestion_id":     suggestion.get("_key") or suggestion.get("id"),
        "title":             title,
        "description":       desc,
        "agent_type":        suggestion.get("agent_type", "workflow_automation"),
        "roi_impact":        suggestion.get("roi_impact"),
        "effort_level":      suggestion.get("effort_level"),
        "accuracy_estimate": suggestion.get("accuracy_estimate"),
        "implementation":    suggestion.get("implementation") or "",
        "workflow_summary": {
            "process":     process.get("title") or "",
            "step_count":  len(steps or []),
            "anchor_step": suggestion.get("step_number") or suggestion.get("step_key", ""),
        },
        "workflow_graph":  workflow_graph,
        "architecture_summary": {
            "pattern":         "Plan-and-Execute Hierarchical Orchestrator",
            "principal_agent": (agents[0].get("name") if agents else "Orchestrator Agent"),
            "agents":          [{"name": a.get("name"), "role": a.get("role")} for a in (agents or [])[:6]],
            "rationale": (
                f"The orchestrator decomposes the '{title}' workflow into "
                "sub-tasks and routes each through specialist agents while "
                "applying policy guardrails on outputs."
            ),
        },
        "automation_logic": automation_logic,
        "recommendations":  recommendations,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PAYLOAD BUILDER
# ═════════════════════════════════════════════════════════════════════════════
def _build_technical_design(ctx: dict, header: dict, dyn: dict) -> dict:
    org   = ctx["organization"]
    date  = ctx["date"]
    year  = ctx["year"]
    title = header["doc_title"]

    domain                = _pick(dyn.get("domain"), f"Agentic AI / {header['process_title']}")
    exec_summary          = _merge_dict(dyn.get("exec_summary"), _DEF_EXEC_SUMMARY)
    principle_apps        = _merge_dict(dyn.get("design_principle_apps"), _DEF_PRINCIPLE_APPS)
    agent_categories      = _pick(dyn.get("agent_categories"), _DEF_AGENT_CATEGORIES)
    pres_components       = _pick(dyn.get("presentation_components"), _DEF_PRESENTATION_COMPONENTS)
    pres_formats          = _pick(dyn.get("presentation_supported_formats"), ["PDF","DOCX","XLSX","CSV","TXT"])
    frontend_stack        = _merge_dict(dyn.get("frontend_stack"), _DEF_FRONTEND_STACK)
    api_gw_components     = _pick(dyn.get("api_gateway_components"), _DEF_API_GATEWAY_COMPONENTS)
    backend_server        = _merge_dict(dyn.get("backend_server"), _DEF_BACKEND_SERVER)
    orchestration_pattern = _merge_dict(dyn.get("orchestration_pattern"), _DEF_ORCHESTRATION_PATTERN)
    agents                = _pick(dyn.get("agents"), _DEF_AGENTS)
    analysis_dimensions   = _pick(dyn.get("analysis_dimensions"), _DEF_ANALYSIS_DIMENSIONS)
    rag_pipeline          = _pick(dyn.get("rag_pipeline"), _DEF_RAG_PIPELINE)
    knowledge_stores      = _pick(dyn.get("knowledge_stores"), _DEF_KNOWLEDGE_STORES)
    chunking_strategies   = _pick(dyn.get("chunking_strategies"), _DEF_CHUNKING)
    frameworks            = _merge_dict(dyn.get("frameworks"), _DEF_FRAMEWORKS)
    tools                 = _pick(dyn.get("tools"), _DEF_TOOLS)
    guardrails            = _pick(dyn.get("guardrails"), _DEF_GUARDRAILS)
    observability         = _merge_dict(dyn.get("observability"), _DEF_OBSERVABILITY)
    governance            = _merge_dict(dyn.get("governance"), _DEF_GOVERNANCE)
    report_structure      = _pick(dyn.get("report_structure"), _DEF_REPORT_STRUCTURE)
    workflows             = _pick(dyn.get("workflows"), _DEF_WORKFLOWS)
    memory_architecture   = _pick(dyn.get("memory_architecture"), _DEF_MEMORY_ARCH)
    memory_practices      = _pick(dyn.get("memory_critical_practices"), _DEF_MEMORY_PRACTICES)
    tech_stack            = _merge_dict(dyn.get("tech_stack"), _DEF_TECH_STACK)
    eval_metrics          = _pick(dyn.get("eval_metrics"), _DEF_EVAL_METRICS)
    chatbot               = _merge_dict(dyn.get("chatbot"), _DEF_CHATBOT)
    automation_logic      = _merge_dict(dyn.get("automation_logic"), _DEF_AUTOMATION_LOGIC)
    recommendations       = _pick(dyn.get("recommendations"), _DEF_RECOMMENDATIONS)

    for c in pres_components:
        if "Upload" in c.get("component_name", "") and "supported_formats" not in c:
            c["supported_formats"] = pres_formats

    principles = [
        {"id": i + 1, "name": name,
         "application": principle_apps.get(name, _DEF_PRINCIPLE_APPS[name])}
        for i, name in enumerate(_DEF_PRINCIPLE_APPS.keys())
    ]

    # ── NEW ENRICHMENTS ─────────────────────────────────────────────────────
    inventory = build_system_module_inventory(
        ctx.get("process") or {},
        ctx.get("process_steps") or [],
        ctx.get("erp_modules") or [],
    )

    process_blob = " ".join(
        f"{s.get('title','')} {s.get('description','')}"
        for s in (ctx.get("process_steps") or [])
    ) + " " + ((ctx.get("process") or {}).get("description") or "")
    data_lineage = detect_document_data_lineage(process_blob)

    workflow_graph = _build_lane_workflow_graph(ctx)

    suggestion_sections = []
    for sug in (ctx.get("all_suggestions") or [ctx.get("suggestion") or {}]):
        if not sug:
            continue
        suggestion_sections.append(_build_suggestion_section(
            suggestion=sug, process=ctx.get("process") or {},
            steps=ctx.get("process_steps") or [],
            automation_logic=automation_logic, recommendations=recommendations,
            workflow_graph=workflow_graph, agents=agents,
        ))

    return {
        "document_metadata": {
            "title": title, "version": "Draft V1.0", "date": date,
            "organization": org, "document_type": "Technical Design Document",
            "pages": 34, "classification": "Draft", "domain": domain,
        },
        "cover_page": {
            "title": title, "subtitle": header["subtitle"], "version": "Draft V1.0",
            "date": date, "organization": org,
        },
        "table_of_contents": [
            # Item 0 — Agentic Process Workflow Graph (rendered immediately
            # after the TOC; not numbered as a section per PwC reference)
            {"section_number": "0",  "title": "Agentic Process Workflow Graph"},
            # 1..6 — PwC reference numbering
            {"section_number": "1",  "title": "Executive Summary"},
            {"section_number": "2",  "title": "Solution Overview & Design Principles"},
            {"section_number": "3",  "title": "Core Architecture Design"},
            {"section_number": "4",  "title": "Agentic Frameworks and SDK Selection"},
            {"section_number": "5",  "title": "Tool Ecosystem and Integrations"},
            {"section_number": "6",  "title": "Guardrails, Observability and Governance and System Prompt"},
            # 7 (Word Report Generation) is INTENTIONALLY OMITTED per spec
            {"section_number": "8",  "title": "End-to-End Workflows"},
            {"section_number": "9",  "title": "Memory"},
            {"section_number": "10", "title": "Tech Stack Summary"},
            {"section_number": "11", "title": "Success Criteria and Eval Metrics"},
            # 12..14 — AgentForge-specific extensions (preserve existing flow)
            {"section_number": "12", "title": "System and Module Inventory"},
            {"section_number": "13", "title": "CSV Source & Document Data Lineage"},
            {"section_number": "14", "title": "Agentic Suggestion Blueprints"},
        ],
        "sections": [
            {"section_number": "1", "title": "Executive Summary",
             "content": {
                 "purpose":           exec_summary.get("purpose"),
                 "problem_statement": exec_summary.get("problem_statement"),
                 "primary_goals":     exec_summary.get("primary_goals"),
                 "design_philosophy": {"statement": exec_summary.get("design_philosophy")},
             }},
            {"section_number": "2", "title": "Solution Overview & Design Principles",
             "subsections": [
                 {"section_number": "2.1", "title": "Design Principles Aligned to Best Practices", "principles": principles},
                 {"section_number": "2.2", "title": "Agent Category Classification", "categories": agent_categories},
             ]},
            {"section_number": "3", "title": "Core Architecture Design",
             # 3.1 — High-Level Architecture Layers (overview block; lists all six layers)
             "subsection_overview": {
                 "section_number": "3.1",
                 "title": "High-Level Architecture Layers",
                 "intro": "The architecture is organized into six layers as shown below.",
                 "layers": [
                     {"id": 1, "name": "PRESENTATION LAYER (React JS Front-End)"},
                     {"id": 2, "name": "API GATEWAY & ORCHESTRATION LAYER"},
                     {"id": 3, "name": "AGENTIC CORE (Multi-Agent Orchestrator)"},
                     {"id": 4, "name": "RAG & KNOWLEDGE SYSTEMS"},
                     {"id": 5, "name": "TOOL ECOSYSTEM & INTEGRATIONS"},
                     {"id": 6, "name": "GUARDRAILS, OBSERVABILITY & GOVERNANCE"},
                 ],
             },
             # Detail subsections 3.2, 3.3, 3.6 ONLY.
             # 3.4 (Agentic Core) and 3.5 (Chatbot considerations) are
             # INTENTIONALLY OMITTED per spec — do not add a layer for them.
             "architecture_layers": [
                 {"layer_id": 1, "section_label": "3.2 Presentation Layer (React JS Front-End)",
                  "name": "Presentation Layer",
                  "components": pres_components, "frontend_stack": frontend_stack},
                 {"layer_id": 2, "section_label": "3.3 API Gateway & Orchestration Layer",
                  "name": "API Gateway & Orchestration Layer",
                  "components": api_gw_components, "backend_server": backend_server},
                 # Layer 3 (Agentic Core / 3.4) is INTENTIONALLY OMITTED.
                 {"layer_id": 4, "section_label": "3.6 RAG and Knowledge Systems",
                  "name": "RAG and Knowledge Systems",
                  "rag_pipeline": rag_pipeline, "knowledge_stores": knowledge_stores,
                  "chunking_strategies": chunking_strategies},
             ]},
            {"section_number": "4", "title": "Agentic Frameworks and SDK Selection", "frameworks": frameworks},
            {"section_number": "5", "title": "Tool Ecosystem and Integrations",
             "tools": tools,
             "tool_interface_standards": {"protocol": "MCP-inspired", "schema_validation": True, "execution_rails": True}},
            {"section_number": "6", "title": "Guardrails, Observability and Governance and System Prompt",
             "guardrails": guardrails, "observability": observability, "governance": governance},
            {"section_number": "8", "title": "End-to-End Workflows", "workflows": workflows},
            {"section_number": "9", "title": "Memory",
             "memory_architecture": memory_architecture, "critical_practices": memory_practices},
            {"section_number": "10", "title": "Tech Stack Summary", "stack": tech_stack},
            {"section_number": "11", "title": "Success Criteria and Eval Metrics", "metrics": eval_metrics},

            # ── NEW section 11 — System & Module Inventory (always populated)
            {
                "section_number": "12",
                "title": "System and Module Inventory",
                "system_module_inventory": inventory,
                "note": (
                    "This section is dynamically generated from the uploaded "
                    "documents and the analysis pipeline. Each row reflects "
                    "either an ERP module the pipeline detected or a module "
                    "inferred from the process content."
                ),
            },

            # ── NEW section 12 — CSV source + document data lineage
            {
                "section_number": "13",
                "title": "CSV Source & Document Data Lineage",
                "csv_source_detection": [],   # populated by /analyze pipeline
                "document_data_lineage": data_lineage,
                "note": (
                    "Source/Target are extracted from the uploaded document. "
                    "If neither could be confidently identified, ADF (Azure "
                    "Data Factory) is assumed as the source so the workflow "
                    "automation can proceed uninterrupted."
                ),
            },

            # ── NEW section 13 — per-suggestion blueprint blocks
            {
                "section_number": "14",
                "title": "Agentic Suggestion Blueprints",
                "suggestion_blueprints": suggestion_sections,
                "note": (
                    "Each suggestion contributes its own workflow summary, "
                    "agentic workflow graph (matching the UI), architecture "
                    "summary, automation logic, and recommendation set."
                ),
            },
        ],

        # ── TOP-LEVEL convenience fields (exporters pick these up easily)
        "workflow_graph":          workflow_graph,
        "system_module_inventory": inventory,
        "document_data_lineage":   data_lineage,
        "automation_logic":        automation_logic,
        "recommendations":         recommendations,
        "suggestion_blueprints":   suggestion_sections,

        "chatbot_special_considerations": {
            "termination_design": {"conditions": chatbot.get("termination_conditions")},
            "loop_prevention":     chatbot.get("loop_prevention"),
            "recovery_strategies": chatbot.get("recovery_strategies"),
            "context_compression": {
                "strategies":        chatbot.get("compression_strategies"),
                "budget_allocation": chatbot.get("budget_allocation"),
            },
            "overflow_prevention":  chatbot.get("overflow_prevention"),
        },
        "footer": {
            "data_classification": "[ ]",
            "legal_notice": f"© {year} AgentForge. All rights reserved.",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7.  ROUTE
# ═════════════════════════════════════════════════════════════════════════════
@technical_design_bp.get("/suggestions/<suggestion_key>/technical-design")
def get_technical_design(suggestion_key: str):
    """
    GET /api/suggestions/<suggestion_key>/technical-design
    Returns the enriched Technical Design Document JSON.
    """
    try:
        ctx = _resolve_doc_context(suggestion_key)
        header = _derive_header_fields(ctx)

        dyn = {}
        if ctx["found"]:
            dyn = _call_llm_full(_build_full_prompt(ctx, header))

        payload = _build_technical_design(ctx, header, dyn)

        payload["document_metadata"]["suggestion_key"] = suggestion_key
        payload["document_metadata"]["suggestion_resolved"] = ctx["found"]
        payload["document_metadata"]["llm_generated"] = bool(dyn)

        # NEW — expose the focused suggestion + its anchor step on the response
        # so the FE PDF can title-card the document around the chosen suggestion.
        # The API response shape is preserved; these are additive fields.
        focused_sug = ctx.get("suggestion") or {}
        focused_step = ctx.get("step") or {}
        if focused_sug or focused_step:
            payload["document_metadata"]["focus"] = {
                "suggestion_id":      focused_sug.get("_key") or focused_sug.get("id"),
                "suggestion_title":   focused_sug.get("title"),
                "suggestion_summary": focused_sug.get("description"),
                "step_number":        focused_step.get("step_number"),
                "step_title":         focused_step.get("title"),
                "automation_potential": focused_step.get("automation_potential"),
                "is_higher_intervention": (
                    int(focused_step.get("automation_potential") or 0) >= 70
                ),
            }

        return jsonify(payload), 200

    except Exception as e:
        logger.error(
            f"[technical-design] failed for suggestion_key={suggestion_key}: {e}",
            exc_info=True,
        )
        return jsonify({
            "status": False,
            "message": "Could not build technical design",
            "data": None,
        }), 500
