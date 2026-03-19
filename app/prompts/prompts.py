"""
Mistral prompt templates for the 3-pass analysis pipeline.
Pass 1: Extract process steps
Pass 2: Score automation potential per step
Pass 3: Generate agentic suggestions
"""

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────

import json


SYSTEM_PROCESS_ANALYST = """You are an expert business process analyst and enterprise architect.
You specialize in analyzing organizational workflows, ERP system data, and process documentation
to identify automation opportunities and inefficiencies.

You always respond with valid JSON only. No markdown, no explanation text outside JSON.
Your JSON must be parseable by Python's json.loads() directly."""

SYSTEM_AUTOMATION_EXPERT = """You are a senior automation and AI agent architect with deep expertise in
RPA, workflow automation, ERP integrations, and agentic AI systems.
You analyze business processes and design practical automation solutions.

You always respond with valid JSON only. No markdown, no explanation text outside JSON."""


# ── PASS 1: EXTRACTION ────────────────────────────────────────────────────────

def build_extraction_prompt(text: str, source_type: str, file_name: str) -> str:
    source_context = {
        "pdf": "a process definition document (PDF)",
        "docx": "a process definition document (Word)",
        "txt": "a plain text process description",
        "csv": "an ERP system data dump (CSV)",
        "erp_dump": "an ERP system data export (Excel/CSV)",
    }.get(source_type, "a business document")

    return f"""Analyze the following content from {source_context} named "{file_name}".

Extract the complete business process described or implied by this data.

Return a JSON object with this exact structure:
{{
  "process_title": "string - clear name for this process",
  "process_description": "string - 2-3 sentence description of the end-to-end process",
  "erp_system": "string or null - detected ERP system (SAP/Oracle/NetSuite/other/null)",
  "process_category": "string - e.g. Order-to-Cash, Procure-to-Pay, HR, Finance, Inventory",
  "steps": [
    {{
      "step_number": 1,
      "title": "string - short step name",
      "description": "string - what happens in this step",
      "actor": "string - who/what performs this (System/Sales Team/Finance/Manager/etc)",
      "step_type": "string - one of: manual|system|decision|approval|notification",
      "inputs": ["list of inputs/triggers for this step"],
      "outputs": ["list of outputs/results from this step"],
      "pain_points": ["list of known inefficiencies or manual effort involved"],
      "erp_module": "string or null - relevant ERP module if applicable",
      "duration_estimate": "string or null - typical time e.g. '2-4 hours', '1 day'"
    }}
  ],
  "erp_modules_identified": [
    {{
      "module_name": "string",
      "description": "string",
      "tables_identified": ["list of table/entity names found"],
      "fields_identified": ["list of field names found"]
    }}
  ],
  "key_insights": [
    {{
      "text": "string - insight text",
      "category": "string - one of: automation|bottleneck|integration|risk|opportunity",
      "impact": "string - one of: high|medium|low"
    }}
  ]
}}

Document content:
---
{text[:12000]}
---"""


# ── PASS 2: AUTOMATION SCORING ────────────────────────────────────────────────

# def build_scoring_prompt(steps: list, process_context: str) -> str:
#     steps_json = str(steps)[:8000]

#     return f"""You are scoring automation potential for each step of this business process.

# Process context: {process_context}

# For each step provided, score its automation potential and identify the best automation approach.

# Steps to score:
# {steps_json}

# Return a JSON array — one entry per step in the same order:
# [
#   {{
#     "step_number": 1,
#     "automation_potential": 85,
#     "automation_reasoning": "string - why this score",
#     "primary_automation_type": "string - rpa|ai_agent|workflow|system_integration|none",
#     "blocking_factors": ["list of what makes it hard to automate"],
#     "quick_win": true
#   }}
# ]

# Scoring guide:
# - 90-100: Fully automatable today with standard tools
# - 70-89: Highly automatable with moderate integration effort
# - 50-69: Partially automatable, some manual oversight needed
# - 20-49: Low automation potential, human judgment critical
# - 0-19: Cannot be meaningfully automated"""


def build_scoring_prompt(steps: list, process_context: str) -> str:
    import json
    # Slim down each step — only fields needed for scoring
    slim_steps = [
        {
            "step_number": s.get("step_number"),
            "title": s.get("title"),
            "description": s.get("description"),
            "actor": s.get("actor"),
            "step_type": s.get("step_type"),
            "pain_points": s.get("pain_points", []),
        }
        for s in steps
    ]
    steps_json = json.dumps(slim_steps, ensure_ascii=False)

    return f"""You are scoring automation potential for each step of this business process.

Process context: {process_context}

You MUST score ALL {len(steps)} steps. Return exactly {len(steps)} entries in the array.

Steps to score:
{steps_json}

Return a JSON array — one entry per step in the same order:
[
  {{
    "step_number": 1,
    "automation_potential": 85,
    "automation_reasoning": "string - why this score",
    "primary_automation_type": "string - rpa|ai_agent|workflow|system_integration|none",
    "blocking_factors": ["list of what makes it hard to automate"],
    "quick_win": true
  }}
]

Scoring guide:
- 90-100: Fully automatable today with standard tools
- 70-89: Highly automatable with moderate integration effort
- 50-69: Partially automatable, some manual oversight needed
- 20-49: Low automation potential, human judgment critical
- 0-19: Cannot be meaningfully automated"""

# ── PASS 3: AGENTIC SUGGESTIONS ───────────────────────────────────────────────

# def build_suggestions_prompt(steps: list, scores: list, process_title: str) -> str:
#     # Focus on steps with automation_potential >= 50
#     high_potential = [
#         {**step, "score_data": next(
#             (s for s in scores if s.get("step_number") == step.get("step_number")), {}
#         )}
#         for step in steps
#         if any(s.get("step_number") == step.get("step_number")
#                and s.get("automation_potential", 0) >= 50 for s in scores)
#     ]

#     return f"""Generate specific agentic automation suggestions for high-potential steps in the "{process_title}" process.

# High-potential steps:
# {str(high_potential)[:8000]}

# For each step, design a concrete automation solution. Return a JSON array:
# [
#   {{
#     "step_number": 1,
#     "title": "string - e.g. 'Automate: Invoice Generation'",
#     "description": "string - 1-2 sentences on what gets automated and how",
#     "agent_type": "string - one of: system_integration|rpa|ai_agent|workflow_automation|communication_agent",
#     "implementation": "string - specific technologies/approach e.g. 'SAP workflow trigger + email agent via SendGrid'",
#     "accuracy_estimate": 95,
#     "execution_speed": "string - instant|minutes|hours|scheduled",
#     "effort_level": "string - low|medium|high",
#     "roi_impact": "string - high|medium|low",
#     "technologies": ["list of specific technologies e.g. SAP BTP, Python, n8n, Zapier"],
#     "prerequisites": ["list of prerequisites e.g. 'ERP API access', 'EDI setup with suppliers'"]
#   }}
# ]

# Focus on practical, implementable solutions. Be specific about technologies."""

# def build_suggestions_prompt(steps: list, scores: list, process_title: str) -> str:
#     import json
#     score_map = {s.get("step_number"): s for s in scores}
#     enriched = [
#         {**step, "score_data": score_map.get(step.get("step_number"), {})}
#         for step in steps
#     ]
#     enriched_json = json.dumps(enriched, ensure_ascii=False)

#     return f"""Generate specific agentic automation suggestions for the "{process_title}" process.

# Steps with their automation scores:
# {enriched_json}

# Generate suggestions for steps with automation_potential >= 50.
# If all steps scored below 50, generate suggestions for the top 3 highest-scoring steps.
# Return a JSON array:
# [
#   {{
#     "step_number": 1,
#     "title": "string - e.g. 'Automate: Invoice Generation'",
#     "description": "string - 1-2 sentences on what gets automated and how",
#     "agent_type": "string - one of: system_integration|rpa|ai_agent|workflow_automation|communication_agent",
#     "implementation": "string - specific technologies/approach",
#     "accuracy_estimate": 95,
#     "execution_speed": "string - instant|minutes|hours|scheduled",
#     "effort_level": "string - low|medium|high",
#     "roi_impact": "string - high|medium|low",
#     "technologies": ["list of specific technologies"],
#     "prerequisites": ["list of prerequisites"]
#   }}
# ]"""


def build_suggestions_prompt(steps, scores, process_title):
    return f"""
You are an expert in ERP automation and AI transformation.

Process: {process_title}

Steps:
{json.dumps(steps, indent=2)}

Automation Scores:
{json.dumps(scores, indent=2)}

Your task:
Generate actionable automation suggestions.

IMPORTANT:
- Always return at least 5 suggestions
- Do NOT return empty list
- Be specific and practical

Return ONLY valid JSON in this format:

{{
  "suggestions": [
    {{
      "title": "Short title",
      "description": "What to automate and how",
      "priority": "high | medium | low",
      "automation_type": "AI | RPA | Workflow | Integration",
      "expected_impact": "Business impact"
    }}
  ]
}}
"""


# ── PASS 4: LOGICAL RELATIONSHIPS (ArangoDB graph data) ───────────────────────

def build_relationships_prompt(process_title: str, steps: list, erp_modules: list) -> str:
    return f"""Identify logical relationships between process components for "{process_title}".

Steps: {str(steps)[:4000]}
ERP Modules: {str(erp_modules)[:2000]}

Return a JSON object with edge data for a graph database:
{{
  "step_sequences": [
    {{"from_step": 1, "to_step": 2, "relationship": "string - e.g. 'triggers'|'feeds into'|'conditionally leads to'", "condition": "string or null"}}
  ],
  "module_relationships": [
    {{"from_module": "module name", "to_module": "module name", "relationship": "string"}}
  ],
  "cross_process_dependencies": [
    {{"description": "string - any external process this connects to"}}
  ]
}}"""
