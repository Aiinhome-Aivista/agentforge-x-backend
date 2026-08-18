"""
Mistral prompt templates for the 3-pass analysis pipeline.
Pass 1: Extract process steps
Pass 2: Score automation potential per step
Pass 3: Generate agentic suggestions
"""

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────

import json


GLOBAL_CONSTRAINTS = """
CRITICAL ANALYSIS RULES:

- Always identify hidden micro-processes
- Always extract decision logic explicitly
- Never assume linear flow if conditions exist
- Prefer decomposition over summarization
"""

SYSTEM_PROCESS_ANALYST = """You are an expert business process analyst and enterprise architect.
You specialize in analyzing organizational workflows, ERP system data, and process documentation
to identify automation opportunities and inefficiencies.

You always respond with valid JSON only. No markdown, no explanation text outside JSON.
Your JSON must be parseable by Python's json.loads() directly."""

SYSTEM_AUTOMATION_EXPERT = """You are a senior automation and AI agent architect with deep expertise in
RPA, workflow automation, ERP integrations, and agentic AI systems.
You analyze business processes and design practical automation solutions.

You always respond with valid JSON only. No markdown, no explanation text outside JSON."""

SYSTEM_ONBOARDING_ASSISTANT = """You are AgentForgeX, an intelligent onboarding assistant. 
Your primary goal is to interact with the user and deeply understand their company's mission and vision before they begin using the platform.

**Guidelines for Understanding Mission & Vision:**
- A **Mission Statement** describes what the organization needs to do now. It defines how the organization differentiates itself, why it exists, and how it creates value. Questions to consider asking: What is your core business? What specific problem are you solving? Who exactly are you serving?
- A **Vision Statement** describes where the organization wants to be in the future (e.g., in 5-10 years). It is a broad, aspirational description of the ultimate desired state. Questions to consider asking: What will your business look like in 5 to 10 years? What is the ultimate impact you want to achieve?
- Ask specific, thoughtful questions inspired by these frameworks to truly understand their core objectives.

**Conversation Flow & Style:**
- STRICT RULE: NEVER use the words "customer" or "customers" in your responses. Use words like "users", "clients", "beneficiaries", or "people you serve" instead.
- Greet the user warmly by their name in your FIRST message only (they introduce themselves in their first message). DO NOT repeat the user's name in every single response.
- Ask questions ONE BY ONE in a conversational, natural manner. Do not overwhelm the user.
- Once you have collected both the mission and the vision, summarize your understanding.
- In your JSON response, when you have collected all necessary context, set "flow_complete" to true and include the "final_summary".
- **CRITICAL**: When "flow_complete" is true, your "response" MUST NOT contain any questions (e.g. no "How does this sound?", "Should we proceed?", etc.). Simply state that you have captured their mission and vision and are ready to proceed.

You always respond with valid JSON only, using this exact structure:
{
  "response": "Your conversational message back to the user. Do not include questions if flow_complete is true.",
  "flow_complete": false,
  "final_summary": null
}"""


def safe_json(data):
    import json
    return json.dumps(data, indent=2).replace("{", "{{").replace("}", "}}")


# ── PASS 1: EXTRACTION ────────────────────────────────────────────────────────

def build_extraction_prompt(text: str, source_type: str, file_name: str) -> str:
    source_context = {
        "pdf": "a process definition document (PDF)",
        "docx": "a process definition document (Word)",
        "txt": "a plain text process description",
        "csv": "an ERP system data dump (CSV)",
        "erp_dump": "an ERP system data export (Excel/CSV)",
    }.get(source_type, "a business document")

    return f"""Analyze the following content from {source_context} (Uploaded files: {file_name}).

Note: The content may contain multiple files separated by '=== File: <filename> ===' headers. You MUST pay attention to these headers to accurately identify which file each piece of data or module comes from.

Extract the complete business process described or implied by this data.

IMPORTANT INSTRUCTIONS:

- Break the process into at least 5 to 10 sequential steps
- Each step MUST be atomic and represent a single action
- DO NOT combine multiple actions into one step
- Ensure steps follow a logical execution order
- Include both system and human actions as separate steps
- If the document is high-level, infer missing intermediate steps logically
- Always return multiple steps even if input is short

STRICT RULES:

- Every step MUST include a "lane" (department)
- Lane examples:
  - Administrative Manager
  - Warehouse
  - Quality Sector
  - Production
  - Expedition

- "actor" is specific person/system
- "lane" is department (used for swimlane)

- If decision exists:
  step_type MUST be "decision"

Return a JSON object with this exact structure:
{{
  "process_title": "string - clear name for this process",
  "process_description": "string - 2-3 sentence description of the end-to-end process",
  "erp_system": "string or null - detected ERP system (SAP/Oracle/NetSuite/other/null)",
  "process_category": "string - e.g. Order-to-Cash, Procure-to-Pay, HR, Finance, Inventory",
  "steps": [
    {{
      "step_number": 1,
      "title": "...",
      "description": "...",
      "actor": "...",
      "lane": "Procurement Department",   
      "role_type": "human",
      "micro_steps": [
        {{
          "micro_step_id": "1.1",
          "title": "Validate input data",
          "type": "validation | decision | action",
          "actor": "system | human",
          "automation_potential": 80
        }}
      ],
      "decisions": [
        {{
          "decision_id": "D1",
          "question": "Is stock available?",
          "type": "rule_based | judgment",
          "branches": [
            {{"condition": "YES", "next_step": 3}},
            {{"condition": "NO", "next_step": 5}}
          ]
        }}
      ]
    }}
  ],
  "erp_modules_identified": [
    {{
      "module_name": "string",
      "description": "string",
      "source_file": "string - name of the uploaded file this module primarily relates to",
      "tables_identified": ["list of EXACT table/entity names found (e.g. 'T001', 'EBAN'). Do NOT include descriptions or definitions. Also include implicit upstream tables if the data is downstream of the process"],
      "fields_identified": ["list of EXACT field names found (e.g. 'BUKRS', 'MANDT'). Do NOT include descriptions or definitions."]
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
{text[:12000].replace("{", "{{").replace("}", "}}")}
---"""


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
    steps_json = safe_json(slim_steps)

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

When scoring automation potential, consider:

- Level of human judgment required (low / medium / high)
- Whether approvals or compliance checks are involved
- Degree of rule-based vs unstructured decision making
- Availability of system APIs or structured inputs
- Exception handling complexity

IMPORTANT:
- Steps requiring significant human judgment or approvals should have lower automation potential
- Only assign 90+ when the step is fully system-driven with minimal or no human involvement
- Do not rely on a single factor; balance all dimensions before assigning the score
- automation_reasoning MUST explain the score based on these factors



Scoring guide:
- 90-100: Fully automatable today with standard tools
- 70-89: Highly automatable with moderate integration effort
- 50-69: Partially automatable, some manual oversight needed
- 20-49: Low automation potential, human judgment critical
- 0-19: Cannot be meaningfully automated"""


def build_suggestions_prompt(steps, scores, process_title):
    import json

    steps_json = json.dumps(steps, indent=2).replace("{", "{{").replace("}", "}}")
    scores_json = json.dumps(scores, indent=2).replace("{", "{{").replace("}", "}}")

    return f"""
You are an expert in ERP automation and AI transformation.

Process: {process_title}

Steps:
{steps_json}

Automation Scores:
{scores_json}

Your task:
Generate actionable automation suggestions mapped to specific steps.

IMPORTANT:
- ONLY generate suggestions for steps where step_type is NOT "manual"
- DO NOT generate suggestions for manual steps
- Focus on step_type: system, decision, approval, notification
- Try to cover ALL non-manual steps
- Prefer high automation_potential steps first
- Each suggestion MUST include "step_number"
- step_number MUST match from the given Steps
- Always return at least 5 suggestions
- Do NOT return empty list

CRITICAL RULES:
- Each suggestion MUST have UNIQUE metrics
- "automation_potential" is MANDATORY inside metrics
- automation_potential MUST be a number between 0 to 100
- DO NOT skip metrics for any suggestion
- Do NOT repeat same metrics
- Metrics must be step-specific

IMPORTANT:
- "accuracy_reason" MUST be different from automation_reasoning
- DO NOT repeat same reasoning
- Focus on WHY this solution works (technology, ROI, speed, implementation)

metrics format:
{{
  "automation_potential": number,
  "outputs": [
    "step-specific impact"
  ]
}}

Return ONLY valid JSON:

{{
  "suggestions": [
    {{
      "step_number": 1,
      "title": "Short title",
      "description": "What to automate and how",
      "priority": "high | medium | low",
      "automation_type": "AI | RPA | Workflow | Integration",
      "accuracy_reason": "Why this automation solution is effective",
      "metrics": {{
        "automation_potential": 75,
        "outputs": [
          "50% faster invoice processing"
        ]
      }},
      "roi_impact": "high | medium | low"
    }}
  ]
}}
"""


# ── PASS 4: LOGICAL RELATIONSHIPS (ArangoDB graph data) ───────────────────────

# def build_relationships_prompt(process_title: str, steps: list, erp_modules: list) -> str:
#     return f"""Identify logical relationships between process components for "{process_title}".

# Steps: {str(steps)[:4000]}
# ERP Modules: {str(erp_modules)[:2000]}

# CRITICAL INSTRUCTIONS:

# - You MUST generate logical relationships between steps
# - Steps can have multiple outgoing edges
# - Not all steps must connect linearly

# FLOW RULES:

# - Support:
#   - Sequential flow
#   - Decision branching (YES / NO / APPROVED / REJECTED)
#   - Loop backs (return to previous steps)
#   - Cross-lane transitions (different roles/departments)

# - If a step is a decision:
#   - It MUST have at least 2 outgoing edges
#   - Each edge MUST include a "condition"

# - You MAY create:
#   - More or fewer edges than (n-1)
#   - Multiple edges from one step

# - Ensure:
#   - No step is completely isolated
#   - Flow remains logically consistent

# EDGE FORMAT:

# "step_sequences": [
#   {{
#     "from_step": 3,
#     "to_step": 4,
#     "type": "normal"
#   }},
#   {{
#     "from_step": 3,
#     "to_step": 5,
#     "type": "conditional",
#     "condition": "YES"
#   }},
#   {{
#     "from_step": 3,
#     "to_step": 2,
#     "type": "loop",
#     "condition": "REJECTED"
#   }}
# ]


# DO NOT:
# - Skip any step
# - Leave any step unconnected
# - Return empty step_sequences


# Return a JSON object with edge data for a graph database:
# {{
#   "step_sequences": [
#     {{"from_step": 2, "to_step": 3, "type": "normal | conditional | loop", "condition": "optional condition"}}
#   ],
#   "module_relationships": [
#     {{"from_module": "module name", "to_module": "module name", "relationship": "string"}}
#   ],
#   "cross_process_dependencies": [
#     {{"description": "string - any external process this connects to"}}
#   ]
# }}"""

# ── PASS 4: LOGICAL RELATIONSHIPS (ArangoDB graph data) ───────────────────────

def build_relationships_prompt(process_title: str, steps: list, erp_modules: list) -> str:
    return f"""
You are an enterprise workflow intelligence engine.

Your task is to generate a CONTEXT GRAPH for the business process:
"{process_title}"

The graph must model:
- Decisions
- Actors
- Goals
- Constraints
- Metrics
- Assumptions
- Risks
- Causal Factors
- Interventions
- Validation Paths
- Step Relationships
- ERP Module Dependencies

==================================================
PROCESS DATA
==================================================

STEPS:
{str(steps)[:5000]}

ERP MODULES:
{str(erp_modules)[:3000]}

==================================================
CONTEXT GRAPH RULES
==================================================

The graph MUST follow this structure:

1. Define Decision
   - What decision is being made?
   - Every major workflow should map to at least one decision node

2. Identify Actor
   - Who owns the decision?
   - Actors may include:
     - Employee
     - Manager
     - Finance Team
     - Procurement Team
     - System
     - ERP Module

3. Set Goal
   - What is the expected outcome?
   - Goals should describe success states

4. Attach Metrics
   - Define measurable KPIs
   - Example:
     - approval_time
     - cost_reduction
     - inventory_accuracy
     - SLA_compliance

5. Surface Constraints
   - Business rules
   - Compliance requirements
   - Budget limitations
   - Authorization limitations

6. Expose Assumptions
   - What assumptions exist?
   - Example:
     - data_is_correct
     - approver_is_available
     - vendor_exists

7. Map Risks
   - What can fail?
   - Include operational risks
   - Include compliance risks
   - Include dependency risks

8. Add Causal Factors
   - What influences outcomes?
   - Include:
     - delays
     - missing_data
     - human_error
     - ERP_sync_failure

9. Define Interventions
   - What actions mitigate failure?
   - Example:
     - escalation
     - retry
     - automated_validation
     - manager_override

10. Validate Scenario
   - Can the workflow trace:
     - failure paths
     - rollback paths
     - recovery paths
     - audit trails

==================================================
FLOW & RELATIONSHIP RULES
==================================================

- Steps can have multiple outgoing edges
- Support:
  - Sequential flow
  - Conditional branching
  - Approval/Rejection flow
  - Loop backs
  - Parallel flow
  - Cross-department flow

- Decision steps MUST contain:
  - at least 2 outgoing edges
  - condition labels

- No node should remain isolated

- Relationships must remain logically valid

==================================================
GRAPH OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

{{
  "process_title": "{process_title}",

  "decision_nodes": [
    {{
      "id": "DEC-1",
      "decision": "string",
      "description": "string"
    }}
  ],

  "actors": [
    {{
      "id": "ACT-1",
      "name": "string",
      "role": "string"
    }}
  ],

  "goals": [
    {{
      "id": "GOAL-1",
      "goal": "string",
      "success_criteria": "string"
    }}
  ],

  "metrics": [
    {{
      "id": "MET-1",
      "metric": "string",
      "target": "string"
    }}
  ],

  "constraints": [
    {{
      "id": "CON-1",
      "constraint": "string",
      "type": "business | technical | compliance"
    }}
  ],

  "assumptions": [
    {{
      "id": "ASM-1",
      "assumption": "string"
    }}
  ],

  "risks": [
    {{
      "id": "RSK-1",
      "risk": "string",
      "severity": "low | medium | high"
    }}
  ],

  "causal_factors": [
    {{
      "id": "CAUSE-1",
      "factor": "string",
      "impact": "string"
    }}
  ],

  "interventions": [
    {{
      "id": "INT-1",
      "action": "string",
      "trigger": "string"
    }}
  ],

  "validation_scenarios": [
    {{
      "id": "VAL-1",
      "scenario": "string",
      "expected_result": "string"
    }}
  ],

  "step_sequences": [
    {{
      "from_step": 1,
      "to_step": 2,
      "type": "normal"
    }},
    {{
      "from_step": 2,
      "to_step": 4,
      "type": "conditional",
      "condition": "APPROVED"
    }},
    {{
      "from_step": 2,
      "to_step": 1,
      "type": "loop",
      "condition": "REJECTED"
    }}
  ],

  "decision_relationships": [
    {{
      "from": "Actor",
      "to": "Decision",
      "relationship": "owns"
    }},
    {{
      "from": "Decision",
      "to": "Goal",
      "relationship": "drives"
    }},
    {{
      "from": "Constraint",
      "to": "Decision",
      "relationship": "limits"
    }},
    {{
      "from": "Metric",
      "to": "Goal",
      "relationship": "measures"
    }},
    {{
      "from": "Assumption",
      "to": "Decision",
      "relationship": "influences"
    }},
    {{
      "from": "Risk",
      "to": "Goal",
      "relationship": "threatens"
    }},
    {{
      "from": "CausalFactor",
      "to": "Risk",
      "relationship": "causes"
    }},
    {{
      "from": "Intervention",
      "to": "Risk",
      "relationship": "mitigates"
    }}
  ],

  "module_relationships": [
    {{
      "from_module": "string",
      "to_module": "string",
      "relationship": "depends_on | syncs_with | validates"
    }}
  ],

  "cross_process_dependencies": [
    {{
      "description": "string"
    }}
  ]
}}

==================================================
IMPORTANT
==================================================

DO NOT:
- Return explanations
- Return markdown
- Return partial JSON
- Leave nodes disconnected
- Create invalid relationships

The output MUST be graph-database ready.
"""


def build_toc_enrichment_prompt(
    process_title: str,
    toc_result: dict,           # output of TOCAnalyzer.to_dict()
    steps: list,                # raw_steps from extraction
) -> str:
    import json
    primary = toc_result.get("primary_constraint")
    phases_summary = [
        {"phase": p["phase"], "name": p["name"], "summary": p["summary"]}
        for p in toc_result.get("phases", [])
    ]

    return f"""You are a Theory of Constraints (TOC) expert and business process optimization consultant.

A rule-based algorithm has identified the following bottleneck analysis for the process: "{process_title}".

PRIMARY CONSTRAINT (identified algorithmically):
{safe_json(primary) if primary else "None identified"}

ALL PROCESS STEPS:
{safe_json(steps[:20])}

CURRENT TOC PHASE SUMMARIES:
{safe_json(phases_summary)}

YOUR TASK:
Review this analysis and enrich it with domain expertise. For each of the 5 TOC phases, 
provide improved, specific action items based on the actual process context.

Return a JSON object with this exact structure:
{{
  "enriched_summary": "string - 2-3 sentence executive summary of the bottleneck situation",
  "constraint_validation": "string - confirm or challenge the algorithmic constraint identification with reasoning",
  "improvement_potential_pct": number,  
  "enriched_phases": [
    {{
      "phase": 1,
      "name": "IDENTIFY",
      "enriched_summary": "string",
      "additional_actions": [
        {{
          "action": "string - specific, actionable step",
          "owner": "string - role/team",
          "impact": "high | medium | low",
          "effort": "low | medium | high",
          "automation_type": "rpa | ai_agent | workflow | none | null"
        }}
      ]
    }}
  ]
}}

Rules:
- Be specific to this process — no generic advice
- Phase 2 (EXPLOIT) must include at least 2 quick-win actions (low effort, immediate)
- Phase 4 (ELEVATE) must include at least 1 technology recommendation
- improvement_potential_pct must be a realistic number between 10 and 80
- Return ONLY valid JSON"""


# ── PASS 5: REACT FLOW GRAPH LAYOUT ──────────────────────────────────────────
def build_react_flow_prompt(process_title: str, steps: list, suggestions: list) -> str:
    return f"""
You are a workflow visualization engine.

Your task is to convert a business process into a STRUCTURED LANE-BASED FLOW JSON.

PROCESS:
"{process_title}"

STEPS:
{safe_json(steps)}

SUGGESTIONS:
{safe_json(suggestions)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT STRUCTURE (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY valid JSON in this EXACT format. YOU MUST NOT add any other root keys like 'graph' or 'data'. YOU MUST include 'lanes' and 'flow' at the root level.

{{
  "title": "string",
  "lanes": [
    {{
      "id": "lane-id",
      "label": "Lane Name",
      "nodes": [
        {{
          "id": "node-id",
          "type": "start | process | decision",
          "label": "Node Label",
          "column": number,
          "agentInfo": {{
            "title": "Agent name",
            "type": "Agent type",
            "tasks": ["task1", "task2"],
            "accentColor": "#optional"
          }}
        }}
      ]
    }}
  ],
  "flow": [
    {{
      "from": "node-id",
      "to": "node-id",
      "type": "inline | down | yes | no | diagonal_down",
      "label": "text"
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. GROUPING:
- Group steps by "lane" → each lane becomes one entry
- lane id = lowercase + hyphen (e.g., "quality-sector")

2. START NODE:
- ALWAYS include one start node in first lane:
  {{
    "id": "start",
    "type": "start",
    "label": "",
    "column": 0
  }}

3. NODE RULES:
- Each step → one node
- type:
  - "decision" if step_type == "decision"
  - otherwise "process"

4. COLUMN LOGIC:
- Assign increasing column numbers (0,1,2...) per lane
- Keep flow visually progressive

5. AGENT INFO:
- If suggestion exists for a step:
  attach:
    {{
      "title": suggestion title,
      "type": automation_type,
      "tasks": derived from description,
      "accentColor": "#optional"
    }}
- If no suggestion → DO NOT include agentInfo

6. FLOW RULES:
- Connect steps sequentially
- If decision:
  - MUST create YES / NO edges
- Use types:
  - inline → same lane forward
  - down → different lane
  - yes / no → decision branches

7. IMPORTANT:
- DO NOT include "color" field anywhere
- DO NOT include extra fields
- ALL nodes must be connected
- IDs must be unique

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT:
ONLY JSON. NO explanation.
"""



def build_inventory_react_flow_prompt(process_title: str, steps: list, suggestions: list) -> str:
    """
    Specialised React Flow prompt for Inventory Check / Sales Order Acceptance workflows.

    Graph structure it produces:
    ┌─────────────────────────────────────────────────────────────────┐
    │  START → Sales Order Agent → [Validate Order Data]             │
    │                │ VALID              │ FAILED                   │
    │                ▼                   ▼                           │
    │        [Query Inventory DB]   [Order Rejected]                 │
    │                │                                               │
    │         Inventory Agent                                        │
    │         ② Check Stock                                          │
    │         ③ Analyze Allocation                                   │
    │         ④ Determine Availability                               │
    │                │                                               │
    │         ◆ Stock Available?                                     │
    │          YES ↙           NO/PARTIAL ↘                         │
    │   [Reserve Stock]         [Consider Alternatives]              │
    │   [Confirm Avail.]        [Calculate Lead Times]  ← Reorder   │
    │   [Update Status]         [Generate Options]         Agent     │
    │   [Accept Order]          [Propose Options]                    │
    │   [Notify Customer/ERP]   [Update Status → Accept]            │
    └─────────────────────────────────────────────────────────────────┘

    Key differences from the generic prompt:
    - Enforces a DECISION NODE for "Stock Available?" (diamond shape)
    - Enforces YES / NO-PARTIAL dual-path edges with colour coding
    - Places Reorder Agent on the NO path with dashed "triggers" edges
    - Produces cross-group edge zIndex:10 so edges render above group containers
    """

    steps_json = safe_json(steps)
    suggestions_json = safe_json(suggestions)

    return f"""
You are a React Flow diagram architect specialising in inventory and order management workflows.
Your output is RAW JSON ONLY — no markdown, no explanation, no code fences.

PROCESS TITLE: "{process_title}"

STEPS (from database):
{steps_json}

AUTOMATION SUGGESTIONS (from database):
{suggestions_json}

════════════════════════════════════════════════════════
GRAPH STRUCTURE YOU MUST PRODUCE
════════════════════════════════════════════════════════

The graph has FOUR AGENT LANES and TWO OUTCOME PATHS.

──────────────────────────────────────────────────────
LANE 1 — Sales Order Agent  (group id: "group-sales-order-agent")
  Position: x=60, y=40, width=380, height=180
  Contains:
    • node "step-validate" — type "processNode" — "Validate Order Data"
        position inside group: x=20, y=60
        Checks completeness and Product Catalog via API

──────────────────────────────────────────────────────
LANE 2 — Inventory Agent  (group id: "group-inventory-agent")
  Position: x=60, y=260, width=380, height=400
  Contains (in order):
    • node "step-query-db"      — "Query Inventory DB"         — y=60
    • node "step-check-stock"   — "Check Stock Levels"         — y=160  (② label in data)
    • node "step-analyze-alloc" — "Analyze Allocation"         — y=245  (③)
    • node "step-determine-avail"— "Determine Availability"    — y=330  (④ — calculates ATP)

──────────────────────────────────────────────────────
DECISION NODE — "Stock Available?"
  id: "node-stock-decision"
  type: "decisionNode"   ← diamond shape in frontend
  position: x=520, y=440   (OUTSIDE all groups — standalone)
  data.label: "Stock Available?"

──────────────────────────────────────────────────────
LANE 3 — YES Path (group id: "group-yes-path")
  Position: x=60, y=700, width=380, height=520
  data.label: "Accept Flow"
  Contains:
    • node "step-reserve"        — "Reserve Stock"          — y=60   (Inventory Agent locks qty)
    • node "step-confirm-avail"  — "Confirm Availability"   — y=145
    • node "step-update-yes"     — "Update Order Status"    — y=230  (Ready to Ship)
    • node "step-accept-order"   — "Accept Order"           — y=315
    • node "step-notify-yes"     — "Order Accepted — Notify Customer/ERP" — y=400

──────────────────────────────────────────────────────
LANE 4 — NO/PARTIAL Path (group id: "group-no-path")
  Position: x=500, y=700, width=400, height=520
  data.label: "Alternatives Flow"
  Contains:
    • node "step-consider-alt"   — "Consider Alternatives"  — y=60   (other WH, lead times, supplier)
    • node "step-calc-lead"      — "Calculate Lead Times"   — y=145
    • node "step-gen-options"    — "Generate Options"       — y=230
    • node "step-propose"        — "Propose Options"        — y=315  (Partial Ship / Wait / Backorder → Sales)
    • node "step-update-no"      — "Update Order Status"    — y=400  (Ready to Ship)
    • node "step-accept-no"      — "Accept Order"           — y=475

──────────────────────────────────────────────────────
AGENT NODES (outside groups, right side x=960)
  • agent-sales-order  — "Sales Order Agent"  — y=100   type "agentNode"  accentColor "#3B82F6"
  • agent-inventory    — "Inventory Agent"    — y=380   type "agentNode"  accentColor "#3B82F6"
  • agent-reorder      — "Reorder Agent"      — y=780   type "agentNode"  accentColor "#8B5CF6"
    data.description: "Checks other warehouses and lead times"

════════════════════════════════════════════════════════
EDGES YOU MUST PRODUCE (exact IDs)
════════════════════════════════════════════════════════

MAIN FLOW EDGES (animated, smoothstep, stroke "#4B5563", strokeWidth 2, zIndex 10):
  seq-start-validate        : agent-sales-order  → step-validate
  seq-validate-querydb      : step-validate       → step-query-db        label "VALID"   stroke "#16A34A"
  seq-querydb-checkstock    : step-query-db       → step-check-stock
  seq-checkstock-analyze    : step-check-stock    → step-analyze-alloc
  seq-analyze-determine     : step-analyze-alloc  → step-determine-avail
  seq-determine-decision    : step-determine-avail → node-stock-decision

DECISION EDGES:
  dec-yes   : node-stock-decision → step-reserve      label "YES"        stroke "#16A34A"  strokeWidth 2
  dec-no    : node-stock-decision → step-consider-alt label "NO/PARTIAL" stroke "#DC2626"  strokeWidth 2

REJECTION EDGE (from validate, type "smoothstep", stroke "#DC2626", strokeDasharray "5 5"):
  edge-validate-rejected : step-validate → node-rejected   label "FAILED"   stroke "#DC2626"

REJECTION NODE (standalone, not in any group):
  id: "node-rejected"
  type: "processNode"
  position: x=520, y=100
  data.label: "Order Rejected / Correction Needed"
  data.accentColor: "#EF4444"
  data.description: "Notification to Sales"

YES PATH SEQUENTIAL EDGES (stroke "#16A34A", animated, zIndex 10):
  seq-reserve-confirm  : step-reserve       → step-confirm-avail
  seq-confirm-update   : step-confirm-avail → step-update-yes
  seq-update-accept    : step-update-yes    → step-accept-order
  seq-accept-notify    : step-accept-order  → step-notify-yes

NO PATH SEQUENTIAL EDGES (stroke "#F59E0B", animated, zIndex 10):
  seq-consider-calc    : step-consider-alt  → step-calc-lead
  seq-calc-gen         : step-calc-lead     → step-gen-options
  seq-gen-propose      : step-gen-options   → step-propose
  seq-propose-updateno : step-propose       → step-update-no
  seq-updateno-acceptno: step-update-no     → step-accept-no

AGENT AUTOMATION EDGES (dashed, stroke "#8B5CF6", strokeDasharray "5 5", zIndex 10):
  auto-sales-validate  : agent-sales-order → step-validate       label "validates"
  auto-inv-checkstock  : agent-inventory   → step-check-stock    label "checks"
  auto-inv-analyze     : agent-inventory   → step-analyze-alloc  label "analyzes"
  auto-reorder-consider: agent-reorder     → step-consider-alt   label "triggers"
  auto-reorder-calc    : agent-reorder     → step-calc-lead      label "calculates"

CONTINUES FLOW EDGES (stroke "#22C55E", animated, zIndex 10):
  cont-inv-determine   : agent-inventory   → step-determine-avail label "continues flow"

════════════════════════════════════════════════════════
NODE DATA TEMPLATE
════════════════════════════════════════════════════════

Every processNode data object must include:
  label, actor, stepType ("system"|"manual"|"decision"),
  automationPotential (0-100 int),
  inputs (string[]), outputs (string[]),
  painPoints (string[]), duration (string),
  erpModule (string),
  accentColor ("#10B981" if >=80, "#F59E0B" if >=60, "#EF4444" otherwise)

Every agentNode data object must include:
  title, description, tasks (string[]),
  agentType ("workflow_automation"|"decision_support"|"reorder"),
  accuracy (int), roiImpact ("high"|"medium"|"low"),
  effortLevel ("high"|"medium"|"low"),
  technologies (string[]),
  accentColor

Every agentGroupNode data object must include:
  label, icon ("Database"|"UserCircle"|"Layers"),
  accentColor ("#6366F1")

GROUP STYLE: {{ "width": <w>, "height": <h>, "zIndex": 0 }}
STEP STYLE:  {{ "width": 340, "zIndex": 2 }}
AGENT STYLE: {{ "width": 300, "zIndex": 2 }}

ALL edges must include:
  id, source, target, type ("smoothstep"), animated (bool),
  label (string|""), zIndex (10),
  style: {{ stroke, strokeWidth (2), zIndex (10),
            strokeDasharray (omit if solid) }}

════════════════════════════════════════════════════════
MAP INCOMING STEPS DATA TO THE GRAPH ABOVE
════════════════════════════════════════════════════════

Use the steps and suggestions from the database to FILL IN the node data fields.
Match each DB step to the closest structural node by step_number or title keywords.
If a DB step doesn't map cleanly, add it as an extra processNode inside the most
appropriate group, appending a sequential edge to maintain connectivity.

Do NOT invent node IDs that differ from the spec above.
Do NOT omit any node or edge listed above.
If suggestions data provides agent tasks/technologies, use them in the agent nodes.

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════

Return ONLY this JSON structure — no other text:

{{
  "nodes": [
    {{
      "id": "...",
      "type": "agentGroupNode"|"processNode"|"decisionNode"|"agentNode",
      "position": {{"x": 0, "y": 0}},
      "parentNode": "group-id-if-inside-group",
      "extent": "parent",
      "style": {{...}},
      "data": {{...}}
    }}
  ],
  "edges": [
    {{
      "id": "...",
      "source": "...",
      "target": "...",
      "type": "smoothstep",
      "animated": true,
      "label": "...",
      "zIndex": 10,
      "style": {{"stroke": "...", "strokeWidth": 2, "zIndex": 10}}
    }}
  ]
}}
""".strip()


def build_workflow_categorization_prompt(process_title: str, steps: list, suggestions: list) -> str:
    return f"""
You are an expert Enterprise Process Architect.

Your task is to classify the given business workflow into EXACTLY 4 layers of an 
"Operating Model: Workflow Automation".

Return a structured JSON with the following 4 categories:

--------------------------------------------------

1. User Interaction Layer
- Anything related to dashboards, UI, human inputs, approvals
- Manual overrides, monitoring, alerts

2. Agent Orchestration
- Workflow engine logic
- Scheduling, retries, parallel execution
- Agent coordination and decision routing

3. ERP Module Integration
- SAP/Oracle/ERP APIs
- Data mapping, transactions, DB operations
- Master data sync, event triggers

4. Governance / Observability
- Logging, audit trails, SLA tracking
- Monitoring, anomaly detection
- Compliance & reporting

--------------------------------------------------

INPUT DATA:

PROCESS:
{process_title}

STEPS:
{json.dumps(steps, indent=2)}

SUGGESTIONS:
{json.dumps(suggestions, indent=2)}

--------------------------------------------------

INSTRUCTIONS:

1. Map EACH step into ONE of the 4 layers
2. Do NOT skip any step
3. Group similar steps together
4. Keep descriptions concise but meaningful
5. Maintain enterprise-level terminology

--------------------------------------------------

OUTPUT FORMAT (STRICT JSON):

{{
  "user_interaction_layer": {{
    "description": "...",
    "steps": [
      {{
        "step_number": 1,
        "title": "...",
        "reason": "Why it belongs here"
      }}
    ]
  }},
  "agent_orchestration": {{
    "description": "...",
    "steps": []
  }},
  "erp_module_integration": {{
    "description": "...",
    "steps": []
  }},
  "governance_observability": {{
    "description": "...",
    "steps": []
  }}
}}

--------------------------------------------------

IMPORTANT RULES:

- No hallucination
- Use ONLY provided steps
- If unclear → choose best logical fit
- Ensure all 4 categories are present
- No extra text outside JSON
"""


def build_agent_architecture_prompt(suggestion, step, erp_modules):
    import json

    return f"""
You are a senior enterprise AI architect.

Your task is to design an agent cluster architecture for the given automation suggestion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT OUTPUT RULES (VERY IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Return ONLY valid JSON
- DO NOT include markdown (no ``` or ```json)
- DO NOT include explanations or extra text
- DO NOT wrap the JSON in quotes
- DO NOT use trailing commas
- Ensure the JSON is directly parsable using json.loads()
- If JSON is invalid, the response will be rejected

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Suggestion:
{json.dumps(suggestion, indent=2)}

Step:
{json.dumps(step, indent=2)}

ERP Modules:
{json.dumps(erp_modules[:3], indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUCTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Design a practical enterprise-grade architecture
- Use real-world ERP + automation concepts (SAP, APIs, RPA, workflows)
- Keep descriptions concise but meaningful
- Ensure all fields are filled
- Maintain consistency across layers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "agent_cluster_architecture": {{
    "operating_model": "string",
    "erp_module": "string",

    "architecture_layers": [
      {{
        "title": "string",
        "description": "string"
      }}
    ],

    "erp_context": [
      {{
        "title": "string",
        "description": "string"
      }}
    ],

    "deployment_steps": [
      {{
        "id": 1,
        "label": "string",
        "description": "string"
      }}
    ]
  }}
}}
""".strip()


# def build_architecture_prompt(suggestion, step, process):
#     return f"""
# You are a principal AI architect designing a fully dynamic enterprise-grade agentic architecture.

# STRICT RULES:
# - Return ONLY valid JSON
# - NO explanation
# - NO markdown
# - JSON must start with {{ and end with }}
# - Do NOT hardcode fixed nodes

# OBJECTIVE:
# Generate a React Flow architecture dynamically based on the given context.

# DYNAMIC DESIGN LOGIC:

# 1. Identify SYSTEM COMPONENTS from context:
#    - Inputs (email, ERP, chat, files, APIs)
#    - Core processing steps → convert into AGENTS
#    - Decision points → validation agents
#    - External systems → ERP, APIs
#    - Data needs → DBs, vector stores, memory

# 2. NODE GENERATION RULES:
#    - Total nodes: 8 to 14 (dynamic)
#    - MUST ALWAYS INCLUDE:
#      - API Gateway
#      - Agentic Orchestrator
#    - Agents MUST be derived from process steps dynamically
#    - Add tools (DB, vector store) ONLY if needed

# 3. LAYERED POSITIONING (MANDATORY):

#    INPUT LAYER:
#    x: 0–200

#    INTERFACE LAYER:
#    x: 250–450

#    ORCHESTRATION LAYER:
#    x: 500–700

#    AGENT LAYER:
#    x: 750–1050

#    DATA/TOOLS LAYER:
#    x: 500–900 (lower y)

#    OUTPUT LAYER:
#    x: 1100–1400

#    - Dynamically space nodes vertically (y-axis)
#    - Avoid overlap

# 4. EDGE RULES:

#    - Primary flow:
#      Inputs → API Gateway → Orchestrator → Agents → Outputs

#    - Orchestrator MUST connect to ALL agents

#    - Agents:
#      - Can call other agents IF process requires
#      - Can connect to DB/tools

#    - Add feedback loops ONLY if logical

  


# MANDATORY ENTERPRISE COMPONENTS (ALWAYS INCLUDE):

# INPUT CHANNELS:
# - Email Input (PDF)
# - Chat Input
# - ERP System Input

# CONNECTOR LAYER:
# - Email Connector
# - Chat/Slack Connector
# - ERP API Connector

# INTERFACE LAYER:
# - Web / Mobile UI
# - Vendor Interaction Portal (Human Node)

# CORE PLATFORM:
# - API Gateway (MANDATORY)
# - Agentic Orchestrator (MANDATORY)
# - LLM Planner (if reasoning exists)

# AGENT LAYER:
# - Derived dynamically from process
# - Minimum 2 agents

# SHARED CONTEXT & TOOLS (MANDATORY):
# - Vector Database (Knowledge Base)
# - Memory Store
# - Relational Database
# - Tool Integration Layer / Middleware
# - Safety / Guardrails

# INFRASTRUCTURE LAYER (MANDATORY):

# - Kubernetes Cluster (EKS/AKS/GKE)
#   - Hosts orchestrator and all agents
#   - Acts as compute layer

# - Redis Cache
#   - Used for session state, orchestration memory, fast retrieval

# - Event Streaming (Kafka or Pub/Sub)
#   - Used for async communication between agents and services

# - Object Storage (S3 / Blob Storage)
#   - Stores documents, artifacts, intermediate outputs

# - Logging & Monitoring
#   - Audit logs, observability (CloudWatch / ELK)

# INFRA RULES:

# - Orchestrator MUST run inside Kubernetes
# - All agents MUST connect to Kubernetes layer
# - Orchestrator SHOULD use Redis for state/cache
# - Agents SHOULD publish/consume via Kafka for async workflows
# - Outputs SHOULD be stored in Object Storage

# DATA STORAGE:
# - Temporary Storage (if documents involved)
# - ERP Database

# OUTPUT LAYER:
# - Document Store
# - ERP Update System
# - Final Business Output (e.g., Payment / PO / Shipping)

# HUMAN-IN-THE-LOOP:
# - Add if any approval / exception exists

# 5. SEMANTIC NAMING:

#    DO NOT use generic names like:
#    ❌ Agent1, NodeA

#    USE:
#    ✔ PR Intake Agent
#    ✔ Sourcing Agent
#    ✔ Validation Agent
#    ✔ Knowledge Base
#    ✔ Memory Store

# 6. TYPE MAPPING:

#    - input → external triggers
#    - system → infra (API GW, DB, orchestrator)
#    - agent → AI/logic units
#    - human → approval / HITL
#    - output → final systems
# ARCHITECTURE STYLE:

# - Think like AWS reference architecture
# - Not just workflow — build FULL PLATFORM
# - Include supporting systems (DBs, UI, connectors)
# - Show where data is stored, not just processed

# OUTPUT FORMAT:

# {{
#   "nodes": [
#     {{
#       "id": "string",
#       "type": "input|system|agent|human|output",
#       "data": {{
#         "label": "string",
#         "description": "context-aware description"
#       }},
#       "position": {{ "x": number, "y": number }}
#     }}
#   ],
#   "edges": [
#     {{
#       "id": "string",
#       "source": "node_id",
#       "target": "node_id",
#       "label": "meaningful data flow"
#     }}
#   ]
# }}

# CONTEXT:
# Suggestion:
# {json.dumps(suggestion)}

# Step:
# {json.dumps(step)}

# Process:
# {json.dumps(process)}

# INTELLIGENCE MODE:

# - Infer architecture like a real system designer
# - If process is complex → more agents
# - If simple → fewer nodes
# - If documents involved → add OCR agent
# - If reasoning required → add LLM planner
# - If memory required → add vector DB
# - If approvals required → add human-in-loop

# FINAL GOAL:
# Create a clean, scalable, production-grade architecture that visually resembles AWS/Azure diagrams but is fully dynamic.
# """



def build_architecture_prompt(suggestion, step, process):
    user_context = str(process.get("raw_text", ""))[:5000] if isinstance(process, dict) else ""
    return f"""
You are a principal AI platform architect designing a PRODUCTION-DEPLOYABLE agentic system.

STRICT RULES:
- Return ONLY valid JSON
- NO explanation
- NO markdown
- JSON must start with {{ and end with }}
- Every node must represent a REAL deployable component

OBJECTIVE:
Generate a React Flow architecture that maps directly to deployable infrastructure and services.

--------------------------------------------------
USER TECHNOLOGY PREFERENCES:
If the user explicitly mentioned specific technologies, databases, or cloud providers in their input (e.g., MySQL, AWS, Azure, GCP, MongoDB, Oracle), you MUST use those exact technologies for the respective layers instead of the default ones. 

User Context/Input:
{user_context}

--------------------------------------------------
CORE DESIGN PRINCIPLES:

- Every agent = containerized microservice
- Orchestrator = stateful service
- Communication = API (sync) + Kafka (async)
- State = Redis + DB
- Deployment = Kubernetes

--------------------------------------------------
DYNAMIC COMPONENT SELECTION LOGIC:

The architecture MUST be generated based on CONTEXT — do NOT include unnecessary components.

INPUT CHANNELS:
- If documents/files are present → include Email/Input + OCR ingestion
- If chat/user interaction → include Chat Input
- If ERP mentioned → include ERP Connector

CONNECTORS:
- Add connectors ONLY for selected inputs

INTERFACE:
- Add Web/Mobile UI ONLY if user interaction is needed
- Add Human-in-the-loop ONLY if approval/exception exists

CORE PLATFORM (ALWAYS INCLUDE):
- API Gateway
- Agentic Orchestrator

OPTIONAL CORE:
- Add LLM Planner ONLY if multi-step reasoning is required

AGENTS:
- Dynamically derive from process steps
- Each major step = one agent
- Validation → Validation Agent
- Matching → Matching Agent
- Extraction → OCR/Extraction Agent

DATA LAYER:
- If semantic retrieval needed → Vector DB
- If transactional data → Relational DB
- If session/state → Redis
- If simple → skip unnecessary DBs

INFRASTRUCTURE:
- ALWAYS include Kubernetes

- Add Redis ONLY if:
  - stateful workflow OR
  - caching needed

- Add Kafka ONLY if:
  - multiple agents OR
  - async/event-driven flow

STORAGE:
- Add Object Storage ONLY if documents/artifacts involved

OUTPUT SYSTEMS:
- Derived from process (ERP update, payment, documents, notifications)

OPTIMIZATION:
- Simple workflows → 8–10 nodes
- Complex workflows → 12–18 nodes

--------------------------------------------------
DEPLOYMENT RULES:

- All agents + orchestrator run INSIDE Kubernetes
- Each agent = separate pod/service
- Orchestrator communicates:
  - Sync → REST via API Gateway
  - Async → Kafka events

- Redis used for:
  - workflow state
  - caching

- Kafka used for:
  - decoupled execution
  - retries / event-driven flows

- DBs must be explicitly connected

--------------------------------------------------
OUTPUT SYSTEMS:

- Document Store
- ERP Update Service
- Final Business Output (PO / Payment / Shipping)

--------------------------------------------------
EDGE RULES:

- Label edges as:
  - "sync API"
  - "async event"
  - "DB read/write"

- Flow:
  Inputs → Gateway → Orchestrator → Agents → Outputs

- Orchestrator must control ALL agents

--------------------------------------------------
POSITIONING (React Flow):

INPUT: x 0–200  
INTERFACE: x 250–450  
CORE: x 500–750  
AGENTS: x 800–1100  
INFRA: x 600–1000, y lower  
OUTPUT: x 1100–1400  

--------------------------------------------------
OUTPUT FORMAT:

{{
  "nodes": [
    {{
      "id": "string",
      "type": "input|system|agent|human|output",
      "data": {{
        "label": "string",
        "description": "deployable service description"
      }},
      "position": {{ "x": number, "y": number }}
    }}
  ],
  "edges": [
    {{
      "id": "string",
      "source": "node_id",
      "target": "node_id",
      "label": "sync API | async event | DB read/write"
    }}
  ]
}}

--------------------------------------------------
CONTEXT:

Suggestion:
{json.dumps(suggestion)}

Step:
{json.dumps(step)}

Process:
{json.dumps(process)}

--------------------------------------------------
FINAL EXPECTATION:

- Output should map 1:1 to real deployment
- No abstract nodes
- Must resemble AWS/Azure production architecture
"""





# ─────────────────────────────────────────────────────────────────────────────
# Process Discovery Assistant
# ─────────────────────────────────────────────────────────────────────────────
def build_discovery_prompt(
    process_title: str,
    process_description: str,
    steps: list,
    suggestions: list,
    erp_modules: list,
) -> tuple:
    """
    Build the (system, user) messages for the Process Discovery Assistant.

    The assistant does NOT give final recommendations. It inspects the
    extracted process map and surfaces the highest-value follow-up questions
    that would refine the map and improve automation analysis.

    Returns a (system_prompt, user_prompt) tuple. The model is instructed to
    return strict JSON matching:
        {
          "summary": "",
          "identified_gaps": [""],
          "follow_up_questions": [{"question": "", "reason": ""}]
        }
    """
    # Compact, token-bounded projection of the process map. We only feed the
    # fields that matter for spotting manual work, delays, approvals and gaps.
    compact_steps = [
        {
            "step_number":          s.get("step_number"),
            "title":                s.get("title"),
            "actor":                s.get("actor"),
            "role_type":            s.get("role_type"),
            "step_type":            s.get("step_type"),
            "automation_potential": s.get("automation_potential"),
            "inputs":               s.get("inputs", []),
            "outputs":              s.get("outputs", []),
            "pain_points":          s.get("pain_points", []),
            "duration_estimate":    s.get("duration_estimate"),
            "decisions":            s.get("decisions", []),
        }
        for s in (steps or [])
    ]
    compact_suggestions = [
        {
            "title":      g.get("title"),
            "agent_type": g.get("agent_type"),
            "step_key":   g.get("step_key"),
        }
        for g in (suggestions or [])
    ]
    compact_modules = [
        {
            "module_name":      m.get("module_name"),
            "erp_system":       m.get("erp_system"),
            "tables_identified": m.get("tables_identified", []),
        }
        for m in (erp_modules or [])
    ]

    system_prompt = (
        "You are a Process Discovery Assistant.\n"
        "You analyze an uploaded business process: its extracted entities, "
        "relationships, process map, and knowledge graph.\n\n"
        "Your goal is NOT to immediately provide final recommendations. Instead:\n"
        "1. Identify areas where the process is unclear.\n"
        "2. Identify missing business context.\n"
        "3. Identify steps that appear manual, delayed, approval-heavy, or ambiguous.\n"
        "4. Identify missing information that would improve automation recommendations.\n"
        "5. Generate 2 to 4 intelligent follow-up questions for the user.\n\n"
        "QUESTION RULES:\n"
        "- Questions must be specific to the uploaded content.\n"
        "- Questions must help refine the process map.\n"
        "- Questions must help identify bottlenecks, delays, manual work, "
        "approval chains, or automation opportunities.\n"
        "- IMPORTANT: You MUST generate at least one question asking the user about their technology stack, specifically their main data source and what type of data is involved, unless it is already explicitly stated.\n"
        "- Do not ask generic questions.\n"
        "- Do not ask questions whose answers already exist in the provided data.\n"
        "- Ask only the highest-value questions.\n\n"
        "OUTPUT FORMAT — return STRICT JSON only, no prose, no markdown fences:\n"
        "{\n"
        '  "summary": "",\n'
        '  "identified_gaps": [""],\n'
        '  "follow_up_questions": [{ "question": "", "reason": "" }]\n'
        "}\n"
        "Each follow_up_question MUST include a concrete 'reason' tying it to "
        "the data. Return between 2 and 4 questions."
    )

    user_prompt = (
        f"PROCESS TITLE: {process_title or 'Untitled process'}\n"
        f"PROCESS DESCRIPTION: {process_description or '(none provided)'}\n\n"
        f"PROCESS MAP — STEPS ({len(compact_steps)}):\n"
        f"{json.dumps(compact_steps, ensure_ascii=False)}\n\n"
        f"AUTOMATION SUGGESTIONS ({len(compact_suggestions)}):\n"
        f"{json.dumps(compact_suggestions, ensure_ascii=False)}\n\n"
        f"ERP MODULES ({len(compact_modules)}):\n"
        f"{json.dumps(compact_modules, ensure_ascii=False)}\n\n"
        "Analyze the above and return the discovery JSON."
    )
    return system_prompt, user_prompt
