"""
Core data models for Process Agentifier.
All models serialize to/from ArangoDB document format (_id, _key, _rev).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


def _new_key() -> str:
    return str(uuid.uuid4()).replace("-", "")


# ── Process Document ──────────────────────────────────────────────────────────

@dataclass
class ProcessDocument:
    title: str
    description: str
    source_type: str           # "pdf" | "docx" | "csv" | "txt" | "erp_dump"
    raw_text: str
    automation_score: float    # 0-100 aggregate
    status: str = "pending"    # pending | analyzing | complete | error
    _key: str = field(default_factory=_new_key)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    erp_system: Optional[str] = None   # SAP | Oracle | NetSuite | other
    file_name: Optional[str] = None
    error: Optional[str] = None

    def to_doc(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_text", None)   # don't store full text in main doc
        return d

    def to_api(self) -> Dict[str, Any]:
        d = self.to_doc()
        d["id"] = self._key
        return d


# ── Process Step ──────────────────────────────────────────────────────────────

@dataclass
class ProcessStep:
    process_key: str
    step_number: int
    title: str
    description: str
    actor: str                  # "System" | "Finance team" | "Sales" etc.
    step_type: str              # "manual" | "system" | "decision" | "approval"
    automation_potential: float # 0-100
    _key: str = field(default_factory=_new_key)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    erp_module: Optional[str] = None
    duration_estimate: Optional[str] = None

    def to_doc(self) -> Dict[str, Any]:
        return asdict(self)

    def to_api(self) -> Dict[str, Any]:
        d = self.to_doc()
        d["id"] = self._key
        return d


# ── Automation Suggestion ─────────────────────────────────────────────────────

@dataclass
class AutomationSuggestion:
    process_key: str
    step_key: str
    title: str
    description: str
    agent_type: str            # "system_integration" | "rpa" | "ai_agent" | "workflow"
    implementation: str        # how to implement
    accuracy_estimate: float   # 0-100
    execution_speed: str       # "instant" | "fast" | "scheduled"
    effort_level: str          # "low" | "medium" | "high"
    roi_impact: str            # "high" | "medium" | "low"
    _key: str = field(default_factory=_new_key)
    technologies: List[str] = field(default_factory=list)  # ["EDI", "SAP workflow"]
    prerequisites: List[str] = field(default_factory=list)

    def to_doc(self) -> Dict[str, Any]:
        return asdict(self)

    def to_api(self) -> Dict[str, Any]:
        d = self.to_doc()
        d["id"] = self._key
        return d


# ── ERP Module ────────────────────────────────────────────────────────────────

@dataclass
class ERPModule:
    process_key: str
    module_name: str           # "Sales and Finance", "Procurement" etc.
    erp_system: str
    source_file: str
    description: str
    _key: str = field(default_factory=_new_key)
    tables_identified: List[str] = field(default_factory=list)
    fields_identified: List[str] = field(default_factory=list)

    def to_doc(self) -> Dict[str, Any]:
        return asdict(self)

    def to_api(self) -> Dict[str, Any]:
        d = self.to_doc()
        d["id"] = self._key
        return d


# ── Key Process Insight ───────────────────────────────────────────────────────

@dataclass
class KeyInsight:
    text: str
    category: str   # "automation" | "bottleneck" | "integration" | "risk"
    impact: str     # "high" | "medium" | "low"


# ── Full Analysis Result (assembled, not stored as-is) ───────────────────────

@dataclass
class AnalysisResult:
    process: ProcessDocument
    steps: List[ProcessStep]
    suggestions: List[AutomationSuggestion]
    erp_modules: List[ERPModule]
    key_insights: List[KeyInsight]
    top_automation_targets: List[Dict[str, Any]]
    graph_url: Optional[str] = None

    def to_api(self) -> Dict[str, Any]:
        return {
            "process": self.process.to_api(),
            "steps": [s.to_api() for s in self.steps],
            "suggestions": [s.to_api() for s in self.suggestions],
            "erp_modules": [m.to_api() for m in self.erp_modules],
            "key_insights": [asdict(i) for i in self.key_insights],
            "top_automation_targets": self.top_automation_targets,
            "graph_url": self.graph_url 
        }
