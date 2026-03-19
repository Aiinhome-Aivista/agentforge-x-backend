"""
Analysis service: orchestrates the full pipeline.
1. Parse uploaded file(s)
2. Run 3-pass Mistral analysis
3. Persist to ArangoDB with graph edges
4. Return structured AnalysisResult
"""

import logging
from typing import List, Tuple, Dict, Any
from pyvis.network import Network
import os
from app.parsers.file_parser import parse_file, detect_source_type
from app.core.mistral_client import get_mistral_client
from app.db.vector_service import store_embeddings
from app.models.models import (
    ProcessDocument, ProcessStep, AutomationSuggestion,
    ERPModule, KeyInsight, AnalysisResult
)
from app.db.arango import get_db, COLLECTIONS, EDGE_COLLECTIONS

logger = logging.getLogger(__name__)


class AnalysisService:

    def analyze(
        self,
        files: List[Tuple[bytes, str]]   # list of (file_bytes, filename)
    ) -> AnalysisResult:
        """
        Full pipeline: parse → extract → score → suggest → persist → return.
        Multiple files are concatenated as context before analysis.
        """
        db = get_db()
        llm = get_mistral_client()

        # ── Step 1: Parse all files ──────────────────────────────────────────
        combined_text_parts = []
        combined_metadata = {}
        primary_file = files[0][1]
        primary_source_type = detect_source_type(primary_file)

        for file_bytes, filename in files:
            text, meta = parse_file(file_bytes, filename)
            combined_text_parts.append(f"=== File: {filename} ===\n{text}")
            combined_metadata[filename] = meta
            logger.info(f"Parsed {filename}: {len(text)} chars")

        combined_text = "\n\n".join(combined_text_parts)

        # ── Step 2: LLM Pass 1 — Extraction ─────────────────────────────────
        extracted = llm.extract_process(combined_text, primary_source_type, primary_file)

        process_title = extracted.get("process_title", "Unnamed Process")
        process_desc = extracted.get("process_description", "")
        raw_steps = extracted.get("steps", [])
        raw_insights = extracted.get("key_insights", [])
        raw_erp_modules = extracted.get("erp_modules_identified", [])
        erp_system = extracted.get("erp_system")

        # ── Step 3: LLM Pass 2 — Automation Scoring ─────────────────────────
        scores = llm.score_automation(raw_steps, f"{process_title}: {process_desc}")

        # Merge scores into steps
        # Normalize step_number to int for reliable matching
        score_map = {int(s.get("step_number")): s for s in scores if s.get("step_number") is not None}
        for step in raw_steps:
            num = step.get("step_number")
            if num is not None and int(num) in score_map:
                matched = score_map[int(num)]
                step["automation_potential"] = matched.get("automation_potential", 50)
                step["automation_reasoning"] = matched.get("automation_reasoning", "")
                step["quick_win"] = matched.get("quick_win", False)
        # ← ADD HERE, after the loop
        logger.debug(f"Score map keys: {list(score_map.keys())}")
        logger.debug(f"Step numbers: {[s.get('step_number') for s in raw_steps]}")
        logger.debug(f"Potentials after merge: {[s.get('automation_potential') for s in raw_steps]}")

        # ── Step 4: LLM Pass 3 — Agentic Suggestions ────────────────────────
        raw_suggestions = llm.generate_suggestions(raw_steps, scores, process_title)

        # ── Step 5: LLM Pass 4 — Graph Relationships ────────────────────────
        relationships = llm.extract_relationships(process_title, raw_steps, raw_erp_modules)

        # ── Step 6: Calculate aggregate automation score ─────────────────────
        potentials = [s.get("automation_potential", 0) for s in raw_steps]
        avg_score = round(sum(potentials) / len(potentials), 1) if potentials else 0

        # ── Step 7: Build domain objects ─────────────────────────────────────
        process_doc = ProcessDocument(
            title=process_title,
            description=process_desc,
            source_type=primary_source_type,
            raw_text=combined_text,
            automation_score=avg_score,
            status="complete",
            erp_system=erp_system,
            file_name=primary_file,
        )

        step_objects: List[ProcessStep] = []
        step_key_map: Dict[int, str] = {}   # step_number → _key

        for raw in raw_steps:
            step = ProcessStep(
                process_key=process_doc._key,
                step_number=raw.get("step_number", 0),
                title=raw.get("title", ""),
                description=raw.get("description", ""),
                actor=raw.get("actor", "Unknown"),
                step_type=raw.get("step_type", "manual"),
                automation_potential=raw.get("automation_potential", 0),
                inputs=raw.get("inputs", []),
                outputs=raw.get("outputs", []),
                pain_points=raw.get("pain_points", []),
                erp_module=raw.get("erp_module"),
                duration_estimate=raw.get("duration_estimate"),
            )
            step_objects.append(step)
            step_key_map[step.step_number] = step._key

        # Map suggestions to step keys
        suggestion_objects: List[AutomationSuggestion] = []
        for raw in raw_suggestions:
            step_num = raw.get("step_number", 0)
            step_key = step_key_map.get(step_num, "")
            sug = AutomationSuggestion(
                process_key=process_doc._key,
                step_key=step_key,
                title=raw.get("title", ""),
                description=raw.get("description", ""),
                agent_type=raw.get("agent_type", "workflow_automation"),
                implementation=raw.get("implementation", ""),
                accuracy_estimate=raw.get("accuracy_estimate", 80),
                execution_speed=raw.get("execution_speed", "fast"),
                effort_level=raw.get("effort_level", "medium"),
                roi_impact=raw.get("roi_impact", "medium"),
                technologies=raw.get("technologies", []),
                prerequisites=raw.get("prerequisites", []),
            )
            suggestion_objects.append(sug)

        erp_module_objects: List[ERPModule] = []
        erp_module_key_map: Dict[str, str] = {}
        for raw in raw_erp_modules:
            mod = ERPModule(
                process_key=process_doc._key,
                module_name=raw.get("module_name", ""),
                erp_system=erp_system or "Unknown",
                source_file=primary_file,
                description=raw.get("description", ""),
                tables_identified=raw.get("tables_identified", []),
                fields_identified=raw.get("fields_identified", []),
            )
            erp_module_objects.append(mod)
            erp_module_key_map[mod.module_name] = mod._key

        insight_objects = [
            KeyInsight(
                text=i.get("text", ""),
                category=i.get("category", "automation"),
                impact=i.get("impact", "medium"),
            )
            for i in raw_insights
        ]

        # Top automation targets
        top_targets = sorted(
            [{"title": s.title, "actor": s.actor, "automation_potential": s.automation_potential}
             for s in step_objects],
            key=lambda x: x["automation_potential"],
            reverse=True
        )[:5]

        # ── Step 8: Persist to ArangoDB ──────────────────────────────────────
        self._persist(
            db, process_doc, step_objects, suggestion_objects,
            erp_module_objects, relationships, step_key_map, erp_module_key_map
        )
        # ── Step 9: Store in Vector DB (ChromaDB) ────────────────────────────
        try:
            print("DEBUG: Calling store_embeddings()")
            store_embeddings(process_doc, step_objects, insight_objects)
            logger.info("Stored embeddings in VectorDB")
        except Exception as e:
            logger.warning(f"VectorDB storage failed: {e}")

        # 🔥 Generate graph HTML file
        generate_graph_html(process_doc._key, step_objects, relationships)   

        # 🔥 ALWAYS generate graph_url (outside try-except)
        import os

        BASE_URL = os.getenv("BASE_URL")

        graph_folder = f"graphs/{process_doc._key}"
        os.makedirs(graph_folder, exist_ok=True)

        html_filename = "graph.html"
        rel_path = graph_folder

        graph_url = f"{BASE_URL}/{rel_path.replace(os.sep, '/')}/{html_filename}"

        return AnalysisResult(
            process=process_doc,
            steps=step_objects,
            suggestions=suggestion_objects,
            erp_modules=erp_module_objects,
            key_insights=insight_objects,
            top_automation_targets=top_targets,
            graph_url=graph_url
        )

    def _persist(self, db, process_doc, steps, suggestions,
                 erp_modules, relationships, step_key_map, erp_module_key_map):
        """Write all objects and graph edges to ArangoDB."""
        try:
            col = db.collection
            graph = db.graph()

            # Insert vertex documents
            col(COLLECTIONS["documents"]).insert(process_doc.to_doc(), overwrite=True)
            proc_id = f"{COLLECTIONS['documents']}/{process_doc._key}"

            for step in steps:
                col(COLLECTIONS["steps"]).insert(step.to_doc(), overwrite=True)

            for sug in suggestions:
                col(COLLECTIONS["suggestions"]).insert(sug.to_doc(), overwrite=True)

            for mod in erp_modules:
                col(COLLECTIONS["erp_modules"]).insert(mod.to_doc(), overwrite=True)

            # Insert edges: process → steps
            ec_has_step = graph.edge_collection(EDGE_COLLECTIONS["has_step"])
            for step in steps:
                step_id = f"{COLLECTIONS['steps']}/{step._key}"
                ec_has_step.insert({
                    "_from": proc_id,
                    "_to": step_id,
                    "step_number": step.step_number,
                })

            # Insert edges: step → next step (sequence)
            ec_seq = graph.edge_collection(EDGE_COLLECTIONS["step_sequence"])
            for seq in relationships.get("step_sequences", []):
                from_key = step_key_map.get(seq.get("from_step"))
                to_key = step_key_map.get(seq.get("to_step"))
                if from_key and to_key:
                    ec_seq.insert({
                        "_from": f"{COLLECTIONS['steps']}/{from_key}",
                        "_to": f"{COLLECTIONS['steps']}/{to_key}",
                        "relationship": seq.get("relationship", "leads_to"),
                        "condition": seq.get("condition"),
                    })

            # Insert edges: step → suggestion
            ec_sug = graph.edge_collection(EDGE_COLLECTIONS["triggers_suggestion"])
            for sug in suggestions:
                if sug.step_key:
                    ec_sug.insert({
                        "_from": f"{COLLECTIONS['steps']}/{sug.step_key}",
                        "_to": f"{COLLECTIONS['suggestions']}/{sug._key}",
                    })

            # Insert edges: process → erp_module
            ec_mod = graph.edge_collection(EDGE_COLLECTIONS["belongs_to_module"])
            for mod in erp_modules:
                ec_mod.insert({
                    "_from": proc_id,
                    "_to": f"{COLLECTIONS['erp_modules']}/{mod._key}",
                })

            # Insert edges: erp_module → erp_module
            ec_mod_rel = graph.edge_collection(EDGE_COLLECTIONS["module_relation"])
            for rel in relationships.get("module_relationships", []):
                from_key = erp_module_key_map.get(rel.get("from_module"))
                to_key = erp_module_key_map.get(rel.get("to_module"))
                if from_key and to_key:
                    ec_mod_rel.insert({
                        "_from": f"{COLLECTIONS['erp_modules']}/{from_key}",
                        "_to": f"{COLLECTIONS['erp_modules']}/{to_key}",
                        "relationship": rel.get("relationship"),
                    })

            logger.info(f"Persisted process {process_doc._key} to ArangoDB")

        except Exception as e:
            logger.error(f"ArangoDB persistence error: {e}", exc_info=True)
            # Don't fail the analysis if persistence fails
            pass

    def get_process(self, process_key: str) -> Dict[str, Any]:
        """Fetch a full process with all related data from ArangoDB."""
        db = get_db()
        col = db.collection

        process = col(COLLECTIONS["documents"]).get(process_key)
        if not process:
            return None

        steps = list(db.aql(
            "FOR s IN process_steps FILTER s.process_key == @key SORT s.step_number RETURN s",
            {"key": process_key}
        ))
        suggestions = list(db.aql(
            "FOR s IN automation_suggestions FILTER s.process_key == @key RETURN s",
            {"key": process_key}
        ))
        erp_modules = list(db.aql(
            "FOR m IN erp_modules FILTER m.process_key == @key RETURN m",
            {"key": process_key}
        ))

        potentials = [s.get("automation_potential", 0) for s in steps]
        top_targets = sorted(
            [{"title": s["title"], "actor": s["actor"],
              "automation_potential": s["automation_potential"]} for s in steps],
            key=lambda x: x["automation_potential"], reverse=True
        )[:5]

        return {
            "process": {**process, "id": process["_key"]},
            "steps": [{**s, "id": s["_key"]} for s in steps],
            "suggestions": [{**s, "id": s["_key"]} for s in suggestions],
            "erp_modules": [{**m, "id": m["_key"]} for m in erp_modules],
            "top_automation_targets": top_targets,
        }

    def list_processes(self) -> List[Dict]:
        """List all processes."""
        db = get_db()
        docs = list(db.aql(
            "FOR p IN processes SORT p.created_at DESC LIMIT 50 RETURN p"
        ))
        return [{**d, "id": d["_key"]} for d in docs]

def generate_graph_html(process_key, steps, relationships):

    graph_folder = f"graphs/{process_key}"
    os.makedirs(graph_folder, exist_ok=True)

    file_path = os.path.join(graph_folder, "graph.html")

    net = Network(height="750px", width="100%", directed=True)

    # 🔹 Add nodes
    for step in steps:
        net.add_node(
            step.step_number,
            label=step.title,
            title=step.description,
            color="#97c2fc"
        )

    # 🔹 Add edges
    step_sequences = relationships.get("step_sequences", [])

    if step_sequences:
        for rel in step_sequences:
            net.add_edge(
                rel.get("from_step"),
                rel.get("to_step"),
                label=rel.get("relationship", "")
            )
    else:
        for i in range(len(steps) - 1):
            net.add_edge(
                steps[i].step_number,
                steps[i + 1].step_number
            )

    # 🔥 Physics (important for layout)
    net.barnes_hut()

    # ✅ FIXED
    net.write_html(file_path)

    return file_path

analysis_service = AnalysisService()
