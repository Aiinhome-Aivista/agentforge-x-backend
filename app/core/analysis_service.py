import logging
import uuid
from typing import List, Tuple, Dict, Any
# pyrefly: ignore [missing-import]
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
import json
from app.db.db_connection import get_mysql_connection
from app.core.toc_analyzer import TOCAnalyzer
from app.prompts.prompts import build_architecture_prompt, build_react_flow_prompt


logger = logging.getLogger(__name__)


import re

from datetime import datetime

def save_uploaded_files(files: List[Tuple[bytes, str]]) -> str:
    """
    Saves uploaded files into uploads/<timestamp>/ with original filenames.
    Returns the folder path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join("uploads", timestamp)

    os.makedirs(base_folder, exist_ok=True)

    for file_bytes, filename in files:
        file_path = os.path.join(base_folder, filename)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

    return base_folder


def escape_braces(text: str) -> str:
    return str(text).replace("{", "{{").replace("}", "}}")


def slugify(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

used_edge_ids = set()

def unique_edge_id(base):
    i = 1
    new_id = base
    while new_id in used_edge_ids:
        new_id = f"{base}-{i}"
        i += 1
    used_edge_ids.add(new_id)
    return new_id

def serialize(obj):
    if isinstance(obj, list):
        return [serialize(o) for o in obj]
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj

# ─────────────────────────────────────────────────────────────────────────────
# Workflow type detector
# ─────────────────────────────────────────────────────────────────────────────
_INVENTORY_KEYWORDS = [
    "inventory", "stock", "sales order", "purchase order",
    "procurement", "warehouse", "shipment", "order acceptance",
    "order management", "fulfillment",
]


def _detect_workflow_type(process_title: str) -> str:
    t = process_title.lower()
    if any(k in t for k in _INVENTORY_KEYWORDS):
        return "inventory"
    return "generic"


class AnalysisService:

    def analyze(
        self,
        files: List[Tuple[bytes, str]],
        user_input: str = "",
        session_id: str = None
    ) -> AnalysisResult:
         
         # 🔥 NEW: Check existing session in MySQL
        if not session_id:
            session_id = str(uuid.uuid4())
            try:
                mysql_db = get_mysql_connection()
                cursor = mysql_db.cursor(dictionary=True)

                cursor.execute("""
                    SELECT * FROM analysis_results
                    WHERE session_id = %s
                    LIMIT 1
                """, (session_id,))

                existing = cursor.fetchone()

                cursor.close()
                mysql_db.close()

                if existing:
                    logger.info(f"✅ Existing result found for session_id={session_id}")

                    return AnalysisResult(
                        process=json.loads(existing["process"]),
                        steps=json.loads(existing["steps"]),
                        suggestions=json.loads(existing["suggestions"]),
                        erp_modules=json.loads(existing["erp_modules"]),
                        key_insights=json.loads(existing["key_insights"]),
                        top_automation_targets=json.loads(existing["automation_targets"]),
                        graph_url=existing["graph_url"],
                        toc_analysis=json.loads(existing["toc_analysis"]),
                        session_id=session_id
                    )

            except Exception as e:
                logger.error(f"MySQL fetch failed: {e}")

        # 🔥 NEW: Save uploaded raw files
        upload_folder = save_uploaded_files(files)
        logger.info(f"Files saved in: {upload_folder}")
        
        db  = get_db()
        llm = get_mistral_client()


        # Step 1: Parse all inputs
        combined_text_parts = []
        combined_metadata   = {}

        # Detect primary source
        if files:
            primary_file = files[0][1]
            primary_source_type = detect_source_type(primary_file)
        else:
            primary_file = "web_input.txt"
            primary_source_type = "text"

        # Parse uploaded files
        if files:
            for file_bytes, filename in files:
                text, meta = parse_file(file_bytes, filename)

                combined_text_parts.append(f"=== File: {filename} ===\n{text}")
                combined_metadata[filename] = meta

                logger.info(f"Parsed {filename}: {len(text)} chars")

        # Add user input
        if user_input:
            combined_text_parts.append(f"=== USER INPUT ===\n{user_input}")

        # 🔥 CRITICAL: Must have at least one input
        if not combined_text_parts:
            raise ValueError("No input to analyze")

        # Final combined text
        combined_text = "\n\n".join(combined_text_parts)

        # Optional instruction wrapper
        if user_input:
            combined_text = (
                f"=== USER INSTRUCTIONS ===\n{user_input}\n\n"
                f"=== CONTEXT DATA ===\n{combined_text}"
            )
            
        # 🔥 NEW: Validate Relevance
        relevance = llm.validate_relevance(user_input, combined_text)
        if not relevance.get("is_relevant", True):
            error_msg = relevance.get("error", "Irrelevant file.")
            solution = relevance.get("recommended_solution", "Upload a relevant file.")
            
            # Minimal ProcessDocument for DB
            err_doc = ProcessDocument(
                title="Irrelevant File",
                description="Validation Failed",
                source_type=primary_source_type,
                raw_text="",
                automation_score=0,
                status="error",
                error=error_msg
            )
            
            # Save to DB
            try:
                mysql_db = get_mysql_connection()
                cursor = mysql_db.cursor()
                cursor.execute("""
                    INSERT INTO analysis_results
                    (erp_modules, graph_url, key_insights, process, steps, suggestions,
                     automation_targets, toc_analysis, session_id, upload_folder)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    "[]", "", "[]", json.dumps(err_doc.to_api()), "[]", "[]", "[]", "{}", session_id, upload_folder
                ))
                mysql_db.commit()
                cursor.close()
                mysql_db.close()
            except Exception as e:
                logger.error(f"MySQL error insert failed: {e}")
                
            raise ValueError(f"IRRELEVANT_FILE|{error_msg}|{solution}")



        # Step 2: LLM Pass 1 — Extraction
        extracted = llm.extract_process(combined_text, primary_source_type, primary_file)


        process_title   = extracted.get("process_title", "Unnamed Process")
        process_desc    = extracted.get("process_description", "")
        raw_steps       = extracted.get("steps", [])
        raw_insights    = extracted.get("key_insights", [])
        raw_erp_modules = extracted.get("erp_modules_identified", [])
        erp_system      = extracted.get("erp_system")


        # Step 2b: Micro-process decomposition
        micro_data = llm.decompose_micro_process(raw_steps)


        for step in raw_steps:
            match = next(
                


            (m for m in micro_data if isinstance(m, dict) and m.get("step_number") == step.get("step_number")),
                {}
            )
            step["micro_steps"] = match.get("micro_steps", [])
            step["decisions"] = match.get("decisions", [])


        # Step 3: LLM Pass 2 — Automation Scoring
        safe_context = escape_braces(f"{process_title}: {process_desc}")


        scores = llm.score_automation(raw_steps, safe_context)






        score_map = {int(s.get("step_number")): s for s in scores if s.get("step_number") is not None}
        for step in raw_steps:
            num = step.get("step_number")
            if num is not None and int(num) in score_map:
                matched = score_map[int(num)]
                step["automation_potential"] = matched.get("automation_potential", 50)
                step["automation_reasoning"] = matched.get("automation_reasoning", "")
                step["quick_win"]            = matched.get("quick_win", False)


        # Step 3b: Decision Analysis (NEW PASS)
        if hasattr(llm, "analyze_decisions"):
            decision_data = llm.analyze_decisions(raw_steps)
        else:
            logger.warning("analyze_decisions not found")
            decision_data = []


        # Merge back into steps
        for step in raw_steps:
            match = next(
                (d for d in decision_data if d.get("step_number") == step.get("step_number")),
                {}
            )
            step["decision_type"] = match.get("decision_type")
            step["decision_confidence"] = match.get("confidence")
            step["decision_automation_feasibility"] = match.get("automation_feasibility")        


        # Step 4: LLM Pass 3 — Agentic Suggestions
        suggestions_response = llm.generate_suggestions(raw_steps, scores, process_title)
        if isinstance(suggestions_response, dict):
            raw_suggestions = suggestions_response.get("suggestions", [])
        elif isinstance(suggestions_response, list):
            raw_suggestions = suggestions_response
        else:
            raw_suggestions = []


        # Step 4b: Theory of Constraints
        toc_dict = {}
        try:
            toc_analyzer = TOCAnalyzer()
            toc_result   = toc_analyzer.analyze(
                process_key="temp",
                process_title=process_title,
                steps=raw_steps,
                suggestions=raw_suggestions,
            )
            toc_dict = toc_analyzer.to_dict(toc_result)
            logger.info(
                f"TOC — primary constraint: Step "
                f"{toc_result.primary_constraint.step_number if toc_result.primary_constraint else 'N/A'}, "
                f"bottleneck={toc_result.bottleneck_index}%"
            )
        except Exception as e:
            logger.warning(f"TOC analysis failed (non-critical): {e}")
        
        # Step 4c — Workflow Categorization (NEW)
        categorization = {}
        try:
            categorization = llm.categorize_workflow(
                process_title,
                raw_steps,
                raw_suggestions
            )
        except Exception as e:
            logger.warning(f"Categorization failed: {e}")


        # Step 5: LLM Pass 4 — Graph Relationships
        relationships = llm.extract_relationships(process_title, raw_steps, raw_erp_modules)


        # Step 6: Aggregate automation score
        potentials = [s.get("automation_potential", 0) for s in raw_steps]
        avg_score  = round(sum(potentials) / len(potentials), 1) if potentials else 0


        # Step 7: Build domain objects
        process_doc = ProcessDocument(
            _key=str(uuid.uuid4()).replace("-", ""),
            title=process_title,
            description=process_desc,
            source_type=primary_source_type,
            raw_text=combined_text,
            automation_score=avg_score,
            status="complete",
            erp_system=erp_system,
            file_name=primary_file,
        )


        if toc_dict:
            toc_dict["process_key"] = process_doc._key


        step_objects: List[ProcessStep] = []
        step_key_map: Dict[int, str]   = {}


        def resolve_step_type(step_type, automation_potential):
            if automation_potential >= 80: return "Higher Agentic intervention"
            if automation_potential >= 60: return "Human + AI Intervention"
            return "Higher Human intervention"


        def clean_step_number(val, index):
            try:    return int(str(val).strip().replace(".", ""))
            except: return index + 1


        for idx, raw in enumerate(raw_steps):
            step_num = clean_step_number(raw.get("step_number"), idx)
            if step_num in step_key_map:
                step_num = max(step_key_map.keys()) + 1


            step = ProcessStep(
                process_key=process_doc._key,
                step_number=step_num,
                title=raw.get("title", ""),
                description=raw.get("description", ""),
                actor=raw.get("actor", "Unknown"),
                lane=raw.get("lane", raw.get("actor")),   
                role_type=raw.get("role_type", "human"),
                step_type=resolve_step_type(raw.get("step_type", "manual"),raw.get("automation_potential", 0)),
                automation_potential=raw.get("automation_potential", 0),
                automation_reasoning=raw.get("automation_reasoning", ""),
                inputs=raw.get("inputs", []),
                outputs=raw.get("outputs", []),
                pain_points=raw.get("pain_points", []),
                erp_module=raw.get("erp_module"),
                duration_estimate=raw.get("duration_estimate"),
                micro_steps=raw.get("micro_steps", []),   
                decisions=raw.get("decisions", []),
            )
            step_objects.append(step)
            step_key_map[step_num] = step._key


        suggestion_objects: List[AutomationSuggestion] = []


        for raw in raw_suggestions:
            step_num       = raw.get("step_number", 0)
            auto_score     = raw.get("metrics", {}).get("automation_potential", 70)
            accuracy       = 90 if auto_score >= 85 else (80 if auto_score >= 70 else 65)
            step_data      = next(
                (s for s in raw_steps if int(s.get("step_number", 0)) == int(step_num)), {}
            )


            if "metrics" not in raw or not isinstance(raw["metrics"], dict):
                raw["metrics"] = {}
            raw["metrics"]["automation_potential"] = step_data.get("automation_potential", 0)
            raw["metrics"]["reason"]               = step_data.get("automation_reasoning", "")


            accuracy_reason = raw.get("accuracy_reason") or raw.get("description", "")


            if not step_num:
                title   = raw.get("title", "").lower()
                matched = next(
                    (s for s in step_objects if s.title.lower() in title or title in s.title.lower()), None
                )
                step_key = matched._key if matched else ""
            else:
                step_key = step_key_map.get(step_num, "")
                if not step_key:
                    title   = raw.get("title", "").lower()
                    matched = next(
                        (s for s in step_objects if s.title.lower() in title or title in s.title.lower()), None
                    )
                    step_key = matched._key if matched else None


            if not step_key:
                continue
            step_obj = next((s for s in step_objects if s._key == step_key), None)
            if step_obj and step_obj.step_type == "Higher Human intervention":
                continue


            suggestion_objects.append(AutomationSuggestion(
                process_key=process_doc._key,
                step_key=step_key,
                title=raw.get("title", ""),
                description=raw.get("description", ""),
                agent_type=raw.get("agent_type", "workflow_automation"),
                implementation=raw.get("implementation", ""),
                accuracy_estimate=accuracy,
                accuracy_reason=accuracy_reason,
                execution_speed=raw.get("execution_speed", "fast"),
                effort_level=raw.get("effort_level", "medium"),
                roi_impact=raw.get("roi_impact", "medium"),
                technologies=raw.get("technologies", []),
                prerequisites=raw.get("prerequisites", []),
                metrics=raw.get("metrics", {})
            ))


        erp_module_objects: List[ERPModule]  = []
        erp_module_key_map: Dict[str, str]   = {}
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


        top_targets = sorted(
            [{"title": s.title, "actor": s.actor, "automation_potential": s.automation_potential}
             for s in step_objects],
            key=lambda x: x["automation_potential"], reverse=True
        )[:5]


        # Step 8: Persist to ArangoDB
        self._persist(db, process_doc, step_objects, suggestion_objects,
                      erp_module_objects, relationships, step_key_map, erp_module_key_map)


        # Step 9: Vector DB
        try:
            store_embeddings(process_doc, step_objects, insight_objects)
            logger.info("Stored embeddings in VectorDB")
        except Exception as e:
            logger.warning(f"VectorDB storage failed: {e}")


        generate_graph_html(process_doc._key, step_objects, relationships)


        BASE_URL     = os.getenv("BASE_URL")
        graph_folder = f"graphs/{process_doc._key}"
        os.makedirs(graph_folder, exist_ok=True)
        graph_url    = f"{BASE_URL}/{graph_folder.replace(os.sep, '/')}/graph.html"


        # Step 10: MySQL
        try:
            mysql_db = get_mysql_connection()
            cursor   = mysql_db.cursor()
            cursor.execute("""
                INSERT INTO analysis_results
                (erp_modules, graph_url, key_insights, process, steps, suggestions,
                 automation_targets, toc_analysis,session_id ,upload_folder)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s,%s)
            """, (
                json.dumps(serialize(erp_module_objects)),
                graph_url,
                json.dumps(serialize(insight_objects)),
                json.dumps(process_doc.to_api()),
                json.dumps(serialize(step_objects)),
                json.dumps(serialize(suggestion_objects)),
                json.dumps(top_targets),
                json.dumps(toc_dict),
                session_id,
                upload_folder
            ))
            mysql_db.commit()
            cursor.close()
            mysql_db.close()
            logger.info("Stored in MySQL")
        except Exception as e:
            logger.error(f"MySQL insert failed: {e}")


        return AnalysisResult(
            process=process_doc,
            steps=step_objects,
            suggestions=suggestion_objects,
            erp_modules=erp_module_objects,
            key_insights=insight_objects,
            top_automation_targets=top_targets,
            graph_url=graph_url,
            workflow_layers=categorization,
            toc_analysis=toc_dict,
            session_id=session_id,
        )


    def _persist(self, db, process_doc, steps, suggestions,
                 erp_modules, relationships, step_key_map, erp_module_key_map):
        try:
            col   = db.collection
            graph = db.graph()


            col(COLLECTIONS["documents"]).insert(process_doc.to_doc(), overwrite=True)
            proc_id = f"{COLLECTIONS['documents']}/{process_doc._key}"


            for step in steps:
                col(COLLECTIONS["steps"]).insert(step.to_doc(), overwrite=True)
            for sug in suggestions:
                col(COLLECTIONS["suggestions"]).insert(sug.to_doc(), overwrite=True)
            for mod in erp_modules:
                col(COLLECTIONS["erp_modules"]).insert(mod.to_doc(), overwrite=True)


            ec_has_step = graph.edge_collection(EDGE_COLLECTIONS["has_step"])
            for step in steps:
                ec_has_step.insert({
                    "_from": proc_id,
                    "_to":   f"{COLLECTIONS['steps']}/{step._key}",
                    "step_number": step.step_number,
                })


            # ec_seq = graph.edge_collection(EDGE_COLLECTIONS["step_sequence", "decision_branch", "micro_flow",])
            # for seq in relationships.get("step_sequences", []):
            #     fk = step_key_map.get(seq.get("from_step"))
            #     tk = step_key_map.get(seq.get("to_step"))
            #     if fk and tk:
            #         ec_seq.insert({
            #             "_from": f"{COLLECTIONS['steps']}/{fk}",
            #             "_to":   f"{COLLECTIONS['steps']}/{tk}",
            #             "relationship": seq.get("relationship", "leads_to"),
            #             "condition":    seq.get("condition"),
            #         })


            edge_types = {
                "step_sequence": graph.edge_collection(EDGE_COLLECTIONS["step_sequence"]),
                "decision_branch": graph.edge_collection(EDGE_COLLECTIONS["decision_branch"]),
                "micro_flow": graph.edge_collection(EDGE_COLLECTIONS["micro_flow"]),
            }


            for seq in relationships.get("step_sequences", []):
                fk = step_key_map.get(seq.get("from_step"))
                tk = step_key_map.get(seq.get("to_step"))


                if not (fk and tk):
                    continue


                rel_type = seq.get("type", "step_sequence")  # default fallback


                ec = edge_types.get(rel_type)
                if not ec:
                    continue


                ec.insert({
                    "_from": f"{COLLECTIONS['steps']}/{fk}",
                    "_to":   f"{COLLECTIONS['steps']}/{tk}",
                    "relationship": seq.get("relationship", "leads_to"),
                    "condition": seq.get("condition"),
                })


            ec_sug = graph.edge_collection(EDGE_COLLECTIONS["triggers_suggestion"])
            for sug in suggestions:
                if sug.step_key:
                    ec_sug.insert({
                        "_from": f"{COLLECTIONS['steps']}/{sug.step_key}",
                        "_to":   f"{COLLECTIONS['suggestions']}/{sug._key}",
                    })


            ec_mod = graph.edge_collection(EDGE_COLLECTIONS["belongs_to_module"])
            for mod in erp_modules:
                ec_mod.insert({
                    "_from": proc_id,
                    "_to":   f"{COLLECTIONS['erp_modules']}/{mod._key}",
                })


            ec_mod_rel = graph.edge_collection(EDGE_COLLECTIONS["module_relation"])
            for rel in relationships.get("module_relationships", []):
                fk = erp_module_key_map.get(rel.get("from_module"))
                tk = erp_module_key_map.get(rel.get("to_module"))
                if fk and tk:
                    ec_mod_rel.insert({
                        "_from": f"{COLLECTIONS['erp_modules']}/{fk}",
                        "_to":   f"{COLLECTIONS['erp_modules']}/{tk}",
                        "relationship": rel.get("relationship"),
                    })


            logger.info(f"Persisted process {process_doc._key} to ArangoDB")
        except Exception as e:
            logger.error(f"ArangoDB persistence error: {e}", exc_info=True)


    def get_process(self, process_key: str) -> Dict[str, Any]:
        db  = get_db()
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


        top_targets = sorted(
            [{"title": s["title"], "actor": s["actor"],
              "automation_potential": s["automation_potential"]} for s in steps],
            key=lambda x: x["automation_potential"], reverse=True
        )[:5]


        return {
            "process":                {**process, "id": process["_key"]},
            "steps":                  [{**s, "id": s["_key"]} for s in steps],
            "suggestions":            [{**s, "id": s["_key"]} for s in suggestions],
            "erp_modules":            [{**m, "id": m["_key"]} for m in erp_modules],
            "top_automation_targets": top_targets,
        }


    def list_processes(self) -> List[Dict]:
        db   = get_db()
        docs = list(db.aql(
            "FOR p IN processes SORT p.created_at DESC LIMIT 50 RETURN p"
        ))
        return [{**d, "id": d["_key"]} for d in docs]

    # ─────────────────────────────────────────────────────────────────────────
    # get_react_flow_data
    # ─────────────────────────────────────────────────────────────────────────
    def get_react_flow_data(self, _key: str) -> Dict[str, Any]:
        db  = get_db()
        col = db.collection
        llm = get_mistral_client()


        # Resolve process key
        try:
            suggestion = col(COLLECTIONS["suggestions"]).get(_key)
        except Exception:
            suggestion = None


        if suggestion:
            actual_process_key = suggestion.get("process_key")
        else:
            try:
                erp_mod = col(COLLECTIONS["erp_modules"]).get(_key)
            except Exception:
                erp_mod = None
            actual_process_key = erp_mod.get("process_key") if erp_mod else _key


        steps = list(db.aql(
            "FOR s IN process_steps FILTER s.process_key == @key SORT s.step_number RETURN s",
            {"key": actual_process_key}
        ))
        suggestions_data = list(db.aql(
            "FOR s IN automation_suggestions FILTER s.process_key == @key RETURN s",
            {"key": actual_process_key}
        ))
        process       = col(COLLECTIONS["documents"]).get(actual_process_key)
        process_title = process.get("title", "Business Process") if process else "Business Process"


        wf_type = _detect_workflow_type(process_title)


        # ── Inventory / order workflows → deterministic agentic graph ─────────
        if wf_type == "inventory":
            logger.info("Using LLM lane-based workflow builder (inventory override)")

            result = llm.generate_react_flow(process_title, steps, suggestions_data)


            return result


        # ── Generic → try LLM, fall back to sequential ───────────────────────
        try:
            result = llm.generate_react_flow(process_title, steps, suggestions_data)

            # --- ADD FALLBACK UNWRAPPING ---
            if "lanes" not in result and "flow" not in result:
                for key, value in result.items():
                    if isinstance(value, dict) and "lanes" in value:
                        result = value
                        break

            if not result.get("lanes") or not result.get("flow"):
                raise ValueError(f"LLM returned invalid lane-based graph: {list(result.keys())}")

            # count nodes inside lanes
            node_count = sum(len(l.get("nodes", [])) for l in result.get("lanes", []))

            if node_count < len(steps):
                raise ValueError(f"Incomplete: {node_count}/{len(steps)} nodes")

            logger.info(f"Lane graph: {len(result['lanes'])} lanes, {len(result['flow'])} flows")

            return result

        except Exception as e:
            logger.error(f"LLM graph failed: {e} — using sequential fallback")
            return self._build_sequential_fallback(steps, suggestions_data)


    

    # ─────────────────────────────────────────────────────────────────────────
    # _build_agentic_workflow_graph
    #
    # Produces the full Image-2 layout deterministically:
    #
    #   Sales Order Agent lane  (validates order)
    #        │ VALID                    │ FAILED → [Order Rejected]
    #        ▼
    #   Inventory Agent lane
    #     ② Check Stock
    #     ③ Analyze Allocation
    #     ④ Determine Availability
    #        │
    #   ◆ Stock Available?
    #     YES ↙                NO/PARTIAL ↘
    #   Accept Flow lane        Alternatives Flow lane  ← Reorder Agent
    # ─────────────────────────────────────────────────────────────────────────
    def _build_agentic_workflow_graph(
        self, process_title: str, steps: list, suggestions: list
    ) -> Dict[str, Any]:


        nodes: list = []
        edges: list = []


        # ── Helpers ────────────────────────────────────────────────────────────
        def accent(p): return "#10B981" if p >= 80 else ("#F59E0B" if p >= 60 else "#EF4444")


        def grp(nid, label, x, y, w, h, icon="Database", color="#6366F1"):
            return {
                "id": nid, "type": "agentGroupNode",
                "position": {"x": x, "y": y},
                "style": {"width": w, "height": h, "zIndex": 0},
                "data": {"label": label, "icon": icon, "accentColor": color},
            }


        def proc(nid, label, x, y, sd=None, parent=None, color="#10B981"):
            d = sd or {}
            n = {
                "id": nid, "type": "processNode",
                "position": {"x": x, "y": y},
                "style": {"width": 340, "zIndex": 2},
                "data": {
                    "label":               label,
                    "actor":               d.get("actor", "System"),
                    "stepNumber":          d.get("step_number", 0),
                    "automationPotential": d.get("automation_potential", 80),
                    "stepType":            d.get("step_type", "system"),
                    "inputs":              d.get("inputs", []),
                    "outputs":             d.get("outputs", []),
                    "painPoints":          d.get("pain_points", []),
                    "duration":            d.get("duration_estimate", ""),
                    "erpModule":           d.get("erp_module", ""),
                    "description":         d.get("description", label),
                    "accentColor":         color,
                },
            }
            if parent:
                n["parentNode"] = parent
                n["extent"]     = "parent"
            return n


        def decision(nid, label, x, y):
            return {
                "id": nid, "type": "decisionNode",
                "position": {"x": x, "y": y},
                "style": {"width": 200, "zIndex": 5},
                "data": {"label": label, "accentColor": "#F59E0B"},
            }


        def agt(nid, title, x, y, desc="", tasks=None, atype="workflow_automation", color="#3B82F6"):
            return {
                "id": nid, "type": "agentNode",
                "position": {"x": x, "y": y},
                "style": {"width": 300, "zIndex": 2},
                "data": {
                    "title": title, "description": desc,
                    "tasks": tasks or [],
                    "agentType": atype,
                    "accuracy": 88, "roiImpact": "high", "effortLevel": "medium",
                    "technologies": ["ERP API", "Python", "n8n"],
                    "accentColor": color,
                },
            }


        def edg(eid, src, tgt, label="", stroke="#4B5563",
                dashed=False, animated=True, sw=2):
            e = {
                "id": eid, "source": src, "target": tgt,
                "type": "smoothstep", "animated": animated,
                "label": label, "zIndex": 10,
                "style": {"stroke": stroke, "strokeWidth": sw, "zIndex": 10},
            }
            if dashed:
                e["style"]["strokeDasharray"] = "5 5"
            return e


        # ── Keyword step matcher ───────────────────────────────────────────────
        def find(kws, fallback_idx=0):
            for kw in kws:
                for s in steps:
                    if kw.lower() in (s.get("title","") + s.get("description","")).lower():
                        return s
            return steps[fallback_idx] if steps and fallback_idx < len(steps) else {}


        # ── Map DB steps to semantic roles ─────────────────────────────────────
        s_val    = find(["validate", "order data", "completeness", "sales order"], 0)
        s_qdb    = find(["query", "inventory db", "database"], 1)
        s_chk    = find(["check stock", "stock level"], 2)
        s_ana    = find(["analy", "allocation", "reserved"], 3)
        s_det    = find(["determine", "availability", "atp", "available-to-promise"], 4)
        s_res    = find(["reserve", "lock qty"], 5)
        s_con    = find(["confirm avail"], 6)
        s_upd    = find(["update", "status", "ready"], 7)
        s_acc    = find(["accept order"], 8)
        s_not    = find(["notify", "customer", "notification"], 9)
        s_alt    = find(["alternative", "other warehouse", "supplier"], 5)
        s_lead   = find(["lead time", "calculate lead"], 6)
        s_gopt   = find(["generate option"], 7)
        s_prop   = find(["propose", "partial ship", "backorder"], 8)


        # ── Agent task lookup ──────────────────────────────────────────────────
        def sug_for(kws):
            for sg in suggestions:
                t = (sg.get("title","") + " " + sg.get("description","")).lower()
                if any(k.lower() in t for k in kws):
                    return sg
            return {}


        so_s  = sug_for(["validate", "sales order"])
        inv_s = sug_for(["inventory", "stock", "check"])
        ro_s  = sug_for(["reorder", "alternative", "lead time"])


        # ═══════════════════════════════════════════════════════════════════════
        # LANE 1 — Sales Order Agent
        # ═══════════════════════════════════════════════════════════════════════
        nodes.append(grp("group-soa", "Sales Order Agent",
                         x=60, y=40, w=400, h=200, icon="UserCircle", color="#6366F1"))
        nodes.append(proc("step-validate",
                          s_val.get("title", "Validate Order Data"),
                          x=20, y=60, sd=s_val, parent="group-soa",
                          color=accent(s_val.get("automation_potential", 85))))


        # Rejection (standalone)
        nodes.append({
            "id": "node-rejected", "type": "processNode",
            "position": {"x": 560, "y": 60},
            "style": {"width": 280, "zIndex": 5},
            "data": {
                "label": "Order Rejected / Correction Needed",
                "actor": "System", "stepType": "notification",
                "automationPotential": 90, "accentColor": "#EF4444",
                "description": "Notification sent to Sales team",
                "inputs": ["Failed validation"], "outputs": ["Rejection notice"],
                "painPoints": [], "duration": "instant", "erpModule": "",
            },
        })


        # ═══════════════════════════════════════════════════════════════════════
        # LANE 2 — Inventory Agent
        # ═══════════════════════════════════════════════════════════════════════
        nodes.append(grp("group-inv", "Inventory Agent",
                         x=60, y=280, w=400, h=430, icon="Database", color="#6366F1"))


        for nid, sd, default_lbl, yy in [
            ("step-qdb",  s_qdb, "Query Inventory DB",        60),
            ("step-chk",  s_chk, "Check Stock Levels",       160),
            ("step-ana",  s_ana, "Analyze Allocation",        260),
            ("step-det",  s_det, "Determine Availability",   350),
        ]:
            nodes.append(proc(nid, sd.get("title", default_lbl),
                              x=20, y=yy, sd=sd, parent="group-inv",
                              color=accent(sd.get("automation_potential", 85))))


        # ═══════════════════════════════════════════════════════════════════════
        # DECISION NODE
        # ═══════════════════════════════════════════════════════════════════════
        nodes.append(decision("node-decision", "Stock Available?", x=210, y=750))


        # ═══════════════════════════════════════════════════════════════════════
        # LANE 3 — YES / Accept Flow
        # ═══════════════════════════════════════════════════════════════════════
        nodes.append(grp("group-yes", "Accept Flow",
                         x=60, y=970, w=400, h=580, icon="Layers", color="#16A34A"))


        for nid, sd, default_lbl, yy in [
            ("step-res",  s_res, "Reserve Stock",             60),
            ("step-con",  s_con, "Confirm Availability",     160),
            ("step-upd",  s_upd, "Update Order Status",      260),
            ("step-acc",  s_acc, "Accept Order",             360),
            ("step-not",  s_not, "Notify Customer / ERP",   460),
        ]:
            nodes.append(proc(nid, sd.get("title", default_lbl),
                              x=20, y=yy, sd=sd, parent="group-yes",
                              color="#10B981"))


        # ═══════════════════════════════════════════════════════════════════════
        # LANE 4 — NO/PARTIAL / Alternatives Flow
        # ═══════════════════════════════════════════════════════════════════════
        nodes.append(grp("group-no", "Alternatives Flow",
                         x=530, y=970, w=400, h=640, icon="Layers", color="#F59E0B"))


        for nid, sd, default_lbl, yy in [
            ("step-alt",  s_alt,  "Consider Alternatives",   60),
            ("step-lead", s_lead, "Calculate Lead Times",   160),
            ("step-gopt", s_gopt, "Generate Options",        260),
            ("step-prop", s_prop, "Propose Options to Sales",360),
            ("step-upd2", s_upd,  "Update Order Status",    460),
            ("step-acc2", s_acc,  "Accept Order",            550),
        ]:
            nodes.append(proc(nid, sd.get("title", default_lbl),
                              x=20, y=yy, sd=sd, parent="group-no",
                              color="#F59E0B"))


        # ═══════════════════════════════════════════════════════════════════════
        # AGENT NODES
        # ═══════════════════════════════════════════════════════════════════════
        nodes.append(agt(
            "agent-soa", "Sales Order Agent", x=1020, y=80,
            desc=so_s.get("description", "Validates incoming sales orders via API"),
            tasks=so_s.get("tasks") or [
                "Check order completeness",
                "Validate Product Catalog via API",
                "Route valid orders to Inventory Agent",
            ],
            color="#3B82F6",
        ))
        nodes.append(agt(
            "agent-inv", "Inventory Agent", x=1020, y=400,
            desc=inv_s.get("description", "Checks and reserves stock levels"),
            tasks=inv_s.get("tasks") or [
                "Query main Warehouse DB",
                "Check reserved stock & pending shipments",
                "Calculate Available-to-Promise (ATP)",
            ],
            color="#3B82F6",
        ))
        nodes.append(agt(
            "agent-ro", "Reorder Agent", x=1020, y=860,
            desc=ro_s.get("description", "Finds alternatives when stock is insufficient"),
            tasks=ro_s.get("tasks") or [
                "Check other warehouses & lead times",
                "Calculate supplier availability",
                "Propose partial ship / backorder options",
            ],
            atype="reorder",
            color="#8B5CF6",
        ))


        # ═══════════════════════════════════════════════════════════════════════
        # EDGES
        # ═══════════════════════════════════════════════════════════════════════


        # Agent triggers validate
        edges.append(edg("e-soa-val",    "agent-soa",    "step-validate",
                         label="triggers", stroke="#3B82F6", dashed=True))


        # Validate → VALID → Query DB
        edges.append(edg("e-val-qdb",    "step-validate","step-qdb",
                         label="VALID",    stroke="#16A34A"))
        # Validate → FAILED → Rejected
        edges.append(edg("e-val-rej",    "step-validate","node-rejected",
                         label="FAILED",   stroke="#EF4444", dashed=True))


        # Inventory agent sub-steps
        for eid, src, tgt in [
            ("e-qdb-chk",  "step-qdb",  "step-chk"),
            ("e-chk-ana",  "step-chk",  "step-ana"),
            ("e-ana-det",  "step-ana",  "step-det"),
        ]:
            edges.append(edg(eid, src, tgt, stroke="#4B5563"))


        # Determine → Decision
        edges.append(edg("e-det-dec",    "step-det",     "node-decision", stroke="#4B5563"))


        # Decision branches
        edges.append(edg("e-dec-yes",    "node-decision","step-res",
                         label="YES",          stroke="#16A34A", sw=2))
        edges.append(edg("e-dec-no",     "node-decision","step-alt",
                         label="NO / PARTIAL", stroke="#EF4444", sw=2))


        # YES path sequential
        for eid, src, tgt in [
            ("e-res-con",  "step-res",  "step-con"),
            ("e-con-upd",  "step-con",  "step-upd"),
            ("e-upd-acc",  "step-upd",  "step-acc"),
            ("e-acc-not",  "step-acc",  "step-not"),
        ]:
            edges.append(edg(eid, src, tgt, stroke="#16A34A"))


        # NO path sequential
        for eid, src, tgt in [
            ("e-alt-lead", "step-alt",  "step-lead"),
            ("e-lead-gopt","step-lead", "step-gopt"),
            ("e-gopt-prop","step-gopt", "step-prop"),
            ("e-prop-upd2","step-prop", "step-upd2"),
            ("e-upd2-acc2","step-upd2", "step-acc2"),
        ]:
            edges.append(edg(eid, src, tgt, stroke="#F59E0B"))


        # Agent automation edges (dashed purple)
        for eid, src, tgt, lbl in [
            ("e-soa-val2",  "agent-soa", "step-validate","validates"),
            ("e-inv-chk",   "agent-inv", "step-chk",     "checks"),
            ("e-inv-ana",   "agent-inv", "step-ana",     "analyzes"),
            ("e-inv-det",   "agent-inv", "step-det",     "continues flow"),
            ("e-ro-alt",    "agent-ro",  "step-alt",     "triggers"),
            ("e-ro-lead",   "agent-ro",  "step-lead",    "calculates"),
        ]:
            edges.append(edg(eid, src, tgt, label=lbl, stroke="#8B5CF6", dashed=True))


        # Catch unmapped extra steps — add inside inventory lane
        mapped_nums = {
            s.get("step_number")
            for s in [s_val,s_qdb,s_chk,s_ana,s_det,s_res,s_con,s_upd,s_acc,s_not,
                      s_alt,s_lead,s_gopt,s_prop]
            if s
        }
        extra = [s for s in steps if s.get("step_number") not in mapped_nums]
        prev  = "step-det"
        for i, es in enumerate(extra):
            nid = f"step-extra-{i}"
            nodes.append(proc(nid, es.get("title", f"Step {es.get('step_number')}"),
                              x=20, y=420 + i*110, sd=es, parent="group-inv",
                              color=accent(es.get("automation_potential", 50))))
            edges.append(edg(f"e-extra-{i}", prev, nid, stroke="#4B5563"))
            prev = nid


        logger.info(f"Agentic graph built: {len(nodes)} nodes, {len(edges)} edges")
        return {"nodes": nodes, "edges": edges}




    # ─────────────────────────────────────────────────────────────────────────
    # Generic sequential fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _build_sequential_fallback(self, steps: list, suggestions: list) -> Dict[str, Any]:
        sorted_steps = sorted(steps, key=lambda x: x.get("step_number", 0))
        nodes: list  = []
        edges: list  = []


        # Group by actor
        actor_groups: Dict[str, list] = {}
        for step in sorted_steps:
            actor_groups.setdefault(step.get("actor", "Process"), []).append(step)


        y_off = 60
        for actor, actor_steps in actor_groups.items():
            gid   = f"group-{slugify(actor)}"
            gh    = 80 + len(actor_steps) * 110
            nodes.append({
                "id": gid, "type": "agentGroupNode",
                "position": {"x": 60, "y": y_off},
                "style": {"width": 400, "height": gh, "zIndex": 0},
                "data": {"label": actor, "icon": "Database", "accentColor": "#6366F1"},
            })
            for slot, step in enumerate(actor_steps):
                auto = step.get("automation_potential", 0)
                nodes.append({
                    "id": f"step-{step['step_number']}", "type": "processNode",
                    "parentNode": gid, "extent": "parent",
                    "position": {"x": 20, "y": 60 + slot * 110},
                    "style": {"width": 360, "zIndex": 2},
                    "data": {
                        "label":               step.get("title", f"Step {step['step_number']}"),
                        "actor":               actor,
                        "stepNumber":          step.get("step_number"),
                        "automationPotential": auto,
                        "stepType":            step.get("step_type", "manual"),
                        "inputs":              step.get("inputs", []),
                        "outputs":             step.get("outputs", []),
                        "painPoints":          step.get("pain_points", []),
                        "duration":            step.get("duration_estimate", ""),
                        "erpModule":           step.get("erp_module", ""),
                        "description":         step.get("description", ""),
                        "accentColor": (
                            "#10B981" if auto >= 80 else
                            "#F59E0B" if auto >= 60 else "#EF4444"
                        ),
                    },
                })
            y_off += gh + 40


        for idx in range(len(sorted_steps) - 1):
            c, n = sorted_steps[idx], sorted_steps[idx+1]
            edges.append({
                "id":     f"seq-{c['step_number']}-{n['step_number']}",
                "source": f"step-{c['step_number']}",
                "target": f"step-{n['step_number']}",
                "type": "smoothstep", "animated": True,
                "label": "next", "zIndex": 10,
                "style": {"stroke": "#4B5563", "strokeWidth": 2, "zIndex": 10},
            })


        for i, sug in enumerate(suggestions):
            target_num = sug.get("step_number") or sug.get("step_key", "")
            target_id  = f"step-{target_num}" if target_num else None
            nodes.append({
                "id": f"agent-{i}", "type": "agentNode",
                "position": {"x": 540, "y": 60 + i * 220},
                "style": {"width": 300, "zIndex": 2},
                "data": {
                    "title":       sug.get("title", "AI Agent"),
                    "description": sug.get("description", ""),
                    "tasks":       [sug.get("implementation") or sug.get("description", "")],
                    "agentType":   sug.get("agent_type", "workflow_automation"),
                    "accentColor": "#8B5CF6",
                },
            })
            if target_id:
                edges.append({
                    "id": f"auto-{i}", "source": f"agent-{i}", "target": target_id,
                    "type": "smoothstep", "animated": True, "label": "automates suggestion", "zIndex": 10,
                    "style": {"strokeDasharray": "5 5", "stroke": "#8B5CF6",
                              "strokeWidth": 2, "zIndex": 10},
                })


        return {"nodes": nodes, "edges": edges}


    # def get_agent_architecture(self, suggestion_key: str) -> Dict[str, Any]:
    #     db = get_db()
    #     col = db.collection


    #     # 1. Fetch suggestion
    #     suggestion = col(COLLECTIONS["suggestions"]).get(suggestion_key)
    #     if not suggestion:
    #         return None


    #     process_key = suggestion.get("process_key")
    #     step_key    = suggestion.get("step_key")


    #     # 2. Fetch step
    #     step = col(COLLECTIONS["steps"]).get(step_key) if step_key else {}


    #     # 3. Fetch ERP modules
    #     erp_modules = list(db.aql(
    #         "FOR m IN erp_modules FILTER m.process_key == @key RETURN m",
    #         {"key": process_key}
    #     ))


    #     # Pick relevant module (basic logic)
    #     erp_module_name = None
    #     if step and step.get("erp_module"):
    #         erp_module_name = step.get("erp_module")
    #     elif erp_modules:
    #         erp_module_name = erp_modules[0].get("module_name")


    #     # 4. Build dynamic architecture
    #     architecture = {
    #         "agent_cluster_architecture": {
    #             "operating_model": suggestion.get("agent_type", "workflow_automation").replace("_", " ").title(),
    #             "erp_module": erp_module_name or "General",


    #             "architecture_layers": [
    #                 {
    #                     "title": "User Interaction Layer",
    #                     "description": f"Handles interaction for step '{step.get('title', '')}', including monitoring and manual overrides."
    #                 },
    #                 {
    #                     "title": "Agent Orchestration",
    #                     "description": f"Executes '{suggestion.get('title')}' using automation logic, retries, and workflow coordination."
    #                 },
    #                 {
    #                     "title": "ERP Module Integration",
    #                     "description": f"Integrates with ERP module '{erp_module_name}' to execute transactions and sync data."
    #                 },
    #                 {
    #                     "title": "Governance / Observability",
    #                     "description": "Tracks logs, monitoring, SLA compliance, and audit trails."
    #                 }
    #             ],


    #             "erp_context": [
    #                 {
    #                     "title": "System Configuration",
    #                     "description": "Includes API configs, authentication, and ERP mappings."
    #                 },
    #                 {
    #                     "title": "Connection Layer",
    #                     "description": "Handles secure API communication with retry and pooling."
    #                 },
    #                 {
    #                     "title": "Access & Security",
    #                     "description": "Role-based access, encryption, and audit compliance."
    #                 }
    #             ],


    #             "deployment_steps": [
    #                 {
    #                     "id": 1,
    #                     "label": "Trigger",
    #                     "description": f"Triggered from step '{step.get('title', '')}'."
    #                 },
    #                 {
    #                     "id": 2,
    #                     "label": "Validate",
    #                     "description": "Validates inputs and business rules."
    #                 },
    #                 {
    #                     "id": 3,
    #                     "label": "Execute",
    #                     "description": suggestion.get("description", "")
    #                 },
    #                 {
    #                     "id": 4,
    #                     "label": "Verify",
    #                     "description": "Checks output accuracy and consistency."
    #                 },
    #                 {
    #                     "id": 5,
    #                     "label": "Complete",
    #                     "description": "Stores results and notifies stakeholders."
    #                 }
    #             ]
    #         }
    #     }


    #     return architecture


    def get_agent_architecture(self, suggestion_key: str):
        db = get_db()
        col = db.collection
        llm = get_mistral_client()

        # 1. Fetch suggestion
        suggestion = col(COLLECTIONS["suggestions"]).get(suggestion_key)
        if not suggestion:
            return None

        # 2. Fetch step
        step = col(COLLECTIONS["steps"]).get(suggestion.get("step_key"))

        # 3. Fetch process
        process = col(COLLECTIONS["documents"]).get(suggestion.get("process_key"))

        # 4. Build dynamic prompt
        prompt = build_architecture_prompt(suggestion, step, process)

        # 5. LLM call
        response = llm._chat(
            "You are an expert AI architect. Return ONLY valid JSON.",
            prompt,
            temperature=0.2
        )

        print("RAW ARCHITECTURE RESPONSE:\n", response[:2000])

        parsed = llm._parse_json(response)

        # ✅ VALIDATION FIX
        if not isinstance(parsed, dict) or "nodes" not in parsed:
            logger.error("Invalid architecture JSON → fallback used")

            return {
                "nodes": [
                    {
                        "id": "start",
                        "type": "input",
                        "data": {"label": "Start"},
                        "position": {"x": 0, "y": 0}
                    },
                    {
                        "id": "agent",
                        "type": "agent",
                        "data": {"label": suggestion.get("title", "Agent")},
                        "position": {"x": 200, "y": 100}
                    },
                    {
                        "id": "erp",
                        "type": "system",
                        "data": {"label": "ERP System"},
                        "position": {"x": 400, "y": 200}
                    }
                ],
                "edges": [
                    {"id": "e1", "source": "start", "target": "agent"},
                    {"id": "e2", "source": "agent", "target": "erp"}
                ]
            }

        return parsed


def generate_graph_html(process_key, steps, relationships):
    graph_folder = f"graphs/{process_key}"
    os.makedirs(graph_folder, exist_ok=True)
    file_path = os.path.join(graph_folder, "graph.html")

    net = Network(height="750px", width="100%", directed=True)

    # ✅ Add nodes
    valid_nodes = set()
    for step in steps:
        node_id = step.step_number
        valid_nodes.add(node_id)

        net.add_node(
            node_id,
            label=step.title,
            title=step.description,
            color="#97c2fc"
        )

    # ✅ Add edges safely
    step_sequences = relationships.get("step_sequences", [])

    if step_sequences:
        for rel in step_sequences:
            from_step = rel.get("from_step")
            to_step   = rel.get("to_step")

            # 🔥 HARD FIX: skip invalid edges
            if from_step not in valid_nodes or to_step not in valid_nodes:
                print(f"⚠️ Skipping invalid edge: {from_step} -> {to_step}")
                continue

            net.add_edge(
                from_step,
                to_step,
                label=rel.get("relationship", "")
            )
    else:
        # fallback sequential flow
        for i in range(len(steps) - 1):
            net.add_edge(
                steps[i].step_number,
                steps[i+1].step_number
            )

    net.barnes_hut()
    net.write_html(file_path)

    return file_path

analysis_service = AnalysisService()

