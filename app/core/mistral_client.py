"""
Mistral LLM client.
Handles the 3-pass analysis pipeline with retry logic and JSON parsing.
"""

import os
import json
import logging
import re
from typing import Any, Dict, List
from mistralai import Mistral
from json_repair import repair_json

from app.prompts.prompts import (
    SYSTEM_PROCESS_ANALYST,
    SYSTEM_AUTOMATION_EXPERT,
    build_extraction_prompt,
    build_react_flow_prompt,
    build_inventory_react_flow_prompt,
    build_scoring_prompt,
    build_suggestions_prompt,
    build_relationships_prompt,
    build_workflow_categorization_prompt,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class MistralClient:
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = Mistral(api_key=api_key)
        self.model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    def _chat(self, system: str, user: str, temperature: float = 0.2) -> str:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=4096,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(f"Mistral attempt {attempt+1} failed: {e}. Retrying...")

    # def _parse_json(self, raw: str):
    #     raw = raw.strip()

    #     # remove markdown fences
    #     raw = re.sub(r"^```(?:json)?\s*", "", raw)
    #     raw = re.sub(r"\s*```$", "", raw)

    #     # try direct parse
    #     try:
    #         return json.loads(raw)
    #     except:
    #         pass

    #     # try extract JSON array FIRST
    #     array_match = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
    #     if array_match:
    #         try:
    #             return json.loads(array_match.group(0))
    #         except:
    #             pass

    #     # try extract JSON object
    #     obj_match = re.search(r"\{\s*\".*?\}\s*", raw, re.DOTALL)
    #     if obj_match:
    #         try:
    #             return json.loads(obj_match.group(0))
    #         except:
    #             pass

    #     # LAST fallback: try fixing common issues
    #     try:
    #         fixed = raw.replace("\n", "").replace("\t", "")
    #         fixed = re.sub(r",\s*}", "}", fixed)  # remove trailing commas
    #         fixed = re.sub(r",\s*]", "]", fixed)
    #         return json.loads(fixed)
    #     except:
    #         logger.error("❌ JSON PARSE FAILED")
    #         logger.error(raw[:1000])  # log first part
    #         return []





    def _parse_json(self, raw: str):
        raw = raw.strip()

        # 🔹 1. Remove markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # 🔹 2. Try direct parse
        try:
            return json.loads(raw)
        except:
            pass

        # 🔹 3. Extract largest JSON block (robust)
        json_candidate = self._extract_largest_json(raw)

        if json_candidate:
            try:
                return json.loads(json_candidate)
            except:
                pass

        # 🔹 4. Use json_repair (BEST FIX)
        try:
            repaired = repair_json(raw)
            return json.loads(repaired)
        except:
            pass

        # 🔹 5. Final fallback logging
        logger.error("❌ JSON PARSE FAILED")
        logger.error(raw[:2000])

        return []



    def _extract_largest_json(self, text: str):
        stack = []
        start = None
        max_json = ""

        for i, char in enumerate(text):
            if char in "{[":
                if not stack:
                    start = i
                stack.append(char)

            elif char in "}]":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidate = text[start:i+1]
                        if len(candidate) > len(max_json):
                            max_json = candidate

        return max_json if max_json else None
    # ── Public API ────────────────────────────────────────────────────────────

    def extract_process(self, text: str, source_type: str, file_name: str) -> Dict:
        """Pass 1: Extract process structure from document."""
        prompt = build_extraction_prompt(text, source_type, file_name)
        raw = self._chat(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1)
        result = self._parse_json(raw)
        logger.info(f"Extraction complete: {len(result.get('steps', []))} steps found")
        return result

    # def score_automation(self, steps: List[Dict], process_context: str) -> List[Dict]:
    #     """Pass 2: Score automation potential for each step."""
    #     prompt = build_scoring_prompt(steps, process_context)
    #     raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.1)
    #     print("RAW SCORING RESPONSE:", raw[:500])
    #     scores = self._parse_json(raw)
    #     print("PARSED TYPE:", type(scores), "VALUE:", scores)
    #     logger.info(f"Scoring complete: {len(scores)} steps scored")
    #     return scores if isinstance(scores, list) else []
    def score_automation(self, steps: List[Dict], process_context: str) -> List[Dict]:
        """Pass 2: Score automation potential for each step."""
        prompt = build_scoring_prompt(steps, process_context)
        raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.1)

        print("RAW SCORING RESPONSE:", raw[:500])

        scores = self._parse_json(raw)

        print("PARSED TYPE:", type(scores), "VALUE:", scores)

        # ✅ FIX: extract list from dict
        if isinstance(scores, dict) and "automation_scores" in scores:
            scores_list = scores["automation_scores"]
        elif isinstance(scores, list):
            scores_list = scores
        else:
            scores_list = []

        logger.info(f"Scoring complete: {len(scores_list)} steps scored")

        return scores_list    


    # def generate_suggestions(
    #     self, steps: List[Dict], scores: List[Dict], process_title: str
    # ) -> List[Dict]:
    #     """Pass 3: Generate agentic automation suggestions."""
    #     prompt = build_suggestions_prompt(steps, scores, process_title)
    #     raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.3)
    #     suggestions = self._parse_json(raw)
    #     logger.info(f"Suggestions generated: {len(suggestions)}")
    #     return suggestions if isinstance(suggestions, list) else []

    def generate_suggestions(
        self, steps: List[Dict], scores: List[Dict], process_title: str
    ) -> List[Dict]:
        
        prompt = build_suggestions_prompt(steps, scores, process_title)
        raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.3)

        print("RAW SUGGESTIONS:", raw[:500])

        try:
            suggestions = self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Suggestion parsing failed: {e}")
            return []

        print("PARSED SUGGESTIONS TYPE:", type(suggestions), "VALUE:", suggestions)

        # ✅ Handle multiple formats
        if isinstance(suggestions, dict):
            if "suggestions" in suggestions:
                suggestions_list = suggestions["suggestions"]
            elif "recommendations" in suggestions:
                suggestions_list = suggestions["recommendations"]
            else:
                # fallback → convert dict to list
                suggestions_list = list(suggestions.values())
        elif isinstance(suggestions, list):
            suggestions_list = suggestions
        else:
            suggestions_list = []

        logger.info(f"Suggestions generated: {len(suggestions_list)}")

        return suggestions

    def extract_relationships(
        self, process_title: str, steps: List[Dict], erp_modules: List[Dict]
    ) -> Dict:
        """Pass 4: Extract logical relationships for graph edges."""
        prompt = build_relationships_prompt(process_title, steps, erp_modules)
        raw = self._chat(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1)
        try:
            return self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
            return {"step_sequences": [], "module_relationships": [], "cross_process_dependencies": []}


    def generate_react_flow(self, process_title: str, steps: list, suggestions: list, workflow_type: str = "generic") -> dict:
        wf_keywords = ["inventory", "stock", "sales order", "purchase order", "procurement"]
        if workflow_type == "inventory" or any(k in process_title.lower() for k in wf_keywords):
            prompt = build_inventory_react_flow_prompt(process_title, steps, suggestions)
        else:
            prompt = build_react_flow_prompt(process_title, steps, suggestions)
        
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_AUTOMATION_EXPERT},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=6000,   # IMPORTANT: must be high for large processes
            temperature=0.1    # low temp = deterministic layout
        )
        
        raw = response.choices[0].message.content
        return self._parse_json(raw)        


    def decompose_micro_process(self, steps):
        import json

        # ✅ FIX 1: escape JSON for f-string
        steps_json = json.dumps(steps, indent=2).replace("{", "{{").replace("}", "}}")

        prompt = f"""
    Break each step into micro-processes and decisions.

    Steps:
    {steps_json}

    Return:
    [
    {{
        "step_number": 1,
        "micro_steps": [...],
        "decisions": [...]
    }}
    ]
    """

        raw = self._chat(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.2)

        # ✅ FIX 2: safe parse
        parsed = self._parse_json(raw)

        # ✅ FIX 3: ensure list of dict
        if not isinstance(parsed, list):
            logger.warning(f"Micro process not list: {type(parsed)} → {parsed}")
            return []

        # filter valid dicts only
        cleaned = [p for p in parsed if isinstance(p, dict)]

        return cleaned


    def categorize_workflow(self, process_title, steps, suggestions):
        prompt = build_workflow_categorization_prompt(
            process_title,
            steps,
            suggestions
        )

        response = self.client.chat.completions.create(
            model="mistral-large-latest",  # or your model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except Exception:
            return {}
# Singleton
_client_instance = None

def get_mistral_client() -> MistralClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = MistralClient()
    return _client_instance
