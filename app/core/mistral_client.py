"""
Mistral LLM client.
Handles the 3-pass analysis pipeline with retry logic and JSON parsing.
"""

import os
import json
import logging
import re
from typing import Any, Dict, List
# pyrefly: ignore [missing-import]
from mistralai import Mistral
# pyrefly: ignore [missing-import]
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
        self.mode = (os.getenv("MISTRAL_MODE") or "").lower()
        self.is_local = self.mode == "local"
        if self.is_local:
            self.model = os.getenv("MISTRAL_LOCAL_MODEL")
            local_url = os.getenv("MISTRAL_LOCAL_URL")
            if not self.model:
                raise ValueError("MISTRAL_LOCAL_MODEL environment variable not set")
            if not local_url:
                raise ValueError("MISTRAL_LOCAL_URL environment variable not set")
            # Local models are much slower than the cloud API, so give them a
            # generous timeout (default SDK timeout can abort long generations
            # and surface as empty/failed responses).
            local_timeout_ms = int(os.getenv("MISTRAL_LOCAL_TIMEOUT_MS") or 600000)
            self.client = Mistral(
                api_key="local",
                server_url=local_url,
                timeout_ms=local_timeout_ms,
            )
        else:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable not set")
            self.client = Mistral(api_key=api_key)
            self.model = os.getenv("MODEL_NAME") or os.getenv("MISTRAL_MODEL")

        temp_str = os.getenv("LLM_TEMPERATURE")
        self.default_temp = float(temp_str) if temp_str else 0.2

        tokens_str = os.getenv("LLM_MAX_TOKENS")
        self.default_max_tokens = int(tokens_str) if tokens_str else 4096

    def _chat(self, system: str, user: str, temperature: float = None, force_json: bool = False) -> str:
        temp = temperature if temperature is not None else self.default_temp
        
        kwargs = dict(
            model=self.model,
            temperature=temp,
            max_tokens=self.default_max_tokens,
        )
        if force_json and not self.is_local:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.chat.complete(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **kwargs
                )
                content = response.choices[0].message.content
                # Some local backends return null content instead of a string.
                # Coerce to "" so downstream _parse_json degrades gracefully
                # instead of raising on None.strip().
                return content if isinstance(content, str) else ""
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

        return {}



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

    @staticmethod
    def _is_empty_parse(parsed) -> bool:
        """True when parsing produced nothing usable (None or empty dict/list)."""
        if parsed is None:
            return True
        if isinstance(parsed, (dict, list, str)) and len(parsed) == 0:
            return True
        return False

    def _chat_json(self, system: str, user: str, temperature: float = None,
                   expect: str = "object", force_json: bool = False):
        """
        Chat and parse JSON. If the model returns prose / non-JSON (common with
        local models that don't follow the 'return JSON' instruction as well as
        the cloud model), reprompt once forcefully and parse again.

        Cloud behavior is unchanged: its first response parses successfully, so
        _is_empty_parse is False and no reprompt happens.
        """
        raw = self._chat(system, user, temperature=temperature, force_json=force_json)
        parsed = self._parse_json(raw)
        if self._is_empty_parse(parsed):
            opener = "[" if expect == "array" else "{"
            kind = "array" if expect == "array" else "object"
            reinforce = (
                user
                + f"\n\nIMPORTANT: Respond with ONLY a single valid JSON {kind}. "
                  f"Do NOT include any explanation, summary, prose, or markdown "
                  f"fences. Your entire reply must start with '{opener}'."
            )
            logger.warning("LLM returned non-JSON; reprompting for strict JSON output.")
            raw = self._chat(system, reinforce, temperature=temperature, force_json=force_json)
            parsed = self._parse_json(raw)
        return parsed

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_process(self, text: str, source_type: str, file_name: str) -> Dict:
        """Pass 1: Extract process structure from document."""
        prompt = build_extraction_prompt(text, source_type, file_name)
        result = self._chat_json(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1, expect="object")
        result = self._normalize_extraction(result)

        # Local models often return valid JSON but with an EMPTY steps array
        # (or prose, which _chat_json already retries). If we still have no
        # steps, reprompt once forcefully and explicitly demand the steps array.
        if not result.get("steps"):
            retry_prompt = prompt + (
                "\n\nYour previous answer did not contain any steps. You MUST return "
                "a single JSON object containing a non-empty \"steps\" array with at "
                "least 5 atomic, sequential steps. Respond with ONLY the JSON object, "
                "starting with '{' and ending with '}'. Do NOT write any summary, "
                "explanation, analysis, or prose."
            )
            retry = self._chat_json(SYSTEM_PROCESS_ANALYST, retry_prompt, temperature=0.1, expect="object")
            retry = self._normalize_extraction(retry)
            if retry.get("steps"):
                result = retry

        logger.info(f"Extraction complete: {len(result.get('steps', []))} steps found")
        return result

    @staticmethod
    def _normalize_extraction(result) -> Dict:
        """Coerce extraction output into a {'steps': [...]} dict.

        Local models sometimes return a bare list of steps instead of the
        expected object, so downstream result.get('steps') stays safe.
        """
        if isinstance(result, list):
            return {"steps": result}
        if isinstance(result, dict):
            return result
        return {}

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
        default = {"step_sequences": [], "module_relationships": [], "cross_process_dependencies": []}
        try:
            parsed = self._chat_json(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1, expect="object")
            return parsed if isinstance(parsed, dict) and parsed else default
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
            return default


    def generate_react_flow(self, process_title: str, steps: list, suggestions: list, workflow_type: str = "generic") -> dict:
        wf_keywords = ["inventory", "stock", "sales order", "purchase order", "procurement"]
        # ALWAYS use lane-based prompt
        prompt = build_react_flow_prompt(process_title, steps, suggestions)

        # Cloud Mistral supports forced JSON output. Many local backends
        # (Ollama / llama.cpp / LM Studio, etc.) do NOT implement
        # response_format=json_object and return EMPTY content when it is sent.
        # So we request JSON mode on the cloud path via force_json=True, and locally 
        # _chat_json handles the fallback and reprompting if it fails to parse.
        parsed = self._chat_json(
            SYSTEM_AUTOMATION_EXPERT, 
            prompt, 
            temperature=self.default_temp, 
            expect="object",
            force_json=True
        )
        return parsed if isinstance(parsed, dict) else {}


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

        # NOTE: the Mistral SDK exposes chat.complete(...), NOT the OpenAI-style
        # chat.completions.create(...). The old call raised AttributeError on
        # both cloud and local, and hardcoded a model that does not exist in
        # local mode. Route through _chat so it works for both backends.
        raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.2)

        parsed = self._parse_json(raw)
        return parsed if isinstance(parsed, dict) else {}
# Singleton
_client_instance = None

def get_mistral_client() -> MistralClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = MistralClient()
    return _client_instance
