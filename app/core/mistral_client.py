"""
LLM client (formerly Mistral client).
Handles the 3-pass analysis pipeline with retry logic and JSON parsing.
Now uses unified openai SDK to support Mistral, OpenAI, Gemini, and Local models.
"""

import os
import json
import logging
import re
from typing import Any, Dict, List
from openai import OpenAI
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
from app.services.settings_service import get_all_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class MistralClient:
    def __init__(self):
        # We don't initialize fixed API keys here anymore.
        # Instead, we will fetch them dynamically inside _chat
        # to ensure any changes from the Admin Panel take effect immediately.
        temp_str = os.getenv("LLM_TEMPERATURE")
        self.default_temp = float(temp_str) if temp_str else 0.2

        tokens_str = os.getenv("LLM_MAX_TOKENS")
        self.default_max_tokens = int(tokens_str) if tokens_str else 4096
        
    def _get_client_and_model(self):
        """Fetch current settings and return an OpenAI client and the target model."""
        settings = get_all_settings()
        
        provider = settings.get("LLM_PROVIDER", "").lower()
        api_key = settings.get("LLM_API_KEY", "")
        model = settings.get("LLM_MODEL_NAME", "mistral-small-latest")
        
        if provider == "local":
            base_url = settings.get("LLM_BASE_URL", "http://localhost:11434/v1")
            client = OpenAI(base_url=base_url, api_key="local-placeholder")
        elif provider == "openai":
            client = OpenAI(api_key=api_key)
        elif provider == "gemini":
            client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key
            )
        else: # Default to Mistral
            client = OpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key=api_key
            )
            
        return client, model, provider

    def _chat(self, system: str, user: str, temperature: float = None, force_json: bool = False) -> str:
        temp = temperature if temperature is not None else self.default_temp
        
        client, model, provider = self._get_client_and_model()
        
        kwargs = dict(
            model=model,
            temperature=temp,
            max_tokens=self.default_max_tokens,
        )
        
        # Only some providers properly support JSON response_format
        if force_json and provider != "local":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **kwargs
                )
                content = response.choices[0].message.content
                
                # Deduct token usage if available and user is authenticated
                try:
                    from flask import g, has_request_context
                    from app.db.db_connection import get_mysql_connection
                    
                    if has_request_context() and hasattr(g, 'user') and g.user and g.user.get('uid'):
                        uid = g.user.get('uid')
                        tokens_used = 0
                        if hasattr(response, 'usage') and response.usage:
                            tokens_used = getattr(response.usage, 'total_tokens', 0)
                            
                        if tokens_used > 0:
                            conn = get_mysql_connection()
                            try:
                                with conn.cursor() as cur:
                                    cur.execute(
                                        "UPDATE users SET llm_tokens_used = llm_tokens_used + %s WHERE id = %s",
                                        (tokens_used, uid)
                                    )
                                conn.commit()
                            finally:
                                conn.close()
                except Exception as e:
                    logger.error(f"Failed to deduct token usage: {e}")
                    
                return content if isinstance(content, str) else ""
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning(f"LLM attempt {attempt+1} failed: {e}. Retrying...")

    def _parse_json(self, raw: str):
        raw = raw.strip()

        # 1. Remove markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # 2. Try direct parse
        try:
            return json.loads(raw)
        except:
            pass

        # 3. Extract largest JSON block
        json_candidate = self._extract_largest_json(raw)

        if json_candidate:
            try:
                return json.loads(json_candidate)
            except:
                pass

        # 4. Use json_repair
        try:
            repaired = repair_json(raw)
            return json.loads(repaired)
        except:
            pass

        # 5. Final fallback logging
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
        if parsed is None:
            return True
        if isinstance(parsed, (dict, list, str)) and len(parsed) == 0:
            return True
        return False

    def _chat_json(self, system: str, user: str, temperature: float = None,
                   expect: str = "object", force_json: bool = False):
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

    
    def validate_relevance(self, user_input: str, file_text: str) -> Dict:
        prompt = f"""
        User's Mission/Vision and Context (may be empty):
        {user_input}
        
        Uploaded File Content Preview (first 2000 chars):
        {file_text[:2000]}
        
        Task: Determine if the uploaded file content is relevant for a Process Automation and Analysis tool.
        The file MUST contain business processes, workflows, standard operating procedures, ERP data, structured tables, or actionable steps for automation.
        If the user provided a Mission/Vision, the file should also be somewhat relevant to that context.
        If the file is a personal chat, a poem, random artwork, or clearly irrelevant non-business data, you must reject it.
        
        CRITICAL: If the file appears to be an image (e.g. a WhatsApp Image or screenshot) or an image-based PDF, you MUST first provide your detailed explanation of why it is irrelevant, and then ADDITIONALLY append this exact statement: "We cannot understand or extract text/content from images."
        
        If it is relevant or if it contains valid business/process data, respond with {{"is_relevant": true}}.
        If it is clearly irrelevant (e.g. a poem, random chat, or entirely unrelated to business processes, or an image), respond with:
        {{
            "is_relevant": false,
            "error": "<Detailed explanation of what the file is and why it lacks business processes. If it is an image, you MUST ALSO append the phrase regarding image extraction as instructed above>",
            "recommended_solution": "<A clear tip on what file they should upload instead>"
        }}
        """
        try:
            return self._chat_json(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1, expect="object")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {"is_relevant": True}

    
    # ── Public API ────────────────────────────────────────────────────────────

    def extract_process(self, text: str, source_type: str, file_name: str) -> Dict:
        prompt = build_extraction_prompt(text, source_type, file_name)
        result = self._chat_json(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1, expect="object")
        result = self._normalize_extraction(result)

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
        if isinstance(result, list):
            return {"steps": result}
        if isinstance(result, dict):
            return result
        return {}

    def score_automation(self, steps: List[Dict], process_context: str) -> List[Dict]:
        prompt = build_scoring_prompt(steps, process_context)
        raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.1)
        scores = self._parse_json(raw)

        if isinstance(scores, dict) and "automation_scores" in scores:
            scores_list = scores["automation_scores"]
        elif isinstance(scores, list):
            scores_list = scores
        else:
            scores_list = []

        logger.info(f"Scoring complete: {len(scores_list)} steps scored")
        return scores_list    

    def generate_suggestions(self, steps: List[Dict], scores: List[Dict], process_title: str) -> List[Dict]:
        prompt = build_suggestions_prompt(steps, scores, process_title)
        raw = self._chat(SYSTEM_AUTOMATION_EXPERT, prompt, temperature=0.3)
        try:
            suggestions = self._parse_json(raw)
        except Exception as e:
            logger.warning(f"Suggestion parsing failed: {e}")
            return []

        if isinstance(suggestions, dict):
            if "suggestions" in suggestions:
                suggestions_list = suggestions["suggestions"]
            elif "recommendations" in suggestions:
                suggestions_list = suggestions["recommendations"]
            else:
                suggestions_list = list(suggestions.values())
        elif isinstance(suggestions, list):
            suggestions_list = suggestions
        else:
            suggestions_list = []

        logger.info(f"Suggestions generated: {len(suggestions_list)}")
        return suggestions

    def extract_relationships(self, process_title: str, steps: List[Dict], erp_modules: List[Dict]) -> Dict:
        prompt = build_relationships_prompt(process_title, steps, erp_modules)
        default = {"step_sequences": [], "module_relationships": [], "cross_process_dependencies": []}
        try:
            parsed = self._chat_json(SYSTEM_PROCESS_ANALYST, prompt, temperature=0.1, expect="object")
            return parsed if isinstance(parsed, dict) and parsed else default
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
            return default

    def generate_react_flow(self, process_title: str, steps: list, suggestions: list, workflow_type: str = "generic") -> dict:
        prompt = build_react_flow_prompt(process_title, steps, suggestions)
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
        parsed = self._parse_json(raw)
        if not isinstance(parsed, list):
            return []
        cleaned = [p for p in parsed if isinstance(p, dict)]
        return cleaned

    def categorize_workflow(self, process_title, steps, suggestions):
        prompt = build_workflow_categorization_prompt(
            process_title,
            steps,
            suggestions
        )
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
