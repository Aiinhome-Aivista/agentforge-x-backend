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

from app.prompts.prompts import (
    SYSTEM_PROCESS_ANALYST,
    SYSTEM_AUTOMATION_EXPERT,
    build_extraction_prompt,
    build_scoring_prompt,
    build_suggestions_prompt,
    build_relationships_prompt,
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

    def _parse_json(self, raw: str) -> Any:
        """Robustly parse JSON from LLM response, stripping any markdown fences."""
        raw = raw.strip()
        # Strip markdown code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object/array within the response
            match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse JSON from LLM response: {raw[:300]}")

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

        return suggestions_list

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


# Singleton
_client_instance = None

def get_mistral_client() -> MistralClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = MistralClient()
    return _client_instance
