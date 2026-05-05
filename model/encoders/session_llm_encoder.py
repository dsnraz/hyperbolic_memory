from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Tuple

import json_repair

from .llm_encoder import LLMEncoder


class SessionLLMEncoder(LLMEncoder):
    """LLM encoder that extracts one four-level tree for a whole session."""

    MEMORY_UNIT_MODES = ("keyword", "fact")

    SESSION_ANALYSIS_PROMPT = """
[INST]
You are a professional session-level dialogue structure extractor.
You ONLY output STANDARD JSON.
You DO NOT output any extra words.

Return JSON with this schema:
{{
  "domain": "macro domain",
  "categories": [
    {{
      "category": "category name",
      "keywords": [
        {{
          "keyword": "keyword or phrase",
          "dialogue_indices": [0, 2]
        }}
      ]
    }}
  ],
}}

Rules:
1. Read the full session before deciding the structure.
2. dialogue_indices are zero-based and must point to the input dialogue list.
3. A dialogue can belong to multiple keywords.
4. Keep the tree compact and globally consistent for the session.
5. Output JSON only.
[/INST]

Session dialogues:
{session_text}

JSON OUTPUT:
"""

    SESSION_FACT_ANALYSIS_PROMPT = """
[INST]
You are a professional session-level dialogue fact extractor.
You ONLY output STANDARD JSON.
You DO NOT output any extra words.

Return JSON with this schema:
{{
  "domains": ["domain A", "domain B"],
  "facts": [
    {{
      "fact": "one self-contained factual statement",
      "dialogue_indices": [0, 2],
      "subject": "main person or entity",
      "predicate": "relation or action",
      "object": "object or complement",
      "time": "time expression if present, else empty string"
    }}
  ],
}}

Rules:
1. Read the full session before extracting.
2. Extract atomic facts that can answer detailed questions. Prefer explicit events, preferences, relationships, plans, dates, locations, and personal details.
3. A fact may be supported by multiple dialogues; dialogue_indices must list every supporting dialogue index.
4. dialogue_indices are zero-based and must point to the input dialogue list.
5. A dialogue can support multiple facts.
6. fact must be self-contained and include the key entity names instead of pronouns whenever possible.
7. predicate must be a concise verb phrase (e.g. "researching", "working at", "attending", "planning to"). object must be the target entity or complement.
8. domains: list 2-3 macro-level domain labels that cover the session content (e.g. "personal life", "career", "health"). Keep labels short and consistent across sessions.
9. Output JSON only.
[/INST]

Session dialogues:
{session_text}

JSON OUTPUT:
"""

    def __init__(self, *args: Any, memory_unit_mode: Literal["keyword", "fact"] = "keyword", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if memory_unit_mode not in self.MEMORY_UNIT_MODES:
            raise ValueError(f"memory_unit_mode must be one of {self.MEMORY_UNIT_MODES}")
        self.memory_unit_mode = memory_unit_mode

    def analyze_session(self, dialogues: List[str], **kwargs: Any) -> Tuple[Dict[str, Any], bool]:
        if not self._init_handler():
            raise RuntimeError("model init failed")

        session_text = self._format_session(dialogues)
        prompt = self._session_prompt_template().format(session_text=session_text)
        response = self._handler.generate(prompt, **kwargs)
        return self._parse_session_response(response, dialogues)

    def batch_analyze_sessions(
        self,
        sessions: List[List[str]],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], List[bool]]:
        if not self._init_handler():
            raise RuntimeError("model init failed")

        prompt_template = self._session_prompt_template()
        prompts = [prompt_template.format(session_text=self._format_session(session)) for session in sessions]
        responses = self._handler.batch_generate(prompts, **kwargs)

        results: List[Dict[str, Any]] = []
        ok_flags: List[bool] = []
        iterator = zip(responses, sessions)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(list(iterator), desc="session analysis", unit="session")
            except ImportError:
                iterator = zip(responses, sessions)

        for response, session in iterator:
            result, is_ok = self._parse_session_response(response, session)
            results.append(result)
            ok_flags.append(is_ok)
        return results, ok_flags

    def _format_session(self, dialogues: List[str]) -> str:
        return "\n".join(f"[{idx}] {self._format_input(text)}" for idx, text in enumerate(dialogues))

    def _session_prompt_template(self) -> str:
        if self.memory_unit_mode == "fact":
            return self.SESSION_FACT_ANALYSIS_PROMPT
        return self.SESSION_ANALYSIS_PROMPT

    def _parse_session_response(
        self,
        response: str,
        dialogues: List[str],
    ) -> Tuple[Dict[str, Any], bool]:
        try:
            response = self._sanitize_json_response(response.strip())
            if not response.startswith("{"):
                start = response.find("{")
                end = response.rfind("}")
                if start >= 0 and end > start:
                    response = response[start : end + 1]
            data = json_repair.loads(response)
            return self._normalize_session_payload(data, dialogues), True
        except Exception:
            return self._fallback_session_payload(dialogues), False

    def _normalize_session_payload(self, data: Any, dialogues: List[str]) -> Dict[str, Any]:
        if self.memory_unit_mode == "fact":
            return self._normalize_fact_session_payload(data, dialogues)
        return self._normalize_keyword_session_payload(data, dialogues)

    def _normalize_keyword_session_payload(self, data: Any, dialogues: List[str]) -> Dict[str, Any]:
        payload = data if isinstance(data, dict) else {}
        dialogue_count = len(dialogues)

        normalized_categories: List[Dict[str, Any]] = []
        for category_item in payload.get("categories", []):
            if not isinstance(category_item, dict):
                continue
            category_name = str(category_item.get("category", "")).strip()
            if not category_name:
                continue

            normalized_keywords: List[Dict[str, Any]] = []
            for keyword_item in category_item.get("keywords", []):
                if not isinstance(keyword_item, dict):
                    continue
                keyword_name = str(keyword_item.get("keyword", "")).strip()
                if not keyword_name:
                    continue
                raw_indices = keyword_item.get("dialogue_indices", [])
                indices: List[int] = []
                for value in raw_indices if isinstance(raw_indices, list) else []:
                    try:
                        idx = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < dialogue_count and idx not in indices:
                        indices.append(idx)
                if not indices:
                    continue
                normalized_keywords.append(
                    {
                        "keyword": keyword_name,
                        "dialogue_indices": sorted(indices),
                    }
                )

            if normalized_keywords:
                normalized_categories.append(
                    {
                        "category": category_name,
                        "keywords": normalized_keywords,
                    }
                )

        return {
            "domain": str(payload.get("domain", "unknown")).strip() or "unknown",
            "categories": normalized_categories,
        }

    def _normalize_fact_session_payload(self, data: Any, dialogues: List[str]) -> Dict[str, Any]:
        payload = data if isinstance(data, dict) else {}
        dialogue_count = len(dialogues)

        domains: List[str] = []
        raw_domains = payload.get("domains", payload.get("domain", []))
        if isinstance(raw_domains, str):
            domains = [raw_domains.strip()] if raw_domains.strip() else []
        elif isinstance(raw_domains, list):
            domains = [str(d).strip() for d in raw_domains if str(d).strip()]
        if not domains:
            domains = ["general"]

        normalized_facts: List[Dict[str, Any]] = []
        raw_facts = payload.get("facts", [])
        if not isinstance(raw_facts, list):
            raw_facts = []
        for fact_item in raw_facts:
            if not isinstance(fact_item, dict):
                continue
            fact_text = str(fact_item.get("fact", "")).strip()
            if not fact_text:
                continue
            indices = self._normalize_dialogue_indices(
                fact_item.get("dialogue_indices", []),
                dialogue_count,
            )
            if not indices:
                continue
            normalized_facts.append(
                {
                    "fact": fact_text,
                    "dialogue_indices": indices,
                    "subject": str(fact_item.get("subject", "")).strip(),
                    "predicate": str(fact_item.get("predicate", "")).strip(),
                    "object": str(fact_item.get("object", "")).strip(),
                    "time": str(fact_item.get("time", "")).strip(),
                }
            )

        return {
            "domains": domains,
            "facts": normalized_facts,
        }

    def _normalize_dialogue_indices(self, raw_indices: Any, dialogue_count: int) -> List[int]:
        indices: List[int] = []
        for value in raw_indices if isinstance(raw_indices, list) else []:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < dialogue_count and idx not in indices:
                indices.append(idx)
        return sorted(indices)

    def _fallback_session_payload(self, dialogues: List[str]) -> Dict[str, Any]:
        if self.memory_unit_mode == "fact":
            return {
                "domains": ["general"],
                "facts": [
                    {
                        "fact": text,
                        "dialogue_indices": [idx],
                        "subject": "",
                        "predicate": "",
                        "object": "",
                        "time": "",
                    }
                    for idx, text in enumerate(dialogues)
                ],
            }
        return {
            "domain": "unknown",
            "categories": [
                {
                    "category": "general",
                    "keywords": [
                        {
                            "keyword": "general",
                            "dialogue_indices": list(range(len(dialogues))),
                        }
                    ],
                }
            ],
        }
