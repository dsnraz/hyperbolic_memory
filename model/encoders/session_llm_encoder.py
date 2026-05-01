from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import json_repair

from .llm_encoder import LLMEncoder


class SessionLLMEncoder(LLMEncoder):
    """LLM encoder that extracts one four-level tree for a whole session."""

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
  "dialogue_summaries": [
    {{
      "dialogue_index": 0,
      "summary": "short summary"
    }}
  ]
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

    def analyze_session(self, dialogues: List[str], **kwargs: Any) -> Tuple[Dict[str, Any], bool]:
        if not self._init_handler():
            raise RuntimeError("model init failed")

        session_text = self._format_session(dialogues)
        prompt = self.SESSION_ANALYSIS_PROMPT.format(session_text=session_text)
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

        prompts = [
            self.SESSION_ANALYSIS_PROMPT.format(session_text=self._format_session(session))
            for session in sessions
        ]
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

        summary_map: Dict[int, str] = {}
        for summary_item in payload.get("dialogue_summaries", []):
            if not isinstance(summary_item, dict):
                continue
            try:
                idx = int(summary_item.get("dialogue_index"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < dialogue_count:
                summary_map[idx] = str(summary_item.get("summary", "")).strip()

        return {
            "domain": str(payload.get("domain", "unknown")).strip() or "unknown",
            "categories": normalized_categories,
            "dialogue_summaries": [
                {
                    "dialogue_index": idx,
                    "summary": summary_map.get(idx, ""),
                    "raw_dialogue": dialogues[idx],
                }
                for idx in range(dialogue_count)
            ],
        }

    def _fallback_session_payload(self, dialogues: List[str]) -> Dict[str, Any]:
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
            "dialogue_summaries": [
                {"dialogue_index": idx, "summary": "", "raw_dialogue": text}
                for idx, text in enumerate(dialogues)
            ],
        }
