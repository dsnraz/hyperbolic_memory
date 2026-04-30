from __future__ import annotations

from typing import Any, Dict

from .llm_encoder import LLMEncoder


class NoCategoryLLMEncoder(LLMEncoder):
    """Dialogue-level extractor without the category layer."""

    ANALYSIS_PROMPT = """
[INST]
You are a professional dialogue structured extractor.
You ONLY output STANDARD JSON.
You DO NOT output any extra words.

Your output MUST strictly follow this format:
{
    "domain": "macro domain",
    "keywords": ["key1", "key2", "key3", "key4", "key5"],
    "summary": "one-sentence summary"
}

Rules:
1. Only extract facts from the dialogue.
2. keywords: 3-8 concise keywords or short phrases.
3. summary: one clear short sentence.
4. Output JSON only.
[/INST]

Dialogue:
{dialogue}

JSON OUTPUT:
"""

    def _build_result(self, data: Any, original_dialogue: str) -> Dict[str, Any]:
        normalized = self._normalize_result_payload(data)
        keywords = normalized.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [keywords] if keywords else []
        return {
            "domain": normalized.get("domain", "unknown"),
            "keywords": keywords,
            "summary": normalized.get("summary"),
            "raw_dialogue": original_dialogue,
        }
