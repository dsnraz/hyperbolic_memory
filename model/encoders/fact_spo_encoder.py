from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import json_repair

from .llm_encoder import LLMEncoder


class FactSPOEncoder(LLMEncoder):
    """Per-fact SPO extractor — Stage 2 of the two-stage extraction pipeline.

    Takes a single fact text (with optional surrounding dialogue context)
    and extracts subject / predicate / object / time.
    """

    FACT_SPO_PROMPT = """[INST]
You are a precise fact analyzer. Extract the subject, predicate, object, and time from the given fact.
You ONLY output STANDARD JSON.
You DO NOT output any extra words.

Return JSON with this schema:
{{
  "subject": "main person or entity",
  "predicate": "relation or action",
  "object": "object or complement",
  "time": "time expression if present, else empty string"
}}

Rules:
1. subject: the main person, entity, or thing the fact is about. Use the original wording from the fact.
2. predicate: a concise verb phrase describing the relation or action (e.g. "has known", "working at", "attended", "was filled with").
3. object: the target, complement, or recipient of the predicate. Use the original wording.
4. time: any temporal expression in the fact (dates, durations, ages, frequencies). If no time is present, use empty string "". Do NOT invent or guess time information.
5. Use the original wording from the fact wherever possible — do not paraphrase.
6. Output JSON only.
[/INST]

Fact:
{fact_text}

JSON OUTPUT:"""

    DIALOGUE_FACT_PROMPT = """[INST]
You are a precise fact extractor. Extract ONE self-contained fact from the dialogue below.
You ONLY output STANDARD JSON.
You DO NOT output any extra words.

Return JSON with this schema:
{{
  "fact": "one self-contained factual statement using original wording",
  "subject": "main person or entity",
  "predicate": "relation or action",
  "object": "object or complement",
  "time": "time expression if present, else empty string"
}}

Rules:
1. Write the fact as a concise, self-contained declarative sentence. NEVER copy the dialogue verbatim. Strip timestamps, speaker names, and conversational filler.
2. Replace pronouns with entity names so the fact is self-contained.
3. subject: main person/entity. predicate: concise verb phrase. object: target/complement.
4. TIME RESOLUTION: The dialogue starts with a posting timestamp. Resolve relative times ("yesterday", "last week") to absolute dates using that timestamp. NEVER output relative expressions like "yesterday" as the time value.
5. If no time expression is present, use empty string for time. Do NOT invent or guess.
6. Output JSON only.
[/INST]

Dialogue:
{dialogue_text}

JSON OUTPUT:"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def extract_spo(self, fact_text: str, **kwargs: Any) -> Dict[str, str]:
        """Extract SPO from a single fact text."""
        if not self._init_handler():
            raise RuntimeError("model init failed")

        prompt = self.FACT_SPO_PROMPT.format(fact_text=fact_text)
        response = self._handler.generate(prompt, **kwargs)
        return self._parse_spo_response(response)

    def batch_extract_spo(
        self,
        fact_texts: List[str],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """Batch extract SPO from multiple fact texts."""
        if not self._init_handler():
            raise RuntimeError("model init failed")

        prompts = [self.FACT_SPO_PROMPT.format(fact_text=ft) for ft in fact_texts]
        responses = self._handler.batch_generate(prompts, **kwargs)

        results: List[Dict[str, str]] = []
        if show_progress:
            try:
                from tqdm import tqdm
                iterator: Any = tqdm(
                    zip(fact_texts, responses),
                    total=len(fact_texts),
                    desc="  SPO extraction",
                    unit="fact",
                )
            except ImportError:
                iterator = zip(fact_texts, responses)
        else:
            iterator = zip(fact_texts, responses)

        for fact_text, response in iterator:
            spo = self._parse_spo_response(response)
            results.append(spo)
            if not show_progress:
                preview = fact_text[:60] + ("..." if len(fact_text) > 60 else "")
                print(f"  [{len(results)}/{len(fact_texts)}] {preview}")
                print(f"    → {spo['subject']} | {spo['predicate']} | {spo['object']}"
                      f" | time={spo['time']}")

        return results

    def _parse_spo_response(self, response: str) -> Dict[str, str]:
        fallback = {"subject": "", "predicate": "", "object": "", "time": ""}
        try:
            response = self._sanitize_json_response(response.strip())
            if not response.startswith("{"):
                start = response.find("{")
                end = response.rfind("}")
                if start >= 0 and end > start:
                    response = response[start : end + 1]
            data = json_repair.loads(response)
            if not isinstance(data, dict):
                return fallback
            return {
                "subject": str(data.get("subject", "")).strip(),
                "predicate": str(data.get("predicate", "")).strip(),
                "object": str(data.get("object", "")).strip(),
                "time": str(data.get("time", "")).strip(),
            }
        except Exception:
            return fallback

    # ------------------------------------------------------------------
    # Stage 3: extract fact + SPO from an orphan dialogue
    # ------------------------------------------------------------------

    def extract_fact_from_dialogue(self, dialogue_text: str, **kwargs: Any) -> Dict[str, Any]:
        """Extract a fact with SPO from a single orphan dialogue."""
        if not self._init_handler():
            raise RuntimeError("model init failed")
        prompt = self.DIALOGUE_FACT_PROMPT.format(dialogue_text=dialogue_text)
        response = self._handler.generate(prompt, **kwargs)
        return self._parse_dialogue_fact_response(response)

    def batch_extract_facts_from_dialogues(
        self,
        dialogue_texts: List[str],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Batch extract facts from orphan dialogues."""
        if not self._init_handler():
            raise RuntimeError("model init failed")
        prompts = [self.DIALOGUE_FACT_PROMPT.format(dialogue_text=dt) for dt in dialogue_texts]
        responses = self._handler.batch_generate(prompts, **kwargs)
        results: List[Dict[str, Any]] = []
        for response in responses:
            results.append(self._parse_dialogue_fact_response(response))
        return results

    def _parse_dialogue_fact_response(self, response: str) -> Dict[str, Any]:
        fallback = {"fact": "", "subject": "", "predicate": "", "object": "", "time": ""}
        try:
            response = self._sanitize_json_response(response.strip())
            if not response.startswith("{"):
                start = response.find("{")
                end = response.rfind("}")
                if start >= 0 and end > start:
                    response = response[start : end + 1]
            data = json_repair.loads(response)
            if not isinstance(data, dict):
                return fallback
            return {
                "fact": str(data.get("fact", "")).strip(),
                "subject": str(data.get("subject", "")).strip(),
                "predicate": str(data.get("predicate", "")).strip(),
                "object": str(data.get("object", "")).strip(),
                "time": str(data.get("time", "")).strip(),
            }
        except Exception:
            return fallback
