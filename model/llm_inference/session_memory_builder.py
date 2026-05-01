from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from model.hierarchical.session_hierarchical_manager import SessionHierarchicalMemoryManager
from model.llm_inference.data_adapter import (
    extract_interactions,
    get_session_numbers,
    turn_to_text,
)


@dataclass
class SessionConversationMemoryBuildResult:
    interaction_texts: List[str]
    nodes: Optional[Dict[str, Any]]
    success: bool


class SessionConversationMemoryBuilder:
    """Build memory for one conversation by analyzing the whole session first."""

    def __init__(
        self,
        manager: SessionHierarchicalMemoryManager,
        llm_batch_size: int = 8,
    ) -> None:
        self.manager = manager
        self.llm_batch_size = llm_batch_size

    def clear(self) -> bool:
        return self.manager.clear_memory()

    def build_from_interactions(
        self,
        interactions: Sequence[Any],
        clear_before_build: bool = True,
        generate_embedding: bool = True,
        session_id: Optional[str] = None,
    ) -> SessionConversationMemoryBuildResult:
        normalized = [str(item) for item in interactions]
        if clear_before_build:
            self.clear()
        nodes, success = self.manager.process_session(
            normalized,
            generate_embedding=generate_embedding,
            session_id=session_id,
        )
        self.manager.flush()
        return SessionConversationMemoryBuildResult(
            interaction_texts=normalized,
            nodes=nodes,
            success=success,
        )

    def build_from_sample(
        self,
        sample: Any,
        dataset_name: Optional[str] = None,
        clear_before_build: bool = True,
        generate_embedding: bool = True,
        session_id: Optional[str] = None,
    ) -> SessionConversationMemoryBuildResult:
        if clear_before_build:
            self.clear()

        if dataset_name == "locomo" and isinstance(sample, dict):
            conversation = sample.get("conversation")
            if isinstance(conversation, dict):
                session_tasks = self._build_locomo_session_tasks(conversation, session_id=session_id)
                if session_tasks:
                    session_ids = [sid for sid, _ in session_tasks]
                    sessions = [interactions for _, interactions in session_tasks]
                    nodes_list, ok_list = self.manager.batch_process_sessions(
                        sessions=sessions,
                        generate_embedding=generate_embedding,
                        show_progress=False,
                        session_ids=session_ids,
                    )
                    self.manager.flush()
                    all_interactions = [text for _, interactions in session_tasks for text in interactions]
                    return SessionConversationMemoryBuildResult(
                        interaction_texts=all_interactions,
                        nodes={"sessions": nodes_list},
                        success=all(ok_list) if ok_list else False,
                    )

        interactions = extract_interactions(sample, dataset_name=dataset_name)
        nodes, success = self.manager.process_session(
            [str(item) for item in interactions],
            generate_embedding=generate_embedding,
            session_id=session_id,
        )
        self.manager.flush()
        return SessionConversationMemoryBuildResult(
            interaction_texts=[str(item) for item in interactions],
            nodes=nodes,
            success=success,
        )

    def _build_locomo_session_tasks(
        self,
        conversation: Dict[str, Any],
        session_id: Optional[str],
    ) -> List[tuple[str, List[str]]]:
        tasks: List[tuple[str, List[str]]] = []
        base_session_id = session_id or "sample"
        for session_number in get_session_numbers(conversation):
            session_key = f"session_{session_number}"
            time_key = f"{session_key}_date_time"
            raw_turns = conversation.get(session_key, [])
            if not isinstance(raw_turns, list):
                continue
            time_value = str(conversation.get(time_key, "")).strip()
            interactions: List[str] = []
            for turn in raw_turns:
                if not isinstance(turn, dict):
                    continue
                text = turn_to_text(turn, time_value=time_value)
                if text:
                    interactions.append(text)
            if interactions:
                tasks.append((f"{base_session_id}_{session_key}", interactions))
        return tasks
