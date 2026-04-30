from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from model.hierarchical.session_hierarchical_manager import SessionHierarchicalMemoryManager
from model.llm_inference.data_adapter import extract_interactions


@dataclass
class SessionConversationMemoryBuildResult:
    interaction_texts: List[str]
    nodes: Optional[Dict[str, Any]]
    success: bool


class SessionConversationMemoryBuilder:
    """Build memory for one conversation by analyzing the whole session first."""

    def __init__(self, manager: SessionHierarchicalMemoryManager) -> None:
        self.manager = manager

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
        interactions = extract_interactions(sample, dataset_name=dataset_name)
        return self.build_from_interactions(
            interactions=interactions,
            clear_before_build=clear_before_build,
            generate_embedding=generate_embedding,
            session_id=session_id,
        )
