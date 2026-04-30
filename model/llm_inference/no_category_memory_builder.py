from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from model.hierarchical.no_category_hierarchical_manager import NoCategoryHierarchicalMemoryManager
from model.llm_inference.data_adapter import extract_interactions


@dataclass
class NoCategoryConversationMemoryBuildResult:
    interaction_texts: List[str]
    nodes_list: List[Optional[Dict[str, Any]]]
    success_flags: List[bool]


class NoCategoryConversationMemoryBuilder:
    """Builder for the three-level no-category hierarchy."""

    def __init__(
        self,
        manager: NoCategoryHierarchicalMemoryManager,
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
        show_progress: bool = False,
    ) -> NoCategoryConversationMemoryBuildResult:
        normalized = [str(item) for item in interactions]
        if clear_before_build:
            self.clear()
        nodes_list, success_flags = self.manager.batch_process_dialogues(
            normalized,
            llm_batch_size=self.llm_batch_size,
            generate_embedding=generate_embedding,
            show_progress=show_progress,
        )
        self.manager.flush()
        return NoCategoryConversationMemoryBuildResult(
            interaction_texts=normalized,
            nodes_list=nodes_list,
            success_flags=success_flags,
        )

    def build_from_sample(
        self,
        sample: Any,
        dataset_name: Optional[str] = None,
        clear_before_build: bool = True,
        generate_embedding: bool = True,
        show_progress: bool = False,
    ) -> NoCategoryConversationMemoryBuildResult:
        interactions = extract_interactions(sample, dataset_name=dataset_name)
        return self.build_from_interactions(
            interactions=interactions,
            clear_before_build=clear_before_build,
            generate_embedding=generate_embedding,
            show_progress=show_progress,
        )
