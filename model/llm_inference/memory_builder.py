from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from model.hierarchical.hierarchical_manager import HierarchicalMemoryManager
from model.llm_inference.data_adapter import extract_interactions


@dataclass
class ConversationMemoryBuildResult:
    """测试阶段单段对话建库结果。"""

    interaction_texts: List[str]
    nodes_list: List[Optional[Dict[str, Any]]]
    success_flags: List[bool]


class ConversationMemoryBuilder:
    """测试阶段建库器，负责 clear 和逐 interaction 建四层节点。"""

    def __init__(
        self,
        manager: HierarchicalMemoryManager,
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
    ) -> ConversationMemoryBuildResult:
        if clear_before_build:
            self.clear()

        nodes_list, success_flags = self.manager.batch_process_dialogues(
            interactions,
            llm_batch_size=self.llm_batch_size,
            generate_embedding=generate_embedding,
            show_progress=show_progress,
        )
        self.manager.flush()

    def build_from_sample(
        self,
        sample: Any,
        dataset_name: str | None = None,
        clear_before_build: bool = True,
        generate_embedding: bool = True,
        show_progress: bool = False,
    ) -> ConversationMemoryBuildResult:
        interactions = extract_interactions(sample, dataset_name=dataset_name)
        print("提取到的数据条数：")
        print(len(interactions))
        self.build_from_interactions(
            interactions=interactions,
            clear_before_build=clear_before_build,
            generate_embedding=generate_embedding,
            show_progress=show_progress,
        )
