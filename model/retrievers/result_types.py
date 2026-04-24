from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel


@dataclass
class BaseRetrievalHit:
    """检索命中结果的公共基类。"""

    node: HierarchicalNode
    score: float


HitT = TypeVar("HitT", bound=BaseRetrievalHit)


@dataclass
class BaseLevelRetrievalResult(Generic[HitT]):
    """单层级检索结果的公共基类。"""

    level: HierarchyLevel
    hits: List[HitT]
    candidate_count: int


LevelResultT = TypeVar("LevelResultT", bound=BaseLevelRetrievalResult)


@dataclass
class BaseHierarchicalRetrievalResult(Generic[HitT, LevelResultT]):
    """完整分层检索结果的公共基类。"""

    query_text: Optional[str]
    query_embedding: List[float]
    top_k: int
    start_level: HierarchyLevel
    target_level: HierarchyLevel
    level_results: List[LevelResultT]

    @property
    def final_hits(self) -> List[HitT]:
        """返回最后一个实际检索层级的命中结果。"""
        if not self.level_results:
            return []
        return self.level_results[-1].hits

    def as_tuples(self) -> List[Tuple[HierarchicalNode, float]]:
        """转换为通用的 `(node, score)` 结构。"""
        return [(hit.node, hit.score) for hit in self.final_hits]
