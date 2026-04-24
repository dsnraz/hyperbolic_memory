"""
基于余弦相似度的分层检索器。

检索流程采用自顶向下的逐层收缩策略：
1. 在起始层级的全部节点中检索 top-k
2. 收集这些节点的所有子节点
3. 在子节点集合中再次检索 top-k
4. 重复直到目标层级
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.retrievers.base_retriever import HierarchicalRetrieverBase
from model.retrievers.result_types import (
    BaseHierarchicalRetrievalResult,
    BaseLevelRetrievalResult,
    BaseRetrievalHit,
)
from model.stores.hierarchical_vector_store import HierarchicalVectorStore


@dataclass
class RetrievalHit(BaseRetrievalHit):
    """单个命中结果。"""


@dataclass
class LevelRetrievalResult(BaseLevelRetrievalResult[RetrievalHit]):
    """单层级检索结果。"""


@dataclass
class HierarchicalRetrievalResult(
    BaseHierarchicalRetrievalResult[RetrievalHit, LevelRetrievalResult]
):
    """完整的分层检索结果。"""


class CosineRetriever(HierarchicalRetrieverBase):
    """基于余弦相似度的分层检索器。"""

    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        reranker: Optional[Any] = None,
    ):
        super().__init__(
            vector_store=vector_store,
            embedding_function=embedding_function,
            reranker=reranker,
        )

    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
    ) -> HierarchicalRetrievalResult:
        """
        执行自顶向下的层级检索。

        参数:
            query_text: 查询文本
            query_embedding: 已计算好的查询向量
            top_k: 每一层保留的候选数
            start_level: 起始检索层级
            target_level: 目标层级
        """
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        self._validate_level_path(start_level, target_level)

        if self.vector_store.get_pending_dirty_count() > 0:
            self.vector_store.flush()

        prepared_query_embedding = self._prepare_query_embedding(query_text, query_embedding)
        level_results: List[LevelRetrievalResult] = []

        current_level = start_level
        current_candidates = self._get_nodes_by_level(current_level)

        while True:
            ranked_hits = self._rank_nodes(prepared_query_embedding, current_candidates, top_k)
            level_results.append(
                LevelRetrievalResult(
                    level=current_level,
                    hits=ranked_hits,
                    candidate_count=len(current_candidates),
                )
            )

            if current_level == target_level or not ranked_hits:
                break

            child_level = current_level.get_child_level()
            if child_level is None:
                break

            current_candidates = self._collect_children([hit.node for hit in ranked_hits])
            current_level = child_level

        return HierarchicalRetrievalResult(
            query_text=query_text,
            query_embedding=prepared_query_embedding,
            top_k=top_k,
            start_level=start_level,
            target_level=target_level,
            level_results=level_results,
        )

    def retrieve_as_tuples(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
    ) -> List[Tuple[HierarchicalNode, float]]:
        """返回目标层级的 `(node, score)` 结果。"""
        result = self.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            start_level=start_level,
            target_level=target_level,
        )
        return result.as_tuples()

    def _get_nodes_by_level(self, level: HierarchyLevel) -> List[HierarchicalNode]:
        nodes = self.vector_store.get_nodes_by_level(level)
        return self._deduplicate_nodes(nodes)

    def _rank_nodes(
        self,
        query_embedding: Sequence[float],
        nodes: List[HierarchicalNode],
        top_k: int,
    ) -> List[RetrievalHit]:
        scored_hits: List[RetrievalHit] = []
        for node in nodes:
            if node.embedding is None:
                continue
            score = self._cosine_similarity(query_embedding, node.embedding)
            scored_hits.append(RetrievalHit(node=node, score=score))

        scored_hits.sort(key=lambda hit: hit.score, reverse=True)
        return scored_hits[:top_k]

    def _cosine_similarity(
        self,
        vector_a: Sequence[float],
        vector_b: Sequence[float],
    ) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError("向量维度不一致，无法计算余弦相似度")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sum(a * a for a in vector_a) ** 0.5
        norm_b = sum(b * b for b in vector_b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return -1.0

        return dot_product / (norm_a * norm_b)
