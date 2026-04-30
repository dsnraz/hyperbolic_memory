from __future__ import annotations

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


NO_CATEGORY_LEVEL_ORDER = [
    HierarchyLevel.DOMAIN,
    HierarchyLevel.KEYWORD,
    HierarchyLevel.DIALOGUE,
]


@dataclass
class NoCategoryRetrievalHit(BaseRetrievalHit):
    pass


@dataclass
class NoCategoryLevelRetrievalResult(BaseLevelRetrievalResult[NoCategoryRetrievalHit]):
    pass


@dataclass
class NoCategoryHierarchicalRetrievalResult(
    BaseHierarchicalRetrievalResult[NoCategoryRetrievalHit, NoCategoryLevelRetrievalResult]
):
    pass


class NoCategoryCosineRetriever(HierarchicalRetrieverBase):
    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        reranker: Optional[Any] = None,
    ) -> None:
        super().__init__(vector_store=vector_store, embedding_function=embedding_function, reranker=reranker)

    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
    ) -> NoCategoryHierarchicalRetrievalResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._validate_no_category_path(start_level, target_level)

        if self.vector_store.get_pending_dirty_count() > 0:
            self.vector_store.flush()

        prepared = self._prepare_query_embedding(query_text, query_embedding)
        level_results: List[NoCategoryLevelRetrievalResult] = []

        current_level = start_level
        current_candidates = self._get_nodes_by_level(current_level)
        while True:
            ranked_hits = self._rank_nodes(prepared, current_candidates, top_k)
            level_results.append(
                NoCategoryLevelRetrievalResult(
                    level=current_level,
                    hits=ranked_hits,
                    candidate_count=len(current_candidates),
                )
            )
            if current_level == target_level or not ranked_hits:
                break
            child_level = self._get_child_level(current_level)
            if child_level is None:
                break
            current_candidates = self._collect_children_at_level([hit.node for hit in ranked_hits], child_level)
            current_level = child_level

        return NoCategoryHierarchicalRetrievalResult(
            query_text=query_text,
            query_embedding=list(prepared),
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
        return self.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            start_level=start_level,
            target_level=target_level,
        ).as_tuples()

    def _get_nodes_by_level(self, level: HierarchyLevel) -> List[HierarchicalNode]:
        return self._deduplicate_nodes(self.vector_store.get_nodes_by_level(level))

    def _rank_nodes(
        self,
        query_embedding: Sequence[float],
        nodes: List[HierarchicalNode],
        top_k: int,
    ) -> List[NoCategoryRetrievalHit]:
        hits: List[NoCategoryRetrievalHit] = []
        for node in nodes:
            if node.embedding is None:
                continue
            score = self._cosine_similarity(query_embedding, node.embedding)
            hits.append(NoCategoryRetrievalHit(node=node, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _collect_children_at_level(
        self,
        parent_nodes: List[HierarchicalNode],
        child_level: HierarchyLevel,
    ) -> List[HierarchicalNode]:
        children: List[HierarchicalNode] = []
        for parent_node in parent_nodes:
            for child_id in parent_node.child_ids:
                child_node = self.vector_store.get_node(child_id, child_level)
                if child_node is not None:
                    children.append(child_node)
        return self._deduplicate_nodes(children)

    def _validate_no_category_path(
        self,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> None:
        if NO_CATEGORY_LEVEL_ORDER.index(start_level) > NO_CATEGORY_LEVEL_ORDER.index(target_level):
            raise ValueError("target_level must not be above start_level")

    def _get_child_level(self, level: HierarchyLevel) -> Optional[HierarchyLevel]:
        mapping = {
            HierarchyLevel.DOMAIN: HierarchyLevel.KEYWORD,
            HierarchyLevel.KEYWORD: HierarchyLevel.DIALOGUE,
            HierarchyLevel.DIALOGUE: None,
        }
        return mapping[level]

    def _get_parent_level(self, level: HierarchyLevel) -> Optional[HierarchyLevel]:
        mapping = {
            HierarchyLevel.DOMAIN: None,
            HierarchyLevel.KEYWORD: HierarchyLevel.DOMAIN,
            HierarchyLevel.DIALOGUE: HierarchyLevel.KEYWORD,
        }
        return mapping[level]

    def _get_parent_context_text(self, node: HierarchicalNode) -> str:
        ancestors: List[str] = []
        seen: set[str] = set()

        for keyword_id in node.parent_ids:
            keyword_node = self.vector_store.get_node(keyword_id, HierarchyLevel.KEYWORD)
            if keyword_node is None or keyword_node.id in seen:
                continue
            seen.add(keyword_node.id)
            ancestors.append(keyword_node.content)
            for domain_id in keyword_node.parent_ids:
                domain_node = self.vector_store.get_node(domain_id, HierarchyLevel.DOMAIN)
                if domain_node is None or domain_node.id in seen:
                    continue
                seen.add(domain_node.id)
                ancestors.append(domain_node.content)
        return ", ".join(ancestors) if ancestors else "none"

    def _cosine_similarity(self, vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError("embedding dimension mismatch")
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = sum(a * a for a in vector_a) ** 0.5
        norm_b = sum(b * b for b in vector_b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return -1.0
        return dot_product / (norm_a * norm_b)
