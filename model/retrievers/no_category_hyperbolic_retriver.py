from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.retrievers.hyperbolic_retriver import (
    BaseHyperbolicRetriever,
    GeodesicHyperbolicRetriever,
    HybridHyperbolicRetriever,
    HyperbolicLevelRetrievalResult,
    HyperbolicRetrievalResult,
    MultiParentAngularHyperbolicRetriever,
)


NO_CATEGORY_LEVEL_ORDER = [
    HierarchyLevel.DOMAIN,
    HierarchyLevel.KEYWORD,
    HierarchyLevel.DIALOGUE,
]


class _NoCategoryTraversalMixin:
    LEVEL_ORDER = NO_CATEGORY_LEVEL_ORDER

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

    def _validate_level_path(
        self,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> None:
        if self.LEVEL_ORDER.index(start_level) > self.LEVEL_ORDER.index(target_level):
            raise ValueError("target_level must not be above start_level")

    def _get_level_range(
        self,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> List[HierarchyLevel]:
        start_idx = self.LEVEL_ORDER.index(start_level)
        target_idx = self.LEVEL_ORDER.index(target_level)
        return self.LEVEL_ORDER[start_idx : target_idx + 1]

    def _select_start_level_by_depth(
        self,
        query_depth: float,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> HierarchyLevel:
        candidate_levels = [
            level
            for level in self._get_level_range(start_level, target_level)
            if self._get_nodes_by_level(level)
        ]
        if not candidate_levels:
            return start_level

        curv = self.get_curvature()
        radius = 1.0 / (curv ** 0.5)
        targets = self.level_depth_targets or {
            "DOMAIN": 0.15,
            "KEYWORD": 0.5,
            "DIALOGUE": 0.8,
        }
        return min(
            candidate_levels,
            key=lambda level: abs(query_depth - targets[level.name] * radius),
        )

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

    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
        force_rebuild_cache: bool = False,
        adaptive_start_level: bool = False,
    ) -> HyperbolicRetrievalResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._validate_level_path(start_level, target_level)

        cache_might_be_stale = False
        if self.vector_store.get_pending_dirty_count() > 0:
            self.vector_store.flush()
            cache_might_be_stale = True
        if force_rebuild_cache or cache_might_be_stale:
            self.clear_cache()

        prepared_query_embedding = self._prepare_query_embedding(query_text, query_embedding)
        query_h = self.project_query(prepared_query_embedding)
        query_depth = self._compute_query_depth(query_h)
        resolved_start_level = (
            self._select_start_level_by_depth(query_depth, start_level, target_level)
            if adaptive_start_level
            else start_level
        )

        level_results: List[HyperbolicLevelRetrievalResult] = []
        current_level = resolved_start_level
        current_candidates = self._get_nodes_by_level(current_level)
        while True:
            ranked_hits = self._rank_nodes(query_h, current_candidates, top_k)
            level_results.append(
                HyperbolicLevelRetrievalResult(
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

        return HyperbolicRetrievalResult(
            query_text=query_text,
            query_embedding=prepared_query_embedding,
            top_k=top_k,
            start_level=start_level,
            target_level=target_level,
            level_results=level_results,
            query_depth=query_depth,
            actual_start_level=resolved_start_level,
            actual_final_level=level_results[-1].level if level_results else None,
            reached_target_level=bool(level_results) and level_results[-1].level == target_level,
        )

    def retrieve_as_tuples(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
        return_opposite_score: bool = True,
        force_rebuild_cache: bool = False,
        adaptive_start_level: bool = False,
    ):
        result = self.retrieve(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k,
            start_level=start_level,
            target_level=target_level,
            force_rebuild_cache=force_rebuild_cache,
            adaptive_start_level=adaptive_start_level,
        )
        return result.as_tuples(return_opposite_score=return_opposite_score)

    def _gather_parent_hyperbolic_matrix(self, node: HierarchicalNode) -> Optional[torch.Tensor]:
        parent_level = self._get_parent_level(node.level)
        if parent_level is None or not node.parent_ids:
            return None
        rows: List[torch.Tensor] = []
        for parent_id in node.parent_ids:
            parent_node = self.vector_store.get_node(parent_id, parent_level)
            if parent_node is None:
                continue
            parent_h = self._get_projected_tensor(parent_node)
            if parent_h is not None:
                rows.append(parent_h.detach().float().cpu())
        if not rows:
            return None
        return torch.stack(rows, dim=0)


class NoCategoryGeodesicHyperbolicRetriever(_NoCategoryTraversalMixin, GeodesicHyperbolicRetriever):
    pass


class NoCategoryMultiParentAngularHyperbolicRetriever(
    _NoCategoryTraversalMixin,
    MultiParentAngularHyperbolicRetriever,
):
    pass


class NoCategoryHybridHyperbolicRetriever(_NoCategoryTraversalMixin, HybridHyperbolicRetriever):
    def __init__(
        self,
        *args: Any,
        hyperbolic_angular_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        BaseHyperbolicRetriever.__init__(self, *args, **kwargs)
        self._hybrid_scoring_geodesic_from_level = HierarchyLevel.KEYWORD
        akw = dict(hyperbolic_angular_kwargs or {})
        self._angular = NoCategoryMultiParentAngularHyperbolicRetriever(
            vector_store=self.vector_store,
            checkpoint_path=None,
            embedding_function=self.embedding_function,
            projector=self.projector,
            device=str(self.device) if self.device else None,
            projection_batch_size=self.projection_batch_size,
            level_depth_targets=self.level_depth_targets,
            reranker=self.reranker,
            **akw,
        )
