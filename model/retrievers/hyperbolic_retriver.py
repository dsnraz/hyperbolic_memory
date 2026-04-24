"""
基于双曲空间的分层检索器。

本模块提供：
1. `BaseHyperbolicRetriever`：双曲检索基类
2. `GeodesicHyperbolicRetriever`：测地线距离检索（`MemoryAugmentedLLMInference` 中 `hyperbolic_geodesic`）
3. `MultiParentAngularHyperbolicRetriever`：多父外角加权检索（`hyperbolic_angular`）
4. `HybridHyperbolicRetriever`：按 query 深度定分界层，上外角/下测地线混合
   （`MemoryAugmentedLLMInference` 中 `hyperbolic_angular_geodesic_hybrid`）

检索流程与余弦检索器一致，采用自顶向下的层级收缩策略：
1. 在起始层级的全部节点中检索 top-k
2. 收集这些节点的所有子节点
3. 在子节点集合中再次检索 top-k
4. 重复直到目标层级
"""

from __future__ import annotations

import glob
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.hyperbolic_utils import lorentz as L
from model.hyperbolic_utils.hyperbolic_projector import Hyperbolic_projector
from model.retrievers.base_retriever import HierarchicalRetrieverBase
from model.retrievers.result_types import (
    BaseHierarchicalRetrievalResult,
    BaseLevelRetrievalResult,
    BaseRetrievalHit,
)
from model.stores.hierarchical_vector_store import HierarchicalVectorStore


@dataclass
class HyperbolicRetrievalHit(BaseRetrievalHit):
    """
    双曲检索单个命中结果。

    opposite_score: 与「相似度/得分」对偶、用于**升序排序**（数值越小越靠前）的标量。
        - 测地线检索：query–节点测地线距离。
        - 多父外角重角加权重：``1 -`` 加权和相似度。
        - **无图父**：Lorentz 模型下参考点 O（空间 0）处三角形 O–q–c 的**双曲内角**（余弦定理由
          三边测地长给出）；该角越小越前，**非**欧氏空间分量夹角。
    """

    opposite_score: float


@dataclass
class HyperbolicLevelRetrievalResult(BaseLevelRetrievalResult[HyperbolicRetrievalHit]):
    """单层级双曲检索结果。"""


@dataclass
class HyperbolicRetrievalResult(
    BaseHierarchicalRetrievalResult[HyperbolicRetrievalHit, HyperbolicLevelRetrievalResult]
):
    """完整的双曲分层检索结果。"""

    query_depth: Optional[float]
    actual_start_level: HierarchyLevel
    actual_final_level: Optional[HierarchyLevel]
    reached_target_level: bool

    def as_tuples(self, return_opposite_score: bool = True) -> List[Tuple[HierarchicalNode, float]]:
        """转换为 `(node, opposite_score)` 或 `(node, score)` 结构。"""
        if return_opposite_score:
            return [(hit.node, hit.opposite_score) for hit in self.final_hits]
        return [(hit.node, hit.score) for hit in self.final_hits]


class BaseHyperbolicRetriever(HierarchicalRetrieverBase, ABC):
    """
    双曲检索基类，负责投影、缓存和分层检索流程。

    注意:
        - 该类会缓存各层节点的双曲投影表示；投射输入为 `HierarchicalNode.level_embedding`
          （带层级前缀的向量，与 `hierarchical_dataset` / 训练时 `use_level_embedding` 一致），
          而非原始 `embedding`。
        - 如果外部更新了 `vector_store` 中的节点或其 `level_embedding`，
          需要调用 `clear_cache()` / `rebuild_cache()`，或在 `retrieve`
          时传入 `force_rebuild_cache=True`。
    """

    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        checkpoint_path: Optional[str] = None,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        projector: Optional[Hyperbolic_projector] = None,
        device: Optional[str] = None,
        projection_batch_size: int = 512,
        level_depth_targets: Optional[Dict[str, float]] = None,
        reranker: Optional[Any] = None,
    ):
        super().__init__(
            vector_store=vector_store,
            embedding_function=embedding_function,
            reranker=reranker,
        )
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.projection_batch_size = projection_batch_size

        self.projector = projector or self._load_projector_from_checkpoint(checkpoint_path)
        self.projector = self.projector.to(self.device)
        self.projector.eval()
        self.level_depth_targets = level_depth_targets or {
            "DOMAIN": 0.1,
            "CATEGORY": 0.3,
            "KEYWORD": 0.5,
            "DIALOGUE": 0.7,
        }

        self._node_cache_by_level: Dict[HierarchyLevel, Dict[str, HierarchicalNode]] = {}
        self._hyperbolic_cache_by_level: Dict[HierarchyLevel, Dict[str, torch.Tensor]] = {}

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
        """执行自顶向下的双曲层级检索。"""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

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
            if adaptive_start_level else start_level
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

            child_level = current_level.get_child_level()
            if child_level is None:
                break

            current_candidates = self._collect_children([hit.node for hit in ranked_hits])
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
    ) -> List[Tuple[HierarchicalNode, float]]:
        """仅返回目标层级结果。"""
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

    def project_query(self, query_embedding: Sequence[float]) -> torch.Tensor:
        """将查询向量投影到双曲空间。"""
        self._validate_embedding_dim(query_embedding, source="query")
        projected = self._project_embeddings([query_embedding])
        return projected[0].unsqueeze(0)

    def clear_cache(self) -> None:
        """清空节点与双曲投影缓存。"""
        self._node_cache_by_level.clear()
        self._hyperbolic_cache_by_level.clear()

    def rebuild_cache(self, levels: Optional[List[HierarchyLevel]] = None) -> None:
        """重建指定层级的双曲缓存。"""
        target_levels = levels or list(HierarchyLevel)
        for level in target_levels:
            self._build_level_cache(level, force=True)

    def get_curvature(self) -> float:
        """获取当前 projector 的实际曲率。"""
        with torch.no_grad():
            return torch.nn.functional.softplus(self.projector.c).item()

    @abstractmethod
    def _rank_nodes(
        self,
        query_h: torch.Tensor,
        nodes: List[HierarchicalNode],
        top_k: int,
    ) -> List[HyperbolicRetrievalHit]:
        """对子节点集合进行排序。"""

    @abstractmethod
    def _similarity(
        self,
        query_h: torch.Tensor,
        node: HierarchicalNode,
    ) -> Tuple[float, float]:
        """
        单对 (query, 节点) 的 (score, opposite_score)：

        在**内部**通过 ``_get_projected_tensor(node)`` 与图结构取候选双曲向量和父信息，
        各子类按测地/外角/混合规则实现。无可用投影时建议 ``(0.0, inf)`` 以便排序沉底。
        """

    def _node_hyperbolic_euclidean(self, node: HierarchicalNode) -> Optional[Sequence[float]]:
        """双曲 projector 的欧氏输入：节点的 `level_embedding`（与训练数据管线对齐）。"""
        return node.level_embedding

    def _get_nodes_by_level(self, level: HierarchyLevel) -> List[HierarchicalNode]:
        self._build_level_cache(level)
        return list(self._node_cache_by_level.get(level, {}).values())

    def _build_level_cache(self, level: HierarchyLevel, force: bool = False) -> None:
        """懒加载某一层的双曲投影缓存。"""
        if level in self._node_cache_by_level and level in self._hyperbolic_cache_by_level and not force:
            return

        raw_nodes = self._deduplicate_nodes(self.vector_store.get_nodes_by_level(level))
        valid_nodes = [
            node for node in raw_nodes if self._node_hyperbolic_euclidean(node) is not None
        ]

        if valid_nodes:
            first_euclid = self._node_hyperbolic_euclidean(valid_nodes[0])
            assert first_euclid is not None
            self._validate_embedding_dim(
                first_euclid,
                source=f"{level.name} node level_embedding",
            )

        self._node_cache_by_level[level] = {node.id: node for node in valid_nodes}
        self._hyperbolic_cache_by_level[level] = {}

        if not valid_nodes:
            return

        euclid_vecs: List[Sequence[float]] = []
        for node in valid_nodes:
            v = self._node_hyperbolic_euclidean(node)
            assert v is not None
            euclid_vecs.append(v)
        projected_embeddings = self._project_embeddings(euclid_vecs)
        for idx, node in enumerate(valid_nodes):
            self._hyperbolic_cache_by_level[level][node.id] = projected_embeddings[idx]

    def _get_projected_tensor(self, node: HierarchicalNode) -> Optional[torch.Tensor]:
        """获取单个节点的双曲投影表示。"""
        self._build_level_cache(node.level)
        cached = self._hyperbolic_cache_by_level.get(node.level, {}).get(node.id)
        if cached is not None:
            return cached

        feat = self._node_hyperbolic_euclidean(node)
        if feat is None:
            return None

        self._validate_embedding_dim(feat, source=f"{node.level.name} node level_embedding")
        projected = self._project_embeddings([feat])[0]
        self._node_cache_by_level.setdefault(node.level, {})[node.id] = node
        self._hyperbolic_cache_by_level.setdefault(node.level, {})[node.id] = projected
        return projected

    def _project_embeddings(self, embeddings: List[Sequence[float]]) -> torch.Tensor:
        """批量将欧氏向量投影到双曲空间。"""
        if not embeddings:
            return torch.empty(0, 0, dtype=torch.float32)

        projected_batches: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(embeddings), self.projection_batch_size):
                end = start + self.projection_batch_size
                batch_embeddings = torch.tensor(
                    embeddings[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                _, batch_h = self.projector(batch_embeddings)
                projected_batches.append(batch_h.detach().cpu())

        return torch.cat(projected_batches, dim=0)

    def _load_projector_from_checkpoint(self, checkpoint_path: Optional[str]) -> Hyperbolic_projector:
        """从 checkpoint 加载训练好的双曲投射器。"""
        if checkpoint_path is None:
            raise ValueError("未提供 checkpoint_path，无法初始化双曲检索器")

        resolved_path = self._resolve_checkpoint_path(checkpoint_path)
        checkpoint = torch.load(resolved_path, map_location="cpu")
        config = checkpoint.get("config", {})

        projector = Hyperbolic_projector(
            input_dim=config.get("embedding_dim", 384),
            hidden_dim=config.get("hidden_dim", 256),
            curvature=config.get("initial_curvature", checkpoint.get("curvature", 0.1)),
            alpha=config.get("alpha", 0.1),
            beta=config.get("beta", 0.8),
        )
        projector.load_state_dict(checkpoint["model_state_dict"])
        return projector

    def _get_projector_input_dim(self) -> int:
        """获取 projector 期望的输入维度。"""
        first_linear = self.projector.phi[0]
        return first_linear.in_features

    def _validate_embedding_dim(self, embedding: Sequence[float], source: str) -> None:
        """校验 embedding 维度是否与 projector 输入一致。"""
        expected_dim = self._get_projector_input_dim()
        actual_dim = len(embedding)
        if actual_dim != expected_dim:
            raise ValueError(
                f"{source} 的 embedding 维度为 {actual_dim}，"
                f"但当前双曲投射器期望输入维度为 {expected_dim}"
            )

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        """解析 checkpoint 路径，支持直接传文件或目录。"""
        if os.path.isfile(checkpoint_path):
            return checkpoint_path

        if os.path.isdir(checkpoint_path):
            pt_files = sorted(
                glob.glob(os.path.join(checkpoint_path, "*.pt")),
                key=os.path.getmtime,
                reverse=True,
            )
            if not pt_files:
                raise FileNotFoundError(f"目录中未找到 checkpoint 文件: {checkpoint_path}")
            return pt_files[0]

        raise FileNotFoundError(f"checkpoint 路径不存在: {checkpoint_path}")

    def _compute_query_depth(self, query_h: torch.Tensor) -> float:
        """计算 query 到双曲原点的测地线距离。"""
        curv = self.get_curvature()
        origin = torch.zeros_like(query_h.cpu())
        distance = L.pairwise_dist_vectors(query_h.cpu(), origin, curv=curv).squeeze()
        return float(distance.item())

    def _select_start_level_by_depth(
        self,
        query_depth: float,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> HierarchyLevel:
        """根据 query 深度选择最接近的起始层级。"""
        candidate_levels = [
            level for level in self._get_level_range(start_level, target_level)
            if self._get_nodes_by_level(level)
        ]
        if not candidate_levels:
            return start_level

        curv = self.get_curvature()
        radius = 1.0 / (curv ** 0.5)

        return min(
            candidate_levels,
            key=lambda level: abs(
                query_depth - self.level_depth_targets[level.name] * radius
            ),
        )

    def _get_level_range(
        self,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> List[HierarchyLevel]:
        """返回从 start_level 到 target_level 的层级序列。"""
        ordered_levels = [
            HierarchyLevel.DOMAIN,
            HierarchyLevel.CATEGORY,
            HierarchyLevel.KEYWORD,
            HierarchyLevel.DIALOGUE,
        ]
        start_idx = ordered_levels.index(start_level)
        target_idx = ordered_levels.index(target_level)
        return ordered_levels[start_idx:target_idx + 1]

class GeodesicHyperbolicRetriever(BaseHyperbolicRetriever):
    """基于双曲测地线距离的检索器。"""

    def _similarity(
        self,
        query_h: torch.Tensor,
        node: HierarchicalNode,
    ) -> Tuple[float, float]:
        """测地线距离 d(q, node)；``opposite_score`` 为 d。"""
        projected = self._get_projected_tensor(node)
        if projected is None:
            return 0.0, float("inf")
        curv = self.get_curvature()
        distance = L.pairwise_dist_vectors(
            query_h.cpu(),
            projected.unsqueeze(0),
            curv=curv,
        ).squeeze()
        distance_value = float(distance.item())
        if not math.isfinite(distance_value):
            return 0.0, float("inf")
        score = 1.0 / (1.0 + distance_value)
        return score, distance_value

    def _rank_nodes(
        self,
        query_h: torch.Tensor,
        nodes: List[HierarchicalNode],
        top_k: int,
    ) -> List[HyperbolicRetrievalHit]:
        scored_hits: List[HyperbolicRetrievalHit] = []

        for node in nodes:
            score, geodesic_dist = self._similarity(query_h, node)
            if not math.isfinite(geodesic_dist):
                continue
            scored_hits.append(
                HyperbolicRetrievalHit(
                    node=node,
                    score=score,
                    opposite_score=geodesic_dist,
                )
            )

        scored_hits.sort(key=lambda hit: hit.opposite_score)
        return scored_hits[:top_k]


def _hyperbolic_spatial_row(h: torch.Tensor) -> torch.Tensor:
    """将双曲空间向量规范为 (D,) CPU float，供 `lorentz` 中成对函数使用。"""
    x = h.detach().float().cpu()
    if x.dim() == 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.dim() != 1:
        raise ValueError(f"期望双曲向量为 1D 或 (1, D)，得到形状 {tuple(h.shape)}")
    return x


def _pair_hyperbolic_angle_at_origin_scores(
    query_h: torch.Tensor,
    candidate_h: torch.Tensor,
    curv: float,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    在**同一 Lorentz 双曲面**表示下（`lorentz.pairwise_dist_vectors` / 空间分量的时间分量补全），
    设参考点 O 为**空间分量为 0** 的锚点，与点 p, q 构成双曲测地三角形；记
    \(a=d(p,q), b=d(O,p), c=d(O,q)\) 为三条**测地边长**，顶点在 O 处的**内角**由双曲
    余弦定理（空间形式，曲率与 ``curv`` 一致）：

    \(\cos \angle O\) 由 `lorentz.hyperbolic_law_of_cosines_angle` 实现（同式）。

    与在 \(\mathbb R^D\) 上对空间分量做**欧氏**夹角不同；此处角为流形上 O 点处两**测地射线**
    OP 与 OQ 的交角。夹角越小越相似。返回 ``(score, opposite_score)`` 与
    `BaseHyperbolicRetriever._similarity` 约定一致，``opposite_score`` 为弧度。
    """
    p = _hyperbolic_spatial_row(query_h).unsqueeze(0)
    n = _hyperbolic_spatial_row(candidate_h).unsqueeze(0)
    d0 = p.shape[1]
    o = torch.zeros(1, d0, dtype=p.dtype, device=p.device)
    a = L.pairwise_dist_vectors(p, n, curv=curv).squeeze()
    b = L.pairwise_dist_vectors(o, p, curv=curv).squeeze()
    c = L.pairwise_dist_vectors(o, n, curv=curv).squeeze()
    ang_t = L.hyperbolic_law_of_cosines_angle(a, b, c, eps=eps)
    if ang_t.numel() != 1:
        raise ValueError("期望成对 (query, 候选) 为单向量，得标量角。")
    ang = float(ang_t.item())
    if not math.isfinite(ang):
        return 0.0, float("inf")
    return (1.0 / (1.0 + ang), ang)


class MultiParentAngularHyperbolicRetriever(BaseHyperbolicRetriever):
    """
    多父场景：相对各父节点的外角向量（`lorentz.pairwise_exterior_angle_vectors`，与
    `HierarchicalAngularContrastiveLoss` 中的外角矩阵一致）。

    「夹角」与「外角」见 `lorentz.pairwise_exterior_angle_vectors`。**有图父**时 ``parents_h``
    为真父行，对 query 与候选分别算外角向量后做重角加权重余弦。
    **无图父**：Lorentz 模型下，用相对参考点 O（空间 0）的**双曲**三角形边长与
    **双曲余弦定理**得 O 处内角，夹角越小越相似；`opposite_score` 为该内角（弧度），
    `score = 1/(1+opposite_score)`。

    有父时排序：对每个父 \(i\) 计算 \(\mathrm{Sim}_i=\frac{1+\cos(\alpha_Q^i-\alpha_D^i)}{2}\)，
    加权后 `opposite_score = 1 - \text{加权和 sim}`。

    父权重（有父时至少开启一种，否则构造时报错）：
    - `weight_by_parent_origin_geodesic`：父到原点测地线越远，归一化后权重越大；
    - `weight_by_parent_anchor_geodesic`：父到锚点（query 或候选 node，见 `parent_geodesic_anchor`）
      测地线越近，归一化后权重越大；二者同时开启时先各自归一化再逐元相乘后再次归一化。
    """

    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        checkpoint_path: Optional[str] = None,
        *,
        weight_by_parent_origin_geodesic: bool = True,
        weight_by_parent_anchor_geodesic: bool = False,
        parent_geodesic_anchor: Literal["query", "node"] = "query",
        weight_eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        if not weight_by_parent_origin_geodesic and not weight_by_parent_anchor_geodesic:
            raise ValueError(
                "MultiParentAngularHyperbolicRetriever 至少需开启一种父节点加权："
                "weight_by_parent_origin_geodesic 或 weight_by_parent_anchor_geodesic。"
            )
        if parent_geodesic_anchor not in ("query", "node"):
            raise ValueError("parent_geodesic_anchor 必须为 'query' 或 'node'。")
        self.weight_by_parent_origin_geodesic = weight_by_parent_origin_geodesic
        self.weight_by_parent_anchor_geodesic = weight_by_parent_anchor_geodesic
        self.parent_geodesic_anchor = parent_geodesic_anchor
        self.weight_eps = weight_eps
        super().__init__(vector_store, checkpoint_path, **kwargs)

    def _similarity(
        self,
        query_h: torch.Tensor,
        node: HierarchicalNode,
    ) -> Tuple[float, float]:
        """
        无图父：O 点双曲内角；有父：多父外角逐父加权重余弦。内部取 ``node`` 的投影与父节点。
        """
        projected = self._get_projected_tensor(node)
        if projected is None:
            return 0.0, float("inf")
        parents_h = self._gather_parent_hyperbolic_matrix(node)
        if parents_h is None or parents_h.shape[0] == 0:
            return _pair_hyperbolic_angle_at_origin_scores(
                query_h, projected, self.get_curvature()
            )
        opposite_score = self._weighted_cosine_angular_opposite_score(
            query_h, projected, parents_h
        )
        return (1.0 - opposite_score), opposite_score

    @staticmethod
    def compute_exterior_angle_vector_relative_to_parents(
        point_h: torch.Tensor,
        parents_h: torch.Tensor,
        curv: float,
    ) -> torch.Tensor:
        """
        计算点相对一组父节点的外角向量，形状 (P,)。

        第 i 个分量表示以 `parents_h[i]` 为锥顶点、`point` 为另一端时，
        `pairwise_exterior_angle_vectors` 给出的外角（与层级角度对比损失中的外角一致）。
        """
        par = parents_h.detach().float().cpu()
        if par.dim() != 2:
            raise ValueError(f"parents_h 期望形状 (P, D)，得到 {tuple(par.shape)}")
        if par.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32)
        point = _hyperbolic_spatial_row(point_h).unsqueeze(0)
        if par.shape[1] != point.shape[1]:
            raise ValueError(
                f"父向量维 {par.shape[1]} 与 point 维 {point.shape[1]} 不一致"
            )
        angles = L.pairwise_exterior_angle_vectors(par, point, curv=curv)
        return angles.squeeze(-1)

    def _gather_parent_hyperbolic_matrix(
        self, node: HierarchicalNode
    ) -> Optional[torch.Tensor]:
        """按 `node.parent_ids` 顺序收集父节点双曲坐标，形状 (P, D)。无父时返回 `None`。"""
        parent_level = node.level.get_parent_level()
        if parent_level is None or not node.parent_ids:
            return None
        rows: List[torch.Tensor] = []
        for pid in node.parent_ids:
            pnode = self.vector_store.get_node(pid, parent_level)
            if pnode is None:
                continue
            ph = self._get_projected_tensor(pnode)
            if ph is None:
                continue
            rows.append(_hyperbolic_spatial_row(ph))
        if not rows:
            return None
        return torch.stack(rows, dim=0)

    def _normalize_parent_weights(self, raw: torch.Tensor) -> torch.Tensor:
        """非负权重归一化为概率向量；全零时退化为均匀。"""
        z = raw.clamp(min=0.0)
        s = z.sum()
        if s < self.weight_eps:
            n = z.numel()
            return torch.full_like(z, 1.0 / max(n, 1))
        return z / s

    def _parent_aggregation_weights(
        self,
        parents_h: torch.Tensor,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        curv: float,
    ) -> torch.Tensor:
        """
        形状 (P,) 的父权重，和为 1。按实例标志组合「原点远则大」与「锚点近则大」两种因子。
        """
        par = parents_h.detach().float().cpu()
        p_count = par.shape[0]
        w = torch.ones(p_count, dtype=torch.float32)

        if self.weight_by_parent_origin_geodesic:
            origin = torch.zeros(1, par.shape[1], dtype=par.dtype, device=par.device)
            d_origin = L.pairwise_dist_vectors(par, origin, curv=curv).squeeze(-1)
            w = w * self._normalize_parent_weights(d_origin)

        if self.weight_by_parent_anchor_geodesic:
            if self.parent_geodesic_anchor == "query":
                anchor = _hyperbolic_spatial_row(query_h).unsqueeze(0)
            else:
                anchor = _hyperbolic_spatial_row(node_h).unsqueeze(0)
            d_anchor = L.pairwise_dist_vectors(par, anchor, curv=curv).squeeze(-1)
            inv = 1.0 / (d_anchor + self.weight_eps)
            w = w * self._normalize_parent_weights(inv)

        return self._normalize_parent_weights(w)

    def _weighted_cosine_angular_opposite_score(
        self,
        query_h: torch.Tensor,
        node_h: torch.Tensor,
        parents_h: torch.Tensor,
    ) -> float:
        """
        逐父 \(\mathrm{Sim}_i=(1+\cos(\alpha_Q^i-\alpha_D^i))/2\)，加权得 `score` 后返回 `1 - score`
        （即写入 `HyperbolicRetrievalHit.opposite_score` 的量，非测地线距离）。
        """
        curv = self.get_curvature()
        aq = self.compute_exterior_angle_vector_relative_to_parents(
            query_h, parents_h, curv
        )
        an = self.compute_exterior_angle_vector_relative_to_parents(
            node_h, parents_h, curv
        )
        if aq.numel() == 0:
            return 0.0
        delta = aq - an
        print(f"delta: {delta}")
        sim_i = (1.0 + torch.cos(delta)) * 0.5
        w = self._parent_aggregation_weights(parents_h, query_h, node_h, curv)
        score = float((w * sim_i).sum().item())
        score = min(1.0, max(0.0, score))
        return 1.0 - score

    def _rank_nodes(
        self,
        query_h: torch.Tensor,
        nodes: List[HierarchicalNode],
        top_k: int,
    ) -> List[HyperbolicRetrievalHit]:
        scored_hits: List[HyperbolicRetrievalHit] = []
        for node in nodes:
            score, opposite_score = self._similarity(query_h, node)
            if not math.isfinite(opposite_score):
                continue
            scored_hits.append(
                HyperbolicRetrievalHit(
                    node=node,
                    score=score,
                    opposite_score=opposite_score,
                )
            )
        scored_hits.sort(key=lambda hit: hit.opposite_score)
        return scored_hits[:top_k]


class HybridHyperbolicRetriever(GeodesicHyperbolicRetriever):
    """
    用与基类 `adaptive_start_level` 相同规则（query 测地深对比各层半深）选出「与 query
    最近」的 `boundary` 仅用于**划分本层用外角分还是测地分**；检索路径**始终**从
    `start_level` 逐层向下到 `target_level`（domain→dialogue），不从 `boundary` 起跳。

    分层与分界：本类 ``_rank_nodes`` **只** 调用 ``self._similarity``；`retrieve` 的每一层
    也**只**调用本类 ``_rank_nodes``。``index(node.level) < index(boundary)`` 时由
    ``_similarity`` 内部走多父外角 / O 点内角（经 ``self._angular._similarity`` 计算）；
    否则走测地线（`GeodesicHyperbolicRetriever._similarity`）。`adaptive_start_level=False` 时
    ``boundary = start_level``，可全程测地。

    分界层在每次 `retrieve` 中写入；未 `retrieve` 且未 `set_hybrid_scoring_boundary` 时
    可为 `None`，此时 `_similarity` 仅测地线。

    返回的 `HyperbolicRetrievalResult` 与其它双曲检索器相同；`actual_start_level` 为这次 walk
    起点，即参数 `start_level`。
    """

    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        checkpoint_path: Optional[str] = None,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        projector: Optional[Hyperbolic_projector] = None,
        device: Optional[str] = None,
        projection_batch_size: int = 512,
        level_depth_targets: Optional[Dict[str, float]] = None,
        reranker: Optional[Any] = None,
        *,
        hyperbolic_angular_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            vector_store=vector_store,
            checkpoint_path=checkpoint_path,
            embedding_function=embedding_function,
            projector=projector,
            device=device,
            projection_batch_size=projection_batch_size,
            level_depth_targets=level_depth_targets,
            reranker=reranker,
        )
        self._hybrid_scoring_geodesic_from_level: Optional[HierarchyLevel] = HierarchyLevel.KEYWORD
        akw = dict(hyperbolic_angular_kwargs or {})
        self._angular = MultiParentAngularHyperbolicRetriever(
            vector_store=vector_store,
            checkpoint_path=checkpoint_path,
            embedding_function=embedding_function,
            projector=self.projector,
            device=str(self.device) if self.device else None,
            projection_batch_size=projection_batch_size,
            level_depth_targets=level_depth_targets,
            reranker=reranker,
            **akw,
        )

    def clear_cache(self) -> None:
        super().clear_cache()
        self._angular.clear_cache()

    def rebuild_cache(self, levels: Optional[List[HierarchyLevel]] = None) -> None:
        super().rebuild_cache(levels)
        self._angular.rebuild_cache(levels)

    def set_hybrid_scoring_boundary(
        self, boundary: Optional[HierarchyLevel]
    ) -> None:
        """
        设置成对算分用分界层（与 `retrieve` 中 ``adaptive_start_level`` 算出的 ``boundary`` 同义）：
        ``index(node.level) < index(boundary)`` 时走角度检索子模块，否则走测地线。
        传入 ``None`` 表示不区分、`_similarity` 仅用测地线。
        """
        self._hybrid_scoring_geodesic_from_level = boundary

    def _rank_nodes(
        self,
        query_h: torch.Tensor,
        nodes: List[HierarchicalNode],
        top_k: int,
    ) -> List[HyperbolicRetrievalHit]:
        """
        仅用 ``Hybrid._similarity`` 按层与 `boundary` 在角分 / 测地分之间切换，不转调
        其它类的 ``_rank_nodes``。
        """
        print("开始检索一层，当前层级为：", nodes[0].level)
        scored_hits: List[HyperbolicRetrievalHit] = []
        for node in nodes:
            score, opposite_score = self._similarity(query_h, node)
            if not math.isfinite(opposite_score):
                continue
            scored_hits.append(
                HyperbolicRetrievalHit(
                    node=node,
                    score=score,
                    opposite_score=opposite_score,
                )
            )
        scored_hits.sort(key=lambda hit: hit.opposite_score)
        return scored_hits[:top_k]

    def _similarity(
        self,
        query_h: torch.Tensor,
        node: HierarchicalNode,
    ) -> Tuple[float, float]:
        print("开始计算相似度")
        print("当前分界层为：", self._hybrid_scoring_geodesic_from_level)
        boundary = self._hybrid_scoring_geodesic_from_level
        if boundary is None:
            print("使用测地线检索")
            return GeodesicHyperbolicRetriever._similarity(self, query_h, node)
        b_idx = self.LEVEL_ORDER.index(boundary)
        n_idx = self.LEVEL_ORDER.index(node.level)
        if n_idx < b_idx:
            print("使用多父外角检索")
            return self._angular._similarity(query_h, node)
        return GeodesicHyperbolicRetriever._similarity(self, query_h, node)

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
            raise ValueError("top_k 必须大于 0")

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

        if adaptive_start_level:
            boundary = self._select_start_level_by_depth(
                query_depth, start_level, target_level
            )
        else:
            boundary = start_level

        self._hybrid_scoring_geodesic_from_level = boundary
        level_results: List[HyperbolicLevelRetrievalResult] = []
        current_level = start_level
        current_candidates = self._get_nodes_by_level(current_level)

        while True:
            ranked_hits = self._rank_nodes(query_h, current_candidates, top_k)
            print("检索完一层，当前层级为：", current_level)
            level_results.append(
                HyperbolicLevelRetrievalResult(
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

        return HyperbolicRetrievalResult(
            query_text=query_text,
            query_embedding=prepared_query_embedding,
            top_k=top_k,
            start_level=start_level,
            target_level=target_level,
            level_results=level_results,
            query_depth=query_depth,
            actual_start_level=start_level,
            actual_final_level=level_results[-1].level if level_results else None,
            reached_target_level=bool(level_results) and level_results[-1].level == target_level,
        )