from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence

from model.hierarchical.hierarchy_types import HierarchicalNode, HierarchyLevel
from model.retrievers.result_types import BaseHierarchicalRetrievalResult, BaseRetrievalHit
from model.stores.hierarchical_vector_store import HierarchicalVectorStore


class HierarchicalRetrieverBase(ABC):
    """分层检索器公共基类，提供上下文拼装接口。"""

    LEVEL_ORDER = [
        HierarchyLevel.DOMAIN,
        HierarchyLevel.CATEGORY,
        HierarchyLevel.KEYWORD,
        HierarchyLevel.DIALOGUE,
    ]

    def __init__(
        self,
        vector_store: HierarchicalVectorStore,
        embedding_function: Optional[Callable[[str], Sequence[float]]] = None,
        reranker: Optional[Any] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_function = embedding_function or vector_store.embedding_function
        self.reranker = reranker

    @abstractmethod
    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
        **kwargs: Any,
    ) -> BaseHierarchicalRetrievalResult:
        """执行检索并返回分层结果。"""

    def get_context(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[Sequence[float]] = None,
        top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
        retrieval_result: Optional[BaseHierarchicalRetrievalResult] = None,
        **retrieve_kwargs: Any,
    ) -> str:
        """
        将最终检索命中拼装成可直接供大模型消费的上下文。

        如果已持有 retrieval_result，可直接传入避免重复检索。
        """
        if retrieval_result is None:
            retrieval_result = self.retrieve(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                start_level=start_level,
                target_level=target_level,
                **retrieve_kwargs,
            )

        reranked_hits = self._rerank_hits(retrieval_result.final_hits)
        if not reranked_hits:
            return ""

        return "\n\n".join(
            self._format_memory_fragment(hit, idx + 1, rerank_score)
            for idx, (hit, rerank_score) in enumerate(reranked_hits)
        )

    def _rerank_hits(
        self,
        hits: Sequence[BaseRetrievalHit],
    ) -> List[tuple[BaseRetrievalHit, Optional[float]]]:
        """
        精排占位接口。

        当前不引入精排模型，因此统一返回 `None` 作为置信度。
        """
        return [(hit, None) for hit in hits]

    def _format_memory_fragment(
        self,
        hit: BaseRetrievalHit,
        memory_idx: int,
        rerank_score: Optional[float],
    ) -> str:
        """把单个命中格式化为记忆片段文本。"""
        parent_context = self._get_parent_context_text(hit.node)
        confidence = "none" if rerank_score is None else str(rerank_score)
        # print(f"节点{memory_idx}分数：",hit.score)
        return (
            f"[记忆片段 #{memory_idx}]\n"
            # f"相关领域/父节点：{parent_context}\n"
            # f"置信度：{confidence}\n"
            f"{hit.node.content}"
        )

    def _get_parent_context_text(self, node: HierarchicalNode) -> str:
        """提取节点的祖先内容，作为片段的结构化上下文。"""
        ancestors = self.vector_store.get_ancestors(node.id, node.level)
        if not ancestors:
            return "无"

        sorted_ancestors = sorted(
            ancestors,
            key=lambda ancestor: self.LEVEL_ORDER.index(ancestor.level),
        )
        seen_contents = set()
        ordered_contents: List[str] = []
        for ancestor in sorted_ancestors:
            if not ancestor.content or ancestor.content in seen_contents:
                continue
            seen_contents.add(ancestor.content)
            ordered_contents.append(ancestor.content)

        return ", ".join(ordered_contents) if ordered_contents else "无"

    def _prepare_query_embedding(
        self,
        query_text: Optional[str],
        query_embedding: Optional[Sequence[float]],
    ) -> List[float]:
        if query_embedding is not None:
            return list(query_embedding)

        if query_text is None:
            raise ValueError("query_text 和 query_embedding 至少需要提供一个")

        if self.embedding_function is None:
            raise ValueError("未配置 embedding_function，无法根据 query_text 生成向量")

        return list(self.embedding_function(query_text))

    def _collect_children(self, parent_nodes: List[HierarchicalNode]) -> List[HierarchicalNode]:
        children: List[HierarchicalNode] = []
        for parent_node in parent_nodes:
            children.extend(self.vector_store.get_children(parent_node.id, parent_node.level))
        return self._deduplicate_nodes(children)

    def _deduplicate_nodes(self, nodes: List[HierarchicalNode]) -> List[HierarchicalNode]:
        unique_nodes = {}
        for node in nodes:
            unique_nodes[node.id] = node
        return list(unique_nodes.values())

    def _validate_level_path(
        self,
        start_level: HierarchyLevel,
        target_level: HierarchyLevel,
    ) -> None:
        if self.LEVEL_ORDER.index(start_level) > self.LEVEL_ORDER.index(target_level):
            raise ValueError("target_level 必须位于 start_level 的同层或更低层")
