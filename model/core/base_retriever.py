"""
记忆检索器抽象基类。

本模块定义记忆检索系统的抽象接口。
记忆检索器从存储中搜索和检索相关记忆。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .memory_item import MemoryItem
from .memory_types import MemoryType, MemoryPriority
from .base_store import BaseMemoryStore


@dataclass
class RetrievalQuery:
    """
    记忆检索查询。
    
    支持多种查询类型和筛选器，实现灵活的检索。
    """
    
    # 查询内容
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    query_type: str = "semantic"  # semantic, keyword, temporal, hybrid
    
    # 筛选器
    memory_types: Optional[List[MemoryType]] = None
    priorities: Optional[List[MemoryPriority]] = None
    tags: Optional[List[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    min_strength: Optional[float] = None
    max_strength: Optional[float] = None
    
    # 关联筛选
    associated_with: Optional[List[str]] = None  # 用于查找关联的记忆ID
    
    # 检索参数
    top_k: int = 10
    min_similarity: float = 0.0
    diversity_threshold: float = 0.0  # 用于多样性检索
    
    # 上下文
    context: Optional[Dict[str, Any]] = None  # 检索的额外上下文


@dataclass
class RetrievalResult:
    """
    记忆检索操作的结果。
    
    包含检索到的记忆及其相关性分数和元数据。
    """
    
    memories: List[Tuple[MemoryItem, float]]  # (记忆条目, 相关性分数)
    query: RetrievalQuery
    total_candidates: int  # 筛选前考虑的记忆总数
    retrieval_time_ms: float  # 检索耗时（毫秒）
    
    def get_memories(self) -> List[MemoryItem]:
        """仅获取记忆条目，不含分数。"""
        return [item for item, score in self.memories]
    
    def get_scores(self) -> List[float]:
        """仅获取相关性分数。"""
        return [score for item, score in self.memories]
    
    def get_top_memory(self) -> Optional[MemoryItem]:
        """获取最相关的记忆。"""
        if not self.memories:
            return None
        return self.memories[0][0]
    
    def get_top_score(self) -> Optional[float]:
        """获取最高的相关性分数。"""
        if not self.memories:
            return None
        return self.memories[0][1]
    
    def filter_by_threshold(self, threshold: float) -> "RetrievalResult":
        """按相关性阈值筛选结果。"""
        filtered_memories = [
            (item, score) for item, score in self.memories 
            if score >= threshold
        ]
        return RetrievalResult(
            memories=filtered_memories,
            query=self.query,
            total_candidates=self.total_candidates,
            retrieval_time_ms=self.retrieval_time_ms
        )


class BaseMemoryRetriever(ABC):
    """
    记忆检索系统的抽象基类。
    
    记忆检索器根据各种查询类型和检索策略
    搜索和检索相关记忆。
    
    遵循认知检索过程：
    1. 提供检索线索（查询）
    2. 搜索记忆痕迹的相关性
    3. 按相关性排序匹配的记忆
    4. 返回最相关的候选
    
    不同的检索器实现可以使用：
    - 向量相似度搜索（语义检索）
    - 关键词匹配（词汇检索）
    - 时间筛选（时间检索）
    - 关联遍历（图检索）
    - 混合方法（组合多种策略）
    """
    
    def __init__(
        self,
        stores: Dict[MemoryType, BaseMemoryStore],
        embedding_model: Optional[Any] = None,
        **kwargs
    ):
        """
        初始化记忆检索器。
        
        参数:
            stores: 记忆类型到其存储的映射字典
            embedding_model: 用于计算查询嵌入的模型（占位符）
            **kwargs: 额外的配置参数
        """
        self.stores = stores
        self.embedding_model = embedding_model
        self.config = kwargs
    
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        根据查询检索记忆。
        
        这是主要的检索方法。实现应该：
        1. 处理查询
        2. 如需要则生成查询嵌入
        3. 搜索相关的记忆存储
        4. 排序和筛选结果
        5. 返回检索结果
        
        参数:
            query: 检索查询
            
        返回:
            包含检索到的记忆的 RetrievalResult
            
        注意:
            此方法应由子类实现。
            具体的检索逻辑取决于查询类型和使用的检索策略。
        """
        pass
    
    @abstractmethod
    def compute_relevance(
        self, 
        query: RetrievalQuery, 
        memory: MemoryItem
    ) -> float:
        """
        计算查询和记忆之间的相关性分数。
        
        这是检索的核心评分函数。
        
        参数:
            query: 检索查询
            memory: 要评分的记忆条目
            
        返回:
            相关性分数（0.0 到 1.0）
            
        注意:
            在此实现你自己的相关性计算逻辑。
            常见考虑因素：
            - 语义相似度（嵌入余弦相似度）
            - 关键词重叠
            - 时间邻近性
            - 记忆强度/重要性
            - 关联强度
        """
        pass
    
    # 查询嵌入的占位符方法
    def generate_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        为查询文本生成嵌入。
        
        这是一个占位符方法。在子类中实现你自己的嵌入逻辑
        或重写此方法。
        
        参数:
            query_text: 要嵌入的查询文本
            
        返回:
            查询嵌入向量，若不支持则返回 None
            
        注意:
            用你实际的嵌入生成逻辑替换此方法。
        """
        # 占位符 - 在此实现你的嵌入逻辑
        return None
    
    # 不同检索策略的辅助方法
    
    def retrieve_by_type(
        self, 
        memory_type: MemoryType,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """
        检索特定类型的所有记忆。
        
        参数:
            memory_type: 要检索的记忆类型
            top_k: 返回的最大记忆数
            
        返回:
            记忆条目列表
        """
        if memory_type not in self.stores:
            return []
        return self.stores[memory_type].get_all()[:top_k]
    
    def retrieve_by_association(
        self, 
        memory_id: str,
        top_k: int = 10
    ) -> List[MemoryItem]:
        """
        检索与给定记忆相关联的记忆。
        
        占位符方法 - 实现你自己的关联检索逻辑。
        
        参数:
            memory_id: 参考记忆的ID
            top_k: 返回的最大记忆数
            
        返回:
            关联的记忆条目列表
        """
        # 占位符 - 在此实现你的关联检索逻辑
        results = []
        for store in self.stores.values():
            for memory in store.get_all():
                if memory_id in memory.associations:
                    results.append(memory)
        return results[:top_k]
    
    def retrieve_recent(
        self,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[MemoryItem]:
        """
        检索最近访问的记忆。
        
        参数:
            top_k: 返回的最大记忆数
            memory_types: 可选的记忆类型筛选
            
        返回:
            最近访问的记忆条目列表
        """
        candidates = []
        types_to_search = memory_types or list(self.stores.keys())
        
        for memory_type in types_to_search:
            if memory_type in self.stores:
                candidates.extend(self.stores[memory_type].get_all())
        
        # 按最后访问时间排序
        sorted_candidates = sorted(
            candidates,
            key=lambda m: m.metadata.last_accessed,
            reverse=True
        )
        return sorted_candidates[:top_k]
    
    def retrieve_strongest(
        self,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[MemoryItem]:
        """
        检索最强的记忆（强度分数最高）。
        
        参数:
            top_k: 返回的最大记忆数
            memory_types: 可选的记忆类型筛选
            
        返回:
            最强的记忆条目列表
        """
        candidates = []
        types_to_search = memory_types or list(self.stores.keys())
        
        for memory_type in types_to_search:
            if memory_type in self.stores:
                candidates.extend(self.stores[memory_type].get_all())
        
        # 按强度排序
        sorted_candidates = sorted(
            candidates,
            key=lambda m: m.get_strength(),
            reverse=True
        )
        return sorted_candidates[:top_k]
    
    # 多查询检索
    def retrieve_multi_query(
        self, 
        queries: List[RetrievalQuery],
        merge_strategy: str = "union"
    ) -> RetrievalResult:
        """
        为多个查询检索记忆。
        
        参数:
            queries: 检索查询列表
            merge_strategy: 如何合并结果（'union', 'intersection', 'weighted'）
            
        返回:
            合并后的 RetrievalResult
        """
        results = [self.retrieve(query) for query in queries]
        
        if merge_strategy == "union":
            # 合并所有结果，按记忆ID去重
            seen_ids = set()
            combined_memories = []
            for result in results:
                for memory, score in result.memories:
                    if memory.id not in seen_ids:
                        seen_ids.add(memory.id)
                        combined_memories.append((memory, score))
            return RetrievalResult(
                memories=combined_memories,
                query=queries[0],  # 使用第一个查询作为参考
                total_candidates=sum(r.total_candidates for r in results),
                retrieval_time_ms=sum(r.retrieval_time_ms for r in results)
            )
        
        elif merge_strategy == "intersection":
            # 只保留在所有结果中都出现的记忆
            memory_scores: Dict[str, List[float]] = {}
            for result in results:
                for memory, score in result.memories:
                    if memory.id not in memory_scores:
                        memory_scores[memory.id] = []
                    memory_scores[memory.id].append((memory, score))
            
            # 筛选在所有结果中都出现的记忆
            intersection_memories = []
            for memory_id, scores_list in memory_scores.items():
                if len(scores_list) == len(queries):
                    # 平均分数
                    avg_score = sum(s for m, s in scores_list) / len(scores_list)
                    intersection_memories.append((scores_list[0][0], avg_score))
            
            return RetrievalResult(
                memories=intersection_memories,
                query=queries[0],
                total_candidates=sum(r.total_candidates for r in results),
                retrieval_time_ms=sum(r.retrieval_time_ms for r in results)
            )
        
        else:
            # 默认使用 union
            return self.retrieve_multi_query(queries, merge_strategy="union")


class SemanticRetriever(BaseMemoryRetriever):
    """
    使用语义相似度的检索器示例。
    
    这是一个展示语义检索器结构的模板类。
    在 retrieve 方法中实现你自己的检索逻辑。
    """
    
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        使用语义相似度检索记忆。
        
        注意：在此实现你实际的检索逻辑。
        这是一个占位符实现。
        """
        import time
        start_time = time.time()
        
        # 占位符 - 在此实现你实际的检索逻辑
        candidates = []
        
        # 从相关存储收集候选
        types_to_search = query.memory_types or list(self.stores.keys())
        for memory_type in types_to_search:
            if memory_type in self.stores:
                candidates.extend(self.stores[memory_type].get_all())
        
        # 计算相关性分数
        scored_memories = []
        for memory in candidates:
            score = self.compute_relevance(query, memory)
            if score >= query.min_similarity:
                scored_memories.append((memory, score))
        
        # 按相关性排序并取 top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = scored_memories[:query.top_k]
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            memories=top_memories,
            query=query,
            total_candidates=len(candidates),
            retrieval_time_ms=retrieval_time
        )
    
    def compute_relevance(
        self, 
        query: RetrievalQuery, 
        memory: MemoryItem
    ) -> float:
        """
        计算查询和记忆之间的语义相似度。
        
        注意：在此实现你实际的相关性计算。
        占位符使用记忆强度作为代理。
        """
        # 占位符 - 在此实现你实际的相关性计算
        # 这里仅使用记忆强度作为代理分数
        return memory.get_strength()


class HybridRetriever(BaseMemoryRetriever):
    """
    组合多种检索策略的检索器示例。
    
    此模板展示如何组合语义、时间和基于关联的检索。
    实现你自己的混合检索逻辑。
    """
    
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        使用混合策略检索记忆。
        
        注意：在此实现你实际的混合检索逻辑。
        """
        import time
        start_time = time.time()
        
        # 占位符实现
        # 组合语义检索和时间筛选
        
        candidates = []
        types_to_search = query.memory_types or list(self.stores.keys())
        for memory_type in types_to_search:
            if memory_type in self.stores:
                candidates.extend(self.stores[memory_type].get_all())
        
        # 应用筛选
        filtered_candidates = candidates
        
        if query.time_range:
            start_time_filter, end_time_filter = query.time_range
            filtered_candidates = [
                m for m in filtered_candidates
                if start_time_filter <= m.metadata.created_at <= end_time_filter
            ]
        
        if query.min_strength:
            filtered_candidates = [
                m for m in filtered_candidates
                if m.get_strength() >= query.min_strength
            ]
        
        # 对剩余候选评分
        scored_memories = []
        for memory in filtered_candidates:
            score = self.compute_relevance(query, memory)
            if score >= query.min_similarity:
                scored_memories.append((memory, score))
        
        # 排序并返回 top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = scored_memories[:query.top_k]
        
        retrieval_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            memories=top_memories,
            query=query,
            total_candidates=len(candidates),
            retrieval_time_ms=retrieval_time
        )
    
    def compute_relevance(
        self, 
        query: RetrievalQuery, 
        memory: MemoryItem
    ) -> float:
        """
        计算混合相关性分数。
        
        注意：在此实现你实际的混合评分逻辑。
        """
        # 占位符 - 实现你自己的逻辑
        base_score = memory.get_strength()
        
        # 添加时间新近度因子
        recency_score = memory.metadata.get_recency_score()
        
        # 组合分数
        return 0.6 * base_score + 0.4 * recency_score