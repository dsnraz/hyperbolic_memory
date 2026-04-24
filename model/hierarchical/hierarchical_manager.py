"""
分层记忆协调器。

整合编码器、存储和检索器，提供统一的分层记忆管理接口。
支持批量推理以提升 GPU 利用率。
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .hierarchy_types import (
    HierarchicalNode,
    HierarchyLevel,
    DialogueAnalysisResult,
    HierarchicalMemoryStats,
)


class HierarchicalMemoryManager:
    """
    分层记忆协调器。
    
    整合以下组件：
    - LLMEncoder: 对话分析（领域、类别、关键词提取）
    - EmbeddingEncoder: 向量嵌入生成
    - HierarchicalVectorStore: 向量存储和检索
    
    提供统一的接口：
    - process_dialogue: 处理单条对话
    - batch_process_dialogues: 批量处理对话（提升 GPU 利用率）
    - search: 语义搜索
    - get_context: 获取对话的上下文
    - flush: 强制写入缓存的节点
    """
    
    def __init__(
        self,
        llm_encoder=None,
        embedding_encoder=None,
        vector_store=None,
        persist_directory: Optional[str] = None,
    ):
        self.llm_encoder = llm_encoder
        self.embedding_encoder = embedding_encoder
        self.vector_store = vector_store
        self.persist_directory = persist_directory
        self._last_batch_perf: Dict[str, float] = {}
        self._perf_stats: Dict[str, float] = {
            "llm_seconds": 0.0,
            "embedding_seconds": 0.0,
            "node_build_seconds": 0.0,
            "relation_update_seconds": 0.0,
            "batches": 0.0,
        }
    
    def process_dialogue(
        self, 
        dialogue: str,
        generate_embedding: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """处理单条对话，构建完整的层级结构。"""
        if self.llm_encoder is None:
            raise ValueError("LLM 编码器未配置")
        
        analysis, parse_is_ok = self.llm_encoder.analyze(dialogue)
        
        if analysis is None or not parse_is_ok:
            return None, False
        
        nodes, build_is_ok = self._build_nodes_from_analysis(analysis, dialogue, generate_embedding)
        
        final_is_ok = parse_is_ok and build_is_ok
        return nodes, final_is_ok
    
    def batch_process_dialogues(
        self,
        dialogues: List[str],
        llm_batch_size: int = 8,
        generate_embedding: bool = True,
        show_progress: bool = True,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[bool]]:
        """
        批量处理对话（使用批量推理提升 GPU 利用率）。
        
        参数:
            dialogues: 对话列表
            llm_batch_size: LLM 批量大小（建议 8-16）
            generate_embedding: 是否生成向量嵌入
            show_progress: 是否显示进度
            
        返回:
            (nodes_list, is_ok_list): 节点列表和成功标志列表
        """
        if self.llm_encoder is None:
            raise ValueError("LLM 编码器未配置")
        
        # 批量分析对话（这是性能优化的关键）
        llm_start = time.perf_counter()
        analyses, parse_is_ok_list = self.llm_encoder.batch_analyze(
            dialogues,
            batch_size=llm_batch_size,
            show_progress=show_progress,
        )
        llm_seconds = time.perf_counter() - llm_start

        nodes_list, final_is_ok_list, build_perf = self._build_nodes_batch_from_analyses(
            analyses,
            dialogues,
            parse_is_ok_list,
            generate_embedding=generate_embedding,
        )

        self._last_batch_perf = {
            "llm_seconds": llm_seconds,
            "embedding_seconds": build_perf["embedding_seconds"],
            "node_build_seconds": build_perf["node_build_seconds"],
            "relation_update_seconds": build_perf["relation_update_seconds"],
            "ok_count": float(sum(final_is_ok_list)),
            "batch_size": float(len(dialogues)),
        }
        for key in ("llm_seconds", "embedding_seconds", "node_build_seconds", "relation_update_seconds"):
            self._perf_stats[key] += self._last_batch_perf[key]
        self._perf_stats["batches"] += 1.0
        
        return nodes_list, final_is_ok_list
    
    def _build_nodes_from_analysis(
        self,
        analysis: Dict[str, Any],
        dialogue: str,
        generate_embedding: bool = True,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        perf_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """根据分析结果构建层级节点。"""
        build_issues = self._collect_analysis_build_issues(analysis)
        if build_issues:
            print(f"[空值警告] 对话片段 '{dialogue[:30]}...' 存在问题: {', '.join(build_issues)}")
            return None, False

        nodes = {}
        normalized_keywords = self._normalize_keywords(analysis.get("keywords", []))
        
        # 生成嵌入
        embedding = None
        level_embedding = None
        if generate_embedding and self.embedding_encoder:
            embedding = self._get_embedding(dialogue, embedding_cache)
            level_embedding = self._get_level_embedding(
                HierarchyLevel.DIALOGUE,
                dialogue,
                embedding_cache,
            )
        
        # 创建领域节点
        domain_node= self.vector_store.get_node_by_content(
            analysis["domain"], 
            HierarchyLevel.DOMAIN
        )
        if not domain_node:
            domain_node = HierarchicalNode(
                content=analysis["domain"],
                level=HierarchyLevel.DOMAIN,
                embedding=self._get_embedding(analysis["domain"], embedding_cache),
                level_embedding=self._get_level_embedding(
                    HierarchyLevel.DOMAIN,
                    analysis["domain"],
                    embedding_cache,
                ),
            )
            self.vector_store.add_node(domain_node)
        nodes["domain"] = domain_node
        
        # 创建类别节点
        category_node = self.vector_store.get_node_by_content(
            analysis["category"],
            HierarchyLevel.CATEGORY
        )
        if not category_node:
            category_node = HierarchicalNode(
                content=analysis["category"],
                level=HierarchyLevel.CATEGORY,
                parent_ids=[domain_node.id],
                embedding=self._get_embedding(analysis["category"], embedding_cache),
                level_embedding=self._get_level_embedding(
                    HierarchyLevel.CATEGORY,
                    analysis["category"],
                    embedding_cache,
                ),
            )
            self.vector_store.add_node(category_node)
        self._append_parent(category_node, domain_node.id)
        domain_node.add_child(category_node.id)
        nodes["category"] = category_node
        
        # 创建关键词节点
        keyword_nodes = []
        for keyword in normalized_keywords:
            keyword_node = self.vector_store.get_node_by_content(
                keyword,
                HierarchyLevel.KEYWORD
            )
            if not keyword_node:
                keyword_node = HierarchicalNode(
                    content=keyword,
                    level=HierarchyLevel.KEYWORD,
                    parent_ids=[category_node.id],
                    embedding=self._get_embedding(keyword, embedding_cache),
                    level_embedding=self._get_level_embedding(
                        HierarchyLevel.KEYWORD,
                        keyword,
                        embedding_cache,
                    ),
                )
                self.vector_store.add_node(keyword_node)
            self._append_parent(keyword_node, category_node.id)
            category_node.add_child(keyword_node.id)
            keyword_nodes.append(keyword_node)
        nodes["keywords"] = keyword_nodes
        
        # 创建对话节点
        parent_keyword_ids = [kw.id for kw in keyword_nodes]
        
        summary = analysis.get("summary")
        all_keywords = [kw.content for kw in keyword_nodes]
        
        dialogue_node = HierarchicalNode(
            content=dialogue,
            level=HierarchyLevel.DIALOGUE,
            parent_ids=parent_keyword_ids,
            embedding=embedding,
            level_embedding=level_embedding,
            metadata={
                "summary": summary,
                "all_keywords": all_keywords,
            }
        )
        self.vector_store.add_node(dialogue_node)
        
        for kw_node in keyword_nodes:
            kw_node.add_child(dialogue_node.id)
        
        relation_update_start = time.perf_counter()
        self.vector_store.update_node(domain_node)
        self.vector_store.update_node(category_node)
        for kw_node in keyword_nodes:
            self.vector_store.update_node(kw_node)
        if perf_stats is not None:
            perf_stats["relation_update_seconds"] += time.perf_counter() - relation_update_start

        nodes["dialogue"] = dialogue_node
        
        return nodes, True

    def _build_nodes_batch_from_analyses(
        self,
        analyses: List[Dict[str, Any]],
        dialogues: List[str],
        parse_is_ok_list: List[bool],
        generate_embedding: bool = True,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[bool], Dict[str, float]]:
        """批量构建节点，复用批量 embedding 与耗时统计。"""
        nodes_list: List[Optional[Dict[str, Any]]] = [None] * len(dialogues)
        build_is_ok_list: List[bool] = [False] * len(dialogues)
        perf_stats = {
            "embedding_seconds": 0.0,
            "node_build_seconds": 0.0,
            "relation_update_seconds": 0.0,
        }

        valid_items = [
            (idx, analyses[idx], dialogues[idx])
            for idx in range(len(dialogues))
            if parse_is_ok_list[idx]
            and analyses[idx] is not None
            and not self._collect_analysis_build_issues(analyses[idx])
        ]
        for idx, analysis, dialogue in (
            (idx, analyses[idx], dialogues[idx])
            for idx in range(len(dialogues))
            if parse_is_ok_list[idx] and analyses[idx] is not None
        ):
            if self._collect_analysis_build_issues(analysis):
                build_issues = self._collect_analysis_build_issues(analysis)
                print(f"[空值警告] 对话片段 '{dialogue[:30]}...' 存在问题: {', '.join(build_issues)}")

        if not valid_items:
            final_is_ok_list = [
                parse_is_ok and build_is_ok
                for parse_is_ok, build_is_ok in zip(parse_is_ok_list, build_is_ok_list)
            ]
            return nodes_list, final_is_ok_list, perf_stats

        embedding_cache: Optional[Dict[str, List[float]]] = None
        if generate_embedding and self.embedding_encoder is not None:
            embedding_start = time.perf_counter()
            embedding_cache = self._build_batch_embedding_cache(valid_items)
            perf_stats["embedding_seconds"] = time.perf_counter() - embedding_start

        build_start = time.perf_counter()
        for idx, analysis, dialogue in valid_items:
            nodes_list[idx], build_is_ok_list[idx] = self._build_nodes_from_analysis(
                analysis,
                dialogue,
                generate_embedding=generate_embedding,
                embedding_cache=embedding_cache,
                perf_stats=perf_stats,
            )
        perf_stats["node_build_seconds"] = time.perf_counter() - build_start
        final_is_ok_list = [
            parse_is_ok and build_is_ok
            for parse_is_ok, build_is_ok in zip(parse_is_ok_list, build_is_ok_list)
        ]
        return nodes_list, final_is_ok_list, perf_stats

    def _append_parent(self, node: HierarchicalNode, parent_id: str) -> None:
        """为节点追加父节点 ID，避免重复写入。"""
        if parent_id not in node.parent_ids:
            node.parent_ids.append(parent_id)

    def _normalize_keywords(self, keywords: Any) -> List[str]:
        """将关键词字段规范化为字符串列表。"""
        if keywords is None:
            return []
        if isinstance(keywords, list):
            return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if isinstance(keywords, str):
            normalized_keyword = keywords.strip()
            return [normalized_keyword] if normalized_keyword else []
        normalized_keyword = str(keywords).strip()
        return [normalized_keyword] if normalized_keyword else []

    def _collect_analysis_build_issues(self, analysis: Dict[str, Any]) -> List[str]:
        """收集会导致构图失败的空值问题。"""
        issues: List[str] = []
        summary = analysis.get("summary")
        keywords = self._normalize_keywords(analysis.get("keywords", []))

        if summary is None or (isinstance(summary, str) and summary.strip() == ""):
            issues.append("summary为空")
        if len(keywords) == 0:
            issues.append("all_keywords为空列表")

        return issues

    def _build_batch_embedding_cache(
        self,
        valid_items: List[Tuple[int, Dict[str, Any], str]],
    ) -> Dict[str, List[float]]:
        """为当前 batch 需要的文本批量生成 embedding。"""
        texts_to_embed: List[str] = []
        seen_texts: Set[str] = set()

        def append_text(text: str) -> None:
            if text and text not in seen_texts:
                seen_texts.add(text)
                texts_to_embed.append(text)

        for _, analysis, dialogue in valid_items:
            append_text(dialogue)
            append_text(self._make_level_aware_text(HierarchyLevel.DIALOGUE, dialogue))
            append_text(analysis["domain"])
            append_text(self._make_level_aware_text(HierarchyLevel.DOMAIN, analysis["domain"]))
            append_text(analysis["category"])
            append_text(self._make_level_aware_text(HierarchyLevel.CATEGORY, analysis["category"]))
            for keyword in analysis.get("keywords", []):
                append_text(keyword)
                append_text(self._make_level_aware_text(HierarchyLevel.KEYWORD, keyword))

        if not texts_to_embed:
            return {}

        embeddings = self._generate_embeddings_in_chunks(texts_to_embed)
        return dict(zip(texts_to_embed, embeddings))

    def _generate_embeddings_in_chunks(
        self,
        texts: List[str],
        chunk_size: int = 256,
    ) -> List[List[float]]:
        """分块批量生成 embedding，避免一次性输入过大。"""
        if self.embedding_encoder is None or not texts:
            return []

        embeddings: List[List[float]] = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start:start + chunk_size]
            embeddings.extend(self.embedding_encoder.generate_embeddings_batch(chunk))
        return embeddings

    def _get_embedding(
        self,
        text: str,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[List[float]]:
        """优先从批量缓存获取 embedding，未命中时再单条编码。"""
        if self.embedding_encoder is None:
            return None
        if embedding_cache is not None and text in embedding_cache:
            return embedding_cache[text]
        return self.embedding_encoder.generate_embedding(text)

    def _make_level_aware_text(self, level: HierarchyLevel, content: str) -> str:
        """构造带层级前缀的文本。"""
        return f"{level.name}: {content}"

    def _get_level_embedding(
        self,
        level: HierarchyLevel,
        content: str,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[List[float]]:
        """获取带层级前缀的 embedding。"""
        return self._get_embedding(
            self._make_level_aware_text(level, content),
            embedding_cache,
        )

    def _generate_level_embedding(
        self,
        level: HierarchyLevel,
        content: str,
    ) -> Optional[List[float]]:
        """生成带层级前缀的 embedding，用于区分跨层同名节点。"""
        return self._get_level_embedding(level, content)

    def get_perf_stats(self, reset: bool = False) -> Dict[str, float]:
        """获取累计批处理耗时统计。"""
        stats = dict(self._perf_stats)
        if reset:
            self._perf_stats = {
                "llm_seconds": 0.0,
                "embedding_seconds": 0.0,
                "node_build_seconds": 0.0,
                "relation_update_seconds": 0.0,
                "batches": 0.0,
            }
        return stats

    def get_last_batch_perf(self) -> Dict[str, float]:
        """获取最近一个 batch 的耗时统计。"""
        return dict(self._last_batch_perf)
    
    def search(
        self,
        query: str,
        level: Optional[HierarchyLevel] = None,
        top_k: int = 10,
    ) -> List[tuple]:
        """语义搜索。"""
        if self.embedding_encoder is None:
            raise ValueError("嵌入编码器未配置")
        
        query_embedding = self.embedding_encoder.generate_embedding(query)
        
        if level:
            return self.vector_store.search_similar(query_embedding, level, top_k)
        else:
            results = []
            for lvl in HierarchyLevel:
                results.extend(
                    self.vector_store.search_similar(query_embedding, lvl, top_k // 4 + 1)
                )
            results.sort(key=lambda x: x[1])
            return results[:top_k]
    
    def get_context(self, dialogue_id: str) -> Dict[str, Any]:
        """获取对话的上下文信息。"""
        dialogue_node = self.vector_store.get_node(dialogue_id, HierarchyLevel.DIALOGUE)
        if dialogue_node is None:
            return {}
        
        context = {
            "dialogue": dialogue_node,
            "ancestors": self.vector_store.get_ancestors(dialogue_id),
        }
        
        if dialogue_node.parent_ids:
            sibling_map = {}
            for parent_id in dialogue_node.parent_ids:
                siblings = self.vector_store.get_children(
                    parent_id,
                    HierarchyLevel.KEYWORD
                )
                for sibling in siblings:
                    if sibling.id != dialogue_id:
                        sibling_map[sibling.id] = sibling
            context["siblings"] = list(sibling_map.values())
        
        return context
    
    def get_stats(self) -> HierarchicalMemoryStats:
        """获取统计信息。"""
        return self.vector_store.get_stats()
    
    def flush(self) -> Dict[str, int]:
        """强制写入所有缓存的节点到 ChromaDB。"""
        return self.vector_store.flush()

    def clear_memory(self) -> bool:
        """清空当前记忆库，避免不同会话之间相互污染。"""
        return self.vector_store.clear_all()
    
    def get_pending_dirty_count(self) -> int:
        """获取待写入与待更新节点数量。"""
        return self.vector_store.get_pending_dirty_count()
    
    def save(self, filepath: str) -> bool:
        """保存到文件。"""
        return self.vector_store.save_to_file(filepath)
    
    def load(self, filepath: str) -> bool:
        """从文件加载。"""
        return self.vector_store.load_from_file(filepath)


def create_hierarchical_manager(
    llm_model_path: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: Optional[str] = None,
    device: str = "auto",
    delayed_write: bool = True,
) -> HierarchicalMemoryManager:
    """创建分层记忆协调器的便捷函数。"""
    from ..encoders import LLMEncoder, EmbeddingEncoder
    from ..stores import HierarchicalVectorStore
    
    llm_encoder = None
    if llm_model_path:
        llm_encoder = LLMEncoder(
            model_path=llm_model_path,
            model_type="transformers",
            device=device,
        )
    
    embedding_encoder = EmbeddingEncoder()
    
    vector_store = HierarchicalVectorStore(
        persist_directory=persist_directory,
        embedding_function=embedding_encoder.generate_embedding,
        delayed_write=delayed_write,
    )
    
    return HierarchicalMemoryManager(
        llm_encoder=llm_encoder,
        embedding_encoder=embedding_encoder,
        vector_store=vector_store,
        persist_directory=persist_directory,
    )