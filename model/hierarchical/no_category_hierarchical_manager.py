from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..encoders import EmbeddingEncoder
from ..encoders.no_category_llm_encoder import NoCategoryLLMEncoder
from ..stores import HierarchicalVectorStore
from .hierarchy_types import HierarchicalMemoryStats, HierarchicalNode, HierarchyLevel


NO_CATEGORY_LEVEL_ORDER = [
    HierarchyLevel.DOMAIN,
    HierarchyLevel.KEYWORD,
    HierarchyLevel.DIALOGUE,
]


class NoCategoryHierarchicalMemoryManager:
    """Three-level hierarchy: DOMAIN -> KEYWORD -> DIALOGUE."""

    def __init__(
        self,
        llm_encoder: Optional[NoCategoryLLMEncoder] = None,
        embedding_encoder: Optional[EmbeddingEncoder] = None,
        vector_store: Optional[HierarchicalVectorStore] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
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
        generate_embedding: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        if self.llm_encoder is None:
            raise ValueError("LLM encoder is required")

        analysis, parse_ok = self.llm_encoder.analyze(dialogue)
        if analysis is None or not parse_ok:
            return None, False
        nodes, build_ok = self._build_nodes_from_analysis(
            analysis=analysis,
            dialogue=dialogue,
            generate_embedding=generate_embedding,
        )
        return nodes, parse_ok and build_ok

    def batch_process_dialogues(
        self,
        dialogues: List[str],
        llm_batch_size: int = 8,
        generate_embedding: bool = True,
        show_progress: bool = True,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[bool]]:
        if self.llm_encoder is None:
            raise ValueError("LLM encoder is required")

        llm_start = time.perf_counter()
        analyses, parse_ok_list = self.llm_encoder.batch_analyze(
            dialogues,
            batch_size=llm_batch_size,
            show_progress=show_progress,
        )
        llm_seconds = time.perf_counter() - llm_start

        nodes_list, final_ok_list, build_perf = self._build_nodes_batch_from_analyses(
            analyses,
            dialogues,
            parse_ok_list,
            generate_embedding=generate_embedding,
        )
        self._last_batch_perf = {
            "llm_seconds": llm_seconds,
            "embedding_seconds": build_perf["embedding_seconds"],
            "node_build_seconds": build_perf["node_build_seconds"],
            "relation_update_seconds": build_perf["relation_update_seconds"],
            "ok_count": float(sum(final_ok_list)),
            "batch_size": float(len(dialogues)),
        }
        for key in ("llm_seconds", "embedding_seconds", "node_build_seconds", "relation_update_seconds"):
            self._perf_stats[key] += self._last_batch_perf[key]
        self._perf_stats["batches"] += 1.0
        return nodes_list, final_ok_list

    def _build_nodes_batch_from_analyses(
        self,
        analyses: List[Dict[str, Any]],
        dialogues: List[str],
        parse_ok_list: List[bool],
        generate_embedding: bool = True,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[bool], Dict[str, float]]:
        nodes_list: List[Optional[Dict[str, Any]]] = [None] * len(dialogues)
        ok_list: List[bool] = [False] * len(dialogues)
        perf = {
            "embedding_seconds": 0.0,
            "node_build_seconds": 0.0,
            "relation_update_seconds": 0.0,
        }

        valid_items = [
            (idx, analyses[idx], dialogues[idx])
            for idx in range(len(dialogues))
            if parse_ok_list[idx] and analyses[idx] is not None
        ]

        embedding_cache: Optional[Dict[str, List[float]]] = None
        if generate_embedding and self.embedding_encoder is not None:
            start = time.perf_counter()
            embedding_cache = self._build_batch_embedding_cache(valid_items)
            perf["embedding_seconds"] = time.perf_counter() - start

        build_start = time.perf_counter()
        for idx, analysis, dialogue in valid_items:
            nodes, ok = self._build_nodes_from_analysis(
                analysis=analysis,
                dialogue=dialogue,
                generate_embedding=generate_embedding,
                embedding_cache=embedding_cache,
                perf_stats=perf,
            )
            nodes_list[idx] = nodes
            ok_list[idx] = ok
        perf["node_build_seconds"] = time.perf_counter() - build_start
        return nodes_list, ok_list, perf

    def _build_nodes_from_analysis(
        self,
        analysis: Dict[str, Any],
        dialogue: str,
        generate_embedding: bool = True,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
        perf_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        domain_name = str(analysis.get("domain", "unknown")).strip() or "unknown"
        keywords = self._normalize_keywords(analysis.get("keywords", []))
        if not keywords:
            keywords = ["general"]

        embedding = None
        level_embedding = None
        if generate_embedding and self.embedding_encoder is not None:
            embedding = self._get_embedding(dialogue, embedding_cache)
            level_embedding = self._get_level_embedding(
                HierarchyLevel.DIALOGUE,
                dialogue,
                embedding_cache,
            )

        domain_node = self.vector_store.get_node_by_content(domain_name, HierarchyLevel.DOMAIN)
        if domain_node is None:
            domain_node = HierarchicalNode(
                content=domain_name,
                level=HierarchyLevel.DOMAIN,
                embedding=self._get_embedding(domain_name, embedding_cache),
                level_embedding=self._get_level_embedding(
                    HierarchyLevel.DOMAIN, domain_name, embedding_cache
                ),
            )
            self.vector_store.add_node(domain_node)

        keyword_nodes: List[HierarchicalNode] = []
        for keyword in keywords:
            keyword_node = self.vector_store.get_node_by_content(keyword, HierarchyLevel.KEYWORD)
            if keyword_node is None:
                keyword_node = HierarchicalNode(
                    content=keyword,
                    level=HierarchyLevel.KEYWORD,
                    parent_ids=[domain_node.id],
                    embedding=self._get_embedding(keyword, embedding_cache),
                    level_embedding=self._get_level_embedding(
                        HierarchyLevel.KEYWORD, keyword, embedding_cache
                    ),
                )
                self.vector_store.add_node(keyword_node)
            self._append_parent(keyword_node, domain_node.id)
            domain_node.add_child(keyword_node.id)
            keyword_nodes.append(keyword_node)

        dialogue_node = HierarchicalNode(
            content=dialogue,
            level=HierarchyLevel.DIALOGUE,
            parent_ids=[node.id for node in keyword_nodes],
            embedding=embedding,
            level_embedding=level_embedding,
            metadata={
                "summary": analysis.get("summary"),
                "all_keywords": [node.content for node in keyword_nodes],
                "domain": domain_name,
            },
        )
        self.vector_store.add_node(dialogue_node)
        for node in keyword_nodes:
            node.add_child(dialogue_node.id)

        relation_start = time.perf_counter()
        self.vector_store.update_node(domain_node)
        for node in keyword_nodes:
            self.vector_store.update_node(node)
        if perf_stats is not None:
            perf_stats["relation_update_seconds"] += time.perf_counter() - relation_start

        return {
            "domain": domain_node,
            "keywords": keyword_nodes,
            "dialogue": dialogue_node,
        }, True

    def _normalize_keywords(self, keywords: Any) -> List[str]:
        if keywords is None:
            return []
        if isinstance(keywords, list):
            return [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        value = str(keywords).strip()
        return [value] if value else []

    def _build_batch_embedding_cache(
        self,
        valid_items: List[Tuple[int, Dict[str, Any], str]],
    ) -> Dict[str, List[float]]:
        texts_to_embed: List[str] = []
        seen: Set[str] = set()

        def add(text: str) -> None:
            if text and text not in seen:
                seen.add(text)
                texts_to_embed.append(text)

        for _, analysis, dialogue in valid_items:
            add(dialogue)
            add(self._make_level_aware_text(HierarchyLevel.DIALOGUE, dialogue))
            domain_name = str(analysis.get("domain", "unknown")).strip() or "unknown"
            add(domain_name)
            add(self._make_level_aware_text(HierarchyLevel.DOMAIN, domain_name))
            for keyword in self._normalize_keywords(analysis.get("keywords", [])):
                add(keyword)
                add(self._make_level_aware_text(HierarchyLevel.KEYWORD, keyword))

        if not texts_to_embed:
            return {}
        embeddings = self.embedding_encoder.generate_embeddings_batch(texts_to_embed)
        return dict(zip(texts_to_embed, embeddings))

    def _append_parent(self, node: HierarchicalNode, parent_id: str) -> None:
        if parent_id not in node.parent_ids:
            node.parent_ids.append(parent_id)

    def _get_embedding(
        self,
        text: str,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[List[float]]:
        if self.embedding_encoder is None:
            return None
        if embedding_cache is not None and text in embedding_cache:
            return embedding_cache[text]
        return self.embedding_encoder.generate_embedding(text)

    def _make_level_aware_text(self, level: HierarchyLevel, content: str) -> str:
        return f"{level.name}: {content}"

    def _get_level_embedding(
        self,
        level: HierarchyLevel,
        content: str,
        embedding_cache: Optional[Dict[str, List[float]]] = None,
    ) -> Optional[List[float]]:
        return self._get_embedding(self._make_level_aware_text(level, content), embedding_cache)

    def get_last_batch_perf(self) -> Dict[str, float]:
        return dict(self._last_batch_perf)

    def get_perf_stats(self, reset: bool = False) -> Dict[str, float]:
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

    def get_stats(self) -> HierarchicalMemoryStats:
        return self.vector_store.get_stats()

    def search(
        self,
        query: str,
        level: Optional[HierarchyLevel] = None,
        top_k: int = 10,
    ) -> List[tuple]:
        if self.embedding_encoder is None:
            raise ValueError("embedding encoder is required")
        query_embedding = self.embedding_encoder.generate_embedding(query)
        if level is not None:
            return self.vector_store.search_similar(query_embedding, level, top_k)

        results = []
        for lvl in NO_CATEGORY_LEVEL_ORDER:
            results.extend(self.vector_store.search_similar(query_embedding, lvl, top_k // 3 + 1))
        results.sort(key=lambda item: item[1])
        return results[:top_k]

    def get_context(self, dialogue_id: str) -> Dict[str, Any]:
        dialogue_node = self.vector_store.get_node(dialogue_id, HierarchyLevel.DIALOGUE)
        if dialogue_node is None:
            return {}

        ancestors = self._get_ancestors(dialogue_node)
        siblings = {}
        for parent_id in dialogue_node.parent_ids:
            parent_node = self.vector_store.get_node(parent_id, HierarchyLevel.KEYWORD)
            if parent_node is None:
                continue
            for sibling_id in parent_node.child_ids:
                if sibling_id == dialogue_id:
                    continue
                sibling_node = self.vector_store.get_node(sibling_id, HierarchyLevel.DIALOGUE)
                if sibling_node is not None:
                    siblings[sibling_id] = sibling_node
        return {
            "dialogue": dialogue_node,
            "ancestors": ancestors,
            "siblings": list(siblings.values()),
        }

    def _get_ancestors(self, dialogue_node: HierarchicalNode) -> List[HierarchicalNode]:
        ancestors: List[HierarchicalNode] = []
        seen: Set[str] = set()
        for keyword_id in dialogue_node.parent_ids:
            keyword_node = self.vector_store.get_node(keyword_id, HierarchyLevel.KEYWORD)
            if keyword_node is None or keyword_node.id in seen:
                continue
            seen.add(keyword_node.id)
            ancestors.append(keyword_node)
            for domain_id in keyword_node.parent_ids:
                domain_node = self.vector_store.get_node(domain_id, HierarchyLevel.DOMAIN)
                if domain_node is not None and domain_node.id not in seen:
                    seen.add(domain_node.id)
                    ancestors.append(domain_node)
        return ancestors

    def flush(self) -> Dict[str, int]:
        return self.vector_store.flush()

    def clear_memory(self) -> bool:
        return self.vector_store.clear_all()

    def get_pending_dirty_count(self) -> int:
        return self.vector_store.get_pending_dirty_count()

    def save(self, filepath: str) -> bool:
        return self.vector_store.save_to_file(filepath)

    def load(self, filepath: str) -> bool:
        return self.vector_store.load_from_file(filepath)


def create_no_category_hierarchical_manager(
    llm_model_path: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: Optional[str] = None,
    device: str = "auto",
    delayed_write: bool = True,
) -> NoCategoryHierarchicalMemoryManager:
    llm_encoder = NoCategoryLLMEncoder(
        model_path=llm_model_path,
        model_type="transformers" if llm_model_path else "ollama",
        device=device,
    )
    embedding_encoder = EmbeddingEncoder()
    vector_store = HierarchicalVectorStore(
        persist_directory=persist_directory,
        embedding_function=embedding_encoder.generate_embedding,
        delayed_write=delayed_write,
    )
    return NoCategoryHierarchicalMemoryManager(
        llm_encoder=llm_encoder,
        embedding_encoder=embedding_encoder,
        vector_store=vector_store,
        persist_directory=persist_directory,
    )
