from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..encoders import EmbeddingEncoder
from ..encoders.session_llm_encoder import SessionLLMEncoder
from ..stores import HierarchicalVectorStore
from .hierarchy_types import HierarchicalMemoryStats, HierarchicalNode, HierarchyLevel


class SessionHierarchicalMemoryManager:
    """Build a four-level hierarchy from one full session at a time."""

    def __init__(
        self,
        llm_encoder: Optional[SessionLLMEncoder] = None,
        embedding_encoder: Optional[EmbeddingEncoder] = None,
        vector_store: Optional[HierarchicalVectorStore] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        self.llm_encoder = llm_encoder
        self.embedding_encoder = embedding_encoder
        self.vector_store = vector_store
        self.persist_directory = persist_directory
        self._last_batch_perf: Dict[str, float] = {}

    def process_session(
        self,
        dialogues: List[str],
        generate_embedding: bool = True,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        if self.llm_encoder is None:
            raise ValueError("LLM encoder is required")

        analysis, parse_ok = self.llm_encoder.analyze_session(dialogues)
        if analysis is None or not parse_ok:
            return None, False

        nodes, build_ok = self._build_nodes_from_session(
            analysis=analysis,
            dialogues=dialogues,
            generate_embedding=generate_embedding,
            session_id=session_id,
        )
        return nodes, parse_ok and build_ok

    def batch_process_sessions(
        self,
        sessions: List[List[str]],
        generate_embedding: bool = True,
        show_progress: bool = True,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[bool]]:
        if self.llm_encoder is None:
            raise ValueError("LLM encoder is required")

        llm_start = time.perf_counter()
        analyses, parse_ok_list = self.llm_encoder.batch_analyze_sessions(
            sessions,
            show_progress=show_progress,
        )
        llm_seconds = time.perf_counter() - llm_start

        nodes_list: List[Optional[Dict[str, Any]]] = [None] * len(sessions)
        ok_list: List[bool] = [False] * len(sessions)

        build_start = time.perf_counter()
        for idx, (analysis, dialogues, parse_ok) in enumerate(zip(analyses, sessions, parse_ok_list)):
            if not parse_ok or analysis is None:
                continue
            nodes, build_ok = self._build_nodes_from_session(
                analysis=analysis,
                dialogues=dialogues,
                generate_embedding=generate_embedding,
                session_id=str(idx),
            )
            nodes_list[idx] = nodes
            ok_list[idx] = build_ok
        build_seconds = time.perf_counter() - build_start

        self._last_batch_perf = {
            "llm_seconds": llm_seconds,
            "embedding_seconds": 0.0,
            "node_build_seconds": build_seconds,
            "relation_update_seconds": 0.0,
            "ok_count": float(sum(ok_list)),
            "batch_size": float(len(sessions)),
        }
        return nodes_list, ok_list

    def _build_nodes_from_session(
        self,
        analysis: Dict[str, Any],
        dialogues: List[str],
        generate_embedding: bool = True,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        if self.vector_store is None:
            raise ValueError("vector store is required")

        embedding_cache: Dict[str, List[float]] = {}
        if generate_embedding and self.embedding_encoder is not None:
            texts_to_embed = self._collect_embedding_texts(analysis, dialogues)
            if texts_to_embed:
                embeddings = self.embedding_encoder.generate_embeddings_batch(texts_to_embed)
                embedding_cache = dict(zip(texts_to_embed, embeddings))

        domain_text = str(analysis.get("domain", "unknown")).strip() or "unknown"
        domain_node = self.vector_store.get_node_by_content(domain_text, HierarchyLevel.DOMAIN)
        if domain_node is None:
            domain_node = HierarchicalNode(
                content=domain_text,
                level=HierarchyLevel.DOMAIN,
                embedding=self._get_embedding(domain_text, embedding_cache),
                level_embedding=self._get_level_embedding(HierarchyLevel.DOMAIN, domain_text, embedding_cache),
            )
            self.vector_store.add_node(domain_node)

        category_nodes: Dict[str, HierarchicalNode] = {}
        keyword_nodes: Dict[Tuple[str, str], HierarchicalNode] = {}
        dialogue_parent_ids: Dict[int, Set[str]] = {idx: set() for idx in range(len(dialogues))}

        for category_item in analysis.get("categories", []):
            category_name = str(category_item.get("category", "")).strip()
            if not category_name:
                continue

            category_node = self.vector_store.get_node_by_content(category_name, HierarchyLevel.CATEGORY)
            if category_node is None:
                category_node = HierarchicalNode(
                    content=category_name,
                    level=HierarchyLevel.CATEGORY,
                    parent_ids=[domain_node.id],
                    embedding=self._get_embedding(category_name, embedding_cache),
                    level_embedding=self._get_level_embedding(
                        HierarchyLevel.CATEGORY, category_name, embedding_cache
                    ),
                )
                self.vector_store.add_node(category_node)
            self._append_parent(category_node, domain_node.id)
            domain_node.add_child(category_node.id)
            category_nodes[category_name] = category_node

            for keyword_item in category_item.get("keywords", []):
                keyword_name = str(keyword_item.get("keyword", "")).strip()
                if not keyword_name:
                    continue
                key = (category_name, keyword_name)
                keyword_node = keyword_nodes.get(key)
                if keyword_node is None:
                    keyword_node = self.vector_store.get_node_by_content(keyword_name, HierarchyLevel.KEYWORD)
                    if keyword_node is None:
                        keyword_node = HierarchicalNode(
                            content=keyword_name,
                            level=HierarchyLevel.KEYWORD,
                            parent_ids=[category_node.id],
                            embedding=self._get_embedding(keyword_name, embedding_cache),
                            level_embedding=self._get_level_embedding(
                                HierarchyLevel.KEYWORD, keyword_name, embedding_cache
                            ),
                            metadata={"session_category": category_name},
                        )
                        self.vector_store.add_node(keyword_node)
                    self._append_parent(keyword_node, category_node.id)
                    category_node.add_child(keyword_node.id)
                    keyword_nodes[key] = keyword_node

                for dialogue_idx in keyword_item.get("dialogue_indices", []):
                    if 0 <= dialogue_idx < len(dialogues):
                        dialogue_parent_ids[dialogue_idx].add(keyword_node.id)

        fallback_category = None
        if not category_nodes:
            fallback_category = self._get_or_create_category("general", domain_node, embedding_cache)
            category_nodes["general"] = fallback_category
        for idx, parent_ids in dialogue_parent_ids.items():
            if parent_ids:
                continue
            if fallback_category is None:
                fallback_category = next(iter(category_nodes.values()))
            fallback_keyword = self._get_or_create_keyword(
                "general",
                fallback_category,
                embedding_cache,
                keyword_nodes,
            )
            parent_ids.add(fallback_keyword.id)

        summary_map = {
            item.get("dialogue_index"): str(item.get("summary", "")).strip()
            for item in analysis.get("dialogue_summaries", [])
            if isinstance(item, dict)
        }

        dialogue_nodes: List[HierarchicalNode] = []
        for idx, text in enumerate(dialogues):
            parent_ids = sorted(dialogue_parent_ids[idx])
            dialogue_node = HierarchicalNode(
                content=text,
                level=HierarchyLevel.DIALOGUE,
                parent_ids=parent_ids,
                embedding=self._get_embedding(text, embedding_cache),
                level_embedding=self._get_level_embedding(HierarchyLevel.DIALOGUE, text, embedding_cache),
                metadata={
                    "summary": summary_map.get(idx, ""),
                    "session_id": session_id or "",
                    "dialogue_index": idx,
                },
            )
            self.vector_store.add_node(dialogue_node)
            for keyword_node in keyword_nodes.values():
                if keyword_node.id in parent_ids:
                    keyword_node.add_child(dialogue_node.id)
            dialogue_nodes.append(dialogue_node)

        self.vector_store.update_node(domain_node)
        for node in category_nodes.values():
            self.vector_store.update_node(node)
        for node in keyword_nodes.values():
            self.vector_store.update_node(node)

        return {
            "domain": domain_node,
            "categories": list(category_nodes.values()),
            "keywords": list(keyword_nodes.values()),
            "dialogues": dialogue_nodes,
        }, True

    def _get_or_create_category(
        self,
        category_name: str,
        domain_node: HierarchicalNode,
        embedding_cache: Dict[str, List[float]],
    ) -> HierarchicalNode:
        category_node = self.vector_store.get_node_by_content(category_name, HierarchyLevel.CATEGORY)
        if category_node is None:
            category_node = HierarchicalNode(
                content=category_name,
                level=HierarchyLevel.CATEGORY,
                parent_ids=[domain_node.id],
                embedding=self._get_embedding(category_name, embedding_cache),
                level_embedding=self._get_level_embedding(HierarchyLevel.CATEGORY, category_name, embedding_cache),
            )
            self.vector_store.add_node(category_node)
        self._append_parent(category_node, domain_node.id)
        domain_node.add_child(category_node.id)
        return category_node

    def _get_or_create_keyword(
        self,
        keyword_name: str,
        category_node: HierarchicalNode,
        embedding_cache: Dict[str, List[float]],
        keyword_nodes: Dict[Tuple[str, str], HierarchicalNode],
    ) -> HierarchicalNode:
        key = (category_node.content, keyword_name)
        keyword_node = keyword_nodes.get(key)
        if keyword_node is not None:
            return keyword_node
        keyword_node = self.vector_store.get_node_by_content(keyword_name, HierarchyLevel.KEYWORD)
        if keyword_node is None:
            keyword_node = HierarchicalNode(
                content=keyword_name,
                level=HierarchyLevel.KEYWORD,
                parent_ids=[category_node.id],
                embedding=self._get_embedding(keyword_name, embedding_cache),
                level_embedding=self._get_level_embedding(HierarchyLevel.KEYWORD, keyword_name, embedding_cache),
            )
            self.vector_store.add_node(keyword_node)
        self._append_parent(keyword_node, category_node.id)
        category_node.add_child(keyword_node.id)
        keyword_nodes[key] = keyword_node
        return keyword_node

    def _collect_embedding_texts(self, analysis: Dict[str, Any], dialogues: List[str]) -> List[str]:
        texts: List[str] = []
        seen: Set[str] = set()

        def add(text: str) -> None:
            if text and text not in seen:
                seen.add(text)
                texts.append(text)

        add(str(analysis.get("domain", "")))
        add(self._make_level_aware_text(HierarchyLevel.DOMAIN, str(analysis.get("domain", ""))))
        for category_item in analysis.get("categories", []):
            category_name = str(category_item.get("category", "")).strip()
            add(category_name)
            add(self._make_level_aware_text(HierarchyLevel.CATEGORY, category_name))
            for keyword_item in category_item.get("keywords", []):
                keyword_name = str(keyword_item.get("keyword", "")).strip()
                add(keyword_name)
                add(self._make_level_aware_text(HierarchyLevel.KEYWORD, keyword_name))
        for dialogue in dialogues:
            add(dialogue)
            add(self._make_level_aware_text(HierarchyLevel.DIALOGUE, dialogue))
        return texts

    def _append_parent(self, node: HierarchicalNode, parent_id: str) -> None:
        if parent_id not in node.parent_ids:
            node.parent_ids.append(parent_id)

    def _get_embedding(self, text: str, embedding_cache: Dict[str, List[float]]) -> Optional[List[float]]:
        if self.embedding_encoder is None:
            return None
        if text in embedding_cache:
            return embedding_cache[text]
        return self.embedding_encoder.generate_embedding(text)

    def _get_level_embedding(
        self,
        level: HierarchyLevel,
        content: str,
        embedding_cache: Dict[str, List[float]],
    ) -> Optional[List[float]]:
        text = self._make_level_aware_text(level, content)
        return self._get_embedding(text, embedding_cache)

    def _make_level_aware_text(self, level: HierarchyLevel, content: str) -> str:
        return f"{level.name}: {content}"

    def get_last_batch_perf(self) -> Dict[str, float]:
        return dict(self._last_batch_perf)

    def get_stats(self) -> HierarchicalMemoryStats:
        return self.vector_store.get_stats()

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


def create_session_hierarchical_manager(
    llm_model_path: Optional[str] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: Optional[str] = None,
    device: str = "auto",
    delayed_write: bool = True,
) -> SessionHierarchicalMemoryManager:
    llm_encoder = SessionLLMEncoder(
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
    return SessionHierarchicalMemoryManager(
        llm_encoder=llm_encoder,
        embedding_encoder=embedding_encoder,
        vector_store=vector_store,
        persist_directory=persist_directory,
    )
