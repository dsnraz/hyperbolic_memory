from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

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
        memory_unit_mode: Literal["keyword", "fact"] = "keyword",
        extraction_mode: Literal["single", "two_stage"] = "single",
        fact_spo_encoder: Optional[Any] = None,
    ) -> None:
        if memory_unit_mode not in ("keyword", "fact"):
            raise ValueError("memory_unit_mode must be 'keyword' or 'fact'")
        if extraction_mode not in ("single", "two_stage"):
            raise ValueError("extraction_mode must be 'single' or 'two_stage'")
        if extraction_mode == "two_stage" and memory_unit_mode != "fact":
            raise ValueError("two_stage extraction is only supported with memory_unit_mode='fact'")
        self.llm_encoder = llm_encoder
        self.embedding_encoder = embedding_encoder
        self.vector_store = vector_store
        self.persist_directory = persist_directory
        self.memory_unit_mode = memory_unit_mode
        self.extraction_mode = extraction_mode
        self.fact_spo_encoder = fact_spo_encoder
        self._last_batch_perf: Dict[str, float] = {}
        self._last_batch_analyses: List[Optional[Dict[str, Any]]] = []
        self._last_batch_parse_ok_list: List[bool] = []

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
        session_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[bool]]:
        if self.llm_encoder is None:
            raise ValueError("LLM encoder is required")

        llm_start = time.perf_counter()
        analyses, parse_ok_list = self.llm_encoder.batch_analyze_sessions(
            sessions,
            show_progress=show_progress,
        )
        self._last_batch_analyses = list(analyses)
        self._last_batch_parse_ok_list = list(parse_ok_list)
        llm_seconds = time.perf_counter() - llm_start

        nodes_list: List[Optional[Dict[str, Any]]] = [None] * len(sessions)
        ok_list: List[bool] = [False] * len(sessions)
        resolved_session_ids = session_ids or [str(idx) for idx in range(len(sessions))]
        if len(resolved_session_ids) != len(sessions):
            raise ValueError("session_ids length must match sessions length")

        build_start = time.perf_counter()
        for idx, (analysis, dialogues, parse_ok, sid) in enumerate(
            zip(analyses, sessions, parse_ok_list, resolved_session_ids)
        ):
            if not parse_ok or analysis is None:
                continue
            nodes, build_ok = self._build_nodes_from_session(
                analysis=analysis,
                dialogues=dialogues,
                generate_embedding=generate_embedding,
                session_id=sid,
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
        if self.memory_unit_mode == "fact":
            return self._build_nodes_from_fact_analysis(
                analysis, dialogues, generate_embedding, session_id
            )
        return self._build_nodes_from_keyword_analysis(
            analysis, dialogues, generate_embedding, session_id
        )

    # ------------------------------------------------------------------
    # fact mode: flat facts → derived categories from predicate + object
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_category_name(fact_item: Dict[str, Any]) -> str:
        subject = str(fact_item.get("subject", "")).strip()
        predicate = str(fact_item.get("predicate", "")).strip()
        object_ = str(fact_item.get("object", "")).strip()
        if predicate and object_:
            return f"{subject} {predicate}"
        if predicate:
            return predicate
        return str(fact_item.get("fact", ""))[:60].strip()

    def _build_nodes_from_fact_analysis(
        self,
        analysis: Dict[str, Any],
        dialogues: List[str],
        generate_embedding: bool,
        session_id: Optional[str],
    ) -> Tuple[Dict[str, Any], bool]:
        if self.vector_store is None:
            raise ValueError("vector store is required")

        domains: List[str] = analysis.get("domains", ["general"])
        facts: List[Dict[str, Any]] = analysis.get("facts", [])

        # --- two-stage: enrich facts with SPO via FactSPOEncoder ---
        if self.extraction_mode == "two_stage" and self.fact_spo_encoder is not None \
                and facts and self.fact_spo_encoder._init_handler():
            fact_texts = [str(f.get("fact", "")).strip() for f in facts]
            spo_results = self.fact_spo_encoder.batch_extract_spo(
                fact_texts, show_progress=False
            )
            for fi, spo in enumerate(spo_results):
                if fi < len(facts):
                    facts[fi]["subject"] = spo.get("subject", "")
                    facts[fi]["predicate"] = spo.get("predicate", "")
                    facts[fi]["object"] = spo.get("object", "")
                    facts[fi]["time"] = spo.get("time", "")

        embedding_cache: Dict[str, List[float]] = {}
        if generate_embedding and self.embedding_encoder is not None:
            texts_to_embed = self._collect_fact_embedding_texts(domains, facts, dialogues)
            if texts_to_embed:
                embeddings = self.embedding_encoder.generate_embeddings_batch(texts_to_embed)
                embedding_cache = dict(zip(texts_to_embed, embeddings))

        # --- domain nodes ---
        domain_nodes: Dict[str, HierarchicalNode] = {}
        for domain_text in domains:
            domain_node = self.vector_store.get_node_by_content(domain_text, HierarchyLevel.DOMAIN)
            if domain_node is None:
                domain_node = HierarchicalNode(
                    content=domain_text,
                    level=HierarchyLevel.DOMAIN,
                    embedding=self._get_embedding(domain_text, embedding_cache),
                    level_embedding=self._get_level_embedding(
                        HierarchyLevel.DOMAIN, domain_text, embedding_cache
                    ),
                )
                emb = self._get_embedding(domain_text, embedding_cache)
                if emb is None:
                    print(f"[WARN] domain '{domain_text}' embedding is None, cache_size={len(embedding_cache)}")
                self.vector_store.add_node(domain_node)
            domain_nodes[domain_text] = domain_node

        # --- derive categories from predicate + object ---
        category_nodes: Dict[str, HierarchicalNode] = {}
        fact_nodes: Dict[str, HierarchicalNode] = {}
        dialogue_parent_ids: Dict[int, Set[str]] = {
            idx: set() for idx in range(len(dialogues))
        }

        for fact_item in facts:
            fact_text = str(fact_item.get("fact", "")).strip()
            if not fact_text:
                continue
            category_name = self._derive_category_name(fact_item)

            if category_name not in category_nodes:
                cat_node = self.vector_store.get_node_by_content(
                    category_name, HierarchyLevel.CATEGORY
                )
                if cat_node is None:
                    cat_node = HierarchicalNode(
                        content=category_name,
                        level=HierarchyLevel.CATEGORY,
                        embedding=self._get_embedding(category_name, embedding_cache),
                        level_embedding=self._get_level_embedding(
                            HierarchyLevel.CATEGORY, category_name, embedding_cache
                        ),
                    )
                    self.vector_store.add_node(cat_node)
                for domain_node in domain_nodes.values():
                    self._append_parent(cat_node, domain_node.id)
                    domain_node.add_child(cat_node.id)
                category_nodes[category_name] = cat_node
            cat_node = category_nodes[category_name]

            fact_node = HierarchicalNode(
                content=fact_text,
                level=HierarchyLevel.KEYWORD,
                parent_ids=[cat_node.id],
                embedding=self._get_embedding(fact_text, embedding_cache),
                level_embedding=self._get_level_embedding(
                    HierarchyLevel.KEYWORD, fact_text, embedding_cache
                ),
                metadata={
                    "memory_unit_mode": "fact",
                    "unit_type": "fact",
                    "session_id": session_id or "",
                    "derived_category": category_name,
                    "subject": str(fact_item.get("subject", "")).strip(),
                    "predicate": str(fact_item.get("predicate", "")).strip(),
                    "object": str(fact_item.get("object", "")).strip(),
                    "time": str(fact_item.get("time", "")).strip(),
                    "dialogue_indices": fact_item.get("dialogue_indices", []),
                },
            )
            self.vector_store.add_node(fact_node)
            cat_node.add_child(fact_node.id)
            fact_nodes[fact_node.id] = fact_node

            for dialogue_idx in fact_item.get("dialogue_indices", []):
                if 0 <= dialogue_idx < len(dialogues):
                    dialogue_parent_ids[dialogue_idx].add(fact_node.id)

        # --- fallback for orphan dialogues ---
        if not category_nodes:
            fallback_cat = self._get_or_create_category(
                "general", next(iter(domain_nodes.values())), embedding_cache
            )
            category_nodes["general"] = fallback_cat
        fallback_category = next(iter(category_nodes.values()))
        for idx, parent_ids in dialogue_parent_ids.items():
            if parent_ids:
                continue
            fallback_fact = self._get_or_create_fact(
                dialogues[idx],
                fallback_category,
                embedding_cache,
                session_id=session_id,
                dialogue_indices=[idx],
            )
            parent_ids.add(fallback_fact.id)

        # --- dialogue nodes ---
        dialogue_nodes: List[HierarchicalNode] = []
        for idx, text in enumerate(dialogues):
            parent_ids = sorted(dialogue_parent_ids[idx])
            dialogue_node = HierarchicalNode(
                content=text,
                level=HierarchyLevel.DIALOGUE,
                parent_ids=parent_ids,
                embedding=self._get_embedding(text, embedding_cache),
                level_embedding=self._get_level_embedding(
                    HierarchyLevel.DIALOGUE, text, embedding_cache
                ),
                metadata={
                    "session_id": session_id or "",
                    "dialogue_index": idx,
                    "parent_unit_type": "fact",
                },
            )
            self.vector_store.add_node(dialogue_node)
            for _fid in parent_ids:
                fn = fact_nodes.get(_fid)
                if fn is not None:
                    fn.add_child(dialogue_node.id)
            dialogue_nodes.append(dialogue_node)

        # --- persist ---
        for node in domain_nodes.values():
            self.vector_store.update_node(node)
        for node in category_nodes.values():
            self.vector_store.update_node(node)
        for node in fact_nodes.values():
            self.vector_store.update_node(node)

        return {
            "domains": list(domain_nodes.values()),
            "categories": list(category_nodes.values()),
            "facts": list(fact_nodes.values()),
            "memory_unit_mode": "fact",
            "dialogues": dialogue_nodes,
        }, True

    def _get_or_create_fact(
        self,
        fact_text: str,
        category_node: HierarchicalNode,
        embedding_cache: Dict[str, List[float]],
        session_id: Optional[str] = None,
        dialogue_indices: Optional[List[int]] = None,
    ) -> HierarchicalNode:
        fact_node = HierarchicalNode(
            content=fact_text,
            level=HierarchyLevel.KEYWORD,
            parent_ids=[category_node.id],
            embedding=self._get_embedding(fact_text, embedding_cache),
            level_embedding=self._get_level_embedding(
                HierarchyLevel.KEYWORD, fact_text, embedding_cache
            ),
            metadata={
                "memory_unit_mode": "fact",
                "unit_type": "fact",
                "session_id": session_id or "",
                "dialogue_indices": dialogue_indices or [],
            },
        )
        self.vector_store.add_node(fact_node)
        category_node.add_child(fact_node.id)
        return fact_node

    def _collect_fact_embedding_texts(
        self,
        domains: List[str],
        facts: List[Dict[str, Any]],
        dialogues: List[str],
    ) -> List[str]:
        texts: List[str] = []
        seen: Set[str] = set()

        def add(text: str) -> None:
            if text and text not in seen:
                seen.add(text)
                texts.append(text)

        for domain_text in domains:
            add(domain_text)
            add(self._make_level_aware_text(HierarchyLevel.DOMAIN, domain_text))
        for fact_item in facts:
            fact_text = str(fact_item.get("fact", "")).strip()
            add(fact_text)
            add(self._make_level_aware_text(HierarchyLevel.KEYWORD, fact_text))
            cat_name = self._derive_category_name(fact_item)
            add(cat_name)
            add(self._make_level_aware_text(HierarchyLevel.CATEGORY, cat_name))
        for dialogue in dialogues:
            add(dialogue)
            add(self._make_level_aware_text(HierarchyLevel.DIALOGUE, dialogue))
        return texts

    # ------------------------------------------------------------------
    # keyword mode (original logic)
    # ------------------------------------------------------------------

    def _build_nodes_from_keyword_analysis(
        self,
        analysis: Dict[str, Any],
        dialogues: List[str],
        generate_embedding: bool,
        session_id: Optional[str],
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
        memory_unit_nodes: Dict[Tuple[str, str], HierarchicalNode] = {}
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

            for memory_unit in self._iter_memory_units(category_item):
                unit_text = str(memory_unit.get("content", "")).strip()
                if not unit_text:
                    continue
                key = (category_name, unit_text)
                memory_unit_node = memory_unit_nodes.get(key)
                if memory_unit_node is None:
                    memory_unit_node = self.vector_store.get_node_by_content(unit_text, HierarchyLevel.KEYWORD)
                    unit_metadata = {
                        "memory_unit_mode": "keyword",
                        "unit_type": "keyword",
                        "session_category": category_name,
                        "session_id": session_id or "",
                        **memory_unit.get("metadata", {}),
                    }
                    if memory_unit_node is None:
                        memory_unit_node = HierarchicalNode(
                            content=unit_text,
                            level=HierarchyLevel.KEYWORD,
                            parent_ids=[category_node.id],
                            embedding=self._get_embedding(unit_text, embedding_cache),
                            level_embedding=self._get_level_embedding(
                                HierarchyLevel.KEYWORD, unit_text, embedding_cache
                            ),
                            metadata=unit_metadata,
                        )
                        self.vector_store.add_node(memory_unit_node)
                    else:
                        memory_unit_node.metadata.update(unit_metadata)
                    self._append_parent(memory_unit_node, category_node.id)
                    category_node.add_child(memory_unit_node.id)
                    memory_unit_nodes[key] = memory_unit_node

                for dialogue_idx in memory_unit.get("dialogue_indices", []):
                    if 0 <= dialogue_idx < len(dialogues):
                        dialogue_parent_ids[dialogue_idx].add(memory_unit_node.id)

        fallback_category = None
        if not category_nodes:
            fallback_category = self._get_or_create_category("general", domain_node, embedding_cache)
            category_nodes["general"] = fallback_category
        for idx, parent_ids in dialogue_parent_ids.items():
            if parent_ids:
                continue
            if fallback_category is None:
                fallback_category = next(iter(category_nodes.values()))
            fallback_unit = self._get_or_create_memory_unit(
                "general",
                fallback_category,
                embedding_cache,
                memory_unit_nodes,
                metadata={
                    "memory_unit_mode": "keyword",
                    "unit_type": "keyword",
                    "session_id": session_id or "",
                    "dialogue_indices": [idx],
                },
            )
            parent_ids.add(fallback_unit.id)

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
                    "session_id": session_id or "",
                    "dialogue_index": idx,
                    "parent_unit_type": "keyword",
                },
            )
            self.vector_store.add_node(dialogue_node)
            for memory_unit_node in memory_unit_nodes.values():
                if memory_unit_node.id in parent_ids:
                    memory_unit_node.add_child(dialogue_node.id)
            dialogue_nodes.append(dialogue_node)

        self.vector_store.update_node(domain_node)
        for node in category_nodes.values():
            self.vector_store.update_node(node)
        for node in memory_unit_nodes.values():
            self.vector_store.update_node(node)

        return {
            "domain": domain_node,
            "categories": list(category_nodes.values()),
            "keywords": list(memory_unit_nodes.values()),
            "memory_unit_mode": "keyword",
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

    def _get_or_create_memory_unit(
        self,
        unit_text: str,
        category_node: HierarchicalNode,
        embedding_cache: Dict[str, List[float]],
        memory_unit_nodes: Dict[Tuple[str, str], HierarchicalNode],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HierarchicalNode:
        key = (category_node.content, unit_text)
        memory_unit_node = memory_unit_nodes.get(key)
        if memory_unit_node is not None:
            return memory_unit_node
        memory_unit_node = (
            None
            if self.memory_unit_mode == "fact"
            else self.vector_store.get_node_by_content(unit_text, HierarchyLevel.KEYWORD)
        )
        if memory_unit_node is None:
            memory_unit_node = HierarchicalNode(
                content=unit_text,
                level=HierarchyLevel.KEYWORD,
                parent_ids=[category_node.id],
                embedding=self._get_embedding(unit_text, embedding_cache),
                level_embedding=self._get_level_embedding(HierarchyLevel.KEYWORD, unit_text, embedding_cache),
                metadata=metadata or {},
            )
            self.vector_store.add_node(memory_unit_node)
        elif metadata:
            memory_unit_node.metadata.update(metadata)
        self._append_parent(memory_unit_node, category_node.id)
        category_node.add_child(memory_unit_node.id)
        memory_unit_nodes[key] = memory_unit_node
        return memory_unit_node

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
            for memory_unit in self._iter_memory_units(category_item):
                unit_text = str(memory_unit.get("content", "")).strip()
                add(unit_text)
                add(self._make_level_aware_text(HierarchyLevel.KEYWORD, unit_text))
        for dialogue in dialogues:
            add(dialogue)
            add(self._make_level_aware_text(HierarchyLevel.DIALOGUE, dialogue))
        return texts

    def _iter_memory_units(self, category_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """keyword 模式专用：从 category 下提取关键词列表。fact 模式走 _build_nodes_from_fact_analysis。"""
        units: List[Dict[str, Any]] = []
        for keyword_item in category_item.get("keywords", []):
            if not isinstance(keyword_item, dict):
                continue
            keyword_name = str(keyword_item.get("keyword", "")).strip()
            if not keyword_name:
                continue
            units.append(
                {
                    "content": keyword_name,
                    "dialogue_indices": keyword_item.get("dialogue_indices", []),
                    "metadata": {"keyword": keyword_name},
                }
            )
        return units

    def _memory_unit_label(self) -> str:
        return "fact" if self.memory_unit_mode == "fact" else "keyword"

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
        if self.memory_unit_mode == "fact":
            return content
        return f"{level.name}: {content}"

    def get_last_batch_perf(self) -> Dict[str, float]:
        return dict(self._last_batch_perf)

    def get_last_batch_analyses(self) -> List[Optional[Dict[str, Any]]]:
        return list(self._last_batch_analyses)

    def get_last_batch_parse_ok_list(self) -> List[bool]:
        return list(self._last_batch_parse_ok_list)

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
    memory_unit_mode: Literal["keyword", "fact"] = "keyword",
    extraction_mode: Literal["single", "two_stage"] = "single",
) -> SessionHierarchicalMemoryManager:
    llm_encoder = SessionLLMEncoder(
        model_path=llm_model_path,
        model_type="transformers" if llm_model_path else "ollama",
        device=device,
        memory_unit_mode=memory_unit_mode,
        extraction_mode=extraction_mode,
    )
    fact_spo_encoder = None
    if extraction_mode == "two_stage" and memory_unit_mode == "fact" and llm_model_path:
        from ..encoders.fact_spo_encoder import FactSPOEncoder

        fact_spo_encoder = FactSPOEncoder(
            model_path=llm_model_path,
            model_type="transformers" if llm_model_path else "ollama",
            device=device,
        )
    embedding_is_path = bool(embedding_model and Path(embedding_model).exists())
    embedding_encoder = EmbeddingEncoder(
        model_path=embedding_model if embedding_is_path else None,
        model_name=None if embedding_is_path else embedding_model,
        device=device,
    )
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
        memory_unit_mode=memory_unit_mode,
        extraction_mode=extraction_mode,
        fact_spo_encoder=fact_spo_encoder,
    )
