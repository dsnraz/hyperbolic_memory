from __future__ import annotations

from typing import Any, Dict, List, Optional

from model.encoders.model_handler import BaseModelHandler, create_model_handler
from model.hierarchical.hierarchical_manager import (
    HierarchicalMemoryManager,
    create_hierarchical_manager,
)
from model.hierarchical.hierarchy_types import HierarchyLevel
from model.retrievers import (
    CosineRetriever,
    GeodesicHyperbolicRetriever,
    HybridHyperbolicRetriever,
    MultiParentAngularHyperbolicRetriever,
)


class MemoryAugmentedLLMInference:
    """检索 + 上下文拼接 +（可选）LLM 生成；建库请用 `memory_builder`。"""

    DEFAULT_PROMPT_TEMPLATE = (
        "You are a helpful assistant that answers questions using retrieved memory.\n\n"
        "Retrieved context from memory:\n"
        "{context}\n\n"
        "Question: {query}\n"
        "Answer straightly"
        "If there is no relevant information in the context, output 'I don't know'."
        "Answer:"
    )

    def __init__(
        self,
        manager: HierarchicalMemoryManager | None = None,
        llm_model_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        device: str = "auto",
        retriever_type: str = "hyperbolic_geodesic",
        projector_checkpoint_path: Optional[str] = None,
        hyperbolic_angular_kwargs: Optional[Dict[str, Any]] = None,
        generation_handler_type: Optional[str] = None,
        generation_model_name: Optional[str] = None,
        generation_model_path: Optional[str] = None,
        generation_api_base: str = "http://localhost:11434",
        retriever_top_k: List[int] = [5, 5, 5, 5],
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
    ) -> None:
        self.manager = manager or create_hierarchical_manager(
            llm_model_path=llm_model_path,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            device=device,
            delayed_write=False,
        )
        self.retriever_top_k = retriever_top_k
        self.start_level = start_level
        self.target_level = target_level

        if retriever_type == "cosine":
            self.retriever = CosineRetriever(vector_store=self.manager.vector_store)
        elif retriever_type == "hyperbolic_geodesic":
            if not projector_checkpoint_path:
                raise ValueError(
                    "使用 hyperbolic_geodesic 检索时必须提供 projector_checkpoint_path"
                )
            self.retriever = GeodesicHyperbolicRetriever(
                vector_store=self.manager.vector_store,
                checkpoint_path=projector_checkpoint_path,
            )
        elif retriever_type == "hyperbolic_angular":
            if not projector_checkpoint_path:
                raise ValueError(
                    "使用 hyperbolic_angular 检索时必须提供 projector_checkpoint_path"
                )
            ang_kw = dict(hyperbolic_angular_kwargs or {})
            self.retriever = MultiParentAngularHyperbolicRetriever(
                vector_store=self.manager.vector_store,
                checkpoint_path=projector_checkpoint_path,
                **ang_kw,
            )
        elif retriever_type == "hyperbolic_angular_geodesic_hybrid":
            if not projector_checkpoint_path:
                raise ValueError(
                    "使用 hyperbolic_angular_geodesic_hybrid 检索时必须提供 projector_checkpoint_path"
                )
            ang_kw = dict(hyperbolic_angular_kwargs or {})
            self.retriever = HybridHyperbolicRetriever(
                vector_store=self.manager.vector_store,
                checkpoint_path=projector_checkpoint_path,
                hyperbolic_angular_kwargs=ang_kw,
            )
        else:
            raise ValueError(
                f"未知检索器类型: {retriever_type}；"
                f"支持 cosine | hyperbolic_geodesic | hyperbolic_angular | "
                f"hyperbolic_angular_geodesic_hybrid"
            )

        self.generation_handler: BaseModelHandler | None = None
        if generation_handler_type is not None:
            model_source = generation_model_path or generation_model_name
            if not model_source:
                raise ValueError("初始化生成模型时必须提供 generation_model_name 或 generation_model_path")
            self.generation_handler = create_model_handler(
                generation_handler_type,
                api_base=generation_api_base,
            )
            if not self.generation_handler.load(model_source, device=device):
                raise RuntimeError("生成模型初始化失败")
        else:
            print("未加载推理模型")

    def clear_retriever_cache(self) -> None:
        if hasattr(self.retriever, "clear_cache"):
            self.retriever.clear_cache()

    def answer(
        self,
        query_text: str,
        prompt_template: Optional[str] = None,
        top_k: Optional[List[int]] = None,
        start_level: Optional[HierarchyLevel] = None,
        target_level: Optional[HierarchyLevel] = None,
        retrieve_kwargs: Optional[Dict[str, Any]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        retrieve_kwargs = retrieve_kwargs or {}
        resolved_top_k = top_k or self.retriever_top_k
        if len(resolved_top_k) != 4:
            raise ValueError("top_k 必须是长度为 4 的列表: [DOMAIN, CATEGORY, KEYWORD, DIALOGUE]")
        retrieval_result = self.retriever.retrieve(
            query_text=query_text,
            top_k=resolved_top_k,
            start_level=start_level or self.start_level,
            target_level=target_level or self.target_level,
            adaptive_start_level = True,
            **retrieve_kwargs,
        )
        # result_euclidean = retrieval_result.level_results
        # print("欧式检索结果：")
        # for i in result_euclidean:
        #     print("--------------------------------")
        #     print("该层level",i.level)
        #     print("该层候选数量",i.candidate_count)
        #     for j in i.hits:
        #         id = j.node.id
        #         print(f"{id} 节点信息")
        #         print(f"节点内容: {j.node.content}")
        #         print(f"节点父节点: {j.node.parent_ids}")
        #         print(f"节点子节点: {j.node.child_ids}")
        #         print(f"节点得分: {j.score}")
        #     print("--------------------------------")
        context = self.retriever.get_context(
            query_text=query_text,
            top_k=resolved_top_k[3],
            start_level=start_level or self.start_level,
            target_level=target_level or self.target_level,
            retrieval_result=retrieval_result,
            **retrieve_kwargs,
        )
        prompt = (prompt_template or self.DEFAULT_PROMPT_TEMPLATE).format(
            context=context if context else "No usable memory was retrieved.",
            query=query_text,
        )

        if self.generation_handler is None:
            print("没找到生成模型, returning None")
            return {
                "answer": None,
                "prompt": prompt,
                "context": context,
                "retrieval_result": retrieval_result,
            }

        answer = self.generation_handler.generate(prompt, **(generate_kwargs or {}))
        return {
            "answer": answer,
            "prompt": prompt,
            "context": context,
            "retrieval_result": retrieval_result,
        }
