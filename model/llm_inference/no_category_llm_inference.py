from __future__ import annotations

from typing import Any, Dict, Optional

from model.encoders.model_handler import BaseModelHandler, create_model_handler
from model.hierarchical.hierarchy_types import HierarchyLevel
from model.hierarchical.no_category_hierarchical_manager import (
    NoCategoryHierarchicalMemoryManager,
    create_no_category_hierarchical_manager,
)
from model.retrievers.no_category_cosine_retriver import NoCategoryCosineRetriever
from model.retrievers.no_category_hyperbolic_retriver import (
    NoCategoryGeodesicHyperbolicRetriever,
    NoCategoryHybridHyperbolicRetriever,
    NoCategoryMultiParentAngularHyperbolicRetriever,
)


class NoCategoryMemoryAugmentedLLMInference:
    DEFAULT_PROMPT_TEMPLATE = (
        "You are a helpful assistant that answers questions using retrieved memory.\n\n"
        "How to read timestamps in the memory fragments below:\n"
        "- The first line inside each fragment (often a clock/calendar line before 'Speaker: ...') is when that **chat turn was posted** in the conversation log, not necessarily when real-world events described in the words happened.\n"
        "- Deictic time in the utterance ('yesterday', 'last year', 'next week', etc.) must be resolved **relative to that posting time** to infer calendar dates.\n"
        "- Do not equate the posting timestamp with the date of an event inside the quote unless the question explicitly asks when the message was sent.\n\n"
        "Retrieved context from memory:\n"
        "{context}\n\n"
        "Question: {query}\n"
        "Answer based on the context above clearly and concisely.\n"
        "If there is no relevant information in the context, reject the question.\n"
        "Answer:"
    )

    def __init__(
        self,
        manager: NoCategoryHierarchicalMemoryManager | None = None,
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
        retriever_top_k: int = 5,
        start_level: HierarchyLevel = HierarchyLevel.DOMAIN,
        target_level: HierarchyLevel = HierarchyLevel.DIALOGUE,
    ) -> None:
        self.manager = manager or create_no_category_hierarchical_manager(
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
            self.retriever = NoCategoryCosineRetriever(vector_store=self.manager.vector_store)
        elif retriever_type == "hyperbolic_geodesic":
            self.retriever = NoCategoryGeodesicHyperbolicRetriever(
                vector_store=self.manager.vector_store,
                checkpoint_path=projector_checkpoint_path,
            )
        elif retriever_type == "hyperbolic_angular":
            self.retriever = NoCategoryMultiParentAngularHyperbolicRetriever(
                vector_store=self.manager.vector_store,
                checkpoint_path=projector_checkpoint_path,
                **dict(hyperbolic_angular_kwargs or {}),
            )
        elif retriever_type == "hyperbolic_angular_geodesic_hybrid":
            self.retriever = NoCategoryHybridHyperbolicRetriever(
                vector_store=self.manager.vector_store,
                checkpoint_path=projector_checkpoint_path,
                hyperbolic_angular_kwargs=hyperbolic_angular_kwargs,
            )
        else:
            raise ValueError(f"unknown retriever_type: {retriever_type}")

        self.generation_handler: BaseModelHandler | None = None
        if generation_handler_type is not None:
            model_source = generation_model_path or generation_model_name
            if not model_source:
                raise ValueError("generation model is required")
            self.generation_handler = create_model_handler(
                generation_handler_type,
                api_base=generation_api_base,
            )
            if not self.generation_handler.load(model_source, device=device):
                raise RuntimeError("generation model init failed")

    def answer(
        self,
        query_text: str,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None,
        start_level: Optional[HierarchyLevel] = None,
        target_level: Optional[HierarchyLevel] = None,
        retrieve_kwargs: Optional[Dict[str, Any]] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        retrieve_kwargs = retrieve_kwargs or {}
        retrieval_result = self.retriever.retrieve(
            query_text=query_text,
            top_k=top_k or self.retriever_top_k,
            start_level=start_level or self.start_level,
            target_level=target_level or self.target_level,
            **retrieve_kwargs,
        )
        context = self.retriever.get_context(
            query_text=query_text,
            top_k=top_k or self.retriever_top_k,
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
