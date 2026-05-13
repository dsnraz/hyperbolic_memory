from __future__ import annotations

from typing import Literal, Optional

from model.hierarchical.session_hierarchical_manager import create_session_hierarchical_manager
from model.llm_inference.llm_inference import MemoryAugmentedLLMInference


class SessionMemoryAugmentedLLMInference(MemoryAugmentedLLMInference):
    """Inference wrapper for the session-level hierarchy build."""

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        device: str = "auto",
        memory_unit_mode: Literal["keyword", "fact"] = "keyword",
        extraction_mode: Literal["single", "two_stage"] = "single",
        **kwargs,
    ) -> None:
        manager = create_session_hierarchical_manager(
            llm_model_path=llm_model_path,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            device=device,
            delayed_write=False,
            memory_unit_mode=memory_unit_mode,
            extraction_mode=extraction_mode,
        )
        super().__init__(
            manager=manager,
            llm_model_path=llm_model_path,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            device=device,
            **kwargs,
        )
