from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from model.llm_inference.data_adapter import extract_interactions

from .session_hierarchical_manager import (
    SessionHierarchicalMemoryManager,
    create_session_hierarchical_manager,
)


class SessionDataProcessor:
    """Dataset-to-store pipeline for the session-level build strategy."""

    def __init__(
        self,
        manager: Optional[SessionHierarchicalMemoryManager] = None,
        datapath: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        device: str = "auto",
        flush_interval: int = 128,
    ) -> None:
        self.manager = manager or create_session_hierarchical_manager(
            llm_model_path=llm_model_path,
            persist_directory=persist_directory,
            device=device,
        )
        self.datapath = datapath
        self.flush_interval = flush_interval
        self.data = self.load_data() if datapath else None

    def load_data(self) -> List[Any]:
        with open(self.datapath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            for key in ("data", "items", "dialogues", "conversations"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
        return data if isinstance(data, list) else [data]

    def process_file(
        self,
        dataset_name: Optional[str] = None,
        max_items: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        if self.data is None:
            raise ValueError("data not loaded")

        total = min(len(self.data), max_items) if max_items is not None else len(self.data)
        iterator = range(total)
        if show_progress:
            iterator = tqdm(iterator, desc="session build", unit="session")

        success_count = 0
        for idx in iterator:
            interactions = extract_interactions(self.data[idx], dataset_name=dataset_name)
            _, ok = self.manager.process_session(interactions, session_id=str(idx))
            if ok:
                success_count += 1
            if success_count > 0 and success_count % self.flush_interval == 0:
                self.manager.flush()

        if self.manager.get_pending_dirty_count() > 0:
            self.manager.flush()

    def get_stats(self) -> Dict[str, int]:
        return self.manager.get_stats().to_dict()
