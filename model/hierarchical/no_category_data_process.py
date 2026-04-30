from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .no_category_hierarchical_manager import (
    NoCategoryHierarchicalMemoryManager,
    create_no_category_hierarchical_manager,
)


class NoCategoryDataProcessor:
    """Build DOMAIN -> KEYWORD -> DIALOGUE memory from raw data."""

    def __init__(
        self,
        manager: Optional[NoCategoryHierarchicalMemoryManager] = None,
        datapath: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        persist_directory: Optional[str] = None,
        device: str = "auto",
        flush_interval: int = 2048,
        llm_batch_size: int = 24,
    ) -> None:
        self.manager = manager or create_no_category_hierarchical_manager(
            llm_model_path=llm_model_path,
            persist_directory=persist_directory,
            device=device,
        )
        self.datapath = datapath
        self.flush_interval = flush_interval
        self.llm_batch_size = llm_batch_size
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
        max_items: Optional[int] = None,
        process_batch_size: int = 128,
        show_progress: bool = True,
    ) -> None:
        if self.data is None:
            raise ValueError("data not loaded")

        total = min(len(self.data), max_items) if max_items is not None else len(self.data)
        iterator = range(0, total, process_batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="no-category build", unit="batch")

        success_count = 0
        for start in iterator:
            end = min(start + process_batch_size, total)
            batch_items = self.data[start:end]
            batch_dialogues = [
                json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                for item in batch_items
            ]
            _, ok_flags = self.manager.batch_process_dialogues(
                batch_dialogues,
                llm_batch_size=self.llm_batch_size,
                show_progress=False,
            )
            success_count += sum(ok_flags)
            if success_count > 0 and success_count % self.flush_interval == 0:
                self.manager.flush()

        if self.manager.get_pending_dirty_count() > 0:
            self.manager.flush()

    def get_stats(self) -> Dict[str, int]:
        return self.manager.get_stats().to_dict()
