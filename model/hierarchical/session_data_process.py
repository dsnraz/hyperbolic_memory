from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from model.llm_inference.data_adapter import extract_interactions, get_session_numbers, turn_to_text

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
        process_batch_size: int = 8,
    ) -> None:
        if self.data is None:
            raise ValueError("data not loaded")
        if process_batch_size <= 0:
            raise ValueError("process_batch_size must be positive")

        total_items = min(len(self.data), max_items) if max_items is not None else len(self.data)
        items = self.data[:total_items]
        tasks = self._build_session_tasks(items, dataset_name=dataset_name)
        total_sessions = len(tasks)
        processed_count = 0
        success_count = 0
        fail_count = 0

        pbar = None
        if show_progress:
            pbar = tqdm(total=total_sessions, desc="session build", unit="session")

        for start in range(0, total_sessions, process_batch_size):
            end = min(start + process_batch_size, total_sessions)
            batch_tasks = tasks[start:end]
            session_ids = [session_id for session_id, _ in batch_tasks]
            sessions = [interactions for _, interactions in batch_tasks]
            _, ok_list = self.manager.batch_process_sessions(
                sessions=sessions,
                show_progress=False,
                session_ids=session_ids,
            )
            batch_success = sum(1 for ok in ok_list if ok)
            batch_fail = len(ok_list) - batch_success
            processed_count += len(ok_list)
            success_count += batch_success
            fail_count += batch_fail

            if processed_count > 0 and processed_count % self.flush_interval == 0:
                self.manager.flush()
            if pbar is not None:
                pbar.update(len(ok_list))

        if pbar is not None:
            pbar.close()

        if self.manager.get_pending_dirty_count() > 0:
            self.manager.flush()

        success_rate = (success_count / total_sessions * 100.0) if total_sessions > 0 else 0.0
        print(f"\n{'=' * 50}")
        print("Session 建库完成")
        print(f"  输入样本数: {total_items}")
        print(f"  总会话数: {total_sessions}")
        print(f"  已处理会话: {processed_count}")
        print(f"  成功会话: {success_count}")
        print(f"  失败会话: {fail_count}")
        print(f"  成功率: {success_rate:.2f}%")
        print(f"{'=' * 50}")

    def _build_session_tasks(
        self,
        items: List[Any],
        dataset_name: Optional[str],
    ) -> List[tuple[str, List[str]]]:
        """将输入样本展开为 session 粒度任务。"""
        tasks: List[tuple[str, List[str]]] = []
        if dataset_name == "locomo":
            for conv_idx, sample in enumerate(items):
                if not isinstance(sample, dict):
                    interactions = extract_interactions(sample, dataset_name=dataset_name)
                    tasks.append((f"{conv_idx}_session_0", interactions))
                    continue

                conversation = sample.get("conversation")
                if not isinstance(conversation, dict):
                    interactions = extract_interactions(sample, dataset_name=dataset_name)
                    tasks.append((f"{conv_idx}_session_0", interactions))
                    continue

                session_numbers = get_session_numbers(conversation)
                if not session_numbers:
                    interactions = extract_interactions(sample, dataset_name=dataset_name)
                    tasks.append((f"{conv_idx}_session_0", interactions))
                    continue

                for session_number in session_numbers:
                    session_key = f"session_{session_number}"
                    time_key = f"{session_key}_date_time"
                    raw_turns = conversation.get(session_key, [])
                    if not isinstance(raw_turns, list):
                        continue

                    time_value = str(conversation.get(time_key, "")).strip()
                    interactions: List[str] = []
                    for turn in raw_turns:
                        if not isinstance(turn, dict):
                            continue
                        text = turn_to_text(turn, time_value=time_value)
                        if text:
                            interactions.append(text)
                    if interactions:
                        tasks.append((f"{conv_idx}_{session_key}", interactions))
            return tasks

        for idx, sample in enumerate(items):
            interactions = extract_interactions(sample, dataset_name=dataset_name)
            tasks.append((f"{idx}_session_0", interactions))
        return tasks

    def get_stats(self) -> Dict[str, int]:
        return self.manager.get_stats().to_dict()


def main() -> None:
    LLM_MODEL_PATH = "/share/home/leiyh5/models/Qwen2.5-7B-Instruct"
    DATA_FILE = "/share/home/leiyh5/Memory/data/locomo/locomo10.json"
    PERSIST_DIR = "/share/home/leiyh5/Memory/data/hierarchical_memory_locomo_session_batch"

    processor = SessionDataProcessor(
        llm_model_path=LLM_MODEL_PATH,
        persist_directory=PERSIST_DIR,
        device="auto",
        datapath=DATA_FILE,
        flush_interval=128,
    )
    print("SessionDataProcessor 创建成功")

    processor.process_file(
        dataset_name="locomo",
        show_progress=True,
        process_batch_size=8,
    )

    stats = processor.get_stats()
    print("\n统计信息:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  领域数: {stats['domain_count']}")
    print(f"  类别数: {stats['category_count']}")
    print(f"  关键词数: {stats['keyword_count']}")
    print(f"  对话数: {stats['dialogue_count']}")


if __name__ == "__main__":
    main()
