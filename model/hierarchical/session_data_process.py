from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

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
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        device: str = "auto",
        flush_interval: int = 128,
        memory_unit_mode: Literal["keyword", "fact"] = "keyword",
    ) -> None:
        self.manager = manager or create_session_hierarchical_manager(
            llm_model_path=llm_model_path,
            embedding_model=embedding_model,
            persist_directory=persist_directory,
            device=device,
            memory_unit_mode=memory_unit_mode,
        )
        self.datapath = datapath
        self.flush_interval = flush_interval
        self.memory_unit_mode = memory_unit_mode
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
        llm_output_save_path: Optional[str] = None,
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
        llm_output_entries: List[Dict[str, Any]] = []

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
            analyses = self.manager.get_last_batch_analyses()
            for sid, analysis in zip(session_ids, analyses):
                llm_output_entries.append(
                    {
                        "session_id": sid,
                        "llm_output": analysis,
                    }
                )
            batch_success = sum(1 for ok in ok_list if ok)
            batch_fail = len(ok_list) - batch_success
            processed_count += len(ok_list)
            success_count += batch_success
            fail_count += batch_fail

            if processed_count > 0 and processed_count % self.flush_interval == 0:
                print(f"[flush] 开始, pending={self.manager.get_pending_dirty_count()}", flush=True)
                self.manager.flush()
                print(f"[flush] 完成", flush=True)
            if pbar is not None:
                pbar.update(len(ok_list))

        if pbar is not None:
            pbar.close()

        if self.manager.get_pending_dirty_count() > 0:
            self.manager.flush()

        if llm_output_save_path:
            self._save_llm_outputs_json(
                output_path=llm_output_save_path,
                dataset_name=dataset_name,
                total_items=total_items,
                entries=llm_output_entries,
            )

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

    def _save_llm_outputs_json(
        self,
        output_path: str,
        dataset_name: Optional[str],
        total_items: int,
        entries: List[Dict[str, Any]],
    ) -> None:
        grouped: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            session_id = str(entry.get("session_id", ""))
            sample_id = session_id.split("_session_", 1)[0] if "_session_" in session_id else session_id
            sample_bucket = grouped.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "sessions": [],
                },
            )
            sample_bucket["sessions"].append(entry)

        sample_outputs = sorted(
            grouped.values(),
            key=lambda item: int(item["sample_id"]) if str(item["sample_id"]).isdigit() else str(item["sample_id"]),
        )
        payload = {
            "dataset_name": dataset_name,
            "total_samples": total_items,
            "total_sessions": len(entries),
            "samples": sample_outputs,
        }
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"LLM 输出已写入: {output_file}")

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
    DATA_FILE = "/share/home/leiyh5/Memory/data/locomo/locomo_qa_test.json"
    PERSIST_DIR = "/share/home/leiyh5/Memory/data/hierarchical_memory_locomo_category"
    parser = argparse.ArgumentParser(description="Build session-level hierarchical memory")
    parser.add_argument("--data-file", type=str, default=DATA_FILE)
    parser.add_argument("--persist-directory", type=str, default=PERSIST_DIR)
    parser.add_argument("--llm-model-path", type=str, default=LLM_MODEL_PATH)
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataset-name", type=str, default="locomo")
    parser.add_argument("--process-batch-size", type=int, default=8)
    parser.add_argument("--flush-interval", type=int, default=64)
    parser.add_argument("--memory-unit-mode", choices=("keyword", "fact"), default="fact")
    parser.add_argument(
        "--llm-output-save-path",
        type=str,
        default="/share/home/leiyh5/Memory/data/locomo/outpu1.json",
        help="Save aggregated per-sample LLM outputs to a pretty-printed JSON file",
    )
    args = parser.parse_args()

    processor = SessionDataProcessor(
        llm_model_path=args.llm_model_path,
        embedding_model=args.embedding_model,
        persist_directory=args.persist_directory,
        device=args.device,
        datapath=args.data_file,
        flush_interval=args.flush_interval,
        memory_unit_mode=args.memory_unit_mode,
    )
    print("SessionDataProcessor 创建成功")
    print(f"Memory unit mode: {args.memory_unit_mode}")

    processor.process_file(
        dataset_name=args.dataset_name,
        show_progress=True,
        process_batch_size=args.process_batch_size,
        llm_output_save_path=args.llm_output_save_path or None,
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
