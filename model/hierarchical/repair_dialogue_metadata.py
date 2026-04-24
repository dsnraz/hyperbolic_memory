"""
汇总处理失败的原始样本，并导出待重试文本清单。

当前逻辑以 DataProcessor 记录的 failed_indices1.json 为主，
problematic_dialogues_after_repair.json 为辅，二者按原始文本索引去重后，
输出 text_id 与 content 到 failed_dialogues.json。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set

from transformers import AutoTokenizer
from ..encoders.llm_encoder import LLMEncoder


def _load_data(datapath: str) -> List[Any]:
    """按 DataProcessor 的规则加载原始数据。"""
    with open(datapath, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)

    if isinstance(data, dict):
        for key in ["data", "items", "dialogues", "conversations"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    return data if isinstance(data, list) else [data]


def _serialize_item(item: Any) -> str:
    """按 DataProcessor 的规则将样本序列化为文本。"""
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)
    return str(item)


def _load_tokenizer(tokenizer_path: str):
    """加载用于统计 token 数的 tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|extra_0|>"
    return tokenizer


def _count_tokens(tokenizer, text: str) -> int:
    """统计文本 token 数，不包含额外 special tokens。"""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _build_llm_encoder(llm_model_path: str, device: str) -> LLMEncoder:
    """构建用于分析失败样本的 LLM 编码器。"""
    return LLMEncoder(
        model_path=llm_model_path,
        model_type="transformers",
        device=device,
    )


def _load_failed_indices(path: str) -> Set[int]:
    """从 failed_indices1.json 中读取失败样本索引。"""
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as failed_file:
        data = json.load(failed_file)

    return {
        int(index)
        for index in data.get("failed_indices", [])
        if isinstance(index, int) or (isinstance(index, str) and index.isdigit())
    }


def _extract_text_id(item: Any) -> Optional[int]:
    """从辅助报告项中提取原始文本索引。"""
    if isinstance(item, int):
        return item
    if isinstance(item, str) and item.isdigit():
        return int(item)
    if not isinstance(item, dict):
        return None

    value = item.get("text_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _load_auxiliary_indices(report_path: str) -> Set[int]:
    """从 problematic_dialogues_after_repair.json 中读取辅助索引。"""
    if not os.path.exists(report_path):
        return set()

    with open(report_path, "r", encoding="utf-8") as report_file:
        data = json.load(report_file)

    collected_indices: Set[int] = set()
    for key in ("nodes", "failed_repairs", "repaired_nodes"):
        for item in data.get(key, []):
            text_id = _extract_text_id(item)
            if text_id is not None:
                collected_indices.add(text_id)
    return collected_indices


def collect_failed_dialogues(
    datapath: str,
    failed_indices_path: str,
    source_report_path: str,
    tokenizer_path: str,
    llm_model_path: str,
    device: str,
    max_length: int,
) -> Dict[str, Any]:
    """汇总失败样本索引并导出 LLM 分析结果。"""
    data_items = _load_data(datapath)
    primary_indices = _load_failed_indices(failed_indices_path)
    auxiliary_indices = _load_auxiliary_indices(source_report_path)
    tokenizer = _load_tokenizer(tokenizer_path)
    llm_encoder = _build_llm_encoder(llm_model_path, device)

    merged_indices = sorted(primary_indices | auxiliary_indices)
    failed_dialogues: List[Dict[str, Any]] = []

    for text_id in merged_indices:
        if text_id < 0 or text_id >= len(data_items):
            continue

        content = _serialize_item(data_items[text_id])
        analysis_result, analyze_ok = llm_encoder.analyze(content, max_length=max_length)
        failed_dialogues.append({
            "text_id": text_id,
            "token_count": _count_tokens(tokenizer, content),
            "analyze_ok": analyze_ok,
            "processed_content": analysis_result,
        })

    return {
        "datapath": datapath,
        "failed_indices_path": failed_indices_path,
        "source_report_path": source_report_path,
        "failed_dialogue_count": len(failed_dialogues),
        "failed_dialogues": failed_dialogues,
        "stats": {
            "primary_index_count": len(primary_indices),
            "auxiliary_index_count": len(auxiliary_indices),
            "merged_index_count": len(merged_indices),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总失败原始样本并导出待重试文本")
    parser.add_argument(
        "--persist-directory",
        type=str,
        default="/share/home/leiyh5/Memory/data/hierarchical_memory1",
        help="失败索引与辅助报告所在目录",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/share/home/leiyh5/Memory/data/hotpot_train_v1.1.json",
        help="原始数据文件路径",
    )
    parser.add_argument(
        "--failed-index-path",
        type=str,
        default=None,
        help="主失败索引文件路径，默认取 persist_directory/failed_indices1.json",
    )
    parser.add_argument(
        "--source-report-path",
        type=str,
        default=None,
        help="辅助问题报告路径，默认取 persist_directory/problematic_dialogues_after_repair.json",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/share/home/leiyh5/models/Qwen2.5-7B-Instruct",
        help="用于统计 token 数的 tokenizer 路径",
    )
    parser.add_argument(
        "--llm-model-path",
        type=str,
        default="/share/home/leiyh5/models/Qwen2.5-7B-Instruct",
        help="用于 analyze 的 LLM 模型路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="分析失败样本时使用的设备",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="传给 analyze 的最大输入长度",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="输出文件路径，默认写入 persist_directory/failed_dialogues.json",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="终端最多打印多少条失败样本",
    )
    args = parser.parse_args()
    failed_index_path = args.failed_index_path or os.path.join(
        args.persist_directory,
        "failed_indices1.json",
    )
    source_report_path = args.source_report_path or os.path.join(
        args.persist_directory,
        "problematic_dialogues_after_repair.json",
    )
    output_path = args.output_path or os.path.join(
        args.persist_directory,
        "failed_dialogues.json",
    )

    failed_dialogue_report = collect_failed_dialogues(
        datapath=args.datapath,
        failed_indices_path=failed_index_path,
        source_report_path=source_report_path,
        tokenizer_path=args.tokenizer_path,
        llm_model_path=args.llm_model_path,
        device=args.device,
        max_length=args.max_length,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(failed_dialogue_report, output_file, ensure_ascii=False, indent=2)

    print(f"failed_dialogues 已写入: {output_path}")
    print(
        "统计: "
        f"主索引 {failed_dialogue_report['stats']['primary_index_count']} 条, "
        f"辅助索引 {failed_dialogue_report['stats']['auxiliary_index_count']} 条, "
        f"去重后 {failed_dialogue_report['stats']['merged_index_count']} 条, "
        f"有效导出 {failed_dialogue_report['failed_dialogue_count']} 条"
    )

    for item in failed_dialogue_report["failed_dialogues"][:args.print_limit]:
        print("-" * 60)
        print(f"text_id: {item['text_id']}")
        print(f"token_count: {item['token_count']}")
        print(f"analyze_ok: {item['analyze_ok']}")
        print(f"processed_content: {item['processed_content']}")


if __name__ == "__main__":
    main()
