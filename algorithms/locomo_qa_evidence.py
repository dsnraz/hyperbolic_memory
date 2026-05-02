"""从 locomo_qa_test.json 按 question 匹配 QA，用 evidence 的 dia_id 拼出与 vector store 一致的原文格式。"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _norm_question(s: str) -> str:
    return " ".join((s or "").split())


def find_qa_match(data: list, query: str) -> tuple[dict, dict] | None:
    """返回 (顶层样本, 命中的 qa 项)；question 与 query 去首尾空白并折叠空白后相等即命中。"""
    q = _norm_question(query)
    for record in data:
        for qa in record.get("qa") or []:
            if _norm_question(qa.get("question", "")) == q:
                return record, qa
    return None


def _session_index_from_dia_id(dia_id: str) -> int:
    m = re.match(r"^D(\d+):\d+$", dia_id.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"invalid dia_id: {dia_id!r}")
    return int(m.group(1))


def format_single_turn(record: dict, dia_id: str) -> str:
    """
    单条 evidence 对应一行存储格式：
    {session_N_date_time}\\n{Speaker}: {text}
    """
    conversation = record.get("conversation") or {}
    sess_idx = _session_index_from_dia_id(dia_id)
    session_key = f"session_{sess_idx}"
    dt_key = f"session_{sess_idx}_date_time"
    turns = conversation.get(session_key) or []
    dt = conversation.get(dt_key, "")
    for turn in turns:
        if turn.get("dia_id") == dia_id.strip():
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            return f"{dt}\n{speaker}: {text}"
    raise KeyError(f"dia_id {dia_id!r} not found under conversation[{session_key!r}]")


def format_evidence_dialogue(record: dict, evidence_ids: list[str]) -> str:
    """多条 evidence 用空行拼接。"""
    blocks = [format_single_turn(record, eid.strip()) for eid in evidence_ids]
    return "\n\n".join(blocks)


def reference_dialogue_for_query(query: str, qa_json: Path) -> tuple[str, list[str]]:
    """
    根据与 --query 一致的 question，在数据集中查找 evidence，
    返回 (供 load_node_embedding(text=...) 的字符串, evidence id 列表)。
    """
    path = qa_json.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QA JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    hit = find_qa_match(data, query)
    if hit is None:
        raise LookupError(
            f'在 {path.name} 中未找到与 query 相同的 question: {query!r}\n'
            "请使用数据集中某条 QA 的完整 question 文案（仅空白可略有不同）。"
        )
    record, qa = hit
    evidence = qa.get("evidence") or []
    if not evidence:
        raise LookupError(f"命中的 QA 没有 evidence 字段: {query!r}")
    text = format_evidence_dialogue(record, evidence)
    return text, evidence
