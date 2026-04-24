from __future__ import annotations

from typing import Any, Dict, List, Sequence


def normalize_interaction(item: Any) -> str:
    """把单条 interaction 统一成字符串。"""
    if isinstance(item, str):
        text = item.strip()
        if text:
            return text
        raise ValueError("interaction 不能为空")

    if isinstance(item, dict):
        for key in ("interaction", "text", "content", "dialogue", "utterance"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    text = str(item).strip()
    if not text:
        raise ValueError("interaction 不能为空")
    return text


def get_session_numbers(conversation: Dict[str, Any]) -> List[int]:
    """提取 LoCoMo conversation 中的 session 编号并排序。"""
    session_numbers = []
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith("date_time"):
            session_numbers.append(int(key.split("_")[-1]))
    return sorted(session_numbers)


def turn_to_text(turn: Dict[str, Any], time_value: str = "") -> str:
    """
    把 LoCoMo 原生 turn 转成可直接建库的文本。

    这里保留 speaker + text，并把 session 时间拼进去，方便后续时间相关检索。
    """
    speaker = str(turn.get("speaker", "")).strip()
    text = str(turn.get("text", "")).strip()
    if not speaker or not text:
        return ""
    if time_value:
        return f"{time_value}\n{speaker}: {text}"
    return f"{speaker}: {text}"


def extract_locomo_conversation_interactions(sample: Dict[str, Any]) -> List[str]:
    """从 locomo10.json 原生样本中提取一整轮 conversation 的所有 interaction。"""
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo 样本缺少 conversation 字段")

    interactions: List[str] = []
    for session_number in get_session_numbers(conversation):
        session_key = f"session_{session_number}"
        time_key = f"{session_key}_date_time"
        session = conversation.get(session_key, [])
        if not isinstance(session, list):
            continue

        time_value = str(conversation.get(time_key, "")).strip()
        for turn in session:
            if not isinstance(turn, dict):
                continue
            interaction = turn_to_text(turn, time_value=time_value)
            if interaction:
                interactions.append(interaction)

    return interactions


def extract_interactions(sample: Any, dataset_name: str | None = None) -> list[str]:
    """
    面向不同数据集的统一接收接口。

    当前支持：
    - 直接传入 interaction 列表
    - LoCoMo 原生 conversation 样本
    - 通用 dict/list 结构
    """
    if isinstance(sample, list):
        return [normalize_interaction(item) for item in sample]

    if isinstance(sample, tuple):
        return [normalize_interaction(item) for item in sample]

    if isinstance(sample, dict):
        if dataset_name == "locomo":
            if "conversation" in sample:
                return extract_locomo_conversation_interactions(sample)
            for key in ("interactions", "dialogues", "turns"):
                value = sample.get(key)
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    return [normalize_interaction(item) for item in value]
            if "interaction" in sample:
                return [normalize_interaction(sample)]

        for key in ("interactions", "conversation", "dialogues", "turns"):
            value = sample.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [normalize_interaction(item) for item in value]

        return [normalize_interaction(sample)]

    raise TypeError(f"不支持的数据样本类型: {type(sample).__name__}")
