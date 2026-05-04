import argparse
import copy
import json
import os
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_INPUT_PATH = "/share/home/leiyh5/Memory/data/locomo/locomo10.json"
DEFAULT_TRAIN_PATH = "/share/home/leiyh5/Memory/data/locomo/locomo_qa_train.json"
DEFAULT_TEST_PATH = "/share/home/leiyh5/Memory/data/locomo/locomo_qa_test.json"
DEFAULT_TRAIN_INTERACTION_OUTPUT_PATH = "/share/home/leiyh5/Memory/data/locomo/locomo_train_interactions.json"


def load_samples(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("LoCoMo file must contain a top-level JSON array.")

    return data


def dump_json(data: Any, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def split_conversations(
    samples: Sequence[Dict[str, Any]],
    train_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not samples:
        return [], []

    rng = random.Random(42)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    train_count = round(len(samples) * train_ratio)
    if len(samples) > 1:
        train_count = min(max(train_count, 1), len(samples) - 1)
    else:
        train_count = 1

    train_indices = sorted(indices[:train_count])
    test_indices = sorted(indices[train_count:])

    train_samples = [copy.deepcopy(samples[idx]) for idx in train_indices]
    test_samples = [copy.deepcopy(samples[idx]) for idx in test_indices]
    return train_samples, test_samples


def get_session_numbers(conversation: Dict[str, Any]) -> List[int]:
    session_numbers = []
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith("date_time"):
            session_numbers.append(int(key.split("_")[-1]))
    return sorted(session_numbers)


def session_to_text(session: List[Dict[str, Any]]) -> str:
    turns: List[str] = []
    for turn in session:
        text = turn_to_text(turn)
        if text:
            turns.append(text)
    return "\n".join(turns)


def image_context_text(turn: Dict[str, Any]) -> str:
    parts: List[str] = []
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        parts.append(f"Image description: {caption}")
    image_query = str(turn.get("query", "")).strip()
    if image_query:
        parts.append(f"Image query: {image_query}")
    return "\n".join(parts)


def turn_to_text(turn: Dict[str, Any]) -> str:
    speaker = str(turn.get("speaker", "")).strip()
    text = str(turn.get("text", "")).strip()
    if speaker and text:
        parts = [f"{speaker}: {text}"]
        image_context = image_context_text(turn)
        if image_context:
            parts.append(image_context)
        return "\n".join(parts)
    return ""


def extract_session_records(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    session_records: List[Dict[str, Any]] = []

    for sample in samples:
        conversation = sample.get("conversation", {})
        if not isinstance(conversation, dict):
            continue

        for session_number in get_session_numbers(conversation):
            session_key = f"session_{session_number}"
            time_key = f"{session_key}_date_time"
            session = conversation.get(session_key, [])

            if not isinstance(session, list):
                continue

            session_records.append(
                {
                    "time": str(conversation.get(time_key, "")),
                    "conversation": session_to_text(session),
                }
            )

    return session_records


def extract_interaction_records(samples: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    interaction_records: List[Dict[str, str]] = []

    for sample in samples:
        conversation = sample.get("conversation", {})
        if not isinstance(conversation, dict):
            continue

        for session_number in get_session_numbers(conversation):
            session_key = f"session_{session_number}"
            time_key = f"{session_key}_date_time"
            session = conversation.get(session_key, [])
            if not isinstance(session, list):
                continue

            time_value = str(conversation.get(time_key, ""))
            for turn in session:
                interaction = turn_to_text(turn)
                if interaction:
                    interaction_records.append(
                        {
                            "time": time_value,
                            "interaction": interaction,
                        }
                    )

    return interaction_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LoCoMo by conversation and flatten training conversations into interaction-level data."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to the source LoCoMo JSON file.",
    )
    parser.add_argument(
        "--train-output-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the output LoCoMo train JSON file.",
    )
    parser.add_argument(
        "--test-output-path",
        type=str,
        default=DEFAULT_TEST_PATH,
        help="Path to the output LoCoMo test JSON file.",
    )
    parser.add_argument(
        "--train-interaction-output-path",
        type=str,
        default=DEFAULT_TRAIN_INTERACTION_OUTPUT_PATH,
        help="Path to the flattened train-interaction JSON file.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio used to split conversations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_path)
    train_samples, test_samples = split_conversations(
        samples,
        train_ratio=args.train_ratio,
    )
    train_interaction_records = extract_interaction_records(train_samples)

    dump_json(train_samples, args.train_output_path)
    dump_json(test_samples, args.test_output_path)
    dump_json(train_interaction_records, args.train_interaction_output_path)

    original_qa_count = sum(len(sample.get("qa", [])) for sample in samples)
    train_qa_count = sum(len(sample.get("qa", [])) for sample in train_samples)
    test_qa_count = sum(len(sample.get("qa", [])) for sample in test_samples)
    train_sample_ids = [str(sample.get("sample_id", "")) for sample in train_samples]
    test_sample_ids = [str(sample.get("sample_id", "")) for sample in test_samples]

    print(f"Input file: {args.input_path}")
    print(f"Train file: {args.train_output_path}")
    print(f"Test file: {args.test_output_path}")
    print(f"Train interaction file: {args.train_interaction_output_path}")
    print(f"Sample count: {len(samples)}")
    print(f"Train conversation count: {len(train_samples)}")
    print(f"Test conversation count: {len(test_samples)}")
    print(f"Original QA count: {original_qa_count}")
    print(f"Train QA count: {train_qa_count}")
    print(f"Test QA count: {test_qa_count}")
    print(f"Original interaction count: {len(extract_interaction_records(samples))}")
    print(f"Train interaction count: {len(train_interaction_records)}")
    print(f"Train sample_ids: {train_sample_ids}")
    print(f"Test sample_ids: {test_sample_ids}")


if __name__ == "__main__":
    main()
