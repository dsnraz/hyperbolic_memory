import argparse
import copy
import json
import os
import random
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT_PATH = "/share/home/leiyh5/Memory/data/locomo10.json"
DEFAULT_TRAIN_PATH = "/share/home/leiyh5/Memory/data/locomo_qa_train.json"
DEFAULT_TEST_PATH = "/share/home/leiyh5/Memory/data/locomo_qa_test.json"
DEFAULT_SESSION_OUTPUT_PATH = "/share/home/leiyh5/Memory/data/locomo_train_sessions.json"


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


def split_qa_list(
    qa_list: List[Dict[str, Any]],
    train_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not qa_list:
        return [], []

    indices = list(range(len(qa_list)))
    rng.shuffle(indices)

    train_count = round(len(qa_list) * train_ratio)
    if len(qa_list) > 1:
        train_count = min(max(train_count, 1), len(qa_list) - 1)
    else:
        train_count = 1

    train_indices = sorted(indices[:train_count])
    test_indices = sorted(indices[train_count:])

    train_qa = [copy.deepcopy(qa_list[idx]) for idx in train_indices]
    test_qa = [copy.deepcopy(qa_list[idx]) for idx in test_indices]
    return train_qa, test_qa


def split_samples_by_qa(
    samples: Iterable[Dict[str, Any]],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    train_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for sample in samples:
        train_sample = copy.deepcopy(sample)
        test_sample = copy.deepcopy(sample)

        train_qa, test_qa = split_qa_list(sample.get("qa", []), train_ratio, rng)
        train_sample["qa"] = train_qa
        test_sample["qa"] = test_qa

        train_samples.append(train_sample)
        test_samples.append(test_sample)

    return train_samples, test_samples


def get_session_numbers(conversation: Dict[str, Any]) -> List[int]:
    session_numbers = []
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith("date_time"):
            session_numbers.append(int(key.split("_")[-1]))
    return sorted(session_numbers)


def session_to_turns(session: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    turns: List[Dict[str, str]] = []
    for turn in session:
        speaker = str(turn.get("speaker", "")).strip()
        text = str(turn.get("text", "")).strip()
        if speaker and text:
            turns.append({speaker: text})
    return turns


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
                    "conversation": session_to_turns(session),
                }
            )

    return session_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LoCoMo QA into train/test and flatten training conversations by session."
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
        "--session-output-path",
        type=str,
        default=DEFAULT_SESSION_OUTPUT_PATH,
        help="Path to the flattened training-session JSON file.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio used to split QA within each sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible QA splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_path)
    train_samples, test_samples = split_samples_by_qa(
        samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    session_records = extract_session_records(train_samples)

    dump_json(train_samples, args.train_output_path)
    dump_json(test_samples, args.test_output_path)
    dump_json(session_records, args.session_output_path)

    original_qa_count = sum(len(sample.get("qa", [])) for sample in samples)
    train_qa_count = sum(len(sample.get("qa", [])) for sample in train_samples)
    test_qa_count = sum(len(sample.get("qa", [])) for sample in test_samples)

    print(f"Input file: {args.input_path}")
    print(f"Train file: {args.train_output_path}")
    print(f"Test file: {args.test_output_path}")
    print(f"Session file: {args.session_output_path}")
    print(f"Sample count: {len(samples)}")
    print(f"Original QA count: {original_qa_count}")
    print(f"Train QA count: {train_qa_count}")
    print(f"Test QA count: {test_qa_count}")
    print(f"Flattened session count: {len(session_records)}")


if __name__ == "__main__":
    main()
