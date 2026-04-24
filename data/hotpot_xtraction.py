import argparse
import json
import os
from typing import Any, Dict, Iterable, Iterator, List


DEFAULT_INPUT_PATH = "/share/home/leiyh5/Memory/data/hotpot_train_v1.1.json"
DEFAULT_OUTPUT_PATH = "/share/home/leiyh5/Memory/data/hotpot_context_texts.json"


def load_hotpot_samples(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Hotpot file must contain a top-level JSON array.")

    return data


def iter_context_texts(samples: Iterable[Dict[str, Any]]) -> Iterator[str]:
    for sample in samples:
        context_items = sample.get("context", [])
        if not isinstance(context_items, list):
            continue

        for context_item in context_items:
            if not isinstance(context_item, list) or len(context_item) < 2:
                continue

            paragraphs = context_item[1]
            if not isinstance(paragraphs, list):
                continue

            cleaned_paragraphs = []
            for paragraph in paragraphs:
                if not isinstance(paragraph, str):
                    continue

                cleaned = paragraph.strip()
                if cleaned:
                    cleaned_paragraphs.append(cleaned)

            if cleaned_paragraphs:
                yield " ".join(cleaned_paragraphs)


def write_json_array(items: Iterable[str], output_path: str) -> int:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("[\n")

        first_item = True
        for item in items:
            if not first_item:
                file.write(",\n")

            json.dump(item, file, ensure_ascii=False)
            first_item = False
            count += 1

        file.write("\n]\n")

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one merged text per HotpotQA context title."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to the Hotpot JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the extracted context-text JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_hotpot_samples(args.input_path)
    context_text_count = write_json_array(
        iter_context_texts(samples),
        args.output_path,
    )

    print(f"Input file: {args.input_path}")
    print(f"Output file: {args.output_path}")
    print(f"Sample count: {len(samples)}")
    print(f"Context text count: {context_text_count}")


if __name__ == "__main__":
    main()
