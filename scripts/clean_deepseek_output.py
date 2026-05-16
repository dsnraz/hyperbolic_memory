"""
Post-process DeepSeek-R1 prediction JSON: strip thinking chain from answers.

Usage:
    python scripts/clean_deepseek_output.py <input.json> [output.json]

If output is omitted, overwrites input in-place (with .bak backup).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def clean_prediction(text: str) -> str:
    """Strip DeepSeek-R1 thinking chain, keep only the final answer."""
    if not text:
        return text
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def clean_file(input_path: str, output_path: str | None = None) -> int:
    src = Path(input_path)
    if not src.is_file():
        print(f"File not found: {src}")
        return 1

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    cleaned = 0
    for sample in data if isinstance(data, list) else [data]:
        for qa in sample.get("qa", []):
            raw = qa.get("memory_prediction", "")
            if not raw:
                continue
            new_val = clean_prediction(raw)
            if new_val != raw:
                qa["memory_prediction"] = new_val
                cleaned += 1

    if output_path is None:
        # In-place with backup
        backup = src.with_suffix(src.suffix + ".bak")
        shutil.copy2(src, backup)
        print(f"Backup: {backup}")
        output_path = input_path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    total_qa = sum(len(s.get("qa", [])) for s in (data if isinstance(data, list) else [data]))
    print(f"Cleaned {cleaned}/{total_qa} predictions → {out}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip DeepSeek-R1 thinking chains")
    parser.add_argument("input", type=str, help="Prediction JSON file")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="Output path (default: overwrite input with .bak)")
    args = parser.parse_args()
    sys.exit(clean_file(args.input, args.output))


if __name__ == "__main__":
    main()
