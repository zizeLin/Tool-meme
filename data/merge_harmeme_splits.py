"""Merge HarMeme source splits into the HarM JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def merge_jsonl(file_paths: Iterable[Path], output_path: Path) -> int:
    merged = []
    seen_ids = set()
    for file_path in file_paths:
        if not file_path.exists():
            print(f"Warning: missing input file: {file_path}")
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: invalid JSON in {file_path}:{line_number}")
                    continue
                if not {"image", "text", "labels"}.issubset(item.keys()):
                    print(f"Warning: missing required keys in {file_path}:{line_number}")
                    continue
                doc_id = item.get("id")
                if doc_id and doc_id in seen_ids:
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                merged.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in merged:
            json.dump(item, handle, ensure_ascii=False)
            handle.write("\n")
    return len(merged)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge HarMeme-C and HarMeme-P JSONL splits.")
    parser.add_argument("--test_inputs", nargs="+", required=True)
    parser.add_argument("--test_output", default="data/HarM/test.jsonl")
    parser.add_argument("--train_inputs", nargs="*", default=[])
    parser.add_argument("--train_output", default="data/HarM/train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_count = merge_jsonl([Path(path) for path in args.test_inputs], Path(args.test_output))
    print(f"Merged test split: {test_count} items -> {args.test_output}")
    if args.train_inputs:
        train_count = merge_jsonl([Path(path) for path in args.train_inputs], Path(args.train_output))
        print(f"Merged train split: {train_count} items -> {args.train_output}")


if __name__ == "__main__":
    main()
