"""Convert raw MAMI files into the JSONL format used by Tool-meme."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, List


MAMI_LABEL_COLUMNS = [
    "misogynous",
    "shaming",
    "stereotype",
    "objectification",
    "violence",
]


def aggregate_binary_label(values: Iterable[str]) -> int:
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError):
            parsed.append(0)
    return 1 if any(value == 1 for value in parsed) else 0


def ensure_output_dirs(output_dir: Path) -> Path:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def write_jsonl(items: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in items:
            json.dump(item, handle, ensure_ascii=False)
            handle.write("\n")


def process_training_data(csv_path: Path, output_path: Path) -> int:
    entries = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "file_name" not in reader.fieldnames:
            raise ValueError(f"Training CSV is missing the file_name column: {reader.fieldnames}")
        for row in reader:
            label = aggregate_binary_label(row.get(column, "0") for column in MAMI_LABEL_COLUMNS)
            entries.append({
                "image": row["file_name"].strip(),
                "text": row.get("Text Transcription", "").strip(),
                "label": label,
            })
    write_jsonl(entries, output_path)
    return len(entries)


def process_test_data(text_csv_path: Path, label_txt_path: Path, output_path: Path) -> int:
    text_map = {}
    with text_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "file_name" not in reader.fieldnames:
            raise ValueError(f"Test CSV is missing the file_name column: {reader.fieldnames}")
        for row in reader:
            text_map[row["file_name"].strip()] = row.get("Text Transcription", "").strip()

    entries = []
    with label_txt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            file_name = parts[0].strip()
            if file_name not in text_map:
                continue
            entries.append({
                "image": file_name,
                "text": text_map[file_name],
                "label": aggregate_binary_label(parts[1:6]),
            })
    write_jsonl(entries, output_path)
    return len(entries)


def copy_images(source_dirs: Iterable[Path], destination_dir: Path) -> int:
    copied = 0
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: image directory does not exist: {source_dir}")
            continue
        for path in source_dir.iterdir():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            target = destination_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
                copied += 1
    return copied


def parse_args():
    parser = argparse.ArgumentParser(description="Build MAMI train/test JSONL files for Tool-meme.")
    parser.add_argument("--training_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--train_image_dir", required=True)
    parser.add_argument("--test_image_dir", required=True)
    parser.add_argument("--output_dir", default="data/MAMI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    images_dir = ensure_output_dirs(output_dir)
    train_count = process_training_data(Path(args.training_csv), output_dir / "train.jsonl")
    test_count = process_test_data(Path(args.test_csv), Path(args.test_labels), output_dir / "test.jsonl")
    copied = copy_images([Path(args.train_image_dir), Path(args.test_image_dir)], images_dir)
    print(f"Built MAMI data: train={train_count}, test={test_count}, new_images={copied}")


if __name__ == "__main__":
    main()
