"""Prepare optional offline inputs for Tool-meme.

This script groups non-core preprocessing tasks behind explicit switches:

- image embeddings used as optional ATR image features,
- image-text alignment scores used by the cross-modal tool,
- event contexts for train samples,
- event-augmented CAMR files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.tool_meme_config import (
    DEFAULT_CAMR_DIR,
    DEFAULT_DATASETS,
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_CAMR_OUTPUT_DIR,
    camr_path,
    resolve_existing_camr_path,
)
from utils.data_utils import get_item_data
from utils.openai_client import OpenAIModelConfig, get_openai_response_with_parts


_CAPTION_KEYS = ("caption", "image_caption", "img_desc", "image_desc", "description")


def _load_items(dataset_name: str, split: str) -> Tuple[List[Dict], str, str]:
    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    jsonl_path = f"{base_data_path}/{split}.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"Missing file: {jsonl_path}")
        return [], image_base_path, jsonl_path

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    print(f"Loaded {len(data)} items for {dataset_name}/{split}.")
    return data, image_base_path, jsonl_path


def extract_image_embeddings(
    dataset_name: str,
    split: str = "test",
    batch_size: int = 128,
    embedding_dir: str = DEFAULT_EMBEDDING_DIR,
) -> None:
    from PIL import Image
    from tqdm import tqdm
    import open_clip
    import torch

    data, image_base_path, _ = _load_items(dataset_name, split)
    if not data:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k", device=device
    )
    model.eval()

    embeddings: List[np.ndarray] = []
    index_map: List[int] = []
    batch_imgs: List[torch.Tensor] = []
    batch_indices: List[int] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        batch_tensor = torch.stack(batch_imgs, dim=0).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor).detach().cpu().numpy()
        for row in feats:
            embeddings.append(row.astype(np.float32))
        index_map.extend(batch_indices)
        batch_imgs.clear()
        batch_indices.clear()

    for idx, item in tqdm(list(enumerate(data)), total=len(data), desc=f"{dataset_name}-{split}"):
        image_file_name, text_content, _ = get_item_data(item, dataset_name)
        if not image_file_name or text_content is None:
            continue

        image_path = os.path.join(image_base_path, image_file_name)
        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        batch_imgs.append(preprocess(image))
        batch_indices.append(idx)

        if len(batch_imgs) >= batch_size:
            flush_batch()

    flush_batch()

    if not embeddings:
        print("No image embeddings generated.")
        return

    out_dir = Path(embedding_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / f"{dataset_name}_{split}_img.npy"
    map_path = out_dir / f"{dataset_name}_{split}_index.json"

    np.save(emb_path, np.stack(embeddings, axis=0))
    with map_path.open("w", encoding="utf-8") as handle:
        json.dump({"index_map": index_map}, handle)

    print(f"Saved image embeddings to {emb_path}")
    print(f"Saved image index map to {map_path}")


def load_image_embeddings(dataset_name: str, split: str, embedding_dir: str = DEFAULT_EMBEDDING_DIR) -> tuple:
    img_path = Path(embedding_dir) / f"{dataset_name}_{split}_img.npy"
    map_path = Path(embedding_dir) / f"{dataset_name}_{split}_index.json"
    if not img_path.exists() or not map_path.exists():
        return None, None
    img_embeddings = np.load(img_path)
    with map_path.open("r", encoding="utf-8") as handle:
        index_map = json.load(handle).get("index_map", [])
    index_to_row = {idx: row for row, idx in enumerate(index_map)}
    return img_embeddings, index_to_row


def extract_alignment_scores(
    dataset_name: str,
    split: str = "test",
    batch_size: int = 128,
    embedding_dir: str = DEFAULT_EMBEDDING_DIR,
) -> None:
    from tqdm import tqdm
    import open_clip
    import torch

    base_data_path = f"data/{dataset_name}"
    jsonl_path = f"{base_data_path}/{split}.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"Missing file: {jsonl_path}")
        return

    img_embeddings, index_to_row = load_image_embeddings(dataset_name, split, embedding_dir)
    if img_embeddings is None or index_to_row is None:
        print(f"Missing image embeddings for {dataset_name}/{split}; run with --image_embeddings first.")
        return

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    print(f"Loaded {len(data)} items for {dataset_name}/{split}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k", device=device
    )
    model.eval()

    align_scores: List[float] = []
    index_map: List[int] = []
    batch_texts: List[str] = []
    batch_imgs: List[np.ndarray] = []
    batch_indices: List[int] = []

    def flush_batch() -> None:
        if not batch_texts:
            return
        tokens = open_clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens).detach().cpu().numpy()
        for img_vec, txt_vec, idx_val in zip(batch_imgs, text_features, batch_indices):
            img = img_vec.astype(np.float32)
            img = img / (np.linalg.norm(img) + 1e-8)
            txt = txt_vec.astype(np.float32)
            txt = txt / (np.linalg.norm(txt) + 1e-8)
            align_scores.append(float(np.dot(img, txt)))
            index_map.append(idx_val)
        batch_texts.clear()
        batch_imgs.clear()
        batch_indices.clear()

    for idx, item in tqdm(list(enumerate(data)), total=len(data), desc=f"{dataset_name}-{split}"):
        image_file_name, text_content, _ = get_item_data(item, dataset_name)
        if not image_file_name or text_content is None:
            continue

        row = index_to_row.get(idx)
        if row is None:
            continue

        batch_texts.append(text_content)
        batch_imgs.append(img_embeddings[row])
        batch_indices.append(idx)

        if len(batch_texts) >= batch_size:
            flush_batch()

    flush_batch()

    if not align_scores:
        print("No alignment scores generated.")
        return

    out_dir = Path(embedding_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / f"{dataset_name}_{split}_align.npy"
    map_path = out_dir / f"{dataset_name}_{split}_align_index.json"

    np.save(score_path, np.array(align_scores, dtype=np.float32))
    with map_path.open("w", encoding="utf-8") as handle:
        json.dump({"index_map": index_map}, handle)

    print(f"Saved alignment scores to {score_path}")
    print(f"Saved alignment index map to {map_path}")


def _extract_caption(raw_item: Dict[str, Any]) -> Optional[str]:
    for key in _CAPTION_KEYS:
        value = raw_item.get(key)
        if value:
            return str(value)
    return None


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        return {}


def _build_event_prompt(text: str, caption: Optional[str]) -> str:
    cap = caption or "None"
    return (
        "Generate a short event/context background (1-2 sentences) "
        "for the meme based on the text and any image caption. "
        "Be factual and concise. If no clear event, return a brief neutral context.\n\n"
        f"Text:\n{text}\n\n"
        f"Image Caption:\n{cap}\n\n"
        "Return ONLY JSON: {\"event_context\": \"...\"}"
    )


def generate_event_contexts(
    dataset_name: str,
    split: str,
    batch_size: int,
    start: int,
    limit: int,
    sleep_seconds: float,
    output_suffix: str,
    log_every: int,
    max_retries: int,
) -> None:
    base_data_path = f"data/{dataset_name}"
    jsonl_path = f"{base_data_path}/{split}.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Missing file: {jsonl_path}")
        return

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle if line.strip()]
    total = len(data)
    end = min(total, start + limit) if limit > 0 else total
    config = OpenAIModelConfig.from_env()

    out_path = f"{base_data_path}/{split}_{output_suffix}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    processed = 0
    failed = 0
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as output_handle:
        for i, item in enumerate(data):
            if i < start or i >= end:
                json.dump(item, output_handle, ensure_ascii=False)
                output_handle.write("\n")
                continue

            _, text_content, _ = get_item_data(item, dataset_name)
            if not text_content:
                json.dump(item, output_handle, ensure_ascii=False)
                output_handle.write("\n")
                continue

            parts = [{"type": "text", "text": _build_event_prompt(text_content, _extract_caption(item))}]
            event_context = ""
            last_err = None
            for _ in range(max_retries + 1):
                try:
                    payload = _parse_json(get_openai_response_with_parts(parts, config))
                    event_context = payload.get("event_context") or ""
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
            if last_err is not None:
                failed += 1

            item["event_context"] = event_context
            json.dump(item, output_handle, ensure_ascii=False)
            output_handle.write("\n")
            processed += 1

            if (i - start + 1) % batch_size == 0:
                output_handle.flush()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if log_every > 0 and processed % log_every == 0:
                elapsed = time.time() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                print(f"[{dataset_name}/{split}] processed={processed} failed={failed} speed={speed:.2f} item/s")

    print(f"Saved event-context data to {out_path}")


def _load_jsonl(path: os.PathLike | str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_event_map(train_items: List[Dict]) -> Dict[int, str]:
    event_map: Dict[int, str] = {}
    for idx, item in enumerate(train_items):
        event_context = item.get("event_context") or item.get("event_contexts")
        if isinstance(event_context, list):
            event_context = "; ".join(str(value) for value in event_context if value)
        if event_context:
            event_map[idx] = str(event_context)
    return event_map


def merge_event_contexts(dataset_name: str, data_suffix: str, camr_dir: str, out_dir: str) -> None:
    train_path = f"data/{dataset_name}/train_{data_suffix}.jsonl"
    source_path = resolve_existing_camr_path(camr_dir, dataset_name)
    if not os.path.exists(train_path):
        print(f"Missing file: {train_path}")
        return
    if source_path is None:
        print(f"Missing CAMR file for {dataset_name} under {camr_dir}")
        return

    train_items = _load_jsonl(train_path)
    event_map = _build_event_map(train_items)
    os.makedirs(out_dir, exist_ok=True)
    output_path = camr_path(out_dir, dataset_name)

    with open(source_path, "r", encoding="utf-8") as input_handle, open(
        output_path, "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            item = json.loads(line)
            event_contexts = []
            for train_idx in item.get("samples") or []:
                if train_idx in event_map:
                    event_contexts.append(event_map[train_idx])
            item["event_contexts"] = event_contexts
            item.setdefault("camr", {})["event_contexts"] = event_contexts
            json.dump(item, output_handle, ensure_ascii=False)
            output_handle.write("\n")

    print(f"Saved event-augmented CAMR file to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare optional offline inputs used by Tool-meme.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Split for feature extraction.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR)

    parser.add_argument("--image_embeddings", action="store_true", help="Extract CLIP image embeddings.")
    parser.add_argument("--alignment_scores", action="store_true", help="Extract image-text alignment scores.")
    parser.add_argument("--event_contexts", action="store_true", help="Generate train/test event_context fields via API.")
    parser.add_argument("--merge_events", action="store_true", help="Merge train event contexts into CAMR files.")
    parser.add_argument("--all", action="store_true", help="Run all preparation stages in dependency order.")

    parser.add_argument("--event_split", default="train", choices=["train", "test"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--event_output_suffix", default="with_event")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_retries", type=int, default=2)

    parser.add_argument("--data_suffix", default="with_event", help="Suffix for train_{suffix}.jsonl.")
    parser.add_argument("--camr_dir", default=DEFAULT_CAMR_DIR)
    parser.add_argument("--ssr_dir", default=None, help="Deprecated alias for --camr_dir.")
    parser.add_argument("--out_dir", default=DEFAULT_CAMR_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_image = args.all or args.image_embeddings
    run_alignment = args.all or args.alignment_scores
    run_events = args.all or args.event_contexts
    run_merge = args.all or args.merge_events

    if not any((run_image, run_alignment, run_events, run_merge)):
        raise SystemExit(
            "No preparation stage selected. Use --image_embeddings, --alignment_scores, "
            "--event_contexts, --merge_events, or --all."
        )

    for dataset in args.datasets:
        if run_image:
            extract_image_embeddings(dataset, args.split, args.batch_size, args.embedding_dir)
        if run_alignment:
            extract_alignment_scores(dataset, args.split, args.batch_size, args.embedding_dir)
        if run_events:
            generate_event_contexts(
                dataset_name=dataset,
                split=args.event_split,
                batch_size=args.batch_size,
                start=args.start,
                limit=args.limit,
                sleep_seconds=args.sleep,
                output_suffix=args.event_output_suffix,
                log_every=args.log_every,
                max_retries=args.max_retries,
            )
        if run_merge:
            merge_event_contexts(dataset, args.data_suffix, args.ssr_dir or args.camr_dir, args.out_dir)


if __name__ == "__main__":
    main()
