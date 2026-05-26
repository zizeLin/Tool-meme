import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import copy
from typing import Any, Dict, List, Optional, Tuple

from utils.tool_meme_config import DEFAULT_CAMR_DIR, DEFAULT_DATASETS, DEFAULT_K, camr_path
from utils.data_utils import get_item_data
from utils.camr_utils import calibrate_retrieval, compute_base_rate, extract_event_context


_CLIP_BUNDLE = None


def get_clip_model():
    global _CLIP_BUNDLE
    if _CLIP_BUNDLE is None:
        import torch
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-g-14", pretrained="laion2b_s34b_b88k", device=device
        )
        model.eval()
        _CLIP_BUNDLE = (torch, open_clip, model, preprocess, device)
    return _CLIP_BUNDLE


def _collect_items(
    data: List[Dict[str, Any]],
    dataset_name: str,
    image_base_path: str,
) -> List[Tuple[int, str, str, Optional[int], Dict[str, Any]]]:
    items = []
    for idx, item in enumerate(data):
        image_file_name, text_content, label = get_item_data(item, dataset_name)
        if not image_file_name or not text_content:
            continue
        image_file_path = os.path.join(image_base_path, image_file_name)
        if not os.path.exists(image_file_path):
            continue
        items.append((idx, image_file_path, text_content, label, item))
    return items


def _batch_encode(items: List[Tuple[int, str, str, Optional[int], Dict[str, Any]]], batch_size: int):
    torch, open_clip, model, preprocess, device = get_clip_model()
    embeddings = []
    index_map = []
    for start in tqdm(range(0, len(items), batch_size), desc="Embeddings"):
        batch = items[start:start + batch_size]
        images = []
        texts = []
        valid_indices = []
        for raw_idx, image_path, text_content, _, _ in batch:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            images.append(preprocess(image))
            texts.append(text_content)
            valid_indices.append(raw_idx)

        if not images:
            continue

        image_tensor = torch.stack(images).to(device)
        text_tensor = open_clip.tokenize(texts).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor).detach().cpu().numpy()
            text_features = model.encode_text(text_tensor).detach().cpu().numpy()

        for i, raw_idx in enumerate(valid_indices):
            embedding = (text_features[i] + image_features[i] * 4) / 5
            embeddings.append(embedding)
            index_map.append(raw_idx)

    return np.array(embeddings), index_map


def process_camr(
    dataset_name: str,
    k: int = DEFAULT_K,
    data_suffix: str = "",
    output_dir: str = DEFAULT_CAMR_DIR,
    batch_size: int = 128,
):
    """
    Context-Augmented Multimodal Retrieval with batching:
    - compute CLIP embeddings in batches
    - retrieve Top-K similar train samples
    - output calibrated CAMR fields used by MCP, ATR, MPRE, and CBDF
    """
    print(f"\n--- Processing dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    suffix = f"_{data_suffix}" if data_suffix else ""
    test_jsonl_path = f"{base_data_path}/test{suffix}.jsonl"
    train_jsonl_path = f"{base_data_path}/train{suffix}.jsonl"
    result_path = camr_path(output_dir, dataset_name)

    try:
        test_data = [json.loads(line) for line in open(test_jsonl_path, "r").readlines()]
        train_data = [json.loads(line) for line in open(train_jsonl_path, "r").readlines()]
    except FileNotFoundError as e:
        print(f"Error: Data files not found for {dataset_name}. Missing file: {e.filename}")
        return

    print(f"Loaded {len(test_data)} test items and {len(train_data)} train items for {dataset_name}.")
    base_rate = compute_base_rate(get_item_data(item, dataset_name)[2] for item in train_data)

    test_items = _collect_items(test_data, dataset_name, image_base_path)
    train_items = _collect_items(train_data, dataset_name, image_base_path)

    if not test_items or not train_items:
        print(f"No valid items found for {dataset_name}. Exiting.")
        return

    print(f"Generating embeddings for {dataset_name} test data (batch_size={batch_size})...")
    embeddings_np, test_index_map = _batch_encode(test_items, batch_size)

    print(f"Generating embeddings for {dataset_name} train data (batch_size={batch_size})...")
    ref_embeddings_np, train_index_map = _batch_encode(train_items, batch_size)

    if embeddings_np.size == 0 or ref_embeddings_np.size == 0:
        print(f"No embeddings generated for {dataset_name}. Exiting.")
        return

    print(f"Calculating similarity scores for {dataset_name}...")
    dot_products = np.dot(embeddings_np, ref_embeddings_np.T)
    norms_embeddings = norm(embeddings_np, axis=1, keepdims=True)
    norms_ref_embeddings = norm(ref_embeddings_np, axis=1, keepdims=True).T

    norms_embeddings[norms_embeddings == 0] = 1
    norms_ref_embeddings[norms_ref_embeddings == 0] = 1
    similarity_scores = dot_products / (norms_embeddings * norms_ref_embeddings)
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)
    similarity_scores[similarity_scores >= 1.0] = 0.0

    similarity_scores_copy = copy.deepcopy(similarity_scores)

    results = []
    print(f"Extracting top-{k} similar samples for {dataset_name}...")
    for i in tqdm(range(len(embeddings_np)), desc="Top-K Extraction"):
        samples = []
        scores = []
        current_scores = similarity_scores_copy[i]

        for _ in range(k):
            if np.max(current_scores) <= 0:
                break
            j = int(np.argmax(current_scores))
            samples.append(j)
            scores.append(float(current_scores[j]))
            current_scores[j] = -1

        retrieved_texts = []
        labels = []
        event_contexts = []
        for pos, train_idx in enumerate(samples):
            if train_idx < 0 or train_idx >= len(train_index_map):
                continue
            raw_idx = train_index_map[train_idx]
            raw_item = train_data[raw_idx]
            image_file_name, text_content, label = get_item_data(raw_item, dataset_name)
            if text_content:
                retrieved_texts.append(text_content)
            if label is not None:
                labels.append(label)
            event_context = extract_event_context(raw_item)
            if event_context:
                event_contexts.append(event_context)
        calibration = calibrate_retrieval(labels=labels, scores=scores, base_rate=base_rate)

        results.append({
            "index": test_index_map[i],
            "samples": samples,
            "scores": scores,
            "retrieved_texts": retrieved_texts,
            "labels": labels,
            "event_contexts": event_contexts,
            "label_prior": calibration["label_prior"],
            "raw_label_prior": calibration["raw_label_prior"],
            "retrieval_reliability": calibration["retrieval_reliability"],
            "context_mode": calibration["context_mode"],
            "camr": {
                "retrieval_size": k,
                "base_rate": base_rate,
                **calibration,
            },
        })

    os.makedirs(os.path.dirname(str(result_path)), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        for result_item in results:
            json.dump(result_item, f)
            f.write("\n")
    print(f"Results saved to {result_path}")
    print(f"--- Finished processing {dataset_name} ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Context-Augmented Multimodal Retrieval with batching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--data_suffix", default="",
                        help="Use data/{dataset}/train_{suffix}.jsonl and test_{suffix}.jsonl")
    parser.add_argument("--output_dir", default=DEFAULT_CAMR_DIR,
                        help="Output directory for CAMR jsonl files")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for CLIP encoding (A800 80G default).")
    args = parser.parse_args()

    for dataset in args.datasets:
        process_camr(dataset, k=args.k, data_suffix=args.data_suffix,
                     output_dir=args.output_dir, batch_size=args.batch_size)
