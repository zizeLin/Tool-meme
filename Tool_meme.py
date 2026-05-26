"""End-to-end Tool-meme inference for harmful meme detection."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from termcolor import colored

from ATR import run_atr
from CBDF import run_cbdf
from MCP import run_mcp
from utils.tool_meme_config import (
    DEFAULT_CAMR_DIR,
    DEFAULT_DATASETS,
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_K,
    DEFAULT_MCP_MAX_DEPTH,
    DEFAULT_RESULT_DIR,
    DEFAULT_ROUTER_PATH,
    DEFAULT_ROUTER_TEMPERATURE,
    DEFAULT_SHORT_CIRCUIT_THRESHOLD,
    DEFAULT_TOOL_BUDGET,
    LABEL_TEXT,
    TOOL_NAMES,
    camr_path,
    label_to_text,
    resolve_existing_camr_path,
    result_path,
)
from MPRE import run_mpre
from utils.camr_utils import calibrate_retrieval
from utils.data_utils import get_item_data
from utils.openai_client import OpenAIModelConfig, get_openai_response_with_parts
from utils.prompting import HARMFUL_MEME_RUBRIC


MAX_EVIDENCE_WORDS = 50


def mask_text(text: str) -> str:
    return re.sub(r"[A-Za-z0-9\u4e00-\u9fff]", "*", text)


def mask_text_strict(text: str) -> str:
    return re.sub(r"\S", "*", text)


def normalize_text(text, max_len: int, text_mode: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len] + "..."
    if text_mode == "mask":
        return mask_text(text)
    if text_mode == "mask_strict":
        return mask_text_strict(text)
    if text_mode == "none":
        return "[TEXT REDACTED]"
    return text


def truncate_words(text: Optional[str], limit: int = MAX_EVIDENCE_WORDS) -> str:
    if not text:
        return ""
    words = str(text).strip().split()
    return " ".join(words[:limit])


def parse_prediction(output_text: str) -> Optional[int]:
    text = str(output_text or "").lower()
    if "answer:" in text:
        text = text.split("answer:", 1)[-1]
    if "harmless" in text:
        return 0
    if "harmful" in text:
        return 1
    return None


def load_jsonl(path: os.PathLike | str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_embedding_map(dataset_name: str):
    emb_path = Path(DEFAULT_EMBEDDING_DIR) / f"{dataset_name}_test_img.npy"
    map_path = Path(DEFAULT_EMBEDDING_DIR) / f"{dataset_name}_test_index.json"
    if not emb_path.exists() or not map_path.exists():
        return None, None
    embeddings = np.load(emb_path)
    with open(map_path, "r", encoding="utf-8") as handle:
        index_map = json.load(handle).get("index_map", [])
    return embeddings, {idx: row for row, idx in enumerate(index_map)}


def load_alignment_map(dataset_name: str):
    score_path = Path(DEFAULT_EMBEDDING_DIR) / f"{dataset_name}_test_align.npy"
    map_path = Path(DEFAULT_EMBEDDING_DIR) / f"{dataset_name}_test_align_index.json"
    if not score_path.exists() or not map_path.exists():
        return None, None
    scores = np.load(score_path)
    with open(map_path, "r", encoding="utf-8") as handle:
        index_map = json.load(handle).get("index_map", [])
    return scores, {idx: row for row, idx in enumerate(index_map)}


def filter_items(data: Sequence[Dict], dataset_name: str, image_base_path: str, check_images: bool) -> List[Dict]:
    filtered = []
    for original_index, item in enumerate(data):
        image_file_name, text_content, label = get_item_data(item, dataset_name)
        if not image_file_name or not text_content:
            continue
        image_path = os.path.join(image_base_path, image_file_name)
        if not os.path.exists(image_path):
            continue
        if check_images:
            try:
                with Image.open(image_path) as img:
                    img.convert("RGB")
            except Exception:
                continue
        filtered.append({
            "original_index": original_index,
            "image_path": image_path,
            "text": text_content,
            "label": label,
        })
    return filtered


def build_camr_package(camr_line: Dict, reference_labels: List[int], base_rate: float = 0.5) -> Dict:
    camr = dict(camr_line.get("camr") or {})
    scores = camr_line.get("scores") or []
    if "label_prior" not in camr or "retrieval_reliability" not in camr:
        camr.update(calibrate_retrieval(reference_labels, scores, base_rate=base_rate))
    camr.setdefault("retrieval_size", len(reference_labels))
    camr.setdefault("event_contexts", camr_line.get("event_contexts") or [])
    return camr


def build_camr_context(
    references: List[Dict],
    camr_package: Dict,
    include_labels: bool,
    max_text_len: int,
    text_mode: str,
    event_contexts: Optional[List[str]] = None,
) -> str:
    event_contexts = event_contexts or []
    top_texts = [
        normalize_text(ref["text"], max_text_len, text_mode)
        for ref in references[:2]
    ]
    lines = [
        f"CAMR retrieved {len(references)} similar samples.",
        f"CAMR mode={camr_package.get('context_mode', 'unknown')}, "
        f"reliability={float(camr_package.get('retrieval_reliability', 0.0)):.3f}, "
        f"harmful_prior={float(camr_package.get('label_prior', 0.5)):.3f}.",
    ]
    if include_labels:
        label_texts = [label_to_text(ref.get("label")) for ref in references if ref.get("label") is not None]
        lines.append(f"Retrieved labels: {', '.join(label_texts[:10])}.")
    if event_contexts:
        safe_events = [normalize_text(ctx, max_text_len, text_mode) for ctx in event_contexts if ctx]
        lines.append(f"Event contexts: {'; '.join(safe_events[:3])}")
    if top_texts:
        lines.append(f"Top retrieved texts: {'; '.join(top_texts)}")
    return "\n".join(lines)


def build_prompt_parts(
    target: Dict,
    references: List[Dict],
    include_labels: bool,
    max_text_len: int,
    text_mode: str,
    context_summary: Optional[str] = None,
    evidence_summary: Optional[str] = None,
    abstract_guidance: bool = False,
) -> List[Dict]:
    header = (
        "You are GPT-4o classifying a multimodal meme into the binary label space harmful/harmless. "
        "Use the target image and target text as primary evidence. Use CAMR references only as calibration. "
        "Inspect who or what is being mocked, whether a protected/vulnerable target is degraded, and whether "
        "the image changes the meaning of the text.\n\n"
        f"{HARMFUL_MEME_RUBRIC}\n\n"
        "Do the reasoning internally. Output exactly one line: Answer: harmful OR Answer: harmless."
    )
    if abstract_guidance:
        header += (
            "\nSensitive-content mode: do not repeat sensitive wording. "
            "Reason at a high level based on harm category, target, intent, and visual-textual relation."
        )
    parts = [{"type": "text", "text": header}]
    if context_summary:
        parts.append({"type": "text", "text": f"\nCAMR Context:\n{context_summary}\n"})
    if evidence_summary:
        parts.append({"type": "text", "text": f"\nTool-meme Evidence:\n{evidence_summary}\n"})

    for idx, ref in enumerate(references, start=1):
        label_line = f"Label: {label_to_text(ref['label'])}\n" if include_labels else ""
        parts.append({
            "type": "text",
            "text": (
                f"\nReference {idx}\n"
                f"Text: {normalize_text(ref['text'], max_text_len, text_mode)}\n"
                f"{label_line}Image:"
            ),
        })
        parts.append({"type": "image", "path": ref["image_path"]})

    parts.append({
        "type": "text",
        "text": f"\nTarget\nText: {normalize_text(target['text'], max_text_len, text_mode)}\nImage:",
    })
    parts.append({"type": "image", "path": target["image_path"]})
    parts.append({"type": "text", "text": (
        "\nFinal decision: prioritize target-specific visual-textual evidence over retrieved-label majority. "
        "Answer format: Answer: harmful OR Answer: harmless."
    )})
    return parts


def is_content_filter_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "content_filter" in message or "content management policy" in message


def disabled_tools_from_args(args: Dict) -> Set[str]:
    disabled = set()
    for tool in TOOL_NAMES:
        flag = f"no_{tool}"
        if args.get(flag):
            disabled.add(tool)
    return disabled


def _processed_result_state(result_file: Path) -> Tuple[Set[int], List[int], List[int], List[int]]:
    processed_indices: Set[int] = set()
    ratio = [0, 0]
    actual_labels: List[int] = []
    predicted_labels: List[int] = []
    if not result_file.exists():
        return processed_indices, ratio, actual_labels, predicted_labels
    for line in result_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "accuracy" in item and "ratio" in item:
            raise RuntimeError(f"{result_file} already contains a final summary. Delete it to re-run.")
        if "index" in item and "actual" in item and "predict" in item:
            processed_indices.add(int(item["index"]))
            actual = item.get("actual")
            predict = item.get("predict")
            if actual is None or predict is None:
                continue
            ratio[0] += 1
            if actual == predict:
                ratio[1] += 1
            actual_labels.append(actual)
            predicted_labels.append(predict)
    return processed_indices, ratio, actual_labels, predicted_labels


def _load_or_create_camr(dataset_name: str, camr_dir: str, k: int, no_camr: bool) -> Path:
    existing = resolve_existing_camr_path(camr_dir, dataset_name)
    if existing:
        return existing
    if no_camr:
        raise FileNotFoundError(f"Missing CAMR file for {dataset_name} under {camr_dir}.")
    from CAMR import process_camr

    process_camr(dataset_name, k=k, output_dir=camr_dir)
    generated = camr_path(camr_dir, dataset_name)
    if not generated.exists():
        raise FileNotFoundError(f"CAMR generation did not create {generated}.")
    return generated


def _pick_target(test_by_original: Dict[int, Dict], test_filtered: List[Dict], target_idx: int) -> Optional[Dict]:
    if target_idx in test_by_original:
        return test_by_original[target_idx]
    if 0 <= target_idx < len(test_filtered):
        return test_filtered[target_idx]
    return None


def _pick_references(
    camr_line: Dict,
    train_filtered: List[Dict],
    k: int,
) -> List[Dict]:
    sample_indices = camr_line.get("samples") or camr_line.get("example") or []
    sample_scores = camr_line.get("scores") or []
    references = []
    for pos, sample_idx in enumerate(sample_indices):
        if len(references) >= k:
            break
        if not isinstance(sample_idx, int) or sample_idx < 0 or sample_idx >= len(train_filtered):
            continue
        ref = train_filtered[sample_idx]
        if ref.get("label") is None:
            continue
        references.append({
            "original_index": ref["original_index"],
            "image_path": ref["image_path"],
            "text": ref["text"],
            "label": ref["label"],
            "score": sample_scores[pos] if pos < len(sample_scores) else None,
        })
    return references


def _run_final_lmm_with_fallbacks(
    target: Dict,
    references: List[Dict],
    include_labels: bool,
    max_text_len: int,
    text_mode: str,
    filter_fallback: str,
    context_summary: str,
    evidence_summary: str,
    config: OpenAIModelConfig,
) -> Tuple[str, str]:
    modes = [text_mode]
    if filter_fallback == "mask" and "mask" not in modes:
        modes.append("mask")
    if filter_fallback == "drop_text" and "none" not in modes:
        modes.append("none")
    if "mask_strict" not in modes:
        modes.append("mask_strict")

    last_error: Optional[Exception] = None
    for mode in modes:
        try:
            parts = build_prompt_parts(
                target=target,
                references=references,
                include_labels=include_labels,
                max_text_len=max_text_len,
                text_mode=mode,
                context_summary=context_summary,
                evidence_summary=evidence_summary,
                abstract_guidance=(mode == "mask_strict"),
            )
            return get_openai_response_with_parts(parts, config), mode
        except Exception as exc:
            last_error = exc
            if not is_content_filter_error(exc):
                raise
    raise RuntimeError(f"OpenAI call failed after fallbacks: {last_error}")


def process_tool_meme_inference(
    dataset_name: str,
    k: int,
    include_labels: bool,
    max_text_len: int,
    check_images: bool,
    sleep_seconds: float,
    text_mode: str,
    filter_fallback: str,
    camr_dir: str = DEFAULT_CAMR_DIR,
    output_dir: str = DEFAULT_RESULT_DIR,
    no_camr: bool = False,
    no_mcp: bool = False,
    no_atr: bool = False,
    no_mpre: bool = False,
    no_cbdf: bool = False,
    selected_indices: Optional[Set[int]] = None,
    disabled_tools: Optional[Set[str]] = None,
    atr_mode: str = "dag",
    router_path: Optional[str] = None,
    router_temperature: float = DEFAULT_ROUTER_TEMPERATURE,
    router_max_steps: Optional[int] = None,
    router_sample: bool = False,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    short_circuit_threshold: float = DEFAULT_SHORT_CIRCUIT_THRESHOLD,
    mcp_max_depth: int = DEFAULT_MCP_MAX_DEPTH,
    decision_source: str = "cbdf",
    cbdf_use_lmm: bool = False,
) -> None:
    print(f"\n--- Starting Tool-meme inference for dataset: {dataset_name} ---")
    base_data_path = Path("data") / dataset_name
    image_base_path = str(base_data_path / "images")
    test_jsonl_path = base_data_path / "test.jsonl"
    train_jsonl_path = base_data_path / "train.jsonl"
    output_file = result_path(output_dir, dataset_name)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        test_data = load_jsonl(test_jsonl_path)
        train_data = load_jsonl(train_jsonl_path)
        camr_file = _load_or_create_camr(dataset_name, camr_dir, k, no_camr)
        camr_lines = load_jsonl(camr_file)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(colored(f"Error: {exc}. Skipping {dataset_name}.", "red"))
        return

    print(f"Loaded {len(test_data)} test items, {len(train_data)} train items, and {len(camr_lines)} CAMR entries.")
    print(f"Using CAMR file: {camr_file}")

    test_filtered = filter_items(test_data, dataset_name, image_base_path, check_images)
    train_filtered = filter_items(train_data, dataset_name, image_base_path, check_images)
    test_by_original = {item["original_index"]: item for item in test_filtered}
    print(f"Filtered to {len(test_filtered)} test items and {len(train_filtered)} train items.")

    embeddings, index_to_row = load_embedding_map(dataset_name)
    align_scores, align_index_to_row = load_alignment_map(dataset_name)
    if embeddings is None:
        print(colored(f"Warning: missing image embeddings for {dataset_name}; ATR receives no image embedding.", "yellow"))
    if align_scores is None:
        print(colored(f"Warning: missing alignment scores for {dataset_name}; cross-modal tools use CAMR hints.", "yellow"))

    try:
        processed_indices, ratio, actual_labels, predicted_labels = _processed_result_state(output_file)
    except RuntimeError as exc:
        print(colored(str(exc), "green"))
        return
    if processed_indices:
        print(colored(f"Continuing from {len(processed_indices)} processed items.", "cyan"))

    openai_config = OpenAIModelConfig.from_env()
    print(f"Using OpenAI model: {openai_config.model}")
    disabled_tools = disabled_tools or set()

    with output_file.open("a", encoding="utf-8") as handle:
        for row_idx, camr_line in enumerate(camr_lines):
            target_idx = int(camr_line.get("index", row_idx))
            if selected_indices is not None and target_idx not in selected_indices:
                continue
            if target_idx in processed_indices:
                continue

            target = _pick_target(test_by_original, test_filtered, target_idx)
            if target is None or target.get("label") is None:
                print(colored(f"Warning: target index {target_idx} is unavailable. Skipping.", "yellow"))
                continue

            references = _pick_references(camr_line, train_filtered, k)
            if not references:
                print(colored(f"Warning: no valid CAMR references for index {target_idx}. Skipping.", "yellow"))
                continue

            image_embedding = None
            if embeddings is not None and index_to_row is not None and target_idx in index_to_row:
                image_embedding = embeddings[index_to_row[target_idx]]
            alignment_score = None
            if align_scores is not None and align_index_to_row is not None and target_idx in align_index_to_row:
                alignment_score = float(align_scores[align_index_to_row[target_idx]])

            reference_labels = [ref["label"] for ref in references]
            camr_package = build_camr_package(camr_line, reference_labels)
            event_contexts = camr_line.get("event_contexts") or []
            retrieved_context = {
                "retrieved_texts": [ref["text"] for ref in references],
                "labels": reference_labels,
                "scores": [ref.get("score") for ref in references if ref.get("score") is not None],
                "event_contexts": event_contexts,
                "alignment_score": alignment_score,
                "camr": camr_package,
            }
            context_summary = build_camr_context(
                references=references,
                camr_package=camr_package,
                include_labels=include_labels,
                max_text_len=max_text_len,
                text_mode=text_mode,
                event_contexts=event_contexts,
            )

            mcp_dag = None
            atr_output = None
            mpre_output = None
            cbdf_output = None
            output = ""
            prompt_text_mode = text_mode

            try:
                if not no_mcp:
                    mcp_dag = run_mcp(
                        target_text=target["text"],
                        retrieved_texts=[ref["text"] for ref in references],
                        labels=reference_labels,
                        event_contexts=event_contexts,
                        config=openai_config,
                        camr_package=camr_package,
                        max_depth=mcp_max_depth,
                    )
                if not no_atr and mcp_dag:
                    atr_output = run_atr(
                        task_dag=mcp_dag,
                        query_text=target["text"],
                        image_embedding=image_embedding,
                        retrieved_context=retrieved_context,
                        context_summary=context_summary,
                        short_circuit_threshold=short_circuit_threshold,
                        disabled_tools=disabled_tools,
                        routing_mode=atr_mode,
                        router_path=router_path,
                        router_temperature=router_temperature,
                        router_max_steps=router_max_steps,
                        router_greedy=not router_sample,
                        tool_budget=tool_budget,
                    )
                if not no_mpre and atr_output:
                    mpre_output = run_mpre(atr_output, openai_config, camr_package=camr_package)
                if not no_cbdf and mpre_output:
                    cbdf_output = run_cbdf(
                        mpre_output,
                        labels=reference_labels,
                        config=openai_config,
                        camr_package=camr_package,
                        atr_output=atr_output,
                        use_lmm=cbdf_use_lmm,
                    )

                if decision_source == "cbdf" and cbdf_output:
                    predict_val = int(cbdf_output["label_id"])
                    output = f"Answer: {cbdf_output['label']}"
                else:
                    evidence_summary = ""
                    if cbdf_output and cbdf_output.get("short_summary"):
                        evidence_summary = cbdf_output["short_summary"]
                    elif mpre_output and mpre_output.get("short_summary"):
                        evidence_summary = mpre_output["short_summary"]
                    output, prompt_text_mode = _run_final_lmm_with_fallbacks(
                        target=target,
                        references=references,
                        include_labels=include_labels,
                        max_text_len=max_text_len,
                        text_mode=text_mode,
                        filter_fallback=filter_fallback,
                        context_summary=context_summary,
                        evidence_summary=truncate_words(evidence_summary),
                        config=openai_config,
                    )
                    predict_val = parse_prediction(output)
                    if predict_val is None and cbdf_output:
                        predict_val = int(cbdf_output["label_id"])
                    if predict_val is None:
                        predict_val = 1 if float(camr_package.get("label_prior", 0.5)) >= 0.5 else 0
            except Exception as exc:
                print(colored(f"Error at index {target_idx}: {exc}", "red"))
                continue

            actual = int(target["label"])
            ratio[0] += 1
            if predict_val == actual:
                ratio[1] += 1
            actual_labels.append(actual)
            predicted_labels.append(int(predict_val))

            result = {
                "index": target_idx,
                "original_index": target["original_index"],
                "ratio": list(ratio),
                "actual": actual,
                "actual_label": LABEL_TEXT[actual],
                "predict": int(predict_val),
                "predict_label": LABEL_TEXT[int(predict_val)],
                "text": target["text"],
                "reference_indices": [ref["original_index"] for ref in references],
                "camr_path": str(camr_file),
                "camr_package": camr_package,
                "mcp_dag": mcp_dag,
                "atr_output": atr_output,
                "mpre_output": mpre_output,
                "cbdf_output": cbdf_output,
                "decision_source": decision_source,
                "prompt_text_mode": prompt_text_mode,
                "output": output,
            }
            json.dump(result, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            print(
                f"Index {target_idx} | Actual: {LABEL_TEXT[actual]} "
                f"Predict: {LABEL_TEXT[int(predict_val)]} Ratio: {ratio}"
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        accuracy = ratio[1] / ratio[0] if ratio[0] else 0.0
        macro_f1 = f1_score(actual_labels, predicted_labels, average="macro") if actual_labels else 0.0
        summary = {
            "ratio": ratio,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "label_space": {"0": "harmless", "1": "harmful"},
            "config": {
                "k": k,
                "camr_dir": camr_dir,
                "atr_mode": atr_mode,
                "tool_budget": tool_budget,
                "short_circuit_threshold": short_circuit_threshold,
                "mcp_max_depth": mcp_max_depth,
                "decision_source": decision_source,
            },
        }
        json.dump(summary, handle, ensure_ascii=False)
        handle.write("\n")

    print(f"\n--- Finished Tool-meme inference for dataset: {dataset_name} ---")
    print(f"Final Accuracy for {dataset_name}: {accuracy:.4f} ({ratio[1]}/{ratio[0]})")
    print(f"Final Macro F1 for {dataset_name}: {macro_f1:.4f}")


def process_ssr_gpt4o(**kwargs) -> None:
    if "ssr_dir" in kwargs and "camr_dir" not in kwargs:
        kwargs["camr_dir"] = kwargs.pop("ssr_dir")
    router_map = {
        "tool_router_mode": "atr_mode",
        "tool_router_path": "router_path",
        "tool_router_temperature": "router_temperature",
        "tool_router_max_steps": "router_max_steps",
        "tool_router_sample": "router_sample",
    }
    for old_key, new_key in router_map.items():
        if old_key in kwargs and new_key not in kwargs:
            kwargs[new_key] = kwargs.pop(old_key)
    disabled_tools = set(kwargs.pop("disabled_tools", set()) or set())
    for tool in TOOL_NAMES:
        flag = f"no_{tool}"
        if kwargs.pop(flag, False):
            disabled_tools.add(tool)
    kwargs["disabled_tools"] = disabled_tools
    return process_tool_meme_inference(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool-meme harmful meme detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of CAMR neighbors.")
    parser.add_argument("--camr_dir", default=DEFAULT_CAMR_DIR, help="Directory containing CAMR jsonl files.")
    parser.add_argument("--ssr_dir", default=None, help="Deprecated alias for --camr_dir.")
    parser.add_argument("--output_dir", default=DEFAULT_RESULT_DIR, help="Directory for Tool-meme result jsonl files.")
    parser.add_argument("--no_camr", action="store_true", help="Require an existing CAMR file.")
    parser.add_argument("--no_labels", dest="include_labels", action="store_false")
    parser.add_argument("--max_text_len", type=int, default=0)
    parser.add_argument("--text_mode", choices=["full", "mask", "none"], default="full")
    parser.add_argument("--filter_fallback", choices=["none", "mask", "drop_text"], default="mask")
    parser.add_argument("--no_check_images", dest="check_images", action="store_false")

    parser.add_argument("--no_mcp", action="store_true")
    parser.add_argument("--no_atr", action="store_true")
    parser.add_argument("--no_mpre", action="store_true")
    parser.add_argument("--no_cbdf", action="store_true")
    for tool in TOOL_NAMES:
        parser.add_argument(f"--no_{tool}", action="store_true", help=f"Disable {tool}.")

    parser.add_argument(
        "--atr_mode",
        "--tool_router_mode",
        dest="atr_mode",
        choices=["dag", "rl", "heuristic", "random", "all_tools"],
        default="dag",
        help="ATR routing strategy.",
    )
    parser.add_argument("--router_path", "--tool_router_path", dest="router_path", default=DEFAULT_ROUTER_PATH)
    parser.add_argument("--router_temperature", "--tool_router_temperature", dest="router_temperature", type=float, default=DEFAULT_ROUTER_TEMPERATURE)
    parser.add_argument("--router_max_steps", "--tool_router_max_steps", dest="router_max_steps", type=int, default=None)
    parser.add_argument("--router_sample", "--tool_router_sample", dest="router_sample", action="store_true")
    parser.add_argument("--tool_budget", type=int, default=DEFAULT_TOOL_BUDGET)
    parser.add_argument("--short_circuit_threshold", type=float, default=DEFAULT_SHORT_CIRCUIT_THRESHOLD)
    parser.add_argument("--mcp_max_depth", type=int, default=DEFAULT_MCP_MAX_DEPTH)
    parser.add_argument("--decision_source", choices=["lmm", "cbdf"], default="cbdf")
    parser.add_argument("--cbdf_use_lmm", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.set_defaults(include_labels=True, check_images=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camr_dir = args.ssr_dir or args.camr_dir
    disabled_tools = disabled_tools_from_args(vars(args))
    for dataset in args.datasets:
        process_tool_meme_inference(
            dataset_name=dataset,
            k=args.k,
            include_labels=args.include_labels,
            max_text_len=args.max_text_len,
            check_images=args.check_images,
            sleep_seconds=args.sleep,
            text_mode=args.text_mode,
            filter_fallback=args.filter_fallback,
            camr_dir=camr_dir,
            output_dir=args.output_dir,
            no_camr=args.no_camr,
            no_mcp=args.no_mcp,
            no_atr=args.no_atr,
            no_mpre=args.no_mpre,
            no_cbdf=args.no_cbdf,
            disabled_tools=disabled_tools,
            atr_mode=args.atr_mode,
            router_path=args.router_path,
            router_temperature=args.router_temperature,
            router_max_steps=args.router_max_steps,
            router_sample=args.router_sample,
            tool_budget=args.tool_budget,
            short_circuit_threshold=args.short_circuit_threshold,
            mcp_max_depth=args.mcp_max_depth,
            decision_source=args.decision_source,
            cbdf_use_lmm=args.cbdf_use_lmm,
        )
    print("\nAll datasets processed for Tool-meme.")


if __name__ == "__main__":
    main()
