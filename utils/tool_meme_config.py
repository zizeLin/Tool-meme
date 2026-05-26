"""Shared configuration for the executable Tool-meme pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_DATASETS = ["FHM", "HarM", "MAMI"]
DEFAULT_K = 10
DEFAULT_TOOL_BUDGET = 6
DEFAULT_SHORT_CIRCUIT_THRESHOLD = 0.80
DEFAULT_MCP_MAX_DEPTH = 4
DEFAULT_ROUTER_TEMPERATURE = 1.0

DEFAULT_CAMR_DIR = "CAMR_output"
DEFAULT_CAMR_OUTPUT_DIR = DEFAULT_CAMR_DIR
DEFAULT_RESULT_DIR = "results"
DEFAULT_EMBEDDING_DIR = "embeddings"
DEFAULT_ROUTER_RUN_ID = "gpt4o_default_seed42"
DEFAULT_ROUTER_DIR = str(Path("rl_atr") / DEFAULT_ROUTER_RUN_ID)
DEFAULT_ROUTER_PATH = str(Path(DEFAULT_ROUTER_DIR) / "checkpoint.pt")

LEGACY_CAMR_DIRS = ["CAMR", "SSR", "SSR_EVENT", "SSR_K3"]
LEGACY_RESULT_DIRS = ["SSR_GPT4o", "SSR_GPT4o_k10"]

CAMR_SUFFIX = "CAMR"
LEGACY_SSR_SUFFIX = "SSR"
RESULT_SUFFIX = "ToolMeme"
LEGACY_RESULT_SUFFIX = "SSR_GPT4o"

TOOL_NAMES = [
    "semantic_dissector",
    "rhetorical_scanner",
    "knowledge_grounding",
    "expectation_deviator",
    "cross_modal_aligner",
    "inconsistency_amplifier",
    "cultural_decoder",
    "evidence_comparator",
]

TOOL_WEIGHTS = {
    "semantic_dissector": 1.0,
    "rhetorical_scanner": 1.0,
    "knowledge_grounding": 1.1,
    "expectation_deviator": 1.0,
    "cross_modal_aligner": 1.1,
    "inconsistency_amplifier": 1.1,
    "cultural_decoder": 1.0,
    "evidence_comparator": 1.0,
}

LABEL_TEXT = {0: "harmless", 1: "harmful"}
LABEL_ID = {"harmless": 0, "harmful": 1}

RETRIEVAL_TEMPERATURE = 0.07
RETRIEVAL_TOP_THRESHOLD = 0.25
RETRIEVAL_DELTA = 0.15
RETRIEVAL_NEIGHBOR_THRESHOLD = 0.20
RETRIEVAL_CONTEXT_THRESHOLD = 0.50

TAU_STOP = 0.80
TAU_CONFLICT = 0.30
TAU_DECISION = 0.70
TAU_CONFIDENCE_SPLIT = 0.30


def label_to_text(label: Optional[int]) -> Optional[str]:
    if label is None:
        return None
    return LABEL_TEXT.get(int(label), "unknown")


def text_to_label(text: str) -> Optional[int]:
    if text is None:
        return None
    return LABEL_ID.get(str(text).strip().lower())


def camr_filename(dataset_name: str, legacy: bool = False) -> str:
    suffix = LEGACY_SSR_SUFFIX if legacy else CAMR_SUFFIX
    return f"{dataset_name}_{suffix}.jsonl"


def result_filename(dataset_name: str, legacy: bool = False) -> str:
    suffix = LEGACY_RESULT_SUFFIX if legacy else RESULT_SUFFIX
    return f"{dataset_name}_{suffix}.jsonl"


def camr_path(camr_dir: str, dataset_name: str, legacy: bool = False) -> Path:
    return Path(camr_dir) / camr_filename(dataset_name, legacy=legacy)


def result_path(output_dir: str, dataset_name: str, legacy: bool = False) -> Path:
    return Path(output_dir) / result_filename(dataset_name, legacy=legacy)


def resolve_existing_camr_path(
    camr_dir: str,
    dataset_name: str,
    extra_legacy_dirs: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    candidates: List[Path] = [
        camr_path(camr_dir, dataset_name, legacy=False),
        camr_path(camr_dir, dataset_name, legacy=True),
    ]
    for legacy_dir in list(extra_legacy_dirs or LEGACY_CAMR_DIRS):
        candidates.append(camr_path(legacy_dir, dataset_name, legacy=False))
        candidates.append(camr_path(legacy_dir, dataset_name, legacy=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
