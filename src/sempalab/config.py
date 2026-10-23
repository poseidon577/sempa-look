from __future__ import annotations

import copy
from pathlib import Path

import yaml


ENCODER_REGISTRY = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    # Re-scoring encoder for the channel-aware results (a different model family).
    "bge-small": "BAAI/bge-small-en-v1.5",
}

ALGORITHMS = {
    "full_search",
    "random_pa",
    "no_packetization",
    "greedy_tss",
    "greedy_rss",
    "sempa_look",
    "genetic",
}

DATASETS = {"inline", "text_file"}

MODES = {"paper_faithful", "research"}

DEFAULTS = {
    "experiment": None,
    "dataset": "inline",
    "data_path": None,
    "messages": [],
    "num_messages": None,
    "tokenization": "word",
    "K": 8,
    "M": 4,
    "B": 100,
    "P": 10,
    "k": 4,
    "snr_db": [2, 4, 6, 8, 10],
    "packet_bits": 300,
    "dictionary_size": 200_000,
    "encoder": "minilm",
    "device": None,
    "objective": "ats",
    "algorithms": ["sempa_look"],
    "trials": 1000,
    "bit_level": False,
    "genetic": {},
    "mode": "paper_faithful",
    "seed": 42,
    "results_dir": "results",
}


def merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` onto a copy of `base`.
    """

    merged = copy.deepcopy(base)

    for key, value in override.items():

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def load_config(path: str | Path) -> dict:
    """
    Load an experiment YAML file.

    Resolution order (later wins):
        DEFAULTS -> file named by `base:` (relative to this file)
        -> this file.
    """

    path = Path(path)

    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a mapping")

    config = DEFAULTS

    base = raw.pop("base", None)

    if base is not None:
        base_path = path.parent / base

        with base_path.open() as handle:
            base_raw = yaml.safe_load(handle) or {}

        base_raw.pop("base", None)

        config = merge(config, base_raw)

    config = merge(config, raw)

    if config["experiment"] is None:
        config["experiment"] = path.stem

    validate_config(config)

    return config


def validate_config(config: dict) -> None:
    """
    Raise ValueError on any inconsistent setting.
    """

    unknown = set(config) - set(DEFAULTS)

    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    K, M, B = config["K"], config["M"], config["B"]

    for name in ("K", "M", "B", "P", "packet_bits", "trials"):
        value = config[name]

        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(config["k"], int) or config["k"] < 0:
        raise ValueError("k must be a non-negative integer")

    if K % M != 0:
        raise ValueError(f"M={M} must divide K={K}")

    if config["k"] > K // M - 1:
        raise ValueError(
            f"k={config['k']} exceeds k_max = N - 1 = {K // M - 1}"
        )

    if config["dataset"] not in DATASETS:
        raise ValueError(
            f"dataset must be one of {sorted(DATASETS)} "
            "(MS-COCO loading is Stage 4)"
        )

    if config["dataset"] == "text_file" and not config["data_path"]:
        raise ValueError("dataset=text_file requires data_path")

    if config["tokenization"] != "word":
        raise ValueError("Only word tokenization is implemented")

    if config["encoder"] not in ENCODER_REGISTRY:
        raise ValueError(
            f"encoder must be one of {sorted(ENCODER_REGISTRY)}"
        )

    if config["objective"] not in {"ats", "wts"}:
        raise ValueError("objective must be 'ats' or 'wts'")

    if config["mode"] not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")

    algorithms = config["algorithms"]

    if not algorithms or set(algorithms) - ALGORITHMS:
        raise ValueError(
            f"algorithms must be a non-empty subset of {sorted(ALGORITHMS)}"
        )

    snr_db = config["snr_db"]

    if isinstance(snr_db, (int, float)):
        config["snr_db"] = [snr_db]
    elif not snr_db:
        raise ValueError("snr_db must not be empty")
