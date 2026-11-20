"""
CASA under a mismatched or estimated channel law.

    python -m sempalab.channel_mismatch --config configs/channel/coco_k12_mismatch.yaml

In docs/channel_aware.md CASA is handed the true joint slot law Q. Here
the aggregator is *designed* with a law Q_design that differs from the
truth, and every partition is *scored* by exact ATS under the true Q.

Design laws (config key `designs`, each a dict with `kind`):
    matched                      the true law (reproduces the main run)
    iid                          i.i.d. at the true p (memory-blind)
    gilbert / rayleigh           that family with `parameter` at the true p
                                 (wrong memory, or wrong family)
    p_scale                      the true family and memory at p * `factor`
    estimate_gilbert             Gilbert channel fitted by maximum
                                 likelihood to `packets` erasures observed
                                 from the true channel
    estimate_empirical           empirical slot-pattern frequencies of
                                 `packets` observed erasures (add-1/2)

Estimated laws use a fresh trace for every (message, M, true point), so
the spread over messages includes the estimation noise.

Algorithms per design: CASA, and swap local search on the RSS sum with
slots ordered for the design law (the cheap channel-aware competitor).
Design-independent references: Random PA, SemPA-Look, swap LS, and the
channel oracle (Full Search over ordered partitions under the true Q).

Rows are tagged algorithm = "<algorithm>@<design label>", so the
channel_experiment summaries apply unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from sempalab.channel.correlated import GilbertElliott, state_matrix
from sempalab.channel_aware import OrderedOracle, best_slot_order, casa, channel_ats, union_masks
from sempalab.channel_experiment import (
    LawBook,
    blind_partitions,
    build_encoder,
    channel_points,
    checkpoint_key,
    load_messages,
    point_seed,
    summarize,
    write_csv,
)
from sempalab.config import ENCODER_REGISTRY, merge
from sempalab.evaluation import SimilarityTable, to_mask
from sempalab.repro import block_unions, count_block_unions


DESIGN_KINDS = {"matched", "iid", "gilbert", "rayleigh", "p_scale", "estimate_gilbert", "estimate_empirical"}

DEFAULTS = {
    "experiment": None,
    "description": "",
    "encoder": "mpnet",
    "device": "cpu",
    "cache_size": 2_000_000,
    "data": {
        "source": "coco",
        "path": "data/coco/annotations/captions_train2017.json",
        "max_articles": None,
        "num_messages": 100,
        "seed": 42,
        "eval_set_dir": "results/stage4/eval_sets",
    },
    "K": 12,
    "M": [2],
    "p": [0.3, 0.5],
    "channels": [{"name": "gilbert", "parameter": 0.6}],
    "designs": [{"kind": "matched"}, {"kind": "iid"}],
    "stride": [1],
    # needed by channel_experiment.blind_partitions
    "algorithms": ["random_pa", "sempa_look", "swap_ls"],
    "full_search": True,
    "P": 10,
    "k": 4,
    "swap_ls": {"restarts": 1},
    "casa": {"budget": None, "restarts": 20, "top": 5, "ridge": 1e-3},
    "smoothing": 0.5,
    "rayleigh_samples": 400_000,
    "packet_bits": 300,
    "prefetch_max_sets": 5000,
    "limit": None,
    "seed": 42,
    "results_dir": "results/channel",
}

REFERENCES = ("casa@matched", "swap_ls+order@matched", "sempa_look", "random_pa")


# -------------------------------------------------------------
# Estimators
# -------------------------------------------------------------


def estimate_gilbert(erased: np.ndarray, smoothing: float = 0.5) -> GilbertElliott:
    """
    Maximum-likelihood Gilbert channel (loss_good = 0, loss_bad = 1, so
    the state is observed) from a 0/1 erasure trace: transition counts
    with `smoothing` pseudo-counts on every transition.
    """

    erased = np.asarray(erased, dtype=bool)

    if len(erased) < 2:
        raise ValueError("need at least two observed packets")

    prev, nxt = erased[:-1], erased[1:]

    gb = np.sum(~prev & nxt) + smoothing
    gg = np.sum(~prev & ~nxt) + smoothing
    bg = np.sum(prev & ~nxt) + smoothing
    bb = np.sum(prev & nxt) + smoothing

    return GilbertElliott(p_gb=gb / (gb + gg), p_bg=bg / (bg + bb))


def empirical_law(erased: np.ndarray, N: int, stride: int = 1, smoothing: float = 0.5) -> np.ndarray:
    """
    Empirical law of the survival pattern of N slots `stride` packets
    apart, from every window of the trace, with add-`smoothing` counts.
    Pattern index s has bit n set when slot n survives (state_matrix).
    """

    erased = np.asarray(erased, dtype=bool)
    span = (N - 1) * stride + 1

    if len(erased) < span:
        raise ValueError(f"trace of {len(erased)} packets is shorter than one window ({span})")

    windows = len(erased) - span + 1
    survive = ~erased
    index = np.zeros(windows, dtype=np.int64)

    for n in range(N):
        index |= survive[n * stride : n * stride + windows].astype(np.int64) << n

    counts = np.bincount(index, minlength=1 << N).astype(float) + smoothing

    return counts / counts.sum()


# -------------------------------------------------------------
# Design laws
# -------------------------------------------------------------


def design_label(design: dict) -> str:
    kind = design["kind"]

    if kind in {"matched", "iid"}:
        return kind
    if kind in {"gilbert", "rayleigh"}:
        return f"{kind}({design['parameter']:g})"
    if kind == "p_scale":
        return f"p_x{design['factor']:g}"
    if kind == "estimate_gilbert":
        return f"est_gilbert(T={design['packets']})"
    if kind == "estimate_empirical":
        return f"est_empirical(T={design['packets']})"

    raise ValueError(f"unknown design kind {kind}")


def design_law(design: dict, point: dict, N: int, stride: int, laws: LawBook, config: dict, rng: np.random.Generator) -> np.ndarray:
    """
    The law Q_design the aggregator is built with, for the true channel
    `point` (dict with channel, parameter, p).
    """

    kind = design["kind"]

    if kind == "matched":
        return laws.law(point, N, stride)

    if kind == "iid":
        return laws.iid(point["p"], N)

    if kind in {"gilbert", "rayleigh"}:
        return laws.law({"channel": kind, "parameter": float(design["parameter"]), "p": point["p"]}, N, stride)

    if kind == "p_scale":
        p = float(np.clip(point["p"] * design["factor"], 0.01, 0.95))
        return laws.law({**point, "p": p}, N, stride)

    if kind in {"estimate_gilbert", "estimate_empirical"}:
        laws.law(point, N, stride)  # builds the channel
        channel = laws.channels[(point["channel"], point["parameter"], point["p"])]
        trace = channel.sample_erasures(int(design["packets"]), rng)

        if kind == "estimate_gilbert":
            return estimate_gilbert(trace, config["smoothing"]).pattern_probabilities(N, stride)
        return empirical_law(trace, N, stride, config["smoothing"])

    raise ValueError(f"unknown design kind {kind}")


# -------------------------------------------------------------
# Config
# -------------------------------------------------------------


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)

    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    config = merge(DEFAULTS, raw)

    if overrides:
        config = merge(config, overrides)

    if config["experiment"] is None:
        config["experiment"] = path.stem

    validate_config(config)

    return config


def validate_config(config: dict) -> None:
    unknown = set(config) - set(DEFAULTS)

    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    if config["encoder"] not in ENCODER_REGISTRY:
        raise ValueError(f"encoder must be one of {sorted(ENCODER_REGISTRY)}")

    for name in ("M", "p", "stride"):
        if isinstance(config[name], (int, float)):
            config[name] = [config[name]]

    for M in config["M"]:
        if config["K"] % M != 0 or M == config["K"]:
            raise ValueError(f"M={M} must divide K={config['K']} and leave N >= 2")

    for spec in config["channels"]:
        if spec.get("name") not in {"iid", "gilbert", "rayleigh"} or set(spec) - {"name", "parameter"}:
            raise ValueError(f"bad channel spec {spec}")

    labels = []

    for design in config["designs"]:
        if design.get("kind") not in DESIGN_KINDS:
            raise ValueError(f"design kind must be one of {sorted(DESIGN_KINDS)}: {design}")
        labels.append(design_label(design))

    if "matched" not in labels:
        raise ValueError("designs must include {kind: matched} (the reference)")

    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate designs: {labels}")

    if config["full_search"]:
        for M in config["M"]:
            if count_block_unions(config["K"], M) > config["prefetch_max_sets"]:
                raise ValueError(f"full search is infeasible at K={config['K']}, M={M}")


# -------------------------------------------------------------
# Runner
# -------------------------------------------------------------


def run_message(message, encoder, config: dict, laws: LawBook) -> list[dict]:
    K = config["K"]
    rows = []

    for M in config["M"]:
        N = K // M
        table = SimilarityTable(encoder, message)

        if count_block_unions(K, M) <= config["prefetch_max_sets"]:
            table.fill(to_mask(subset) for subset in block_unions(K, M))

        seed = config["seed"] * 1_000_003 + message.message_id
        blind = blind_partitions(message, encoder, M, config, seed)
        oracle = OrderedOracle(table, K, M) if config["full_search"] else None

        for point in channel_points(config):
            for stride in config["stride"]:
                if point["channel"] == "iid" and stride != config["stride"][0]:
                    continue

                Q = laws.law(point, N, stride)
                outcomes: dict[str, tuple] = {}

                for name, (partition, calls) in blind.items():
                    outcomes[name] = (partition, calls, {})

                if oracle is not None:
                    partition, _, _ = oracle.best(Q)
                    outcomes["full_search"] = (partition, len(oracle.partitions) * (1 << N), {})

                for design in config["designs"]:
                    label = design_label(design)
                    trace_rng = np.random.default_rng(point_seed(config, message.message_id, M, point["channel"], point["parameter"], point["p"], stride, "trace", label))
                    Q_design = design_law(design, point, N, stride, laws, config, trace_rng)

                    # Same seed as channel_experiment's CASA, so casa@matched
                    # reproduces the main run's partitions exactly.
                    rng = np.random.default_rng(point_seed(config, message.message_id, M, point["channel"], point["parameter"], point["p"], stride))
                    result = casa(table, K, M, Q_design, rng, **config["casa"])
                    tv = 0.5 * float(np.abs(Q_design - Q).sum())
                    outcomes[f"casa@{label}"] = (result.partition, result.logical_calls, {"tv_distance": tv})

                    if "swap_ls" in blind:
                        partition, calls = blind["swap_ls"]
                        ordered, _ = best_slot_order(table, partition, Q_design)
                        extra = len({int(m) for m in union_masks(partition) if m})
                        outcomes[f"swap_ls+order@{label}"] = (ordered, calls + extra, {"tv_distance": tv})

                for name, (partition, calls, extra) in outcomes.items():
                    rows.append({
                        "experiment": config["experiment"],
                        "encoder": config["encoder"],
                        "K": K,
                        "M": M,
                        "N": N,
                        "channel": point["channel"],
                        "parameter": point["parameter"],
                        "stride": stride,
                        "p": point["p"],
                        "algorithm": name,
                        "message_id": message.message_id,
                        "partition": json.dumps([[int(i) for i in g] for g in partition]),
                        "ats": channel_ats(table, partition, Q),
                        "logical_calls": calls,
                        **extra,
                    })

    return rows


def add_oracle_share(summary: list[dict]) -> list[dict]:
    """
    share_of_oracle_gain = (ATS - Random PA) / (Oracle - Random PA) per setting.
    """

    keys = ("K", "M", "channel", "parameter", "stride", "p")
    by_setting: dict[tuple, dict[str, float]] = {}

    for row in summary:
        by_setting.setdefault(tuple(row[k] for k in keys), {})[row["algorithm"]] = row["mean_ats"]

    for row in summary:
        ref = by_setting[tuple(row[k] for k in keys)]
        if "full_search" in ref and "random_pa" in ref and ref["full_search"] > ref["random_pa"]:
            row["share_of_oracle_gain"] = (row["mean_ats"] - ref["random_pa"]) / (ref["full_search"] - ref["random_pa"])

    return summary


def run(config: dict, encoder=None) -> list[dict]:
    if encoder is None:
        encoder = build_encoder(config["encoder"], config)

    started = time.time()
    laws = LawBook(config)
    rows = []
    messages = load_messages(config)

    out_dir = Path(config["results_dir"]) / config["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = out_dir / "_checkpoints" / checkpoint_key(config)
    checkpoints.mkdir(parents=True, exist_ok=True)

    for count, message in enumerate(messages, 1):
        path = checkpoints / f"{message.message_id}.json"

        if path.exists():
            rows.extend(json.loads(path.read_text()))
            continue

        message_rows = run_message(message, encoder, config, laws)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(message_rows))
        tmp.replace(path)
        rows.extend(message_rows)
        print(f"[{config['experiment']}] message {count}/{len(messages)} ({time.time() - started:.0f}s)", flush=True)

    summary = add_oracle_share(summarize(rows, references=REFERENCES))

    write_csv(out_dir / "results.csv.gz", rows)
    write_csv(out_dir / "summary.csv", summary)

    provenance = copy.deepcopy(config)
    provenance["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    provenance["wall_seconds"] = round(time.time() - started, 1)
    provenance["model_name"] = ENCODER_REGISTRY[config["encoder"]]

    with (out_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(provenance, handle, sort_keys=False)

    shutil.rmtree(out_dir / "_checkpoints", ignore_errors=True)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CASA with a mismatched or estimated channel law")
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-dir")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.encoder:
        overrides["encoder"] = args.encoder
    if args.limit:
        overrides["limit"] = args.limit
    if args.results_dir:
        overrides["results_dir"] = args.results_dir

    summary = run(load_config(args.config, overrides))

    for row in summary:
        if row["algorithm"].startswith("casa@") or row["algorithm"] in {"full_search", "random_pa"}:
            share = row.get("share_of_oracle_gain")
            print(
                f"M={row['M']} {row['channel']}({row['parameter']}) p={float(row['p']):.2f} "
                f"{row['algorithm']:>28} ATS={row['mean_ats']:.4f}" + (f" share={share:.2f}" if share is not None else "")
            )


if __name__ == "__main__":
    main()
