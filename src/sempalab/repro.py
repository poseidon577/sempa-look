"""
Stage 4 reproduction runner (MS-COCO captions).

    python -m sempalab.repro --config configs/repro/fig6.yaml

One config describes one paper figure or table: a message length K,
subpacket size M, SemPA-Look parameters (P, k), an SNR list, a list of
algorithms, and optionally a sweep over one parameter (M, k, P or K).

For every sweep point, message, algorithm and SNR the runner records
exact ATS, WTS and the algorithm's own logical encoder calls.

Encoder cost accounting: before any algorithm runs on a message, the
embeddings it can possibly need (every union of blocks, when that set
is small) are computed in large batches and cached. Logical calls, the
paper's complexity metric, are unaffected; miss counts and runtimes of
individual algorithms are therefore not meaningful in these outputs.

Outputs go to <results_dir>/<experiment>/:
    results.csv, summary.csv, config.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import yaml

from sempalab.channel.awgn_bpsk import packet_error_probability
from sempalab.config import ENCODER_REGISTRY, merge
from sempalab.datasets import coco_eval_set
from sempalab.metrics import exact_ats, wts
from sempalab.partition import sample_random_partition


ALGORITHMS = (
    "full_search",
    "random_pa",
    "no_packetization",
    "greedy_tss",
    "greedy_rss",
    "sempa_look",
    "genetic",
)

# Partition does not depend on the channel: computed once per message.
P_INDEPENDENT = {"random_pa", "no_packetization", "greedy_tss", "greedy_rss", "sempa_look"}

SWEEPABLE = {"M", "k", "P", "K"}

DEFAULTS = {
    "experiment": None,
    "description": "",
    "encoder": "minilm",
    "device": None,
    "coco": {
        "annotations": "data/coco/annotations/captions_train2017.json",
        "num_messages": 100,
        "seed": 42,
        "eval_set_dir": "results/stage4/eval_sets",
    },
    "K": 8,
    "M": 4,
    "P": 10,
    "k": 4,
    "packet_bits": 300,
    "snr_db": [2, 4, 6, 8, 10],
    "algorithms": ["random_pa", "sempa_look"],
    "genetic": {"P": 10, "G": 5},
    "sweep": None,
    "metrics": ["ats", "wts"],
    "prefetch_max_sets": 5000,
    "seed": 42,
    "results_dir": "results/stage4",
}


# -------------------------------------------------------------
# Config
# -------------------------------------------------------------


def load_repro_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)

    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    config = merge(DEFAULTS, raw)

    if overrides:
        config = merge(config, overrides)

    if config["experiment"] is None:
        config["experiment"] = path.stem

    validate_repro_config(config)

    return config


def validate_repro_config(config: dict) -> None:
    unknown = set(config) - set(DEFAULTS)

    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")

    if config["encoder"] not in ENCODER_REGISTRY:
        raise ValueError(f"encoder must be one of {sorted(ENCODER_REGISTRY)}")

    algorithms = config["algorithms"]

    if not algorithms or set(algorithms) - set(ALGORITHMS):
        raise ValueError(f"algorithms must be a non-empty subset of {ALGORITHMS}")

    if set(config["metrics"]) - {"ats", "wts"}:
        raise ValueError("metrics must be a subset of ['ats', 'wts']")

    if isinstance(config["snr_db"], (int, float)):
        config["snr_db"] = [config["snr_db"]]

    sweep = config["sweep"]

    if sweep is not None:
        if set(sweep) != {"param", "values"} or sweep["param"] not in SWEEPABLE:
            raise ValueError(
                f"sweep must be {{param: one of {sorted(SWEEPABLE)}, values: [...]}}"
            )

        if not sweep["values"]:
            raise ValueError("sweep values must not be empty")

    for point in sweep_points(config):
        K, M = point["K"], point["M"]

        if K % M != 0:
            raise ValueError(f"M={M} must divide K={K}")

        if point["P"] <= 0 or point["k"] < 0:
            raise ValueError("P must be positive and k non-negative")


def sweep_points(config: dict) -> list[dict]:
    """
    The (K, M, P, k) settings this config visits, in order.
    """

    base = {name: config[name] for name in ("K", "M", "P", "k")}

    sweep = config["sweep"]

    if sweep is None:
        return [base]

    return [{**base, sweep["param"]: value} for value in sweep["values"]]


# -------------------------------------------------------------
# Embedding prefetch
# -------------------------------------------------------------


def union_states(partition) -> list[tuple[int, ...]]:
    """
    Surviving index set for every one of the 2^N erasure patterns.
    """

    N = len(partition)

    return [
        tuple(
            sorted(
                index
                for n, group in enumerate(partition)
                if mask & (1 << n)
                for index in group
            )
        )
        for mask in range(1 << N)
    ]


def count_block_unions(K: int, M: int) -> int:
    """
    Number of index subsets whose size is a multiple of M; these are
    exactly the subsets that can survive under some partition.
    """

    return sum(comb(K, size) for size in range(0, K + 1, M))


def block_unions(K: int, M: int):
    for size in range(0, K + 1, M):
        yield from combinations(range(K), size)


def prefetch(encoder, message, index_sets, chunk: int = 4096) -> None:
    """
    Encode, in large batches, every set not already cached.
    """

    encoder.embed_original(message.message_id, message.tokens)

    sets = list(dict.fromkeys(tuple(sorted(s)) for s in index_sets))

    for start in range(0, len(sets), chunk):
        encoder.embed_sets(sets[start:start + chunk], message.message_id, message.tokens)


# -------------------------------------------------------------
# Algorithms
# -------------------------------------------------------------


def packetize(name, message, encoder, point, config, p, seed):
    """
    Returns (partition, logical_calls, partitions_evaluated, runtime).
    """

    M = point["M"]
    K = len(message.tokens)

    if name == "random_pa":
        start = time.perf_counter()
        partition = sample_random_partition(K, M, np.random.default_rng(seed))
        return partition, 0, 1, time.perf_counter() - start

    if name == "no_packetization":
        return (tuple(range(K)),), 0, 1, 0.0

    if name == "full_search":
        from sempalab.algorithms.full_search import full_search

        result = full_search(message, encoder, M, objective="ats", p=p)

    elif name == "greedy_tss":
        from sempalab.algorithms.greedy_tss import greedy_tss

        result = greedy_tss(message, encoder, M)

    elif name == "greedy_rss":
        from sempalab.algorithms.greedy_rss import greedy_rss

        result = greedy_rss(message, encoder, M)

    elif name == "sempa_look":
        from sempalab.algorithms.sempa_look import sempa_look

        result = sempa_look(message, encoder, M, P=point["P"], k=point["k"], seed=seed)

    elif name == "genetic":
        from sempalab.algorithms.genetic import genetic

        result = genetic(
            message,
            encoder,
            M,
            objective="ats",
            p=p,
            seed=seed,
            **config["genetic"],
        )

    else:
        raise ValueError(f"Unknown algorithm: {name}")

    return (
        result.best_partition,
        result.logical_calls,
        result.partitions_evaluated,
        result.runtime_seconds,
    )


# -------------------------------------------------------------
# Runner
# -------------------------------------------------------------


def build_encoder(config: dict):
    from sempalab.encoder import InstrumentedEncoder

    return InstrumentedEncoder(
        model_name=ENCODER_REGISTRY[config["encoder"]],
        device=config["device"],
    )


def load_messages(config: dict, K: int):
    coco = config["coco"]

    return coco_eval_set(
        coco["annotations"],
        K=K,
        num_messages=coco["num_messages"],
        seed=coco["seed"],
        cache_dir=coco["eval_set_dir"],
    )


def run_point(point, messages, encoder, config, sweep_value=None) -> list[dict]:
    K, M = point["K"], point["M"]

    snrs = list(config["snr_db"])
    ps = [packet_error_probability(snr, config["packet_bits"]) for snr in snrs]

    sweep_param = config["sweep"]["param"] if config["sweep"] else None

    do_prefetch = count_block_unions(K, M) <= config["prefetch_max_sets"]

    rows = []

    for message in messages:

        if do_prefetch:
            prefetch(encoder, message, block_unions(K, M))

        seed = config["seed"] * 1_000_003 + message.message_id

        for name in config["algorithms"]:

            if name == "full_search" and not do_prefetch:
                raise ValueError(f"full_search is infeasible at K={K}, M={M}")

            cached = None

            for snr, p in zip(snrs, ps):

                if name in P_INDEPENDENT and cached is not None:
                    outcome = cached
                else:
                    outcome = packetize(name, message, encoder, point, config, p, seed)
                    cached = outcome

                partition, logical_calls, evaluated, runtime = outcome

                if not do_prefetch:
                    prefetch(encoder, message, union_states(partition))

                row = {
                    "experiment": config["experiment"],
                    "encoder": config["encoder"],
                    "sweep_param": sweep_param,
                    "sweep_value": sweep_value,
                    "K": K,
                    "M": M,
                    "N": K // M,
                    "P": point["P"],
                    "k": point["k"],
                    "algorithm": name,
                    "snr_db": snr,
                    "p": p,
                    "message_id": message.message_id,
                    "text": message.text,
                    "partition": json.dumps([[int(i) for i in g] for g in partition]),
                    "logical_calls": logical_calls,
                    "partitions_evaluated": evaluated,
                    "runtime_seconds": runtime,
                }

                if "ats" in config["metrics"]:
                    row["ats"] = exact_ats(partition, message, encoder, p)

                if "wts" in config["metrics"]:
                    row["wts"] = wts(partition, message, encoder, p)

                rows.append(row)

    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}

    for row in rows:
        key = (row["sweep_value"], row["K"], row["M"], row["P"], row["k"], row["algorithm"], row["snr_db"])
        groups.setdefault(key, []).append(row)

    table = []

    for (sweep_value, K, M, P, k, name, snr), group in groups.items():

        entry = {
            "experiment": group[0]["experiment"],
            "encoder": group[0]["encoder"],
            "sweep_param": group[0]["sweep_param"],
            "sweep_value": sweep_value,
            "K": K,
            "M": M,
            "P": P,
            "k": k,
            "algorithm": name,
            "snr_db": snr,
            "p": group[0]["p"],
            "messages": len(group),
            "mean_logical_calls": float(np.mean([r["logical_calls"] for r in group])),
        }

        for metric in ("ats", "wts"):
            if metric in group[0]:
                values = np.array([r[metric] for r in group])
                entry[f"mean_{metric}"] = float(values.mean())
                entry[f"sem_{metric}"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0

        table.append(entry)

    return table


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict, encoder=None) -> list[dict]:
    if encoder is None:
        encoder = build_encoder(config)

    started = time.time()

    rows = []
    messages_by_K = {}

    sweep = config["sweep"]

    for point in sweep_points(config):

        K = point["K"]

        if K not in messages_by_K:
            messages_by_K[K] = load_messages(config, K)

        sweep_value = point[sweep["param"]] if sweep else None

        rows.extend(run_point(point, messages_by_K[K], encoder, config, sweep_value))

    summary = summarize(rows)

    out_dir = Path(config["results_dir"]) / config["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "results.csv", rows)
    write_csv(out_dir / "summary.csv", summary)

    provenance = copy.deepcopy(config)
    provenance["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    provenance["wall_seconds"] = round(time.time() - started, 1)
    provenance["model_name"] = ENCODER_REGISTRY[config["encoder"]]

    with (out_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(provenance, handle, sort_keys=False)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Stage 4 reproduction experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder", help="override the encoder (minilm | mpnet)")
    parser.add_argument("--num-messages", type=int, help="override coco.num_messages")
    parser.add_argument("--results-dir", help="override results_dir")
    args = parser.parse_args(argv)

    overrides: dict = {}

    if args.encoder:
        overrides["encoder"] = args.encoder

    if args.num_messages:
        overrides["coco"] = {"num_messages": args.num_messages}

    if args.results_dir:
        overrides["results_dir"] = args.results_dir

    config = load_repro_config(args.config, overrides)

    summary = run(config)

    for row in summary:
        extra = f" {row['sweep_param']}={row['sweep_value']}" if row["sweep_param"] else ""
        print(
            f"{row['algorithm']:>16}{extra} | SNR={row['snr_db']:>4} dB | "
            f"ATS={row.get('mean_ats', float('nan')):.4f} | "
            f"WTS={row.get('mean_wts', float('nan')):.4f} | "
            f"calls={row['mean_logical_calls']:.0f}"
        )


if __name__ == "__main__":
    main()
