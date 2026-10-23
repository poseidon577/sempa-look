"""
Channel-aware aggregation experiments (bursty and fading erasures).

    python -m sempalab.channel_experiment --config configs/channel/coco_k12.yaml
    python -m sempalab.channel_experiment --config configs/channel/coco_k12.yaml --rescore minilm

For every message, subpacket size M, channel, average packet error
rate p and layout stride, each algorithm's partition (in slot order) is
scored by exact ATS under the channel's joint slot law Q
(`channel.correlated`). See docs/channel_aware.md.

Algorithms
    random_pa          uniform random partition (paper baseline)
    interleave         token i -> slot i mod N
    sempa_look         the paper's method (channel- and p-blind)
    swap_ls            swap local search on the RSS surrogate
    <blind>+order      a channel-blind partition with its slots
                       reordered for Q (2^N extra calls)
    casa_iid           CASA fitted and searched as if the channel were
                       i.i.d. at the same p (p-aware, memory-blind)
    casa               channel-aware semantic aggregation
    full_search_iid    ATS-optimal partition under i.i.d. erasures at p
    full_search        ATS_Q-optimal ordered partition (oracle, K = 12)

`--rescore ENCODER` re-scores the stored partitions under another
encoder (direction 4 of docs/limitations.md): partitions are kept,
only phi changes.

Outputs go to <results_dir>/<experiment>/:
    results.csv.gz, summary.csv, config.yaml
    rescored_<encoder>.csv.gz, summary_<encoder>.csv   (with --rescore)
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import time
import zlib
from pathlib import Path

import numpy as np
import yaml

from sempalab.algorithms.interleave import interleave_partition
from sempalab.channel.correlated import make_channel
from sempalab.channel_aware import (
    OrderedOracle,
    best_slot_order,
    casa,
    channel_ats,
    sample_pool,
    sample_subsets,
    surrogate_r2,
    fitting_law,
    union_masks,
)
from sempalab.config import ENCODER_REGISTRY, merge
from sempalab.datasets import eval_set
from sempalab.evaluation import SimilarityTable, to_mask
from sempalab.partition import sample_random_partition
from sempalab.repro import block_unions, count_block_unions


BLIND = ("random_pa", "interleave", "sempa_look", "swap_ls")

ALGORITHMS = (
    *BLIND,
    *(f"{name}+order" for name in BLIND),
    "casa_iid",
    "casa",
    "full_search_iid",
    "full_search",
)

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
    "M": [4],
    "p": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    # Each entry: {name: iid | gilbert | rayleigh, parameter: rho | alpha}.
    "channels": [{"name": "iid"}],
    "stride": [1],
    "algorithms": list(ALGORITHMS),
    # SemPA-Look
    "P": 10,
    "k": 4,
    "swap_ls": {"restarts": 1},
    "casa": {"budget": None, "restarts": 20, "top": 5, "ridge": 1e-3},
    # Fit every CASA run of a (message, M) on one shared, reweighted pool
    # of subsets (channel_aware.sample_pool). Saves encoder time when
    # nothing is prefetched (K = 24); reported calls are unchanged.
    "shared_pool": False,
    # Held-out subsets for the surrogate's R^2 (0 disables).
    "r2_samples": 200,
    "rayleigh_samples": 400_000,
    "packet_bits": 300,
    "prefetch_max_sets": 5000,
    # Use only the first `limit` messages of the evaluation set (smoke runs).
    "limit": None,
    "seed": 42,
    "results_dir": "results/channel",
}


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

    if not config["algorithms"] or set(config["algorithms"]) - set(ALGORITHMS):
        raise ValueError(f"algorithms must be a non-empty subset of {ALGORITHMS}")

    for M in config["M"]:
        if config["K"] % M != 0 or M == config["K"]:
            raise ValueError(f"M={M} must divide K={config['K']} and leave N >= 2")

    for p in config["p"]:
        if not 0.0 < p < 1.0:
            raise ValueError("every p must be in (0, 1)")

    for spec in config["channels"]:
        if spec.get("name") not in {"iid", "gilbert", "rayleigh"} or set(spec) - {"name", "parameter"}:
            raise ValueError(f"bad channel spec {spec}")

    if "full_search" in config["algorithms"] or "full_search_iid" in config["algorithms"]:
        for M in config["M"]:
            if count_block_unions(config["K"], M) > config["prefetch_max_sets"]:
                raise ValueError(f"full search is infeasible at K={config['K']}, M={M}")


def channel_points(config: dict) -> list[dict]:
    """
    Every (channel, parameter, p) the config visits; iid appears once per p.
    """

    points = []

    for spec in config["channels"]:
        for p in config["p"]:
            points.append({"channel": spec["name"], "parameter": float(spec.get("parameter", 0.0)), "p": float(p)})

    return points


def build_channel(point: dict, config: dict):
    options = {}

    if point["channel"] == "rayleigh":
        options = {"samples": config["rayleigh_samples"], "d": config["packet_bits"]}

    return make_channel(point["channel"], point["p"], point["parameter"], **options)


class LawBook:
    """
    Q for every (channel point, N, stride), built once.
    """

    def __init__(self, config: dict):
        self.config = config
        self.channels: dict[tuple, object] = {}

    def law(self, point: dict, N: int, stride: int) -> np.ndarray:
        key = (point["channel"], point["parameter"], point["p"])

        if key not in self.channels:
            self.channels[key] = build_channel(point, self.config)

        return self.channels[key].pattern_probabilities(N, stride)

    def iid(self, p: float, N: int) -> np.ndarray:
        return self.law({"channel": "iid", "parameter": 0.0, "p": p}, N, 1)


# -------------------------------------------------------------
# Runner
# -------------------------------------------------------------


def build_encoder(name: str, config: dict):
    from sempalab.encoder import InstrumentedEncoder

    return InstrumentedEncoder(
        model_name=ENCODER_REGISTRY[name],
        device=config["device"],
        cache_size=config["cache_size"],
    )


def load_messages(config: dict):
    data = config["data"]

    messages = eval_set(
        data["source"],
        data["path"],
        K=config["K"],
        num_messages=data["num_messages"],
        seed=data["seed"],
        cache_dir=data["eval_set_dir"],
        max_articles=data["max_articles"],
    )

    return messages[: config["limit"]] if config["limit"] else messages


def blind_partitions(message, encoder, M: int, config: dict, seed: int) -> dict:
    """
    Channel-blind partitions (slot order as each algorithm emits it)
    with their logical encoder calls.
    """

    K = len(message.tokens)
    wanted = {name.split("+")[0] for name in config["algorithms"]} & set(BLIND)
    out = {}

    if "random_pa" in wanted:
        out["random_pa"] = (sample_random_partition(K, M, np.random.default_rng(seed)), 0)

    if "interleave" in wanted:
        out["interleave"] = (interleave_partition(K, M), 0)

    if "sempa_look" in wanted:
        from sempalab.algorithms.sempa_look import sempa_look

        result = sempa_look(message, encoder, M, P=config["P"], k=config["k"], seed=seed)
        out["sempa_look"] = (result.best_partition, result.logical_calls)

    if "swap_ls" in wanted:
        from sempalab.algorithms.swap_local_search import swap_local_search

        result = swap_local_search(message, encoder, M, seed=seed, **config["swap_ls"])
        out["swap_ls"] = (result.best_partition, result.logical_calls)

    return out


def point_seed(config: dict, *parts) -> int:
    """
    Deterministic seed for one (message, M, channel point, stride).
    """

    return int(np.random.SeedSequence([config["seed"], zlib.crc32(repr(parts).encode())]).generate_state(1)[0])


def run_message(message, encoder, config: dict, laws: LawBook) -> list[dict]:
    K = config["K"]
    rows = []
    algorithms = config["algorithms"]

    for M in config["M"]:
        N = K // M
        table = SimilarityTable(encoder, message)

        if count_block_unions(K, M) <= config["prefetch_max_sets"]:
            table.fill(to_mask(subset) for subset in block_unions(K, M))

        seed = config["seed"] * 1_000_003 + message.message_id
        blind = blind_partitions(message, encoder, M, config, seed)

        oracle = OrderedOracle(table, K, M) if {"full_search", "full_search_iid"} & set(algorithms) else None

        iid_cache: dict[float, dict] = {}

        casa_options = dict(config["casa"])

        if config["shared_pool"]:
            size = casa_options.pop("budget") or 4 * (1 + K + K * (K - 1) // 2)
            casa_options["pool"] = sample_pool(K, M, size, np.random.default_rng(point_seed(config, message.message_id, M, "pool")))

        def iid_outcomes(p: float) -> dict:
            if p not in iid_cache:
                Q_iid = laws.iid(p, N)
                out = {}
                if "full_search_iid" in algorithms:
                    partition, _, _ = oracle.best(Q_iid)
                    # Under i.i.d. erasures order is irrelevant: keep canonical order.
                    out["full_search_iid"] = (tuple(sorted(partition)), len(oracle.partitions) * (1 << N), {})
                if "casa_iid" in algorithms:
                    result = casa(table, K, M, Q_iid, np.random.default_rng(point_seed(config, message.message_id, M, "iid", p)), **casa_options)
                    out["casa_iid"] = (result.partition, result.logical_calls, {"surrogate_ats": result.surrogate_ats})
                iid_cache[p] = out
            return iid_cache[p]

        for point in channel_points(config):
            for stride in config["stride"]:

                if point["channel"] == "iid" and stride != config["stride"][0]:
                    continue  # stride is irrelevant without memory

                Q = laws.law(point, N, stride)
                outcomes: dict[str, tuple] = {}

                for name, (partition, calls) in blind.items():
                    if name in algorithms:
                        outcomes[name] = (partition, calls, {})

                    if f"{name}+order" in algorithms:
                        ordered, _ = best_slot_order(table, partition, Q)
                        extra = len({int(m) for m in union_masks(partition) if m})
                        outcomes[f"{name}+order"] = (ordered, calls + extra, {})

                outcomes.update(iid_outcomes(point["p"]))

                if "casa" in algorithms:
                    rng = np.random.default_rng(point_seed(config, message.message_id, M, point["channel"], point["parameter"], point["p"], stride))
                    result = casa(table, K, M, Q, rng, **casa_options)
                    extra = {"surrogate_ats": result.surrogate_ats}

                    if config["r2_samples"]:
                        held_out = sample_subsets(K, M, fitting_law(Q), config["r2_samples"], np.random.default_rng(rng.integers(1 << 31)))
                        extra["surrogate_r2"] = surrogate_r2(result.model, table, held_out)

                    outcomes["casa"] = (result.partition, result.logical_calls, extra)

                if "full_search" in algorithms:
                    partition, _, _ = oracle.best(Q)
                    outcomes["full_search"] = (partition, len(oracle.partitions) * (1 << N), {})

                for name in algorithms:
                    partition, calls, extra = outcomes[name]

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


# -------------------------------------------------------------
# Summaries
# -------------------------------------------------------------


GROUP = ("K", "M", "channel", "parameter", "stride", "p", "algorithm")


def bootstrap_ci(values: np.ndarray, resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], references=("sempa_look", "random_pa", "swap_ls+order", "full_search_iid")) -> list[dict]:
    """
    Mean ATS, SEM and mean calls per setting, plus the paired mean
    gain over each reference algorithm with a bootstrap 95% CI.
    """

    by_setting: dict[tuple, dict[str, dict[int, dict]]] = {}

    for row in rows:
        setting = tuple(row[key] for key in GROUP[:-1])
        by_setting.setdefault(setting, {}).setdefault(row["algorithm"], {})[int(row["message_id"])] = row

    table = []

    for setting, algorithms in by_setting.items():
        for name, by_message in algorithms.items():
            ids = sorted(by_message)
            ats = np.array([float(by_message[i]["ats"]) for i in ids])

            entry = dict(zip(GROUP, (*setting, name)))
            entry.update({
                "encoder": next(iter(by_message.values()))["encoder"],
                "messages": len(ids),
                "mean_ats": float(ats.mean()),
                "sem_ats": float(ats.std(ddof=1) / np.sqrt(len(ats))) if len(ats) > 1 else 0.0,
                "mean_logical_calls": float(np.mean([float(by_message[i]["logical_calls"]) for i in ids])),
            })

            for key in ("surrogate_r2",):
                values = [by_message[i].get(key) for i in ids]
                values = [float(v) for v in values if v not in (None, "")]
                if values:
                    entry[f"mean_{key}"] = float(np.mean(values))

            for reference in references:
                if reference in algorithms and reference != name:
                    ref = np.array([float(algorithms[reference][i]["ats"]) for i in ids])
                    diff = ats - ref
                    low, high = bootstrap_ci(diff) if len(diff) > 1 else (float(diff[0]),) * 2
                    entry[f"gain_vs_{reference}"] = float(diff.mean())
                    entry[f"gain_vs_{reference}_ci_low"] = low
                    entry[f"gain_vs_{reference}_ci_high"] = high
                    entry[f"rel_gain_vs_{reference}"] = float(diff.mean() / ref.mean())

            table.append(entry)

    return table


def _open(path: Path, mode: str):
    """
    Per-message results are stored gzipped (*.csv.gz) to keep the repo small.
    """

    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", newline="")
    return path.open(mode, newline="")


def write_csv(path: Path, rows: list[dict]) -> None:
    with _open(path, "w") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with _open(path, "r") as handle:
        return list(csv.DictReader(handle))


def run(config: dict, encoder=None) -> list[dict]:
    if encoder is None:
        encoder = build_encoder(config["encoder"], config)

    started = time.time()
    laws = LawBook(config)
    rows = []

    messages = load_messages(config)

    for count, message in enumerate(messages, 1):
        rows.extend(run_message(message, encoder, config, laws))
        print(f"[{config['experiment']}] message {count}/{len(messages)} ({time.time() - started:.0f}s)", flush=True)

    out_dir = Path(config["results_dir"]) / config["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(rows)

    write_csv(out_dir / "results.csv.gz", rows)
    write_csv(out_dir / "summary.csv", summary)

    provenance = copy.deepcopy(config)
    provenance["run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    provenance["wall_seconds"] = round(time.time() - started, 1)
    provenance["model_name"] = ENCODER_REGISTRY[config["encoder"]]

    with (out_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(provenance, handle, sort_keys=False)

    return summary


def rescore(config: dict, encoder_name: str, encoder=None) -> list[dict]:
    """
    Re-score the stored partitions of `config`'s experiment under
    another encoder. Partitions, channels and laws are unchanged.
    """

    out_dir = Path(config["results_dir"]) / config["experiment"]
    rows = read_csv(out_dir / "results.csv.gz")

    if encoder is None:
        encoder = build_encoder(encoder_name, config)

    laws = LawBook(config)
    messages = {message.message_id: message for message in load_messages(config)}

    tables: dict[tuple[int, int], SimilarityTable] = {}
    rescored = []

    for row in rows:
        message_id, M = int(row["message_id"]), int(row["M"])
        key = (message_id, M)

        if key not in tables:
            tables.clear()  # rows are grouped by message
            tables[key] = SimilarityTable(encoder, messages[message_id])

        partition = tuple(tuple(group) for group in json.loads(row["partition"]))
        point = {"channel": row["channel"], "parameter": float(row["parameter"]), "p": float(row["p"])}
        Q = laws.law(point, int(row["N"]), int(row["stride"]))

        new = dict(row)
        new["encoder"] = encoder_name
        new["partition_encoder"] = row["encoder"]
        new["ats"] = channel_ats(tables[key], partition, Q)
        rescored.append(new)

    write_csv(out_dir / f"rescored_{encoder_name}.csv.gz", rescored)

    summary = summarize(rescored)
    write_csv(out_dir / f"summary_{encoder_name}.csv", summary)

    return summary


def resummarize(config: dict) -> None:
    """
    Rebuild summary.csv (and summary_<encoder>.csv) from stored rows.
    """

    out_dir = Path(config["results_dir"]) / config["experiment"]

    write_csv(out_dir / "summary.csv", summarize(read_csv(out_dir / "results.csv.gz")))

    for path in sorted(out_dir.glob("rescored_*.csv.gz")):
        encoder = path.name.removeprefix("rescored_").removesuffix(".csv.gz")
        write_csv(out_dir / f"summary_{encoder}.csv", summarize(read_csv(path)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a channel-aware aggregation experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder", help="override the encoder")
    parser.add_argument("--num-messages", type=int, help="override data.num_messages")
    parser.add_argument("--results-dir", help="override results_dir")
    parser.add_argument("--limit", type=int, help="use only the first LIMIT messages")
    parser.add_argument("--rescore", metavar="ENCODER", help="re-score stored partitions under ENCODER")
    parser.add_argument("--summarize-only", action="store_true", help="rebuild summary*.csv from stored results")
    args = parser.parse_args(argv)

    overrides: dict = {}

    if args.encoder:
        overrides["encoder"] = args.encoder
    if args.num_messages:
        overrides["data"] = {"num_messages": args.num_messages}
    if args.results_dir:
        overrides["results_dir"] = args.results_dir
    if args.limit:
        overrides["limit"] = args.limit

    config = load_config(args.config, overrides)

    if args.summarize_only:
        resummarize(config)
        return

    summary = rescore(config, args.rescore) if args.rescore else run(config)

    for row in summary:
        if row["algorithm"] in {"sempa_look", "casa", "full_search"}:
            print(
                f"M={row['M']} {row['channel']}({row['parameter']}) D={row['stride']} p={row['p']:.2f} "
                f"{row['algorithm']:>12} ATS={row['mean_ats']:.4f} calls={row['mean_logical_calls']:.0f}"
            )


if __name__ == "__main__":
    main()
