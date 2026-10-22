"""
Stage 3 experiment entry point.

    python -m sempalab.run --config configs/stage3_demo.yaml

For every SNR in the config:
    1. p = 1 - (1 - Q(sqrt(2 gamma)))^d
    2. each algorithm packetizes every message
    3. batches of B messages are aggregated into packets,
       sent over BPSK/AWGN, and deaggregated
    4. exact ATS (analytical) and empirical ATS (physical
       simulation) are recorded with the algorithm's cost counters

Outputs go to <results_dir>/<experiment>/:
    results.csv, summary.json, config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from sempalab.aggregation import packet_bits
from sempalab.channel.awgn_bpsk import packet_error_probability
from sempalab.config import ENCODER_REGISTRY, load_config
from sempalab.metrics import exact_ats
from sempalab.system import simulate_batch_ats
from sempalab.types import Message


# Algorithms whose partition does not depend on p
# are computed once per message and reused across SNRs.
P_INDEPENDENT = {"greedy_tss", "greedy_rss", "sempa_look"}


def load_messages(config: dict) -> list[Message]:
    """
    Load and word-tokenize messages, keeping those with exactly K tokens.
    """

    if config["dataset"] == "inline":
        texts = list(config["messages"])
    else:
        # Relative paths resolve against the working directory,
        # like results_dir.
        texts = Path(config["data_path"]).read_text().splitlines()

    K = config["K"]

    messages = []

    for text in texts:

        tokens = tuple(text.split())

        if len(tokens) != K:
            continue

        messages.append(
            Message(
                message_id=len(messages),
                text=" ".join(tokens),
                tokens=tokens,
            )
        )

        if config["num_messages"] and len(messages) >= config["num_messages"]:
            break

    if not messages:
        raise ValueError(f"No messages with exactly K={K} tokens")

    return messages


def build_encoder(config: dict):
    # Imported lazily: torch is only needed for real runs.
    from sempalab.encoder import InstrumentedEncoder

    return InstrumentedEncoder(
        model_name=ENCODER_REGISTRY[config["encoder"]],
        device=config["device"],
    )


def packetize(name: str, message: Message, encoder, config: dict, p: float, seed: int):
    """
    Dispatch to one packetization algorithm.
    """

    M = config["M"]
    objective = config["objective"]

    if name == "full_search":
        from sempalab.algorithms.full_search import full_search

        return full_search(message, encoder, M, objective=objective, p=p)

    if name == "random_pa":
        from sempalab.algorithms.random_pa import random_pa

        return random_pa(
            message,
            encoder,
            M,
            objective=objective,
            p=p,
            rng=np.random.default_rng(seed),
        )

    if name == "no_packetization":
        from sempalab.algorithms.no_packetization import no_packetization

        return no_packetization(message, encoder, objective=objective, p=p)

    if name == "greedy_tss":
        from sempalab.algorithms.greedy_tss import greedy_tss

        return greedy_tss(message, encoder, M)

    if name == "greedy_rss":
        from sempalab.algorithms.greedy_rss import greedy_rss

        return greedy_rss(message, encoder, M)

    if name == "sempa_look":
        from sempalab.algorithms.sempa_look import sempa_look

        return sempa_look(
            message,
            encoder,
            M,
            P=config["P"],
            k=config["k"],
            seed=seed,
        )

    if name == "genetic":
        from sempalab.algorithms.genetic import genetic

        return genetic(
            message,
            encoder,
            M,
            objective=objective,
            p=p,
            seed=seed,
            **config["genetic"],
        )

    raise ValueError(f"Unknown algorithm: {name}")


def run(config: dict, encoder=None) -> dict:
    """
    Run one experiment. Returns the summary dict and writes outputs.
    """

    if encoder is None:
        encoder = build_encoder(config)

    messages = load_messages(config)

    seed = config["seed"]
    d = config["packet_bits"]
    B = config["B"]

    budget = packet_bits(config["K"], config["M"], B, config["dictionary_size"])

    if not budget.fits(d):
        print(
            f"Warning: packet needs {budget.total_bits} bits "
            f"but packet_bits={d}"
        )

    batches = [messages[i:i + B] for i in range(0, len(messages), B)]

    rows = []
    cached_results = {}

    for snr_index, snr_db in enumerate(config["snr_db"]):

        p = packet_error_probability(snr_db, d)

        for algo_index, name in enumerate(config["algorithms"]):

            # Distinct, reproducible stream per (SNR, algorithm).
            rng = np.random.default_rng([seed, snr_index, algo_index])

            results = []

            for message in messages:

                key = (name, message.message_id)

                if name in P_INDEPENDENT and key in cached_results:
                    results.append(cached_results[key])
                    continue

                result = packetize(
                    name,
                    message,
                    encoder,
                    config,
                    p,
                    seed + message.message_id,
                )

                cached_results[key] = result
                results.append(result)

            by_id = {
                message.message_id: result
                for message, result in zip(messages, results)
            }

            for batch in batches:

                partitions = [
                    by_id[message.message_id].best_partition
                    for message in batch
                ]

                empirical, per = simulate_batch_ats(
                    batch,
                    partitions,
                    encoder,
                    snr_db,
                    config["trials"],
                    d,
                    rng,
                    bit_level=config["bit_level"],
                )

                for message, partition in zip(batch, partitions):

                    result = by_id[message.message_id]

                    rows.append(
                        {
                            "experiment": config["experiment"],
                            "algorithm": name,
                            "snr_db": snr_db,
                            "p": p,
                            "message_id": message.message_id,
                            "text": message.text,
                            "partition": json.dumps(partition),
                            "exact_ats": exact_ats(partition, message, encoder, p),
                            "empirical_ats": empirical[message.message_id],
                            "empirical_per": per,
                            "batch_size": len(batch),
                            "partitions_evaluated": result.partitions_evaluated,
                            "logical_calls": result.logical_calls,
                            "miss_calls": result.miss_calls,
                            "cache_hits": result.cache_hits,
                            "encoder_batches": result.encoder_batches,
                            "runtime_seconds": result.runtime_seconds,
                        }
                    )

    summary = summarize(rows, config, budget)

    write_outputs(rows, summary, config)

    return summary


def summarize(rows: list[dict], config: dict, budget) -> dict:
    groups = {}

    for row in rows:
        groups.setdefault((row["algorithm"], row["snr_db"]), []).append(row)

    table = []

    for (name, snr_db), group in groups.items():
        table.append(
            {
                "algorithm": name,
                "snr_db": snr_db,
                "p": group[0]["p"],
                "messages": len(group),
                "mean_exact_ats": float(np.mean([r["exact_ats"] for r in group])),
                "mean_empirical_ats": float(np.mean([r["empirical_ats"] for r in group])),
                "mean_empirical_per": float(np.mean([r["empirical_per"] for r in group])),
                "mean_logical_calls": float(np.mean([r["logical_calls"] for r in group])),
                "mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in group])),
            }
        )

    return {
        "experiment": config["experiment"],
        "mode": config["mode"],
        "seed": config["seed"],
        "packet_budget": budget.__dict__,
        "results": table,
    }


def write_outputs(rows: list[dict], summary: dict, config: dict) -> Path:
    out_dir = Path(config["results_dir"]) / config["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    # Provenance: the fully resolved config used for this run.
    with (out_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(
            {**config, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            handle,
            sort_keys=False,
        )

    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a SEMPA-LAB experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    summary = run(config)

    for row in summary["results"]:
        print(
            f"{row['algorithm']:>18} | SNR={row['snr_db']:>5} dB | "
            f"p={row['p']:.4f} | exact ATS={row['mean_exact_ats']:.4f} | "
            f"empirical ATS={row['mean_empirical_ats']:.4f}"
        )


if __name__ == "__main__":
    main()
