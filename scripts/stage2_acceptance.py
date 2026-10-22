"""
Stage 2 acceptance experiment.

Runs every packetization method on a 100-message evaluation set at
K=8, M=4 and checks the Stage 2 acceptance criteria from the README:

    1. Full Search >= every other method on every message.
    2. SemPA-Look stays within 3% of Full Search.
    3. Lookahead (k >= 0.2 * k_max) beats the greedy / k=0 baselines.

At K=8, M=4 we have N=2, so k_max = N - 1 = 1 and the target regime
is k=1.

The evaluation set is built from MS-COCO 2017 caption annotations:
captions that tokenize to exactly K words are de-duplicated and
B of them are sampled with a fixed seed. The chosen set is written
next to the results so later runs can reuse exactly the same messages.

Usage:

    python scripts/stage2_acceptance.py \\
        --captions data/coco/captions_val2017.json

`--captions` also accepts a plain text file with one caption per line.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from sempalab.algorithms.full_search import full_search
from sempalab.algorithms.greedy_rss import greedy_rss
from sempalab.algorithms.greedy_tss import greedy_tss
from sempalab.algorithms.random_pa import random_pa
from sempalab.algorithms.sempa_look import sempa_look
from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats, wts
from sempalab.types import Message


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

# Full Search optimality is checked with a small tolerance because
# embeddings are float32.
OPTIMALITY_TOLERANCE = 1e-6


# ============================================================
# Evaluation set
# ============================================================

def tokenize(text):
    return tuple(TOKEN_PATTERN.findall(text.lower()))


def load_captions(path):
    path = Path(path)

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return [
            annotation["caption"]
            for annotation in data["annotations"]
        ]

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def build_messages(captions, K, B, seed):
    seen = set()
    pool = []

    for caption in captions:
        tokens = tokenize(caption)

        if len(tokens) != K or tokens in seen:
            continue

        seen.add(tokens)
        pool.append(tokens)

    if len(pool) < B:
        raise ValueError(
            f"Only {len(pool)} unique {K}-token captions found; "
            f"need {B}."
        )

    # Sort first so the sample depends only on the caption set,
    # not on file order.
    pool.sort()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(pool), size=B, replace=False)

    return [
        Message(
            message_id=message_id,
            text=" ".join(pool[index]),
            tokens=pool[index],
        )
        for message_id, index in enumerate(sorted(chosen))
    ]


# ============================================================
# Experiment
# ============================================================

def reset_cache(encoder):
    """
    Start every method cold so cost counters are comparable.
    """
    encoder.cache.clear()
    encoder.original_cache.clear()


def run_methods(message, encoder, M, p, P, k_values, seed):
    runs = {}

    runs["full_search"] = lambda: full_search(
        message, encoder, M, objective="ats", p=p,
    )

    runs["random_pa"] = lambda: random_pa(
        message, encoder, M, objective="ats", p=p,
        rng=np.random.default_rng(seed + message.message_id),
    )

    runs["greedy_tss"] = lambda: greedy_tss(message, encoder, M)

    runs["greedy_rss"] = lambda: greedy_rss(message, encoder, M)

    for k in k_values:
        runs[f"sempa_look_k{k}"] = (
            lambda k=k: sempa_look(
                message, encoder, M, P=P, k=k,
                seed=seed + message.message_id,
            )
        )

    rows = []

    for method, run in runs.items():
        reset_cache(encoder)

        result = run()

        # Every method is scored with the same exact objectives,
        # regardless of the surrogate it optimized.
        ats = exact_ats(result.best_partition, message, encoder, p)
        wts_score = wts(result.best_partition, message, encoder, p)

        rows.append(
            {
                "message_id": message.message_id,
                "text": message.text,
                "p": p,
                "method": method,
                "partition": json.dumps(result.best_partition),
                "ats": ats,
                "wts": wts_score,
                "logical_calls": result.logical_calls,
                "miss_calls": result.miss_calls,
                "cache_hits": result.cache_hits,
                "encoder_batches": result.encoder_batches,
                "runtime_seconds": result.runtime_seconds,
            }
        )

    return rows


def summarize(rows, methods, p, look_method, tolerance):
    by_message = {}

    for row in rows:
        if row["p"] != p:
            continue

        by_message.setdefault(row["message_id"], {})[row["method"]] = row

    full = np.array(
        [m["full_search"]["ats"] for m in by_message.values()]
    )

    methods_summary = {}

    for method in methods:
        ats = np.array(
            [m[method]["ats"] for m in by_message.values()]
        )
        wts_scores = np.array(
            [m[method]["wts"] for m in by_message.values()]
        )
        relative_gap = (full - ats) / full

        methods_summary[method] = {
            "mean_ats": float(ats.mean()),
            "mean_wts": float(wts_scores.mean()),
            "mean_relative_gap": float(relative_gap.mean()),
            "max_relative_gap": float(relative_gap.max()),
            "fraction_within_tolerance": float(
                (relative_gap <= tolerance).mean()
            ),
            "fraction_matching_full": float(
                (np.abs(full - ats) <= OPTIMALITY_TOLERANCE).mean()
            ),
            "mean_logical_calls": float(np.mean(
                [m[method]["logical_calls"] for m in by_message.values()]
            )),
            "mean_miss_calls": float(np.mean(
                [m[method]["miss_calls"] for m in by_message.values()]
            )),
            "mean_runtime_seconds": float(np.mean(
                [m[method]["runtime_seconds"] for m in by_message.values()]
            )),
        }

    full_search_violations = [
        {
            "message_id": message_id,
            "method": method,
            "excess": float(m[method]["ats"] - m["full_search"]["ats"]),
        }
        for message_id, m in by_message.items()
        for method in methods
        if m[method]["ats"]
        > m["full_search"]["ats"] + OPTIMALITY_TOLERANCE
    ]

    look = methods_summary[look_method]

    checks = {
        "full_search_dominates": not full_search_violations,
        "sempa_look_within_tolerance": (
            look["mean_relative_gap"] <= tolerance
        ),
        "lookahead_beats_k0": (
            look["mean_ats"] >= methods_summary["sempa_look_k0"]["mean_ats"]
        ),
        "lookahead_beats_greedy_tss": (
            look["mean_ats"] >= methods_summary["greedy_tss"]["mean_ats"]
        ),
    }

    return {
        "p": p,
        "messages": len(by_message),
        "methods": methods_summary,
        "full_search_violations": full_search_violations,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--P", type=int, default=10)
    parser.add_argument(
        "--p", type=float, nargs="+", default=[0.1, 0.3, 0.5],
        help="Packet erasure probabilities.",
    )
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder", default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/stage2")
    args = parser.parse_args()

    if args.K % args.M != 0:
        raise ValueError("M must divide K")

    N = args.K // args.M
    k_max = N - 1
    k_values = list(range(k_max + 1))

    # Smallest k in the target regime k >= 0.2 * k_max (at least 1).
    k_target = max(1, int(np.ceil(0.2 * k_max)))
    look_method = f"sempa_look_k{k_target}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    messages = build_messages(
        load_captions(args.captions), args.K, args.B, args.seed,
    )

    (out / "messages.json").write_text(
        json.dumps(
            [
                {"message_id": m.message_id, "tokens": m.tokens}
                for m in messages
            ],
            indent=2,
        )
    )

    encoder = InstrumentedEncoder(
        model_name=args.encoder, device=args.device,
    )

    rows = []
    start = time.perf_counter()

    for p in args.p:
        for index, message in enumerate(messages, start=1):
            rows.extend(
                run_methods(
                    message, encoder, args.M, p, args.P,
                    k_values, args.seed,
                )
            )

            if index % 10 == 0:
                print(
                    f"p={p}: {index}/{len(messages)} messages "
                    f"({time.perf_counter() - start:.1f}s)"
                )

    with open(out / "acceptance.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    methods = list(dict.fromkeys(row["method"] for row in rows))

    summary = {
        "config": {
            "K": args.K,
            "M": args.M,
            "N": N,
            "B": args.B,
            "P": args.P,
            "k_max": k_max,
            "k_target": k_target,
            "p": args.p,
            "tolerance": args.tolerance,
            "seed": args.seed,
            "encoder": args.encoder,
            "device": encoder.device,
            "captions": str(args.captions),
        },
        "results": [
            summarize(rows, methods, p, look_method, args.tolerance)
            for p in args.p
        ],
    }

    summary["passed"] = all(r["passed"] for r in summary["results"])

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    for result in summary["results"]:
        print(f"\np = {result['p']}")
        print(
            f"{'Method':<16} {'ATS':>8} {'WTS':>8} "
            f"{'gap':>8} {'max gap':>8} {'=full':>6} {'calls':>7}"
        )

        for method, s in result["methods"].items():
            print(
                f"{method:<16} {s['mean_ats']:>8.4f} {s['mean_wts']:>8.4f} "
                f"{s['mean_relative_gap']:>8.2%} "
                f"{s['max_relative_gap']:>8.2%} "
                f"{s['fraction_matching_full']:>6.0%} "
                f"{s['mean_logical_calls']:>7.0f}"
            )

        for check, ok in result["checks"].items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {check}")

    print(f"\nOverall: {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"Results written to {out}/")


if __name__ == "__main__":
    main()
