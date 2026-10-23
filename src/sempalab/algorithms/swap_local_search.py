"""
Swap local search on the paper's RSS surrogate.

The RSS surrogate of a partition is the sum of its single-loss
similarities,

    Psi(G) = sum_n phi(W \\ G_n, W),

the first-order (p -> 0) term that SemPA-Look's candidate score
approximates greedily (eqs. 15, 18). This baseline maximises the same
quantity by best-improvement token swaps between subpackets, so it is
channel- and p-blind like SemPA-Look but searches the surrogate
directly. It was missing from the paper (docs/limitations.md, L9).

Encoder accounting: every distinct subset the search asks the encoder
for counts once (one logical call); repeats are served from a memo.
"""

from __future__ import annotations

import time

import numpy as np

from sempalab.metrics import batch_semantic_similarity
from sempalab.partition import sample_random_partition
from sempalab.types import SearchResult


def rss_sum(partition, rss: dict) -> float:
    return float(sum(rss[tuple(sorted(group))] for group in partition))


def swap_local_search(
    message,
    encoder,
    M: int,
    seed=None,
    init=None,
    restarts: int = 1,
    max_passes: int = 100,
    tol: float = 1e-9,
) -> SearchResult:
    """
    Best-improvement swap search on the RSS surrogate.

    Starts from `init` (a partition) or from `restarts` uniformly random
    partitions, and in each pass evaluates every swap of two tokens in
    different subpackets, applying the best one while it improves Psi.
    Returns the best local optimum found, groups in slot order.
    """

    K = len(message.tokens)

    if M <= 0 or K % M != 0:
        raise ValueError(f"M={M} must be positive and divide K={K}")

    if restarts < 1:
        raise ValueError("restarts must be positive")

    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()
    initial = (encoder.logical_calls, encoder.miss_calls, encoder.cache_hits, encoder.encoder_batches)

    all_indices = tuple(range(K))
    rss: dict[tuple[int, ...], float] = {}

    def ensure(groups) -> None:
        missing = list(dict.fromkeys(g for g in groups if g not in rss))

        if missing:
            removed = [tuple(i for i in all_indices if i not in set(g)) for g in missing]
            scores = batch_semantic_similarity(encoder, message, removed)
            rss.update(zip(missing, map(float, scores)))

    starts = [init] if init is not None else [sample_random_partition(K, M, rng) for _ in range(restarts)]

    best_partition, best_score = None, -np.inf
    evaluated = 0

    for start in starts:

        groups = [tuple(sorted(g)) for g in start]
        ensure(groups)

        for _ in range(max_passes):

            moves = []

            for a in range(len(groups)):
                for b in range(a + 1, len(groups)):
                    for i in groups[a]:
                        for j in groups[b]:
                            new_a = tuple(sorted([x for x in groups[a] if x != i] + [j]))
                            new_b = tuple(sorted([x for x in groups[b] if x != j] + [i]))
                            moves.append((a, b, new_a, new_b))

            ensure([g for move in moves for g in move[2:]])
            evaluated += len(moves)

            gains = [
                rss[new_a] + rss[new_b] - rss[groups[a]] - rss[groups[b]]
                for a, b, new_a, new_b in moves
            ]

            if not moves or max(gains) <= tol:
                break

            a, b, new_a, new_b = moves[int(np.argmax(gains))]
            groups[a], groups[b] = new_a, new_b

        score = rss_sum(groups, rss)

        if score > best_score:
            best_partition, best_score = tuple(groups), score

    return SearchResult(
        best_partition=best_partition,
        best_score=float(best_score),
        objective="rss-sum",
        partitions_evaluated=evaluated + len(starts),
        logical_calls=encoder.logical_calls - initial[0],
        miss_calls=encoder.miss_calls - initial[1],
        cache_hits=encoder.cache_hits - initial[2],
        encoder_batches=encoder.encoder_batches - initial[3],
        runtime_seconds=time.perf_counter() - start_time,
    )
