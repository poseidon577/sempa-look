import time

import numpy as np

from sempalab.objectives import evaluate_partition
from sempalab.partition import generate_partitions
from sempalab.types import SearchResult


def random_pa(
    message,
    encoder,
    M,
    objective="ats",
    p=0.0,
    rng=None,
):
    """
    Select one valid partition uniformly at random
    and evaluate it.
    """

    K = len(message.tokens)

    if K == 0:
        raise ValueError("Message must contain at least one token")

    if M <= 0:
        raise ValueError("M must be positive")

    if K % M != 0:
        raise ValueError(
            f"M={M} must divide K={K}"
        )

    if objective not in {"ats", "wts"}:
        raise ValueError(
            f"Unknown objective: {objective}"
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "p must be between 0 and 1"
        )

    if rng is None:
        rng = np.random.default_rng()

    start_time = time.perf_counter()

    initial_logical_calls = encoder.logical_calls
    initial_miss_calls = encoder.miss_calls
    initial_cache_hits = encoder.cache_hits
    initial_encoder_batches = encoder.encoder_batches

    partitions = list(
        generate_partitions(K, M)
    )

    selected_index = rng.integers(
        len(partitions)
    )

    selected_partition = partitions[
        selected_index
    ]

    score = evaluate_partition(
        selected_partition,
        message,
        encoder,
        objective,
        p,
    )

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    return SearchResult(
    best_partition=selected_partition,
    best_score=score,
    objective=objective,
    partitions_evaluated=1,
    logical_calls=encoder.logical_calls - initial_logical_calls,
    miss_calls=encoder.miss_calls - initial_miss_calls,
    cache_hits=encoder.cache_hits - initial_cache_hits,
    encoder_batches=(
        encoder.encoder_batches
        - initial_encoder_batches
    ),
    runtime_seconds=runtime_seconds,
    )