import time

from sempalab.objectives import evaluate_partition
from sempalab.partition import generate_partitions
from sempalab.types import SearchResult


def full_search(
    message,
    encoder,
    M,
    objective="ats",
    p=0.0,
):
    """
    Exhaustively evaluate every valid partition.

    Returns the highest-scoring partition according
    to the selected objective.
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

    start_time = time.perf_counter()

    initial_logical_calls = encoder.logical_calls
    initial_miss_calls = encoder.miss_calls
    initial_cache_hits = encoder.cache_hits
    initial_encoder_batches = encoder.encoder_batches

    best_partition = None
    best_score = float("-inf")

    partitions_evaluated = 0

    for partition in generate_partitions(K, M):

        score = evaluate_partition(
            partition,
            message,
            encoder,
            objective,
            p,
        )

        partitions_evaluated += 1

        if score > best_score:
            best_score = score
            best_partition = partition

    runtime_seconds = time.perf_counter() - start_time

    return SearchResult(
    best_partition=best_partition,
    best_score=best_score,
    objective=objective,
    partitions_evaluated=partitions_evaluated,
    logical_calls=encoder.logical_calls - initial_logical_calls,
    miss_calls=encoder.miss_calls - initial_miss_calls,
    cache_hits=encoder.cache_hits - initial_cache_hits,
    encoder_batches=(
        encoder.encoder_batches
        - initial_encoder_batches
    ),
    runtime_seconds=runtime_seconds,
    )