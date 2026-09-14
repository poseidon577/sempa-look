import time

from sempalab.objectives import evaluate_partition
from sempalab.types import SearchResult


def no_packetization(
    message,
    encoder,
    objective="ats",
    p=0.0,
):
    """
    Treat the entire message as a single packet.
    """

    if len(message.tokens) == 0:
        raise ValueError(
            "Message must contain at least one token"
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

    # One packet containing the entire message.
    partition = (
        tuple(range(len(message.tokens))),
    )

    score = evaluate_partition(
        partition,
        message,
        encoder,
        objective,
        p,
    )

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    return SearchResult(
        best_partition=partition,
        best_score=score,
        objective=objective,
        partitions_evaluated=1,
        logical_calls=(
            encoder.logical_calls
            - initial_logical_calls
        ),
        miss_calls=(
            encoder.miss_calls
            - initial_miss_calls
        ),
        cache_hits=(
            encoder.cache_hits
            - initial_cache_hits
        ),
        runtime_seconds=runtime_seconds,
    )