import time
from itertools import combinations

from sempalab.metrics import semantic_similarity
from sempalab.types import SearchResult


def greedy_tss(
    message,
    encoder,
    M,
):
    """
    Greedy-TSS packetization.

    At each step, evaluate every candidate subpacket C
    of size M from the remaining token pool and select
    the candidate maximizing:

        phi(C, W)

    where W is the original message.

    This is the high-p surrogate branch described
    in the implementation plan.
    """

    K = len(message.tokens)

    if K == 0:
        raise ValueError(
            "Message must contain at least one token"
        )

    if M <= 0:
        raise ValueError("M must be positive")

    if K % M != 0:
        raise ValueError(
            f"M={M} must divide K={K}"
        )

    start_time = time.perf_counter()

    initial_logical_calls = encoder.logical_calls
    initial_miss_calls = encoder.miss_calls
    initial_cache_hits = encoder.cache_hits
    initial_encoder_batches = encoder.encoder_batches

    remaining = tuple(range(K))
    selected_groups = []

    while remaining:

        candidates = combinations(
            remaining,
            M,
        )

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:

            score = semantic_similarity(
                encoder,
                message,
                candidate,
            )

            if score > best_score:
                best_score = score
                best_candidate = tuple(candidate)

        selected_groups.append(
            best_candidate
        )

        selected = set(best_candidate)

        remaining = tuple(
            index
            for index in remaining
            if index not in selected
        )

    partition = tuple(selected_groups)

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    return SearchResult(
        best_partition=partition,
        best_score=best_score,
        objective="tss",
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
        encoder_batches=(
        encoder.encoder_batches
        - initial_encoder_batches
        ),
    )