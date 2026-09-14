import time
from itertools import combinations

from sempalab.metrics import semantic_similarity
from sempalab.types import SearchResult


def greedy_rss(message, encoder, M):
    """
    Greedy-RSS packetization.

    At each step:
        - Consider every M-token candidate C from the remaining tokens.
        - Score the remaining tokens W \\ C.
        - Commit the candidate with the highest RSS score.

    RSS surrogate:
        psi(C, W) = phi(W \\ C, W)
    """

    K = len(message.tokens)

    # Validation
    if K == 0:
        raise ValueError("Message must contain at least one token.")

    if M <= 0:
        raise ValueError("M must be positive.")

    if K % M != 0:
        raise ValueError("M must divide K exactly.")

    start_time = time.perf_counter()

    # Snapshot instrumentation counters
    initial_logical_calls = encoder.logical_calls
    initial_miss_calls = encoder.miss_calls
    initial_cache_hits = encoder.cache_hits
    initial_encoder_batches = encoder.encoder_batches

    remaining = tuple(range(K))
    selected_groups = []

    # Continue until all tokens have been assigned
    while remaining:

        best_candidate = None
        best_score = float("-inf")

        # Consider every possible M-token candidate
        for candidate in combinations(remaining, M):

            candidate_set = set(candidate)

            # Tokens that would remain if this candidate were removed
            remaining_after_candidate = tuple(
                index
                for index in remaining
                if index not in candidate_set
            )

            # RSS objective:
            # semantic quality of everything EXCEPT candidate C
            score = semantic_similarity(
                encoder,
                message,
                remaining_after_candidate,
            )

            if score > best_score:
                best_score = score
                best_candidate = tuple(candidate)

        # Commit the best candidate
        selected_groups.append(best_candidate)

        selected_set = set(best_candidate)

        remaining = tuple(
            index
            for index in remaining
            if index not in selected_set
        )

    runtime_seconds = time.perf_counter() - start_time

    return SearchResult(
        best_partition=tuple(selected_groups),
        best_score=best_score,
        objective="rss",
        partitions_evaluated=1,
        logical_calls=encoder.logical_calls - initial_logical_calls,
        miss_calls=encoder.miss_calls - initial_miss_calls,
        cache_hits=encoder.cache_hits - initial_cache_hits,
        runtime_seconds=runtime_seconds,
        encoder_batches=(
        encoder.encoder_batches
        - initial_encoder_batches
        ),
    )