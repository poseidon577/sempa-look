import time
from itertools import combinations

import numpy as np

from sempalab.metrics import batch_semantic_similarity
from sempalab.types import SearchResult


def _rss_index_set(candidate, all_indices):
    """
    RSS reconstruction set.

    For candidate C:
        RSS(C, W) = phi(W \\ C, W)

    Returns the token indices remaining after removing C.
    """
    candidate_set = set(candidate)

    return tuple(
        index
        for index in all_indices
        if index not in candidate_set
    )


def _sample_disjoint_lookaheads(remaining, candidate, M, k, rng):
    """
    Sample k mutually disjoint lookahead groups of size M.

    Lookaheads are sampled from:

        remaining \\ candidate

    The groups are mutually disjoint.
    """

    if k == 0:
        return []

    candidate_set = set(candidate)

    available = [
        index
        for index in remaining
        if index not in candidate_set
    ]

    required = k * M

    if len(available) < required:
        raise ValueError(
            f"Cannot sample {k} disjoint lookahead groups of "
            f"size {M}: only {len(available)} tokens available."
        )

    # Randomly permute the available tokens.
    shuffled = list(rng.permutation(available))

    lookaheads = []

    for j in range(k):
        start = j * M
        end = start + M

        group = tuple(
            sorted(shuffled[start:end])
        )

        lookaheads.append(group)

    return lookaheads


def sempa_look(
    message,
    encoder,
    M,
    P=10,
    k=4,
    seed=None,
):
    """
    Paper-faithful SemPA-Look.

    Parameters
    ----------
    message:
        Message being packetized.

    encoder:
        Instrumented semantic encoder.

    M:
        Subpacket size.

    P:
        Number of candidate subpackets sampled at each depth.

    k:
        Number of mutually disjoint lookahead subpackets
        sampled for each candidate.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    SearchResult
        best_partition contains the packetization produced by
        SemPA-Look.

    Notes
    -----
    Candidate sampling:
        With replacement across candidates.

    Lookahead sampling:
        Mutually disjoint groups within each candidate.

    Surrogate:
        RSS:

            psi(C, W) = phi(W \\ C, W)

    Candidate score:

            Psi(C) =
                [psi(C,W) + sum_j psi(C_hat_j,W)] / (k+1)
    """

    K = len(message.tokens)

    # -------------------------
    # Validation
    # -------------------------

    if K == 0:
        raise ValueError("Message must contain at least one token.")

    if M <= 0:
        raise ValueError("M must be positive.")

    if K % M != 0:
        raise ValueError("M must divide K exactly.")

    if P <= 0:
        raise ValueError("P must be positive.")

    if k < 0:
        raise ValueError("k must be non-negative.")

    N = K // M

    # At the first decision depth, there are N-1
    # future groups available after selecting a candidate.

    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    # Snapshot instrumentation counters.
    initial_logical_calls = encoder.logical_calls
    initial_miss_calls = encoder.miss_calls
    initial_cache_hits = encoder.cache_hits
    initial_encoder_batches = encoder.encoder_batches

    all_indices = tuple(range(K))
    remaining = all_indices

    selected_groups = []

    # ---------------------------------------------------------
    # Main depth-wise lookahead search.
    #
    # We make N-1 lookahead decisions.
    # The final remaining group is committed directly.
    # ---------------------------------------------------------

    while len(remaining) > M:

        # -----------------------------------------------------
        # 1. Candidate Generation
        #
        # Sample P candidate subpackets of size M.
        #
        # Sampling is WITH replacement across candidates.
        # -----------------------------------------------------

        candidate_pool = list(
            combinations(remaining, M)
        )

        sampled_indices = rng.integers(
            low=0,
            high=len(candidate_pool),
            size=P,
        )

        candidates = [
            tuple(candidate_pool[index])
            for index in sampled_indices
        ]

        # -----------------------------------------------------
        # 2. Lookahead Search
        # -----------------------------------------------------

        evaluation_sets = []
        candidate_lookahead_counts = []

        for candidate in candidates:

            # Current candidate RSS reconstruction:
            #
            # phi(W \ C, W)
            candidate_rss_set = _rss_index_set(
                candidate,
                all_indices,
            )

            evaluation_sets.append(candidate_rss_set)

            # Number of future subpackets available after selecting C.
            future_groups = (len(remaining) // M) - 1

            # The requested k cannot exceed the number of future groups.
            effective_k = min(k, future_groups)

            lookaheads = _sample_disjoint_lookaheads(
                remaining=remaining,
                candidate=candidate,
                M=M,
                k=effective_k,
                rng=rng,
            )

            candidate_lookahead_counts.append(
                len(lookaheads)
            )

            # Each lookahead is also evaluated using RSS:
            #
            # phi(W \ C_hat, W)
            for lookahead in lookaheads:

                lookahead_rss_set = _rss_index_set(
                    lookahead,
                    all_indices,
                )

                evaluation_sets.append(
                    lookahead_rss_set
                )

        # -----------------------------------------------------
        # 3. Batch RSS Evaluation
        #
        # There are exactly P(k+1) semantic evaluations.
        # They are sent through the encoder in one batch.
        # -----------------------------------------------------

        scores = batch_semantic_similarity(
            encoder=encoder,
            message=message,
            surviving_index_sets=evaluation_sets,
        )

        # -----------------------------------------------------
        # 4. Average RSS Computation
        # -----------------------------------------------------

        candidate_scores = []

        offset = 0

        for number_of_lookaheads in candidate_lookahead_counts:

            count = number_of_lookaheads + 1

            candidate_score = float(
                np.mean(
                    scores[offset:offset + count]
                )
            )

            candidate_scores.append(candidate_score)

            offset += count

        # Sanity check:
        #
        # Every candidate should have exactly k lookaheads,
        # hence P(k+1) evaluations.
        expected_evaluations = P * (effective_k + 1)

        assert len(evaluation_sets) == expected_evaluations
        assert len(scores) == expected_evaluations
        assert len(candidate_scores) == P

        # -----------------------------------------------------
        # 5. Candidate Selection
        # -----------------------------------------------------

        best_candidate_index = int(
            np.argmax(candidate_scores)
        )

        best_candidate = candidates[
            best_candidate_index
        ]

        best_score = candidate_scores[
            best_candidate_index
        ]

        # Commit selected candidate.
        selected_groups.append(best_candidate)

        selected_set = set(best_candidate)

        remaining = tuple(
            index
            for index in remaining
            if index not in selected_set
        )

    # ---------------------------------------------------------
    # Final group
    #
    # No lookahead is possible/necessary.
    # ---------------------------------------------------------

    if remaining:
        assert len(remaining) == M

        selected_groups.append(
            tuple(sorted(remaining))
        )

    partition = tuple(selected_groups)

    # ---------------------------------------------------------
    # Instrumentation
    # ---------------------------------------------------------

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    logical_calls = (
        encoder.logical_calls
        - initial_logical_calls
    )

    miss_calls = (
        encoder.miss_calls
        - initial_miss_calls
    )

    cache_hits = (
        encoder.cache_hits
        - initial_cache_hits
    )

    return SearchResult(
        best_partition=partition,
        best_score=float(best_score),
        objective="rss-lookahead",
        partitions_evaluated=1,
        logical_calls=logical_calls,
        miss_calls=miss_calls,
        cache_hits=cache_hits,
        runtime_seconds=runtime_seconds,
        encoder_batches=(
        encoder.encoder_batches
        - initial_encoder_batches
        ),
    )