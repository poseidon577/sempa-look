from __future__ import annotations

from itertools import combinations

import numpy as np

from sempalab.encoder import InstrumentedEncoder
from sempalab.types import Message


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Cosine similarity between two L2-normalized vectors.
    """
    return float(np.dot(a, b))


def semantic_similarity(
    encoder,
    message,
    surviving_indices,
):
    encoder.logical_calls += 1

    original_embedding = encoder.embed_original(
        message_id=message.message_id,
        tokens=message.tokens,
    )

    reconstructed_embedding = encoder.embed_sets(
        [tuple(sorted(surviving_indices))],
        message_id=message.message_id,
        tokens=message.tokens,
    )[0]

    return cosine_similarity(
        original_embedding,
        reconstructed_embedding,
    )

    return cosine_similarity(embeddings[0], embeddings[1])

def batch_semantic_similarity(
    encoder,
    message,
    surviving_index_sets,
):
    """
    Compute multiple semantic similarities while reusing
    the cached original-message embedding.
    """

    if not surviving_index_sets:
        return np.empty(
            0,
            dtype=np.float32,
        )

    encoder.logical_calls += len(
        surviving_index_sets
    )

    original_embedding = encoder.embed_original(
        message_id=message.message_id,
        tokens=message.tokens,
    )

    canonical_sets = [
        tuple(sorted(indices))
        for indices in surviving_index_sets
    ]

    reconstructed_embeddings = encoder.embed_sets(
        canonical_sets,
        message_id=message.message_id,
        tokens=message.tokens,
    )

    return np.array(
        [
            cosine_similarity(
                original_embedding,
                embedding,
            )
            for embedding in reconstructed_embeddings
        ],
        dtype=np.float32,
    )

def exact_ats(
    partition: tuple[tuple[int, ...], ...],
    message: Message,
    encoder: InstrumentedEncoder,
    p: float,
) -> float:
    """
    Compute exact Average Transmission Similarity (ATS)
    for a fixed partition.

    Enumerates every possible subset of surviving packets.

    P(H survives) =
        (1-p)^|H| * p^(N-|H|)
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")

    N = len(partition)

    total = 0.0

    for mask in range(1 << N):

        surviving_indices = tuple(
            index
            for packet_index, group in enumerate(partition)
            if mask & (1 << packet_index)
            for index in group
        )

        surviving_count = mask.bit_count()

        probability = (
            (1 - p) ** surviving_count
            * p ** (N - surviving_count)
        )

        similarity = semantic_similarity(
            encoder,
            message,
            surviving_indices,
        )

        total += probability * similarity

    return total


def wts(
    partition: tuple[tuple[int, ...], ...],
    message: Message,
    encoder: InstrumentedEncoder,
    p: float,
) -> float:
    """
    Compute Worst-case Transmission Similarity (WTS).

    For every possible number L of lost packets, find the
    minimum semantic similarity over all ways to lose L packets,
    then weight according to the probability of exactly L losses.

    This corresponds to the WTS definition used in the
    implementation plan.
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")

    N = len(partition)

    total = 0.0

    for losses in range(N + 1):

        best_similarity = float("inf")

        for lost_indices in combinations(
            range(N),
            losses,
        ):

            lost = set(lost_indices)

            surviving_indices = tuple(
                index
                for packet_index, group in enumerate(partition)
                if packet_index not in lost
                for index in group
            )

            similarity = semantic_similarity(
                encoder,
                message,
                surviving_indices,
            )

            best_similarity = min(
                best_similarity,
                similarity,
            )

        probability = (
            _binomial_probability(
                N,
                losses,
                p,
            )
        )

        total += probability * best_similarity

    return total


def _binomial_probability(
    N: int,
    losses: int,
    p: float,
) -> float:
    """
    Probability of exactly `losses` erasures
    among N packets.
    """

    from math import comb

    return (
        comb(N, losses)
        * p ** losses
        * (1 - p) ** (N - losses)
    )

def monte_carlo_ats(
    partition: tuple[tuple[int, ...], ...],
    message: Message,
    encoder: InstrumentedEncoder,
    p: float,
    samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Estimate ATS using Monte Carlo packet-loss sampling.
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")

    if samples <= 0:
        raise ValueError("samples must be positive")

    if rng is None:
        rng = np.random.default_rng()

    N = len(partition)

    total_similarity = 0.0

    for _ in range(samples):

        # True = packet survives
        survives = rng.random(N) >= p

        surviving_indices = tuple(
            index
            for packet_index, group in enumerate(partition)
            if survives[packet_index]
            for index in group
        )

        similarity = semantic_similarity(
            encoder,
            message,
            surviving_indices,
        )

        total_similarity += similarity

    return total_similarity / samples