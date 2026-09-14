from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from sempalab.metrics import exact_ats
from sempalab.partition import validate_partition
from sempalab.types import SearchResult


def _random_partition(K: int, M: int, rng: np.random.Generator):
    """
    Generate a uniformly shuffled valid partition.

    The chromosome is represented as a tuple of groups.
    """
    indices = list(rng.permutation(K))

    groups = []

    for start in range(0, K, M):
        group = tuple(sorted(indices[start:start + M]))
        groups.append(group)

    return tuple(groups)


def _canonicalize_partition(partition):
    """
    Put groups into deterministic order.

    Group order does not affect the actual partition.
    """
    return tuple(sorted(
        tuple(sorted(group))
        for group in partition
    ))


def _crossover(parent_a, parent_b, K, M, rng):
    """
    Partition-aware crossover.

    Start with complete groups from parent A, then fill the
    remaining token positions using groups from parent B.
    Any remaining positions are completed arbitrarily.

    The result is always a valid partition.
    """

    N = K // M

    # Randomly choose how many groups to inherit from parent A.
    num_from_a = int(rng.integers(1, N))

    selected_a_indices = rng.choice(
        N,
        size=num_from_a,
        replace=False,
    )

    child_groups = []
    used = set()

    for index in selected_a_indices:
        group = tuple(parent_a[index])

        child_groups.append(group)
        used.update(group)

    # Try to inherit compatible groups from parent B.
    b_order = rng.permutation(N)

    for index in b_order:
        group = tuple(parent_b[index])

        if all(token not in used for token in group):
            child_groups.append(group)
            used.update(group)

        if len(child_groups) == N:
            break

    # Fill any remaining positions.
    remaining = [
        token
        for token in range(K)
        if token not in used
    ]

    rng.shuffle(remaining)

    while remaining:
        group = tuple(
            sorted(remaining[:M])
        )

        child_groups.append(group)
        used.update(group)
        remaining = remaining[M:]

    child = _canonicalize_partition(child_groups)

    assert validate_partition(child, K, M)

    return child


def _mutate(partition, K, M, rng):
    """
    Mutation that preserves partition validity.

    Select two groups and swap one token between them.
    """

    groups = [
        list(group)
        for group in partition
    ]

    N = len(groups)

    if N < 2:
        return partition

    group_a, group_b = rng.choice(
        N,
        size=2,
        replace=False,
    )

    index_a = int(
        rng.integers(len(groups[group_a]))
    )

    index_b = int(
        rng.integers(len(groups[group_b]))
    )

    groups[group_a][index_a], groups[group_b][index_b] = (
        groups[group_b][index_b],
        groups[group_a][index_a],
    )

    mutated = _canonicalize_partition(groups)

    assert validate_partition(mutated, K, M)

    return mutated


def _tournament_select(
    population,
    fitnesses,
    tournament_size,
    rng,
):
    """
    Tournament selection.
    """

    population_size = len(population)

    indices = rng.integers(
        0,
        population_size,
        size=tournament_size,
    )

    best_index = max(
        indices,
        key=lambda index: fitnesses[int(index)],
    )

    return population[int(best_index)]


def genetic(
    message,
    encoder,
    M,
    objective="ats",
    p=0.0,
    P=10,
    G=5,
    mutation_rate=0.1,
    tournament_size=3,
    seed=None,
):
    """
    Genetic Algorithm baseline.

    Fitness is exact ATS.

    Parameters
    ----------
    message:
        Message being packetized.

    encoder:
        InstrumentedEncoder.

    M:
        Subpacket size.

    objective:
        Currently only "ats" is supported because GA fitness
        must use exact ATS for the Stage-2 baseline.

    p:
        Packet loss probability.

    P:
        Population size.

    G:
        Number of generations.

    mutation_rate:
        Probability of mutating a child.

    tournament_size:
        Tournament size for parent selection.

    seed:
        Random seed.
    """

    K = len(message.tokens)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if K == 0:
        raise ValueError(
            "Message must contain at least one token."
        )

    if M <= 0:
        raise ValueError(
            "M must be positive."
        )

    if K % M != 0:
        raise ValueError(
            "M must divide K exactly."
        )

    if objective != "ats":
        raise ValueError(
            "Genetic Algorithm fitness must use exact ATS."
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            "p must be between 0 and 1."
        )

    if P <= 0:
        raise ValueError(
            "Population size P must be positive."
        )

    if G < 0:
        raise ValueError(
            "Number of generations G cannot be negative."
        )

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "mutation_rate must be between 0 and 1."
        )

    if tournament_size <= 0:
        raise ValueError(
            "tournament_size must be positive."
        )

    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    initial_logical_calls = encoder.logical_calls
    initial_miss_calls = encoder.miss_calls
    initial_cache_hits = encoder.cache_hits
    initial_encoder_batches = encoder.encoder_batches

    # --------------------------------------------------------
    # Initial population
    # --------------------------------------------------------

    population = [
        _random_partition(
            K,
            M,
            rng,
        )
        for _ in range(P)
    ]

    # --------------------------------------------------------
    # Fitness evaluation
    # --------------------------------------------------------

    def evaluate_population(population):
        fitnesses = []

        for partition in population:

            assert validate_partition(
                partition,
                K,
                M,
            )

            score = exact_ats(
                partition=partition,
                message=message,
                encoder=encoder,
                p=p,
            )

            fitnesses.append(score)

        return fitnesses

    fitnesses = evaluate_population(
        population
    )

    # --------------------------------------------------------
    # Track global best
    # --------------------------------------------------------

    best_index = int(
        np.argmax(fitnesses)
    )

    best_partition = population[best_index]
    best_score = float(
        fitnesses[best_index]
    )

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    for _ in range(G):

        new_population = [
            best_partition
        ]

        while len(new_population) < P:

            parent_a = _tournament_select(
                population,
                fitnesses,
                tournament_size,
                rng,
            )

            parent_b = _tournament_select(
                population,
                fitnesses,
                tournament_size,
                rng,
            )

            child = _crossover(
                parent_a,
                parent_b,
                K,
                M,
                rng,
            )

            if rng.random() < mutation_rate:
                child = _mutate(
                    child,
                    K,
                    M,
                    rng,
                )

            assert validate_partition(
                child,
                K,
                M,
            )

            new_population.append(child)

        population = new_population

        fitnesses = evaluate_population(
            population
        )

        generation_best_index = int(
            np.argmax(fitnesses)
        )

        generation_best_score = float(
            fitnesses[generation_best_index]
        )

        if generation_best_score > best_score:
            best_score = generation_best_score
            best_partition = population[
                generation_best_index
            ]

    # --------------------------------------------------------
    # Instrumentation
    # --------------------------------------------------------

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    return SearchResult(
        best_partition=_canonicalize_partition(
            best_partition
        ),
        best_score=float(best_score),
        objective="ats",
        partitions_evaluated=(
            P * (G + 1)
        ),
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
        encoder_batches=(
            encoder.encoder_batches
            - initial_encoder_batches
        ),
        runtime_seconds=runtime_seconds,
    )