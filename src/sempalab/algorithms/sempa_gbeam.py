"""
SemPA-GBeam: semantic packet aggregation via genetic beam search.

Baseline of the TCOM paper (Sec. V, ref. [29]); the algorithm itself is
specified in the earlier conference paper, Lee, Park, Choi, Park,
"Semantic Packet Aggregation for Token Communication via Genetic Beam
Search" (arXiv:2504.19591), Sec. III-B, steps 1-4:

1. Initialization: L random groupings (valid partitions into N
   subpackets of size M).
2. Fitness: exact ATS over all 2^N erasure patterns.
3. Mutation: from each of the top-B groupings (the beams), create L/B
   children by choosing two distinct subpackets and swapping one
   randomly chosen token between them. No crossover.
4. Selection: keep the top B of the L new children as the next beams.

Steps 2-4 repeat for G generations. Counting the initial population as
the first generation, as for GA (assumption P6), the encoder cost is
G * L * 2^N, the closed form both papers give.

Choices the papers leave open (docs/assumptions.md, A13-A15):
- Selection pool: the L children only, as eq. (11) reads; parents are
  not carried over.
- Returned grouping: the best one evaluated in any generation. This
  needs no extra encoder calls, and only differs from "best of the last
  beam" when a generation of children is worse than its parents.
- If L is not a multiple of B, the first L mod B beams get one extra
  child, so every generation still evaluates exactly L groupings.
"""

from __future__ import annotations

import time

import numpy as np

from sempalab.algorithms.genetic import (
    _canonicalize_partition,
    _mutate,
    _random_partition,
)
from sempalab.metrics import exact_ats
from sempalab.partition import validate_partition
from sempalab.types import SearchResult


def _children_per_beam(L: int, B: int) -> list[int]:
    base, extra = divmod(L, B)
    return [base + (1 if b < extra else 0) for b in range(B)]


def sempa_gbeam(
    message,
    encoder,
    M,
    p=0.0,
    L=10,
    B=2,
    G=5,
    seed=None,
):
    """
    Parameters
    ----------
    message, encoder:
        Message and InstrumentedEncoder (or any encoder with the same
        interface).
    M:
        Subpacket size; must divide K.
    p:
        Subpacket erasure probability used in the ATS fitness.
    L:
        Population size: groupings evaluated per generation (the TCOM
        paper's P).
    B:
        Beam width (2 in the GBeam paper's simulations).
    G:
        Generations, counting the initial random population. G = 1 is
        "best of L random partitions".
    seed:
        Random seed.
    """

    K = len(message.tokens)

    if K == 0:
        raise ValueError("Message must contain at least one token.")
    if M <= 0 or K % M != 0:
        raise ValueError("M must be positive and divide K exactly.")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1.")
    if L <= 0:
        raise ValueError("Population size L must be positive.")
    if not 1 <= B <= L:
        raise ValueError("Beam width B must satisfy 1 <= B <= L.")
    if G < 1:
        raise ValueError("G must be at least 1.")

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    initial = (
        encoder.logical_calls,
        encoder.miss_calls,
        encoder.cache_hits,
        encoder.encoder_batches,
    )

    def evaluate(population):
        # Encode every surviving set of the generation in one batch so
        # exact_ats only hits the cache; logical calls are unchanged.
        sets = {
            tuple(sorted(
                index
                for n, group in enumerate(partition)
                if mask >> n & 1
                for index in group
            ))
            for partition in population
            for mask in range(1, 1 << len(partition))
        }
        encoder.embed_sets(sorted(sets), message.message_id, message.tokens)

        return [exact_ats(partition, message, encoder, p) for partition in population]

    # Generation 1: L random groupings.
    population = [_canonicalize_partition(_random_partition(K, M, rng)) for _ in range(L)]
    fitnesses = evaluate(population)

    best_index = int(np.argmax(fitnesses))
    best_partition, best_score = population[best_index], float(fitnesses[best_index])

    for _ in range(G - 1):
        # Top-B beams of the current generation (stable for ties).
        order = np.argsort(-np.asarray(fitnesses), kind="stable")[:B]
        beams = [population[int(i)] for i in order]

        population = [
            _mutate(beam, K, M, rng)
            for beam, count in zip(beams, _children_per_beam(L, B))
            for _ in range(count)
        ]

        assert len(population) == L
        assert all(validate_partition(child, K, M) for child in population)

        fitnesses = evaluate(population)

        index = int(np.argmax(fitnesses))
        if fitnesses[index] > best_score:
            best_partition, best_score = population[index], float(fitnesses[index])

    return SearchResult(
        best_partition=_canonicalize_partition(best_partition),
        best_score=best_score,
        objective="ats",
        partitions_evaluated=L * G,
        logical_calls=encoder.logical_calls - initial[0],
        miss_calls=encoder.miss_calls - initial[1],
        cache_hits=encoder.cache_hits - initial[2],
        encoder_batches=encoder.encoder_batches - initial[3],
        runtime_seconds=time.perf_counter() - start_time,
    )
