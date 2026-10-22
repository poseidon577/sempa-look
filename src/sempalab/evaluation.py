"""
Fast exact evaluation for the Stage 4 experiments.

A `SimilarityTable` holds phi(subset, W) for one message, keyed by the
subset's bitmask, and fills itself through the encoder's batched,
cached `embed_sets`. ATS, the paper's WTS and full search are then
plain array arithmetic over the table, which keeps K=12 full search
(10,395 partitions x 64 erasure patterns) to well under a second.

These functions give the same numbers as `metrics.exact_ats`,
`metrics.wts_lost` and `algorithms.full_search` (tests check this);
they only avoid one Python call per erasure pattern.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from sempalab.metrics import EMPTY_SIMILARITY
from sempalab.partition import generate_partitions


def to_mask(indices) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def from_mask(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(mask.bit_length()) if mask >> i & 1)


class SimilarityTable:
    """
    phi(F, W) for subsets F of one message, keyed by bitmask.
    """

    def __init__(self, encoder, message, chunk: int = 4096):
        self.encoder = encoder
        self.message = message
        self.chunk = chunk
        self.original = encoder.embed_original(message.message_id, message.tokens)
        self.sims: dict[int, float] = {0: EMPTY_SIMILARITY}

    def fill(self, masks) -> None:
        missing = [mask for mask in dict.fromkeys(int(m) for m in masks) if mask not in self.sims]

        for start in range(0, len(missing), self.chunk):
            batch = missing[start:start + self.chunk]

            embeddings = self.encoder.embed_sets(
                [from_mask(mask) for mask in batch],
                self.message.message_id,
                self.message.tokens,
            )

            for mask, embedding in zip(batch, embeddings):
                self.sims[mask] = float(np.dot(self.original, embedding))

    def lookup(self, masks) -> np.ndarray:
        masks = np.asarray(masks, dtype=np.int64)

        self.fill(np.unique(masks).tolist())

        flat = np.fromiter((self.sims[int(m)] for m in masks.ravel()), dtype=np.float64, count=masks.size)

        return flat.reshape(masks.shape)


def _state_matrix(N: int) -> np.ndarray:
    """
    (2^N, N) 0/1 matrix: row s marks which subpackets survive in pattern s.
    """

    return ((np.arange(1 << N)[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int64)


def _state_probabilities(N: int, p: float) -> np.ndarray:
    survived = _state_matrix(N).sum(axis=1)
    return (1 - p) ** survived * p ** (N - survived)


def ats(table: SimilarityTable, partition, p: float) -> float:
    block_masks = np.array([to_mask(group) for group in partition], dtype=np.int64)

    states = _state_matrix(len(partition)) @ block_masks

    return float(table.lookup(states) @ _state_probabilities(len(partition), p))


def wts(table: SimilarityTable, partition, L: int) -> float:
    """
    Paper WTS for one message: min over losing exactly L subpackets.
    """

    N = len(partition)

    if not 0 <= L <= N:
        raise ValueError(f"L must be between 0 and N={N}")

    full = to_mask(i for group in partition for i in group)
    block_masks = [to_mask(group) for group in partition]

    survivors = [
        full & ~sum(block_masks[n] for n in lost)
        for lost in combinations(range(N), L)
    ]

    return float(table.lookup(survivors).min())


def full_search(table: SimilarityTable, K: int, M: int, p: float):
    """
    Exhaustive ATS-optimal partition.

    Returns (partition, best_ats, partitions_evaluated, logical_calls),
    where logical_calls = partitions * 2^N, the same count the
    reference `algorithms.full_search` reports.
    """

    partitions = list(generate_partitions(K, M))
    N = K // M

    block_masks = np.array(
        [[to_mask(group) for group in partition] for partition in partitions],
        dtype=np.int64,
    )

    states = block_masks @ _state_matrix(N).T  # (partitions, 2^N)

    scores = table.lookup(states) @ _state_probabilities(N, p)

    best = int(np.argmax(scores))

    return partitions[best], float(scores[best]), len(partitions), len(partitions) * (1 << N)
