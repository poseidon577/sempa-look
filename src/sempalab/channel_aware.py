"""
Channel-aware semantic packet aggregation (CASA).

Under the paper's i.i.d. erasures only the partition matters: every
slot (packet position) fails alike, so ATS is invariant to which group
rides in which slot. On a channel with memory (`channel.correlated`)
the N slots of a message fail jointly with law Q, and both the
partition and the slot order change ATS:

    ATS_Q(G) = sum_s Q[s] * phi(union of the groups in surviving slots).

This module provides

  * exact ATS under any Q, the best slot order for a fixed partition
    (2^N encoder calls, then free), and an exhaustive oracle over
    ordered partitions (feasible at K = 12);

  * a p- and channel-aware pairwise surrogate (direction 3 of
    docs/limitations.md): phi is approximated by an additive-plus-
    pairwise model over token survival indicators x,

        phi(x) ~ a0 + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j,

    fitted by least squares on random subsets whose sizes follow the
    channel's law of the number of surviving slots. Taking expectations
    under Q, with token i in slot g(i),

        ATS_Q ~ a0 + sum_i a_i r[g(i)] + sum_{i<j} b_ij R[g(i), g(j)],

    where r and R are the slot survival and pairwise joint survival
    probabilities of Q. The channel enters only through (r, R), and the
    search over labelled assignments becomes a quadratic assignment
    problem that needs no encoder calls;

  * `casa`: fit the surrogate, run swap local search on it from random
    restarts, then re-rank the best distinct local optima by exact
    ATS_Q with their best slot order.

Encoder accounting follows the paper's convention of one logical call
per similarity evaluation, counted here as distinct non-empty subsets
an algorithm asks for (repeats are free; phi of the empty set is 0 and
needs no call).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import comb

import numpy as np

from sempalab.channel.correlated import state_matrix, survival_moments
from sempalab.evaluation import SimilarityTable, to_mask
from sempalab.partition import generate_partitions


# -------------------------------------------------------------
# Exact evaluation under a channel law Q
# -------------------------------------------------------------


def union_masks(partition) -> np.ndarray:
    """
    Bitmask of the surviving token set for each of the 2^N slot patterns.
    """

    blocks = np.array([to_mask(group) for group in partition], dtype=np.int64)
    return state_matrix(len(partition)) @ blocks


def channel_ats(table: SimilarityTable, partition, Q: np.ndarray) -> float:
    """
    Exact ATS of a partition whose group n rides in slot n.
    """

    return float(table.lookup(union_masks(partition)) @ Q)


def slot_orders(N: int) -> list[tuple[int, ...]]:
    return list(permutations(range(N)))


def permuted_laws(Q: np.ndarray, N: int) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """
    For every slot order (order[n] = group placed in slot n), the law of
    the *group* survival pattern: column k of the (2^N, N!) result.
    """

    S = state_matrix(N)
    orders = slot_orders(N)

    columns = []

    for order in orders:
        # group pattern s -> slot pattern t with t_n = s_{order[n]}
        slot_index = (S[:, list(order)] << np.arange(N)).sum(axis=1)
        columns.append(Q[slot_index])

    return orders, np.stack(columns, axis=1)


def best_slot_order(table: SimilarityTable, partition, Q: np.ndarray):
    """
    Reorder the groups of `partition` to maximise exact ATS under Q.

    Returns (reordered partition, ATS). Needs only the 2^N similarities
    of the partition's unions, whatever the order.
    """

    N = len(partition)
    orders, laws = permuted_laws(Q, N)

    phi = table.lookup(union_masks(partition))
    scores = phi @ laws

    best = int(np.argmax(scores))

    return tuple(partition[g] for g in orders[best]), float(scores[best])


class OrderedOracle:
    """
    Exhaustive optimum over ordered partitions, for many channel laws.

    The 2^N union similarities of every unordered partition are looked
    up once (K = 12, M = 2: 10,395 x 64); each law Q then costs one
    matrix product over all N! slot orders.
    """

    def __init__(self, table: SimilarityTable, K: int, M: int):
        self.N = K // M
        self.partitions = list(generate_partitions(K, M))

        blocks = np.array([[to_mask(group) for group in partition] for partition in self.partitions], dtype=np.int64)
        self.phi = table.lookup(blocks @ state_matrix(self.N).T)  # (partitions, 2^N)

    def best(self, Q: np.ndarray):
        """
        (partition in slot order, ATS, ordered partitions evaluated).
        """

        orders, laws = permuted_laws(Q, self.N)
        scores = self.phi @ laws  # (partitions, N!)

        best_partition, best_order = np.unravel_index(int(np.argmax(scores)), scores.shape)
        partition = self.partitions[best_partition]

        return (
            tuple(partition[g] for g in orders[best_order]),
            float(scores[best_partition, best_order]),
            scores.size,
        )


def full_search_channel(table: SimilarityTable, K: int, M: int, Q: np.ndarray):
    """
    Exhaustive optimum over ordered partitions under Q.

    Returns (partition in slot order, ATS, ordered partitions evaluated).
    """

    return OrderedOracle(table, K, M).best(Q)


# -------------------------------------------------------------
# Pairwise surrogate
# -------------------------------------------------------------


def survivor_count_law(Q: np.ndarray) -> np.ndarray:
    """
    P(exactly c of the N slots survive), c = 0..N.
    """

    N = int(round(np.log2(len(Q))))
    counts = state_matrix(N).sum(axis=1)
    return np.bincount(counts, weights=Q, minlength=N + 1)


def fitting_law(Q: np.ndarray, partial_only: bool = True) -> np.ndarray:
    """
    Survivor-count law the surrogate is fitted on: the channel's law,
    conditioned on 1 <= c <= N - 1 when `partial_only`.
    """

    law = survivor_count_law(Q)

    if partial_only and len(law) > 2:
        law = law.copy()
        law[0] = law[-1] = 0.0

    return law / law.sum()


def pairwise_features(X: np.ndarray) -> np.ndarray:
    """
    [1, x_i, x_i x_j for i < j] for each row of the 0/1 matrix X.
    """

    K = X.shape[1]
    rows, cols = np.triu_indices(K, k=1)

    return np.hstack([np.ones((X.shape[0], 1)), X, X[:, rows] * X[:, cols]])


def bits(masks, K: int) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int64)
    return ((masks[:, None] >> np.arange(K)[None, :]) & 1).astype(float)


@dataclass
class PairwiseSurrogate:
    """
    phi(x) ~ a0 + a . x + sum_{i<j} B[i, j] x_i x_j  (B symmetric, zero diagonal).
    """

    a0: float
    a: np.ndarray
    B: np.ndarray

    @property
    def K(self) -> int:
        return len(self.a)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.a0 + X @ self.a + 0.5 * np.einsum("ni,ij,nj->n", X, self.B, X)

    def expected(self, slots: np.ndarray, r: np.ndarray, R: np.ndarray) -> float:
        """
        Surrogate ATS of the labelled assignment `slots` (token -> slot).
        """

        slots = np.asarray(slots)
        joint = R[slots[:, None], slots[None, :]]
        return float(self.a0 + self.a @ r[slots] + 0.5 * np.sum(self.B * joint))


def sample_pool(K: int, M: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Masks with survivor count uniform on 1..N-1, for `fit_pairwise(pool=...)`.
    """

    N = K // M
    law = np.zeros(N + 1)
    law[1:N] = 1.0
    return sample_subsets(K, M, law, size, rng)


def sample_subsets(K: int, M: int, count_law: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    `size` random token subsets, as bitmasks: the number of surviving
    slots c is drawn from `count_law`, then a uniform subset of c * M
    tokens. Under a uniformly random partition this is exactly the law
    of the surviving token set, so the fit targets the subsets that ATS
    integrates over.
    """

    counts = rng.choice(len(count_law), size=size, p=count_law / count_law.sum())

    masks = np.empty(size, dtype=np.int64)

    for n, c in enumerate(counts):
        chosen = rng.choice(K, size=int(c) * M, replace=False)
        masks[n] = int(np.sum(1 << chosen.astype(np.int64))) if c else 0

    return masks


def fit_pairwise(
    table: SimilarityTable,
    K: int,
    M: int,
    Q: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    ridge: float = 1e-3,
    partial_only: bool = True,
    pool: np.ndarray | None = None,
) -> tuple[PairwiseSurrogate, set[int]]:
    """
    Least-squares pairwise surrogate from `budget` sampled subsets.

    With `partial_only` (default) the survivor count is drawn from the
    channel's law conditioned on 1 <= c <= N - 1: losing all or no
    slots adds a constant to ATS whatever the partition, so those
    subsets carry no information for the search, and a bursty channel
    would otherwise spend most of the budget on them.

    `pool` (optional) is a fixed sample of masks whose survivor counts
    are uniform on 1..N-1 (`sample_pool`). The fit then reweights it to
    the channel's law by importance weights instead of drawing a fresh
    sample, which lets many channel settings share one set of encoder
    calls in an experiment. Each fit still reports the whole pool as
    its own calls, as a standalone run would need them.

    Returns the model and the set of distinct non-empty masks it asked
    the encoder for (phi(empty) = 0 needs no call).
    """

    law = fitting_law(Q, partial_only)

    if pool is None:
        if budget < 1:
            raise ValueError("budget must be positive")
        masks = sample_subsets(K, M, law, budget, rng)
        weights = np.ones(len(masks))
    else:
        masks = np.asarray(pool, dtype=np.int64)
        counts = np.array([bin(int(m)).count("1") // M for m in masks])
        uniform = 1.0 / (len(law) - 2)
        weights = law[counts] / uniform

    y = table.lookup(masks)
    F = pairwise_features(bits(masks, K))
    Fw = F * weights[:, None]

    penalty = np.full(F.shape[1], ridge * weights.sum())
    penalty[0] = 0.0

    coef = np.linalg.solve(Fw.T @ F + np.diag(penalty), Fw.T @ y)

    rows, cols = np.triu_indices(K, k=1)
    B = np.zeros((K, K))
    B[rows, cols] = coef[K + 1:]
    B = B + B.T

    queried = {int(m) for m in masks if m}

    return PairwiseSurrogate(float(coef[0]), coef[1:K + 1].copy(), B), queried


def surrogate_r2(model: PairwiseSurrogate, table: SimilarityTable, masks) -> float:
    """
    Out-of-sample R^2 of the surrogate on the given subsets.
    """

    masks = np.asarray(list(masks), dtype=np.int64)
    y = table.lookup(masks)
    residual = y - model.predict(bits(masks, model.K))
    return float(1 - residual @ residual / np.sum((y - y.mean()) ** 2))


# -------------------------------------------------------------
# Quadratic-assignment local search on the surrogate
# -------------------------------------------------------------


def swap_gains(model: PairwiseSurrogate, slots: np.ndarray, r: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    (K, K) change in surrogate ATS from swapping the slots of tokens i
    and j; -inf where i and j share a slot.
    """

    a, B = model.a, model.B
    s = np.asarray(slots)

    C = B @ R[:, s].T  # C[i, n] = sum_j B[i, j] R[n, s_j]

    ri = r[s]
    Ci_own = C[np.arange(len(s)), s]  # C[i, s_i]
    C_cross = C[:, s]  # C_cross[i, j] = C[i, s_j]

    gains = (
        a[:, None] * (ri[None, :] - ri[:, None])
        + a[None, :] * (ri[:, None] - ri[None, :])
        + C_cross - Ci_own[:, None]
        + C_cross.T - Ci_own[None, :]
        - B * (R[s, s][None, :] + R[s, s][:, None] - 2 * R[s[:, None], s[None, :]])
    )

    gains[s[:, None] == s[None, :]] = -np.inf

    return gains


def surrogate_local_search(
    model: PairwiseSurrogate,
    M: int,
    r: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator,
    restarts: int = 20,
    max_passes: int = 1000,
    tol: float = 1e-12,
) -> list[tuple[float, np.ndarray]]:
    """
    Best-improvement swap search over labelled assignments from random
    starts. Returns (surrogate value, slots) of every local optimum.
    """

    K = model.K
    N = K // M

    optima = []

    for _ in range(restarts):
        slots = np.repeat(np.arange(N), M)[rng.permutation(K)]

        for _ in range(max_passes):
            gains = swap_gains(model, slots, r, R)
            i, j = np.unravel_index(int(np.argmax(gains)), gains.shape)

            if gains[i, j] <= tol:
                break

            slots[i], slots[j] = slots[j], slots[i]

        optima.append((model.expected(slots, r, R), slots.copy()))

    return optima


def slots_to_partition(slots: np.ndarray, N: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(i) for i in np.flatnonzero(slots == n)) for n in range(N))


@dataclass
class CasaResult:
    partition: tuple[tuple[int, ...], ...]
    ats: float
    surrogate_ats: float
    logical_calls: int
    model: PairwiseSurrogate


def casa(
    table: SimilarityTable,
    K: int,
    M: int,
    Q: np.ndarray,
    rng: np.random.Generator,
    budget: int | None = None,
    restarts: int = 20,
    top: int = 5,
    ridge: float = 1e-3,
    partial_only: bool = True,
    pool: np.ndarray | None = None,
) -> CasaResult:
    """
    Channel-aware semantic aggregation.

    1. Fit the pairwise surrogate on `budget` subsets drawn from the
       channel's partial-survival law (default: four times the number
       of model parameters, 1 + K + K(K-1)/2, i.e. 316 at K = 12).
    2. Swap local search on the closed-form surrogate ATS_Q from
       `restarts` random labelled assignments (no encoder calls).
    3. Re-rank the `top` best distinct local optima by exact ATS_Q with
       each one's best slot order (2^N calls per candidate).
    """

    N = K // M

    if budget is None:
        budget = 4 * (1 + K + comb(K, 2))

    model, queried = fit_pairwise(table, K, M, Q, budget, rng, ridge, partial_only, pool)

    r, R = survival_moments(Q)

    optima = sorted(surrogate_local_search(model, M, r, R, rng, restarts), key=lambda item: -item[0])

    candidates = []
    seen = set()

    for value, slots in optima:
        partition = slots_to_partition(slots, N)
        key = frozenset(partition)

        if key not in seen:
            seen.add(key)
            candidates.append((value, partition))

        if len(candidates) == top:
            break

    best = None

    for value, partition in candidates:
        queried.update(int(m) for m in union_masks(partition) if m)
        ordered, score = best_slot_order(table, partition, Q)

        if best is None or score > best[1]:
            best = (ordered, score, value)

    return CasaResult(
        partition=best[0],
        ats=best[1],
        surrogate_ats=best[2],
        logical_calls=len(queried),
        model=model,
    )
