from __future__ import annotations

import numpy as np


def sample_survival(
    N: int,
    p: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample which of N packets survive.

    Parameters
    ----------
    N:
        Number of packets.

    p:
        Probability that a packet is erased.

    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Boolean array of length N.
        True  = survived
        False = erased
    """

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    if rng is None:
        rng = np.random.default_rng()

    return rng.random(N) >= p


def survival_probability(
    surviving_count: int,
    N: int,
    p: float,
) -> float:
    """
    Probability that exactly `surviving_count`
    packets survive out of N.
    """

    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    if not 0 <= surviving_count <= N:
        return 0.0

    # P(H survives) =
    # C(N, h) (1-p)^h p^(N-h)

    from math import comb

    h = surviving_count

    return (
        comb(N, h)
        * (1 - p) ** h
        * p ** (N - h)
    )