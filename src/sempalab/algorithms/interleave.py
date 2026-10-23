"""
Deterministic (modular) token interleaving: token i goes to
subpacket i mod N, so adjacent tokens always travel in different
packets. A channel- and content-blind baseline that costs no encoder
calls (cf. the modular interleaving of TokCode, arXiv 2604.12931).
"""

from __future__ import annotations


def interleave_partition(K: int, M: int) -> tuple[tuple[int, ...], ...]:
    """
    Slot n carries tokens n, n + N, n + 2N, ... (N = K / M).
    """

    if K <= 0 or M <= 0:
        raise ValueError("K and M must be positive")

    if K % M != 0:
        raise ValueError(f"M={M} must divide K={K}")

    N = K // M

    return tuple(tuple(range(n, K, N)) for n in range(N))
