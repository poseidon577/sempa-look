"""
Delay accounting for packet interleaving (docs/channel_aware.md, sec. 3).

In the diagonal layout (`aggregation.aggregate_diagonal`) slot n of a
message rides in packet b + n * D, so the receiver has all N
subpackets of a message only after a span of (N - 1) * D packets.
D = 1 is the paper's layout. Interleaving deeper trades delay for
decorrelated losses; channel-aware aggregation trades encoder calls
for the same thing. These helpers put both on one axis.
"""

from __future__ import annotations

import numpy as np


def span_packets(N: int, depth: int) -> int:
    """
    Packets between a message's first and last subpacket.
    """

    if N < 1 or depth < 1:
        raise ValueError("N and depth must be positive")

    return (N - 1) * depth


def matching_depth(depths, ats, target: float) -> float:
    """
    Smallest interleaver depth at which a method whose mean ATS at the
    given depths is `ats` reaches `target`, by linear interpolation of
    ATS over log2(depth).

    Returns the first depth if the method already reaches `target`
    there, and inf if it never does within the depths given. ATS need
    not be monotone in depth; the first upward crossing is used.
    """

    depths = np.asarray(depths, dtype=float)
    ats = np.asarray(ats, dtype=float)

    if depths.ndim != 1 or depths.shape != ats.shape or len(depths) == 0:
        raise ValueError("depths and ats must be equal-length 1-D sequences")

    order = np.argsort(depths)
    depths, ats = depths[order], ats[order]

    if np.any(depths <= 0):
        raise ValueError("depths must be positive")

    if ats[0] >= target:
        return float(depths[0])

    logs = np.log2(depths)

    for i in range(1, len(depths)):
        if ats[i] >= target:
            fraction = (target - ats[i - 1]) / (ats[i] - ats[i - 1])
            return float(2 ** (logs[i - 1] + fraction * (logs[i] - logs[i - 1])))

    return float("inf")
