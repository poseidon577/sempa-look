"""
SemPA-Look samples candidate subpackets by unranking a uniform index
instead of listing every M-subset, so it runs at K = 48, M = 8. The
draws, and therefore the partitions, are unchanged.
"""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from sempalab.algorithms.sempa_look import sempa_look, unrank_combination
from sempalab.types import Message


def test_unrank_combination_matches_itertools_order():
    for pool, M in [((0, 1, 2, 3, 4), 2), ((3, 5, 8, 9, 11, 12, 20), 3), (tuple(range(8)), 8), (tuple(range(6)), 1)]:
        listed = list(combinations(pool, M))
        assert [unrank_combination(pool, M, i) for i in range(comb(len(pool), M))] == listed

    with pytest.raises(IndexError):
        unrank_combination((0, 1, 2), 2, 3)


def test_unrank_combination_handles_large_pools():
    pool = tuple(range(48))
    assert unrank_combination(pool, 8, 0) == tuple(range(8))
    assert unrank_combination(pool, 8, comb(48, 8) - 1) == tuple(range(40, 48))
    assert len(set(unrank_combination(pool, 8, comb(48, 8) // 3))) == 8


def _listing_candidates(remaining, M, rng, P):
    """
    The previous implementation's candidate draw.
    """

    pool = list(combinations(remaining, M))
    return [tuple(pool[i]) for i in rng.integers(low=0, high=len(pool), size=P)]


def test_candidate_draws_match_listing_implementation():
    remaining = (0, 2, 3, 5, 7, 8, 10, 11, 13, 14, 15, 17)
    for M in (2, 3, 4):
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        expected = _listing_candidates(remaining, M, a, 10)
        got = [unrank_combination(remaining, M, int(i)) for i in b.integers(low=0, high=comb(len(remaining), M), size=10)]
        assert got == expected
        # the two generators stay in lockstep afterwards
        assert a.random() == b.random()


def test_sempa_look_runs_at_k48(fake_encoder):
    tokens = tuple(f"w{i}" for i in range(48))
    message = Message(0, " ".join(tokens), tokens)
    result = sempa_look(message, fake_encoder(), 8, P=10, k=4, seed=0)
    assert sorted(i for g in result.best_partition for i in g) == list(range(48))
    assert all(len(g) == 8 for g in result.best_partition)
