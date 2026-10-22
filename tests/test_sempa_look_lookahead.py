import pytest

from sempalab.algorithms.sempa_look import (
    _sample_independent_lookaheads,
    sempa_look,
)
from sempalab.partition import validate_partition
from sempalab.types import Message

import numpy as np


def message(K):
    words = "a man riding a red bicycle near the sandy beach at sunset with his small brown dog today while two kids fly a big kite".split()
    tokens = tuple(words[:K])
    return Message(message_id=K, text=" ".join(tokens), tokens=tokens)


@pytest.mark.parametrize("K, M, P, k", [(8, 4, 10, 4), (24, 4, 10, 4), (12, 3, 7, 2), (12, 2, 5, 9)])
def test_independent_call_count_is_closed_form(fake_encoder, K, M, P, k):
    N = K // M

    result = sempa_look(message(K), fake_encoder(), M, P=P, k=k, seed=0, lookahead_mode="independent")

    assert result.logical_calls == (N - 1) * P * (k + 1)
    assert validate_partition(result.best_partition, K, M)


def test_table_iii_sempa_look_count(fake_encoder):
    # Paper Table III: K=24, M=4 (N=6), P=10, k=4 -> 250 logical calls.
    result = sempa_look(message(24), fake_encoder(), 4, P=10, k=4, seed=1, lookahead_mode="independent")

    assert result.logical_calls == 250


def test_disjoint_caps_k_per_depth(fake_encoder):
    # N=6, k=4: future groups per depth are 5,4,3,2,1 -> k_eff 4,4,3,2,1.
    result = sempa_look(message(24), fake_encoder(), 4, P=10, k=4, seed=1, lookahead_mode="disjoint")

    assert result.logical_calls == 10 * (5 + 5 + 4 + 3 + 2)


def test_independent_lookaheads_avoid_the_candidate():
    rng = np.random.default_rng(0)

    remaining = tuple(range(12))
    candidate = (0, 1, 2)

    lookaheads = _sample_independent_lookaheads(remaining, candidate, 3, 50, rng)

    assert len(lookaheads) == 50
    assert all(len(set(group)) == 3 for group in lookaheads)
    assert all(not set(group) & set(candidate) for group in lookaheads)
    # Independent draws: overlaps between lookaheads occur.
    assert any(set(a) & set(b) for a, b in zip(lookaheads, lookaheads[1:]))


def test_modes_agree_when_k_is_one_at_N_two(fake_encoder):
    # With N=2 the only lookahead is W \ C in both modes.
    a = sempa_look(message(8), fake_encoder(), 4, P=10, k=1, seed=3, lookahead_mode="disjoint")
    b = sempa_look(message(8), fake_encoder(), 4, P=10, k=1, seed=3, lookahead_mode="independent")

    assert a.best_partition == b.best_partition


def test_rejects_unknown_mode(fake_encoder):
    with pytest.raises(ValueError):
        sempa_look(message(8), fake_encoder(), 4, lookahead_mode="bogus")
