import numpy as np
import pytest

from sempalab.algorithms.mcts_uct import mcts_uct
from sempalab.algorithms.sempa_look import sempa_look
from sempalab.partition import validate_partition
from sempalab.repro import sempa_look_calls
from sempalab.types import Message


WORDS = (
    "to keep your garden healthy water the young plants early in the morning "
    "and check the soil every few days for pests and weeds"
).split()[:24]
MESSAGE = Message(message_id=3, text=" ".join(WORDS), tokens=tuple(WORDS))


def test_sempa_look_call_formula_matches_measured(fake_encoder):
    # 190 at the Table II setting; the formula must match what SemPA-Look does.
    assert sempa_look_calls(24, 4, 10, 4) == 190
    assert sempa_look_calls(24, 4, 10, 4, "independent") == 250
    assert sempa_look_calls(24, 12, 130, 4) == 260

    for M, P, k in [(4, 10, 4), (6, 10, 4), (3, 5, 2), (12, 130, 4)]:
        result = sempa_look(MESSAGE, fake_encoder(), M, P=P, k=k, seed=0)
        assert result.logical_calls == sempa_look_calls(24, M, P, k)


@pytest.mark.parametrize("budget", [0, 6, 50, 190, 250])
def test_respects_budget_and_returns_valid_partition(fake_encoder, budget):
    result = mcts_uct(MESSAGE, fake_encoder(), 4, budget=budget, seed=1)

    assert result.logical_calls <= budget
    assert validate_partition(result.best_partition, 24, 4)


def test_uses_most_of_the_budget(fake_encoder):
    # A rollout costs at most N new psi values, so at most N calls go unused.
    result = mcts_uct(MESSAGE, fake_encoder(), 4, budget=190, seed=2)

    assert result.logical_calls >= 190 - 24 // 4


def test_reproducible_with_seed(fake_encoder):
    a = mcts_uct(MESSAGE, fake_encoder(), 4, budget=100, seed=9)
    b = mcts_uct(MESSAGE, fake_encoder(), 4, budget=100, seed=9)

    assert a.best_partition == b.best_partition
    assert a.logical_calls == b.logical_calls


def test_small_action_space_terminates(fake_encoder):
    # K=6, M=2: 15 root actions; memoized rollouts become free and must not loop.
    words = WORDS[:6]
    message = Message(message_id=4, text=" ".join(words), tokens=tuple(words))

    result = mcts_uct(message, fake_encoder(), 2, budget=10_000, seed=0)

    assert validate_partition(result.best_partition, 6, 2)
    # Only C(6,2) + ... distinct subpackets exist: 15 psi values at most.
    assert result.logical_calls <= 15


def test_trivial_sizes(fake_encoder):
    whole = mcts_uct(MESSAGE, fake_encoder(), 24, budget=50, seed=0)
    assert whole.best_partition == (tuple(range(24)),)
    assert whole.logical_calls == 0

    singles = mcts_uct(MESSAGE, fake_encoder(), 1, budget=50, seed=0)
    assert validate_partition(singles.best_partition, 24, 1)


def test_search_improves_rss_reward_over_random(fake_encoder):
    # With a real budget the committed partition should score better on the
    # MCTS reward (mean RSS over subpackets) than a random partition.
    from sempalab.metrics import batch_semantic_similarity

    def reward(partition, encoder):
        survivors = [tuple(i for i in range(24) if i not in set(g)) for g in partition]
        return float(np.mean(batch_semantic_similarity(encoder, MESSAGE, survivors)))

    rng = np.random.default_rng(0)
    encoder = fake_encoder()

    searched = np.mean([reward(mcts_uct(MESSAGE, fake_encoder(), 4, budget=400, seed=s).best_partition, encoder)
                        for s in range(5)])
    random = np.mean([reward(tuple(tuple(sorted(int(i) for i in row)) for row in rng.permutation(24).reshape(6, 4)), encoder)
                      for _ in range(30)])

    assert searched > random


@pytest.mark.parametrize("kwargs", [dict(budget=-1), dict(c=-0.1), dict(widening=0), dict(widening=1.5)])
def test_rejects_bad_parameters(fake_encoder, kwargs):
    with pytest.raises(ValueError):
        mcts_uct(MESSAGE, fake_encoder(), 4, **kwargs)
