import numpy as np
import pytest

from sempalab.algorithms.sempa_gbeam import _children_per_beam, sempa_gbeam
from sempalab.evaluation import SimilarityTable, full_search
from sempalab.metrics import exact_ats
from sempalab.partition import validate_partition
from sempalab.types import Message


WORDS = "a man riding a red bicycle near the sandy beach at sunset".split()
MESSAGE = Message(message_id=1, text=" ".join(WORDS), tokens=tuple(WORDS))


def test_children_split_keeps_population_size():
    assert _children_per_beam(10, 2) == [5, 5]
    assert _children_per_beam(10, 3) == [4, 3, 3]
    assert sum(_children_per_beam(7, 7)) == 7


@pytest.mark.parametrize("M, L, G", [(4, 10, 5), (3, 8, 3), (2, 10, 5)])
def test_calls_follow_closed_form(fake_encoder, M, L, G):
    # G generations of L groupings, 2^N exact-ATS terms each (Table III).
    result = sempa_gbeam(MESSAGE, fake_encoder(), M, p=0.3, L=L, B=2, G=G, seed=0)

    N = len(WORDS) // M
    assert result.logical_calls == G * L * 2**N
    assert result.partitions_evaluated == G * L
    assert validate_partition(result.best_partition, len(WORDS), M)


def test_score_is_exact_ats_of_returned_partition(fake_encoder):
    result = sempa_gbeam(MESSAGE, fake_encoder(), 3, p=0.4, seed=3)

    assert result.best_score == pytest.approx(exact_ats(result.best_partition, MESSAGE, fake_encoder(), 0.4))


def test_reproducible_with_seed(fake_encoder):
    a = sempa_gbeam(MESSAGE, fake_encoder(), 3, p=0.3, seed=11)
    b = sempa_gbeam(MESSAGE, fake_encoder(), 3, p=0.3, seed=11)

    assert a.best_partition == b.best_partition


def test_more_generations_never_hurt(fake_encoder):
    # Same seed: generation 1 is shared, and the best-so-far can only rise.
    scores = [sempa_gbeam(MESSAGE, fake_encoder(), 3, p=0.3, G=G, seed=7).best_score for G in (1, 3, 8)]

    assert scores == sorted(scores)


def test_near_full_search_and_above_random_on_average(fake_encoder):
    encoder = fake_encoder()
    rng = np.random.default_rng(0)

    gaps, gains = [], []

    for message_id in range(20):
        words = [WORDS[i] for i in rng.permutation(len(WORDS))[:8]]
        message = Message(message_id=message_id, text=" ".join(words), tokens=tuple(words))

        table = SimilarityTable(encoder, message)
        _, optimum, _, _ = full_search(table, 8, 2, 0.3)

        beam = sempa_gbeam(message, encoder, 2, p=0.3, L=10, B=2, G=5, seed=message_id).best_score
        first = sempa_gbeam(message, encoder, 2, p=0.3, L=10, B=2, G=1, seed=message_id).best_score

        assert beam <= optimum + 1e-9
        gaps.append(optimum - beam)
        gains.append(beam - first)

    assert np.mean(gains) > 0
    assert np.mean(gaps) < 0.01


@pytest.mark.parametrize("kwargs", [dict(B=0), dict(B=11), dict(G=0), dict(L=0), dict(p=1.5)])
def test_rejects_bad_parameters(fake_encoder, kwargs):
    with pytest.raises(ValueError):
        sempa_gbeam(MESSAGE, fake_encoder(), 3, **{"p": 0.3, **kwargs})


def test_rejects_m_not_dividing_k(fake_encoder):
    with pytest.raises(ValueError):
        sempa_gbeam(MESSAGE, fake_encoder(), 5, p=0.3)
