import numpy as np
import pytest

from sempalab import evaluation
from sempalab.algorithms.full_search import full_search as reference_full_search
from sempalab.metrics import exact_ats, wts_lost
from sempalab.partition import sample_random_partition
from sempalab.types import Message


def message(K):
    words = "a man riding a red bicycle near the sandy beach at sunset with his dog".split()[:K]
    return Message(message_id=K, text=" ".join(words), tokens=tuple(words))


def test_mask_round_trip():
    assert evaluation.to_mask((0, 3, 5)) == 0b101001
    assert evaluation.from_mask(0b101001) == (0, 3, 5)
    assert evaluation.from_mask(0) == ()


@pytest.mark.parametrize("K, M, p", [(8, 4, 0.5), (12, 3, 0.2), (12, 2, 0.05), (6, 1, 0.9)])
def test_ats_matches_reference(fake_encoder, K, M, p):
    encoder = fake_encoder()
    table = evaluation.SimilarityTable(encoder, message(K))
    rng = np.random.default_rng(K + M)

    for _ in range(5):
        partition = sample_random_partition(K, M, rng)
        assert evaluation.ats(table, partition, p) == pytest.approx(
            exact_ats(partition, message(K), encoder, p), abs=1e-6
        )


@pytest.mark.parametrize("K, M", [(8, 2), (12, 2), (12, 4)])
def test_wts_matches_reference(fake_encoder, K, M):
    encoder = fake_encoder()
    table = evaluation.SimilarityTable(encoder, message(K))
    partition = sample_random_partition(K, M, np.random.default_rng(0))

    for L in range(K // M + 1):
        assert evaluation.wts(table, partition, L) == pytest.approx(
            wts_lost(partition, message(K), encoder, L), abs=1e-6
        )


@pytest.mark.parametrize("K, M, p", [(8, 4, 0.512), (8, 2, 0.207), (9, 3, 0.0557)])
def test_full_search_matches_reference(fake_encoder, K, M, p):
    encoder = fake_encoder()
    table = evaluation.SimilarityTable(encoder, message(K))

    partition, score, evaluated, calls = evaluation.full_search(table, K, M, p)
    reference = reference_full_search(message(K), encoder, M, objective="ats", p=p)

    assert score == pytest.approx(reference.best_score, abs=1e-6)
    assert evaluated == reference.partitions_evaluated
    assert calls == reference.logical_calls
    assert evaluation.ats(table, partition, p) == pytest.approx(reference.best_score, abs=1e-6)


def test_table_encodes_each_subset_once(fake_encoder):
    encoder = fake_encoder()
    table = evaluation.SimilarityTable(encoder, message(8))

    evaluation.full_search(table, 8, 4, 0.3)
    misses = encoder.miss_calls

    # 70 four-token subsets + the full message; the empty set is never encoded.
    assert misses == 71

    evaluation.full_search(table, 8, 4, 0.7)
    assert encoder.miss_calls == misses
