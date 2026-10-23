import numpy as np
import pytest

from sempalab.algorithms.interleave import interleave_partition
from sempalab.algorithms.swap_local_search import swap_local_search
from sempalab.channel.correlated import (
    GilbertElliott,
    iid_pattern_probabilities,
    state_matrix,
    survival_moments,
)
from sempalab.channel_aware import (
    PairwiseSurrogate,
    best_slot_order,
    bits,
    casa,
    channel_ats,
    fit_pairwise,
    fitting_law,
    full_search_channel,
    pairwise_features,
    permuted_laws,
    sample_subsets,
    slots_to_partition,
    surrogate_local_search,
    surrogate_r2,
    survivor_count_law,
    swap_gains,
)
from sempalab import evaluation
from sempalab.evaluation import SimilarityTable, to_mask
from sempalab.metrics import exact_ats
from sempalab.partition import generate_partitions, validate_partition
from sempalab.types import Message


TOKENS = tuple("a man rides a red bike near the old harbour at dusk".split())


@pytest.fixture
def table(fake_encoder):
    message = Message(0, " ".join(TOKENS), TOKENS)
    return SimilarityTable(fake_encoder(), message), message


def gilbert(p=0.4, rho=0.8):
    return GilbertElliott.from_correlation(p, rho)


# -------------------------------------------------------------
# Exact evaluation
# -------------------------------------------------------------


def test_channel_ats_under_iid_matches_paper_ats(table):
    tab, message = table
    partition = ((0, 5, 7, 9), (1, 2, 3, 4), (6, 8, 10, 11))
    Q = iid_pattern_probabilities(3, 0.3)

    assert channel_ats(tab, partition, Q) == pytest.approx(evaluation.ats(tab, partition, 0.3))
    assert channel_ats(tab, partition, Q) == pytest.approx(exact_ats(partition, message, tab.encoder, 0.3))


def test_channel_ats_by_brute_force(table):
    tab, _ = table
    partition = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
    Q = gilbert().pattern_probabilities(3)

    total = 0.0
    for s, row in enumerate(state_matrix(3)):
        survivors = [i for n, g in enumerate(partition) if row[n] for i in g]
        total += Q[s] * tab.lookup([to_mask(survivors)])[0]

    assert channel_ats(tab, partition, Q) == pytest.approx(total)


def test_slot_order_irrelevant_under_iid_but_not_under_bursts(table):
    tab, _ = table
    partition = ((0, 5, 7, 9), (1, 2, 3, 4), (6, 8, 10, 11))
    reordered = (partition[1], partition[2], partition[0])

    iid = iid_pattern_probabilities(3, 0.4)
    assert channel_ats(tab, partition, iid) == pytest.approx(channel_ats(tab, reordered, iid))

    swapped = (partition[1], partition[0], partition[2])
    bursty = gilbert().pattern_probabilities(3)
    scores = {round(channel_ats(tab, q, bursty), 12) for q in (partition, reordered, swapped)}
    assert len(scores) > 1


def test_permuted_laws_match_reordering(table):
    tab, _ = table
    partition = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    Q = gilbert().pattern_probabilities(4)
    orders, laws = permuted_laws(Q, 4)

    phi = tab.lookup(evaluation._state_matrix(4) @ np.array([to_mask(g) for g in partition]))

    for k in [0, 5, 13, 23]:
        reordered = tuple(partition[g] for g in orders[k])
        assert phi @ laws[:, k] == pytest.approx(channel_ats(tab, reordered, Q))


def test_best_slot_order_beats_every_order(table):
    tab, _ = table
    partition = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    Q = gilbert(0.5, 0.9).pattern_probabilities(4)
    orders, _ = permuted_laws(Q, 4)

    best, score = best_slot_order(tab, partition, Q)

    assert sorted(best) == sorted(partition)
    assert score == pytest.approx(channel_ats(tab, best, Q))
    assert score >= max(channel_ats(tab, tuple(partition[g] for g in o), Q) for o in orders) - 1e-12


def test_full_search_channel_is_the_ordered_optimum(fake_encoder):
    tokens = TOKENS[:8]
    message = Message(1, " ".join(tokens), tokens)
    tab = SimilarityTable(fake_encoder(), message)
    Q = gilbert(0.3, 0.9).pattern_probabilities(4)

    partition, score, evaluated = full_search_channel(tab, 8, 2, Q)

    assert validate_partition(partition, 8, 2)
    assert evaluated == 105 * 24
    assert score == pytest.approx(channel_ats(tab, partition, Q))

    brute = max(best_slot_order(tab, q, Q)[1] for q in generate_partitions(8, 2))
    assert score == pytest.approx(brute)


def test_full_search_channel_reduces_to_paper_full_search_under_iid(table):
    tab, _ = table
    Q = iid_pattern_probabilities(3, 0.5)
    _, score, _ = full_search_channel(tab, 12, 4, Q)
    _, paper_score, _, _ = evaluation.full_search(tab, 12, 4, 0.5)
    assert score == pytest.approx(paper_score)


# -------------------------------------------------------------
# Surrogate
# -------------------------------------------------------------


def test_survivor_count_law_iid_is_binomial():
    law = survivor_count_law(iid_pattern_probabilities(4, 0.3))
    from math import comb
    expected = [comb(4, c) * 0.7**c * 0.3 ** (4 - c) for c in range(5)]
    assert np.allclose(law, expected)


def test_sample_subsets_sizes_follow_count_law():
    rng = np.random.default_rng(0)
    law = np.array([0.1, 0.2, 0.3, 0.4])
    masks = sample_subsets(12, 4, law, 20_000, rng)
    sizes = np.array([bin(int(m)).count("1") for m in masks])
    assert set(np.unique(sizes)) <= {0, 4, 8, 12}
    assert np.allclose(np.bincount(sizes // 4, minlength=4) / len(sizes), law, atol=0.01)


def test_pairwise_features_layout():
    X = np.array([[1, 0, 1], [1, 1, 1]], dtype=float)
    F = pairwise_features(X)
    # [1, x0, x1, x2, x0x1, x0x2, x1x2]
    assert F.tolist() == [[1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1]]


def test_surrogate_recovers_an_exact_pairwise_function():
    K = 6
    rng = np.random.default_rng(0)
    a = rng.normal(size=K)
    B = np.triu(rng.normal(size=(K, K)), 1)
    B = B + B.T
    truth = PairwiseSurrogate(0.2, a, B)

    class Table:
        def lookup(self, masks):
            return truth.predict(bits(masks, K))

    model, queried = fit_pairwise(Table(), K, 1, iid_pattern_probabilities(K, 0.5), 400, rng, ridge=0.0)
    assert model.a0 == pytest.approx(0.2)
    assert np.allclose(model.a, a)
    assert np.allclose(model.B, B)
    assert 0 not in queried


def test_surrogate_expectation_matches_monte_carlo():
    """
    E_Q[model(x)] for x induced by a labelled assignment equals the
    closed form a0 + a.r[g] + sum b_ij R[g_i, g_j].
    """

    K, N, M = 6, 3, 2
    rng = np.random.default_rng(1)
    B = np.triu(rng.normal(size=(K, K)), 1)
    model = PairwiseSurrogate(0.1, rng.normal(size=K), B + B.T)

    Q = gilbert(0.4, 0.7).pattern_probabilities(N)
    r, R = survival_moments(Q)
    slots = np.array([2, 0, 1, 0, 2, 1])

    S = state_matrix(N)
    exact = sum(Q[s] * model.predict(S[s][slots][None, :])[0] for s in range(1 << N))

    assert model.expected(slots, r, R) == pytest.approx(exact)


def test_swap_gains_match_recomputation():
    K, N, M = 8, 4, 2
    rng = np.random.default_rng(2)
    B = np.triu(rng.normal(size=(K, K)), 1)
    model = PairwiseSurrogate(0.0, rng.normal(size=K), B + B.T)
    r, R = survival_moments(gilbert(0.3, 0.8).pattern_probabilities(N))

    slots = np.repeat(np.arange(N), M)[rng.permutation(K)]
    gains = swap_gains(model, slots, r, R)
    base = model.expected(slots, r, R)

    for i in range(K):
        for j in range(K):
            if slots[i] == slots[j]:
                assert gains[i, j] == -np.inf
                continue
            swapped = slots.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            assert gains[i, j] == pytest.approx(model.expected(swapped, r, R) - base)


def test_local_search_reaches_swap_local_optima():
    K, N, M = 8, 4, 2
    rng = np.random.default_rng(3)
    B = np.triu(rng.normal(size=(K, K)), 1)
    model = PairwiseSurrogate(0.0, rng.normal(size=K), B + B.T)
    r, R = survival_moments(gilbert(0.3, 0.8).pattern_probabilities(N))

    for value, slots in surrogate_local_search(model, M, r, R, rng, restarts=5):
        assert np.bincount(slots, minlength=N).tolist() == [M] * N
        assert value == pytest.approx(model.expected(slots, r, R))
        assert swap_gains(model, slots, r, R).max() <= 1e-9


def test_slots_to_partition():
    assert slots_to_partition(np.array([1, 0, 1, 0]), 2) == ((1, 3), (0, 2))


def test_fake_encoder_surrogate_fits_reasonably(table):
    tab, _ = table
    rng = np.random.default_rng(4)
    Q = gilbert(0.3, 0.6).pattern_probabilities(3)
    model, queried = fit_pairwise(tab, 12, 4, Q, 400, rng)
    held_out = sample_subsets(12, 4, fitting_law(Q), 200, np.random.default_rng(99))
    assert surrogate_r2(model, tab, held_out) > 0.5
    assert len(queried) <= 400
    sizes = {bin(m).count("1") for m in queried}
    assert sizes == {4, 8}  # only partial survival: 1 or 2 of 3 slots


# -------------------------------------------------------------
# CASA
# -------------------------------------------------------------


def test_casa_returns_valid_partition_close_to_oracle(table):
    tab, _ = table
    Q = gilbert(0.5, 0.9).pattern_probabilities(3)

    result = casa(tab, 12, 4, Q, np.random.default_rng(0))
    _, oracle, _ = full_search_channel(tab, 12, 4, Q)

    assert validate_partition(result.partition, 12, 4)
    assert result.ats == pytest.approx(channel_ats(tab, result.partition, Q))
    assert result.ats <= oracle + 1e-12
    assert result.ats >= oracle - 0.05
    # Default budget 4 * (1 + 12 + 66) = 316 samples plus re-ranking.
    assert result.logical_calls <= 316 + 5 * 7


def test_casa_is_deterministic_given_seed(table):
    tab, _ = table
    Q = gilbert(0.3, 0.6).pattern_probabilities(3)
    first = casa(tab, 12, 4, Q, np.random.default_rng(7))
    second = casa(tab, 12, 4, Q, np.random.default_rng(7))
    assert first.partition == second.partition


# -------------------------------------------------------------
# Baselines
# -------------------------------------------------------------


def test_interleave_partition():
    assert interleave_partition(12, 4) == ((0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11))
    assert interleave_partition(6, 3) == ((0, 2, 4), (1, 3, 5))
    assert validate_partition(interleave_partition(12, 2), 12, 2)
    with pytest.raises(ValueError):
        interleave_partition(12, 5)


def test_swap_local_search_finds_rss_optimum_on_small_case(fake_encoder):
    tokens = TOKENS[:8]
    message = Message(2, " ".join(tokens), tokens)
    encoder = fake_encoder()
    tab = SimilarityTable(encoder, message)

    def rss_sum(partition):
        full = (1 << 8) - 1
        return sum(tab.lookup([full & ~to_mask(g)])[0] for g in partition)

    best = max(rss_sum(q) for q in generate_partitions(8, 4))

    result = swap_local_search(message, fake_encoder(), 4, seed=0, restarts=3)

    assert validate_partition(result.best_partition, 8, 4)
    assert result.best_score == pytest.approx(rss_sum(result.best_partition))
    # N = 2: every partition is one swap-path away; the search must reach the optimum.
    assert result.best_score == pytest.approx(best)
    assert result.logical_calls > 0


def test_swap_local_search_counts_distinct_subsets(fake_encoder):
    message = Message(3, " ".join(TOKENS), TOKENS)
    encoder = fake_encoder()
    result = swap_local_search(message, encoder, 4, seed=1)
    assert result.logical_calls == encoder.logical_calls
    # Each distinct group is scored once: at most C(12, 4) groups exist.
    assert result.logical_calls <= 495


def test_pool_fit_matches_fresh_fit_on_exact_pairwise_function():
    """
    A reweighted shared pool recovers the same exact pairwise model
    as a fresh sample drawn from the channel's law.
    """

    from sempalab.channel_aware import sample_pool

    K, M = 8, 2
    rng = np.random.default_rng(5)
    B = np.triu(rng.normal(size=(K, K)), 1)
    truth = PairwiseSurrogate(0.3, rng.normal(size=K), B + B.T)

    class Table:
        def lookup(self, masks):
            return truth.predict(bits(masks, K))

    pool = sample_pool(K, M, 300, rng)
    sizes = {bin(int(m)).count("1") for m in pool}
    assert sizes == {2, 4, 6}

    Q = gilbert(0.4, 0.8).pattern_probabilities(4)
    model, queried = fit_pairwise(Table(), K, M, Q, 0, rng, ridge=0.0, pool=pool)
    assert np.allclose(model.B, truth.B, atol=1e-8)
    assert len(queried) == len({int(m) for m in pool})


def test_casa_with_pool_runs(table):
    from sempalab.channel_aware import sample_pool

    tab, _ = table
    Q = gilbert(0.5, 0.6).pattern_probabilities(3)
    pool = sample_pool(12, 4, 316, np.random.default_rng(0))
    result = casa(tab, 12, 4, Q, np.random.default_rng(1), pool=pool)
    assert validate_partition(result.partition, 12, 4)
