import numpy as np
import pytest

from sempalab.aggregation import aggregate_diagonal, deaggregate_packets
from sempalab.channel.awgn_bpsk import packet_error_probability
from sempalab.channel.correlated import (
    GaussMarkovRayleigh,
    GilbertElliott,
    IIDErasure,
    iid_pattern_probabilities,
    make_channel,
    packet_error_from_snr,
    rayleigh_average_per,
    state_matrix,
    survival_moments,
)
from sempalab.types import Message


@pytest.mark.parametrize("N", [1, 3, 6])
def test_iid_law_matches_product(N):
    Q = IIDErasure(0.3).pattern_probabilities(N)
    S = state_matrix(N)
    expected = np.prod(np.where(S == 1, 0.7, 0.3), axis=1)
    assert np.allclose(Q, expected)
    assert Q.sum() == pytest.approx(1.0)


def test_gilbert_with_zero_correlation_is_iid():
    channel = GilbertElliott.from_correlation(0.4, 0.0)
    assert np.allclose(channel.pattern_probabilities(5), iid_pattern_probabilities(5, 0.4))


@pytest.mark.parametrize("p,rho,stride", [(0.1, 0.5, 1), (0.5, 0.9, 1), (0.3, 0.8, 3)])
def test_gilbert_marginals_and_pair_correlation(p, rho, stride):
    channel = GilbertElliott.from_correlation(p, rho)
    assert channel.p == pytest.approx(p)

    N = 4
    Q = channel.pattern_probabilities(N, stride)
    assert Q.sum() == pytest.approx(1.0)
    assert (Q >= 0).all()

    r, R = survival_moments(Q)
    assert np.allclose(r, 1 - p)

    # Two-state chain: cov of the Good indicator at lag d is p(1-p) rho^d.
    for n in range(N):
        for m in range(N):
            lag = abs(n - m) * stride
            expected = (1 - p) if n == m else (1 - p) ** 2 + p * (1 - p) * rho**lag
            assert R[n, m] == pytest.approx(expected)


def test_gilbert_mean_burst_length():
    p, rho = 0.2, 0.75
    channel = GilbertElliott.from_correlation(p, rho)
    assert channel.mean_burst_length == pytest.approx(1 / ((1 - p) * (1 - rho)))


def test_gilbert_elliott_with_partial_losses():
    channel = GilbertElliott.from_correlation(0.3, 0.6, loss_good=0.05, loss_bad=0.8)
    Q = channel.pattern_probabilities(3)
    r, _ = survival_moments(Q)
    assert channel.p == pytest.approx(0.3)
    assert np.allclose(r, 0.7)


def test_gilbert_simulation_matches_exact_law():
    channel = GilbertElliott.from_correlation(0.3, 0.8)
    rng = np.random.default_rng(1)
    erased = channel.sample_erasures(400_000, rng)

    assert erased.mean() == pytest.approx(0.3, abs=0.01)

    # Empirical law of 3 consecutive packets vs exact.
    survive = (~erased).astype(int)
    patterns = survive[:-2] + 2 * survive[1:-1] + 4 * survive[2:]
    empirical = np.bincount(patterns, minlength=8) / len(patterns)
    assert np.allclose(empirical, channel.pattern_probabilities(3), atol=0.01)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        GilbertElliott.from_correlation(0.3, 1.0)
    with pytest.raises(ValueError):
        IIDErasure(0.0)
    with pytest.raises(ValueError):
        make_channel("awgn", 0.3)
    with pytest.raises(ValueError):
        IIDErasure(0.3).pattern_probabilities(3, stride=0)


def test_vectorised_packet_error_matches_scalar():
    for snr_db in [2, 6, 7, 8]:
        gamma = 10 ** (snr_db / 10)
        assert packet_error_from_snr(np.array([gamma]))[0] == pytest.approx(packet_error_probability(snr_db))


@pytest.mark.parametrize("p", [0.05, 0.3, 0.6])
def test_rayleigh_matches_target_per(p):
    channel = GaussMarkovRayleigh(p, 0.9, samples=200_000)
    assert rayleigh_average_per(channel.mean_snr) == pytest.approx(p, rel=1e-4)

    Q = channel.pattern_probabilities(3)
    r, _ = survival_moments(Q)
    assert Q.sum() == pytest.approx(1.0)
    assert np.allclose(r, 1 - p, atol=0.005)


def test_rayleigh_with_no_memory_is_iid():
    channel = GaussMarkovRayleigh(0.3, 0.0, samples=400_000)
    assert np.allclose(channel.pattern_probabilities(3), iid_pattern_probabilities(3, 0.3), atol=0.004)


def test_rayleigh_memory_raises_joint_survival():
    weak = GaussMarkovRayleigh(0.3, 0.5, samples=200_000)
    strong = GaussMarkovRayleigh(0.3, 0.99, samples=200_000)

    _, R_weak = survival_moments(weak.pattern_probabilities(3))
    _, R_strong = survival_moments(strong.pattern_probabilities(3))

    assert R_strong[0, 1] > R_weak[0, 1] > 0.7**2
    # Correlation decays with slot distance.
    assert R_weak[0, 1] > R_weak[0, 2]


def test_rayleigh_simulation_matches_law():
    channel = GaussMarkovRayleigh(0.3, 0.8, samples=200_000)
    erased = channel.sample_erasures(300_000, np.random.default_rng(3))
    survive = (~erased).astype(int)
    patterns = survive[:-1] + 2 * survive[1:]
    empirical = np.bincount(patterns, minlength=4) / len(patterns)
    assert np.allclose(empirical, channel.pattern_probabilities(2), atol=0.01)


def test_diagonal_layout_places_slots_stride_apart():
    messages = [Message(b, "", tuple("abcdef")) for b in range(5)]
    partitions = [((0, 1), (2, 3), (4, 5))] * 5

    packets = aggregate_diagonal(messages, partitions, stride=2)
    assert len(packets) == 5 + 2 * 2

    where = {}
    for packet in packets:
        ids = [s.message_id for s in packet.subpackets]
        assert len(ids) == len(set(ids))
        assert len(packet.subpackets) + packet.dummy_slots == 3
        for s in packet.subpackets:
            where.setdefault(s.message_id, []).append(packet.packet_id)

    for b in range(5):
        assert where[b] == [b, b + 2, b + 4]


def test_system_simulation_matches_exact_slot_law():
    """
    End to end: Gilbert stream -> diagonal layout -> deaggregation gives
    each message the slot law `pattern_probabilities(N, stride)`.
    """

    N, stride, B = 3, 2, 2000
    messages = [Message(b, "", tuple("abcdef")) for b in range(B)]
    partitions = [((0, 1), (2, 3), (4, 5))] * B

    channel = GilbertElliott.from_correlation(0.4, 0.85)
    packets = aggregate_diagonal(messages, partitions, stride=stride)

    rng = np.random.default_rng(0)
    counts = np.zeros(1 << N)

    for _ in range(40):
        received = deaggregate_packets(packets, channel.sample_erasures(len(packets), rng))
        for subpackets in received.values():
            pattern = sum(1 << (s.indices[0] // 2) for s in subpackets)
            counts[pattern] += 1

    assert np.allclose(counts / counts.sum(), channel.pattern_probabilities(N, stride), atol=0.01)
