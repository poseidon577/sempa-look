import math

import numpy as np
import pytest

from sempalab.channel.awgn_bpsk import (
    bit_error_probability,
    empirical_packet_error_rate,
    packet_error_probability,
    sample_packet_erasures,
)


# Packet-error anchors stated in the paper (Fig. 9 caption, Table V).
#
# 6 dB: the project plan lists 0.5137 (from p_b = 2.403e-3), but
# Q(sqrt(2 gamma)) at 6 dB is 2.388e-3, giving p = 0.5120. That
# matches the paper's Table V value delta = 51.2%, so 0.5120 is
# used here and the plan's 0.5137 is treated as a typo.
ANCHORS = {
    4: 0.9770,
    6: 0.5120,
    7: 0.2070,
    8: 0.0557,
}


@pytest.mark.parametrize("snr_db, expected", ANCHORS.items())
def test_packet_error_anchors(snr_db, expected):
    assert packet_error_probability(snr_db, d=300) == pytest.approx(
        expected, abs=5e-4
    )


def test_bit_error_anchor():
    # p_b at 4 dB is 1.250e-2 in the plan's anchor table.
    assert bit_error_probability(4) == pytest.approx(1.250e-2, rel=1e-3)


@pytest.mark.parametrize("snr_db", [4, 6, 7, 8])
def test_bit_level_simulation_matches_formula(snr_db):
    # Independent check: simulated BPSK + AWGN + hard decisions
    # must agree with the Q-function formula within 4 sigma.
    n = 20_000
    p = packet_error_probability(snr_db)
    sigma = math.sqrt(p * (1 - p) / n)

    empirical = empirical_packet_error_rate(
        n,
        snr_db,
        rng=np.random.default_rng(snr_db),
    )

    assert abs(empirical - p) < 4 * sigma + 1e-3


def test_bit_level_bit_error_rate():
    # With d = 1 a packet error is a bit error.
    p_b = bit_error_probability(2)
    n = 200_000

    empirical = empirical_packet_error_rate(
        n,
        2,
        d=1,
        rng=np.random.default_rng(0),
    )

    assert abs(empirical - p_b) < 4 * math.sqrt(p_b * (1 - p_b) / n)


def test_sample_packet_erasures_shapes_and_extremes():
    rng = np.random.default_rng(1)

    erased = sample_packet_erasures(50, 8, rng=rng)
    assert erased.shape == (50,)
    assert erased.dtype == bool

    # Very low SNR: every 300-bit packet is lost.
    assert sample_packet_erasures(100, -10, rng=rng).all()

    # Very high SNR: nothing is lost, in either mode.
    assert not sample_packet_erasures(100, 20, rng=rng).any()
    assert not sample_packet_erasures(100, 20, rng=rng, bit_level=True).any()


def test_invalid_packet_length():
    with pytest.raises(ValueError):
        packet_error_probability(6, d=0)
