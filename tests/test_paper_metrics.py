import pytest

from sempalab.metrics import (
    batch_semantic_similarity,
    exact_ats,
    semantic_similarity,
    wts_lost,
)
from sempalab.types import Message


MESSAGE = Message(
    message_id=7,
    text="a man riding a bicycle near the beach",
    tokens=tuple("a man riding a bicycle near the beach".split()),
)


def test_empty_reconstruction_has_zero_similarity(fake_encoder):
    encoder = fake_encoder()

    assert semantic_similarity(encoder, MESSAGE, ()) == 0.0

    sims = batch_semantic_similarity(encoder, MESSAGE, [(), (0, 1), ()])
    assert sims[0] == 0.0 and sims[2] == 0.0 and sims[1] != 0.0

    # Empty sets still count as logical evaluations, never as encodes.
    assert encoder.logical_calls == 4
    assert encoder.miss_calls == 1


def test_ats_limits_with_empty_zero(fake_encoder):
    partition = ((0, 2, 4, 6), (1, 3, 5, 7))

    assert exact_ats(partition, MESSAGE, fake_encoder(), 1.0) == 0.0
    assert exact_ats(partition, MESSAGE, fake_encoder(), 0.0) == pytest.approx(1.0)


def test_no_packetization_ats_is_one_minus_p(fake_encoder):
    # Paper Fig. 7/8: M = K gives exactly 1 - p (0.793 at 7 dB).
    p = 0.2070
    whole = (tuple(range(8)),)

    assert exact_ats(whole, MESSAGE, fake_encoder(), p) == pytest.approx(1 - p)


def test_wts_lost_matches_brute_force(fake_encoder):
    encoder = fake_encoder()
    partition = ((0, 1), (2, 3), (4, 5), (6, 7))

    for L in range(0, 5):
        expected = min(
            semantic_similarity(
                encoder,
                MESSAGE,
                tuple(i for n, g in enumerate(partition) if not (mask >> n) & 1 for i in g),
            )
            for mask in range(16)
            if bin(mask).count("1") == L
        )

        assert wts_lost(partition, MESSAGE, encoder, L) == pytest.approx(expected)

    assert wts_lost(partition, MESSAGE, encoder, 0) == pytest.approx(1.0)
    assert wts_lost(partition, MESSAGE, encoder, 4) == 0.0


def test_wts_lost_rejects_bad_L(fake_encoder):
    with pytest.raises(ValueError):
        wts_lost(((0, 1), (2, 3)), MESSAGE, fake_encoder(), 3)
