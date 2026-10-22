from collections import Counter

import numpy as np
import pytest

from sempalab.aggregation import (
    aggregate_into_packets,
    deaggregate_packets,
    packet_bits,
)
from sempalab.types import Message


def make_batch(B, K, M):
    messages = [
        Message(
            message_id=100 + b,
            text=" ".join(f"w{b}_{i}" for i in range(K)),
            tokens=tuple(f"w{b}_{i}" for i in range(K)),
        )
        for b in range(B)
    ]

    partitions = [
        tuple(tuple(range(n * M, (n + 1) * M)) for n in range(K // M))
        for _ in range(B)
    ]

    return messages, partitions


def check_layout(packets, messages, N):
    # Every subpacket appears exactly once.
    seen = Counter(
        (sp.message_id, sp.indices)
        for packet in packets
        for sp in packet.subpackets
    )
    assert len(seen) == len(messages) * N
    assert set(seen.values()) == {1}

    for packet in packets:
        ids = [sp.message_id for sp in packet.subpackets]

        # No packet carries two subpackets of one message.
        assert len(ids) == len(set(ids))

        # Every packet has exactly N slots.
        assert len(packet.subpackets) + packet.dummy_slots == N


@pytest.mark.parametrize("B, K, M", [(4, 8, 4), (3, 12, 4), (10, 12, 3), (1, 4, 4)])
def test_no_padding_when_batch_is_large_enough(B, K, M):
    messages, partitions = make_batch(B, K, M)
    N = K // M

    packets = aggregate_into_packets(messages, partitions)

    assert len(packets) == B
    assert all(packet.dummy_slots == 0 for packet in packets)
    check_layout(packets, messages, N)


@pytest.mark.parametrize("B, K, M", [(1, 8, 4), (2, 12, 3), (3, 12, 2)])
def test_dummy_padding_when_batch_is_small(B, K, M):
    messages, partitions = make_batch(B, K, M)
    N = K // M

    packets = aggregate_into_packets(messages, partitions)

    assert len(packets) == N
    assert sum(packet.dummy_slots for packet in packets) == N * N - B * N
    check_layout(packets, messages, N)


def test_deaggregate_round_trip():
    messages, partitions = make_batch(4, 12, 4)
    packets = aggregate_into_packets(messages, partitions)

    # Nothing lost: every message gets all N subpackets back.
    received = deaggregate_packets(packets, np.zeros(len(packets), dtype=bool))

    for message, partition in zip(messages, partitions):
        assert sorted(sp.indices for sp in received[message.message_id]) == sorted(partition)

    # Everything lost: every message is present with nothing received.
    received = deaggregate_packets(packets, np.ones(len(packets), dtype=bool))
    assert received == {message.message_id: [] for message in messages}


def test_one_lost_packet_costs_each_message_at_most_one_subpacket():
    messages, partitions = make_batch(5, 12, 4)
    packets = aggregate_into_packets(messages, partitions)

    erased = np.zeros(len(packets), dtype=bool)
    erased[2] = True

    received = deaggregate_packets(packets, erased)

    for message in messages:
        assert len(received[message.message_id]) >= 3 - 1


def test_aggregation_rejects_bad_batches():
    messages, partitions = make_batch(2, 8, 4)

    with pytest.raises(ValueError):
        aggregate_into_packets(messages, partitions[:1])

    with pytest.raises(ValueError):
        aggregate_into_packets(messages, [partitions[0], ((0, 1, 2, 3, 4, 5, 6, 7),)])

    with pytest.raises(ValueError):
        aggregate_into_packets([messages[0], messages[0]], partitions)


def test_packet_bits_closes_near_300():
    # 12 tokens * 18 bits + 12 positions * 4 bits + 3 slots * 7 bits.
    budget = packet_bits(K=12, M=4, B=100)

    assert budget.bits_per_token == 18
    assert budget.payload_bits == 216
    assert budget.position_bits == 48
    assert budget.message_id_bits == 21
    assert budget.total_bits == 285
    assert budget.fits(300)


def test_packet_bits_k8():
    budget = packet_bits(K=8, M=4, B=100)

    assert budget.total_bits == 8 * 18 + 8 * 3 + 2 * 7
    assert budget.fits(300)


def test_packet_bits_rejects_overflow():
    assert not packet_bits(K=24, M=4, B=100).fits(300)

    with pytest.raises(ValueError):
        packet_bits(K=10, M=4, B=10)
