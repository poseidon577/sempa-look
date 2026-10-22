from __future__ import annotations

import numpy as np

from sempalab.aggregation import aggregate_into_packets, deaggregate_packets
from sempalab.channel.awgn_bpsk import (
    DEFAULT_PACKET_BITS,
    sample_packet_erasures,
)
from sempalab.metrics import batch_semantic_similarity
from sempalab.types import Message


def transmit_batch(
    messages: list[Message],
    partitions: list[tuple[tuple[int, ...], ...]],
    snr_db: float,
    d: int = DEFAULT_PACKET_BITS,
    rng: np.random.Generator | None = None,
    bit_level: bool = False,
) -> tuple[dict[int, tuple[int, ...]], np.ndarray]:
    """
    Send one batch through the full physical chain:

        aggregate -> BPSK/AWGN packet erasures -> deaggregate

    Returns
    -------
    surviving:
        message_id -> sorted surviving token indices.

    erased:
        Boolean erasure pattern of the physical packets.
    """

    if rng is None:
        rng = np.random.default_rng()

    packets = aggregate_into_packets(messages, partitions)

    erased = sample_packet_erasures(
        len(packets),
        snr_db,
        d,
        rng,
        bit_level=bit_level,
    )

    received = deaggregate_packets(packets, erased)

    surviving = {
        message_id: tuple(
            sorted(
                index
                for subpacket in subpackets
                for index in subpacket.indices
            )
        )
        for message_id, subpackets in received.items()
    }

    return surviving, erased


def simulate_batch_ats(
    messages: list[Message],
    partitions: list[tuple[tuple[int, ...], ...]],
    encoder,
    snr_db: float,
    trials: int,
    d: int = DEFAULT_PACKET_BITS,
    rng: np.random.Generator | None = None,
    bit_level: bool = False,
) -> tuple[dict[int, float], float]:
    """
    Empirical ATS of each message under the physical system,
    averaged over `trials` independent transmissions of the batch.

    Returns
    -------
    ats:
        message_id -> mean semantic similarity.

    packet_error_rate:
        Fraction of physical packets erased across all trials.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")

    if rng is None:
        rng = np.random.default_rng()

    received_sets = {message.message_id: [] for message in messages}

    erased_packets = 0
    total_packets = 0

    for _ in range(trials):

        surviving, erased = transmit_batch(
            messages,
            partitions,
            snr_db,
            d,
            rng,
            bit_level=bit_level,
        )

        erased_packets += int(erased.sum())
        total_packets += len(erased)

        for message_id, indices in surviving.items():
            received_sets[message_id].append(indices)

    # One batched (and cached) encoder call per message.
    ats = {
        message.message_id: float(
            batch_semantic_similarity(
                encoder,
                message,
                received_sets[message.message_id],
            ).mean()
        )
        for message in messages
    }

    return ats, erased_packets / total_packets
