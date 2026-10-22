from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sempalab.types import Message, Packet, Subpacket


DEFAULT_DICTIONARY_SIZE = 200_000


def aggregate_into_packets(
    messages: list[Message],
    partitions: list[tuple[tuple[int, ...], ...]],
) -> list[Packet]:
    """
    Multiplex the subpackets of a batch of B messages into
    physical packets.

    Every message has N subpackets. Each packet has N slots,
    so it carries K tokens, the same as one message.

    Layout (cyclic):

        num_packets = max(B, N)
        subpacket n of message b -> packet (b + n) mod num_packets

    For a fixed packet j, distinct n give distinct messages
    b = (j - n) mod num_packets, so no packet ever carries two
    subpackets of the same message and the N subpackets of a
    message are erased independently.

    When B < N there are not enough messages to fill every
    slot, and the slots with b >= B become dummy padding.
    """

    if len(messages) != len(partitions):
        raise ValueError(
            "messages and partitions must have the same length"
        )

    if not messages:
        raise ValueError("Batch must contain at least one message")

    message_ids = [message.message_id for message in messages]

    if len(set(message_ids)) != len(message_ids):
        raise ValueError("message_id values must be unique in a batch")

    group_counts = {len(partition) for partition in partitions}

    if len(group_counts) != 1:
        raise ValueError(
            "All messages in a batch must have the same number "
            "of subpackets N"
        )

    N = group_counts.pop()
    B = len(messages)

    num_packets = max(B, N)

    packets = []

    for j in range(num_packets):

        subpackets = []
        dummy_slots = 0

        for n in range(N):

            b = (j - n) % num_packets

            if b >= B:
                dummy_slots += 1
                continue

            subpackets.append(
                Subpacket(
                    message_id=messages[b].message_id,
                    indices=tuple(partitions[b][n]),
                )
            )

        packets.append(
            Packet(
                packet_id=j,
                subpackets=tuple(subpackets),
                dummy_slots=dummy_slots,
            )
        )

    return packets


def deaggregate_packets(
    packets: list[Packet],
    erased: np.ndarray | list[bool],
) -> dict[int, list[Subpacket]]:
    """
    Recover each message's surviving subpackets from the
    packets that were not erased.

    Returns
    -------
    dict
        message_id -> surviving subpackets (possibly empty).
        Every message present in `packets` has an entry.
    """

    if len(packets) != len(erased):
        raise ValueError("erased must have one entry per packet")

    surviving: dict[int, list[Subpacket]] = {}

    for packet, is_erased in zip(packets, erased):

        for subpacket in packet.subpackets:

            received = surviving.setdefault(subpacket.message_id, [])

            if not is_erased:
                received.append(subpacket)

    return surviving


@dataclass(frozen=True)
class PacketBudget:
    """
    Bit accounting for one physical packet.
    """

    slots: int
    tokens: int
    bits_per_token: int
    bits_per_position: int
    bits_per_message_id: int
    payload_bits: int
    position_bits: int
    message_id_bits: int
    total_bits: int

    def fits(self, d: int) -> bool:
        return self.total_bits <= d


def _bits_for(values: int) -> int:
    """
    Bits needed to label `values` distinct values.
    """

    return math.ceil(math.log2(values)) if values > 1 else 0


def packet_bits(
    K: int,
    M: int,
    B: int,
    dictionary_size: int = DEFAULT_DICTIONARY_SIZE,
) -> PacketBudget:
    """
    Bits needed by one packet of the aggregation above.

    A packet has N = K / M slots of M tokens. It carries:

        payload:     ceil(log2 |D|) bits per token
        position:    ceil(log2 K) bits per token, so the receiver
                     can restore original token order
        message id:  ceil(log2 (B + 1)) bits per slot; the extra
                     value marks a dummy (padding) slot

    With |D| = 200,000, K = 12, M = 4, B = 100:
        12*18 + 12*4 + 3*7 = 216 + 48 + 21 = 285 <= 300.
    """

    if M <= 0 or K <= 0 or B <= 0:
        raise ValueError("K, M and B must be positive")

    if K % M != 0:
        raise ValueError(f"M={M} must divide K={K}")

    if dictionary_size <= 0:
        raise ValueError("dictionary_size must be positive")

    N = K // M

    bits_per_token = _bits_for(dictionary_size)
    bits_per_position = _bits_for(K)
    bits_per_message_id = _bits_for(B + 1)

    payload_bits = K * bits_per_token
    position_bits = K * bits_per_position
    message_id_bits = N * bits_per_message_id

    return PacketBudget(
        slots=N,
        tokens=K,
        bits_per_token=bits_per_token,
        bits_per_position=bits_per_position,
        bits_per_message_id=bits_per_message_id,
        payload_bits=payload_bits,
        position_bits=position_bits,
        message_id_bits=message_id_bits,
        total_bits=payload_bits + position_bits + message_id_bits,
    )
