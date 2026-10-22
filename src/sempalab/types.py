from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    """
    Represents one tokenized message.
    """

    message_id: int
    text: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Subpacket:
    """
    Represents a group of token indices belonging to a message.
    """

    message_id: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class Packet:
    """
    Physical packet carrying one or more subpackets.

    Every subpacket in a packet belongs to a distinct message,
    so the N subpackets of one message are erased independently.
    `dummy_slots` counts padding slots that carry no tokens.
    """

    packet_id: int
    subpackets: tuple[Subpacket, ...]
    dummy_slots: int = 0

@dataclass(frozen=True)
class SearchResult:
    best_partition: tuple[tuple[int, ...], ...]
    best_score: float

    objective: str

    partitions_evaluated: int
    logical_calls: int
    miss_calls: int
    cache_hits: int
    encoder_batches: int
    runtime_seconds: float

    runtime_seconds: float