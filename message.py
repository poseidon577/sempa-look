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

    For now, we assume one subpacket per packet.
    """

    packet_id: int
    subpacket: Subpacket