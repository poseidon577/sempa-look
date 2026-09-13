from message import Message, Subpacket

def render(
    indices: tuple[int, ...],
    tokens: tuple[str, ...],
) -> str:
    """
    Reconstruct text from token indices.

    Indices are sorted so that tokens always appear
    in their original order.
    """

    return " ".join(
        tokens[i]
        for i in sorted(indices)
    )


def reconstruct(
    surviving_subpackets: list[Subpacket],
    message: Message,
) -> str:
    """
    Reconstruct a message from surviving subpackets.
    """

    indices = tuple(
        i
        for subpacket in surviving_subpackets
        for i in subpacket.indices
    )

    return render(indices, message.tokens)