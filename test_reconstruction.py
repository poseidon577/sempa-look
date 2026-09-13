from message import Message, Subpacket
from reconstruction import render, reconstruct


message = Message(
    message_id=0,
    text="a small motor bike",
    tokens=("a", "small", "motor", "bike"),
)


# Group 1 = a, motor
group_1 = Subpacket(
    message_id=0,
    indices=(0, 2),
)

# Group 2 = small, bike
group_2 = Subpacket(
    message_id=0,
    indices=(1, 3),
)


# Test rendering.
assert render(
    (3, 1),
    message.tokens,
) == "small bike"


# Simulate losing group 1.
reconstructed = reconstruct(
    [group_2],
    message,
)

assert reconstructed == "small bike"


# If both survive, we should recover the original.
reconstructed = reconstruct(
    [group_1, group_2],
    message,
)

assert reconstructed == "a small motor bike"


print("Original:", message.text)
print("After losing group 1:", reconstruct([group_2], message))
print("After losing nothing:", reconstructed)
print("✅ Reconstruction tests passed")