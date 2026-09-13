from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import wts, semantic_similarity
from sempalab.types import Message


message = Message(
    message_id=3,
    text="a small motor bike",
    tokens=(
        "a",
        "small",
        "motor",
        "bike",
    ),
)

partition = (
    (0, 2),
    (1, 3),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

p = 0.3

# Similarity when packet 0 is lost.
s_loss_0 = semantic_similarity(
    encoder,
    message,
    surviving_indices=(1, 3),
)

# Similarity when packet 1 is lost.
s_loss_1 = semantic_similarity(
    encoder,
    message,
    surviving_indices=(0, 2),
)

# Similarity when both packets are lost.
s_empty = semantic_similarity(
    encoder,
    message,
    surviving_indices=(),
)

worst_one_loss = min(s_loss_0, s_loss_1)

manual_wts = (
    (1 - p) ** 2
    + 2 * p * (1 - p) * worst_one_loss
    + p ** 2 * s_empty
)

implementation_wts = wts(
    partition,
    message,
    encoder,
    p,
)

print("Similarity losing packet 0:", s_loss_0)
print("Similarity losing packet 1:", s_loss_1)
print("Worst one-loss similarity:", worst_one_loss)
print("Empty similarity:", s_empty)

print("\nManual WTS:", manual_wts)
print("Implementation WTS:", implementation_wts)
print(
    "Absolute difference:",
    abs(manual_wts - implementation_wts),
)

assert abs(manual_wts - implementation_wts) < 1e-6

print("\n✅ WTS hand-verification test passed")