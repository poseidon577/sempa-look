from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.partition import generate_partitions
from sempalab.types import Message


message = Message(
    message_id=5,
    text="a small motor bike",
    tokens=(
        "a",
        "small",
        "motor",
        "bike",
    ),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

p = 0.3
M = 2

partitions = list(
    generate_partitions(
        len(message.tokens),
        M,
    )
)

print("All partitions:")
print("=" * 50)

scores = []

for partition in partitions:
    score = exact_ats(
        partition,
        message,
        encoder,
        p,
    )

    scores.append((score, partition))

    print(
        f"{partition} -> ATS = {score:.6f}"
    )


best_score, best_partition = max(
    scores,
    key=lambda x: x[0],
)

print("\nBest:")
print("Partition:", best_partition)
print("ATS:", best_score)

assert len(partitions) == 3

assert best_partition in partitions

assert best_score == max(
    score for score, _ in scores
)

print("\n✅ All-partitions test passed")