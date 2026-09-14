from sempalab.algorithms.full_search import full_search
from sempalab.encoder import InstrumentedEncoder
from sempalab.types import Message


message = Message(
    message_id=4,
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

result = full_search(
    message=message,
    encoder=encoder,
    M=2,
    objective="ats",
    p=0.3,
)

print("\nBest partition:")
print(result.best_partition)

print("Best ATS:", result.best_score)
print("Objective:", result.objective)
print("Partitions evaluated:", result.partitions_evaluated)
print("Logical calls:", result.logical_calls)
print("Miss calls:", result.miss_calls)
print("Cache hits:", result.cache_hits)
print("Runtime:", result.runtime_seconds)


# K=4, M=2:
#
# 4! / ((2!)^2 * 2!) = 3
#
# Therefore exactly 3 unique partitions exist.
assert result.partitions_evaluated == 3

assert result.objective == "ats"

assert result.best_partition is not None

assert 0.0 <= result.best_score <= 1.0

assert result.logical_calls > 0

assert result.miss_calls > 0

assert result.runtime_seconds >= 0.0

print("\n✅ Full-search test passed")