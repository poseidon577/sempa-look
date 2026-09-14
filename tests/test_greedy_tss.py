from sempalab.algorithms.greedy_tss import greedy_tss
from sempalab.encoder import InstrumentedEncoder
from sempalab.types import Message


message = Message(
    message_id=8,
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

result = greedy_tss(
    message=message,
    encoder=encoder,
    M=2,
)

print("\nGreedy-TSS partition:")
print(result.best_partition)

print("TSS score of final selected group:", result.best_score)
print("Objective:", result.objective)
print("Logical calls:", result.logical_calls)
print("Miss calls:", result.miss_calls)
print("Cache hits:", result.cache_hits)
print("Runtime:", result.runtime_seconds)

# K=4, M=2.
#
# First greedy step:
# C(4,2) = 6 candidates.
#
# Second greedy step:
# C(2,2) = 1 candidate.
#
# Therefore:
#
# 6 + 1 = 7 semantic evaluations.

assert result.logical_calls == 7

assert len(result.best_partition) == 2

assert all(
    len(group) == 2
    for group in result.best_partition
)

flattened = [
    index
    for group in result.best_partition
    for index in group
]

assert sorted(flattened) == list(range(4))

assert result.objective == "tss"

assert 0.0 <= result.best_score <= 1.0

print("\n✅ Greedy-TSS test passed")