from sempalab.algorithms.greedy_rss import greedy_rss
from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.types import Message


message = Message(
    message_id=9,
    text="a small motor bike",
    tokens=("a", "small", "motor", "bike"),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

p = 0.3
M = 2

# Run Greedy-RSS
rss_result = greedy_rss(
    message=message,
    encoder=encoder,
    M=M,
)

partition = rss_result.best_partition

print("Greedy-RSS partition:")
print(partition)

print("\nRSS selection score:")
print(rss_result.best_score)

# Check that we produced the expected number of groups
assert len(partition) == len(message.tokens) // M

# Check every group has exactly M tokens
assert all(len(group) == M for group in partition)

# Check that every token index appears exactly once
all_indices = [
    index
    for group in partition
    for index in group
]

assert sorted(all_indices) == list(range(len(message.tokens)))

# Evaluate the resulting partition using the COMMON ATS metric
ats = exact_ats(
    partition=partition,
    message=message,
    encoder=encoder,
    p=p,
)

print("\nATS of Greedy-RSS partition:")
print(ats)

# Basic sanity check
assert partition is not None
assert 0.0 <= rss_result.best_score <= 1.0
assert 0.0 <= ats <= 1.0

print("\nInstrumentation:")
print(f"Logical calls: {rss_result.logical_calls}")
print(f"Miss calls: {rss_result.miss_calls}")
print(f"Cache hits: {rss_result.cache_hits}")
print(f"Runtime: {rss_result.runtime_seconds}")

print("\n✅ Greedy-RSS test passed")