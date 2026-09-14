from sempalab.algorithms.no_packetization import no_packetization
from sempalab.encoder import InstrumentedEncoder
from sempalab.types import Message


message = Message(
    message_id=7,
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

result = no_packetization(
    message=message,
    encoder=encoder,
    objective="ats",
    p=0.3,
)

print("\nPartition:")
print(result.best_partition)

print("ATS:", result.best_score)
print("Partitions evaluated:", result.partitions_evaluated)
print("Logical calls:", result.logical_calls)
print("Miss calls:", result.miss_calls)
print("Cache hits:", result.cache_hits)
print("Runtime:", result.runtime_seconds)

assert result.best_partition == ((0, 1, 2, 3),)

assert result.partitions_evaluated == 1

assert result.objective == "ats"

assert 0.0 <= result.best_score <= 1.0

# One packet means two possible survival states:
# survive / lost.
assert result.logical_calls == 2

print("\n✅ No-packetization test passed")