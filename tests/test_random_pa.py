import numpy as np

from sempalab.algorithms.random_pa import random_pa
from sempalab.encoder import InstrumentedEncoder
from sempalab.types import Message


message = Message(
    message_id=6,
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

result = random_pa(
    message=message,
    encoder=encoder,
    M=2,
    objective="ats",
    p=0.3,
    rng=np.random.default_rng(42),
)

print("\nSelected partition:")
print(result.best_partition)

print("ATS:", result.best_score)
print("Partitions evaluated:", result.partitions_evaluated)
print("Logical calls:", result.logical_calls)
print("Miss calls:", result.miss_calls)
print("Cache hits:", result.cache_hits)
print("Runtime:", result.runtime_seconds)

assert result.partitions_evaluated == 1

assert result.best_partition is not None

assert 0.0 <= result.best_score <= 1.0

assert result.logical_calls == 4

print("\n✅ Random-PA test passed")