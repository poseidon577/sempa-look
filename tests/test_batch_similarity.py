import numpy as np

from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import (
    batch_semantic_similarity,
    semantic_similarity,
)
from sempalab.types import Message


message = Message(
    message_id=10,
    text="a small motor bike",
    tokens=("a", "small", "motor", "bike"),
)

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Evaluate the same three reconstructions individually
individual_scores = []

for indices in [
    (0, 1),
    (2, 3),
    (0, 2),
]:
    individual_scores.append(
        semantic_similarity(
            encoder,
            message,
            indices,
        )
    )

# Now evaluate them through the batched function
batch_scores = batch_semantic_similarity(
    encoder,
    message,
    [
        (0, 1),
        (2, 3),
        (0, 2),
    ],
)

print("Individual scores:")
print(individual_scores)

print("\nBatch scores:")
print(batch_scores)

# The two approaches should give the same semantic similarities
assert np.allclose(
    individual_scores,
    batch_scores,
    atol=1e-6,
)

# Three semantic evaluations should have happened in the batch call
assert len(batch_scores) == 3

print("\nLogical calls:")
print(encoder.logical_calls)

print("\nMiss calls:")
print(encoder.miss_calls)

print("\nCache hits:")
print(encoder.cache_hits)

print("\n✅ Batch semantic similarity test passed")