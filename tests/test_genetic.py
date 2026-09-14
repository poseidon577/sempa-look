from sempalab.algorithms.genetic import genetic
from sempalab.encoder import InstrumentedEncoder
from sempalab.partition import validate_partition
from sempalab.types import Message


message = Message(
    message_id=40,
    text="a man riding a bicycle",
    tokens=(
        "a",
        "man",
        "riding",
        "a",
        "bicycle",
        "near",
        "beach",
        "today",
    ),
)

K = len(message.tokens)
M = 4

encoder = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

result = genetic(
    message=message,
    encoder=encoder,
    M=M,
    objective="ats",
    p=0.3,
    P=10,
    G=5,
    mutation_rate=0.1,
    tournament_size=3,
    seed=42,
)

print("GA partition:")
print(result.best_partition)

print("GA ATS:")
print(result.best_score)

print("Logical calls:")
print(result.logical_calls)

print("Miss calls:")
print(result.miss_calls)

print("Cache hits:")
print(result.cache_hits)

print("Encoder batches:")
print(result.encoder_batches)

print("Runtime:")
print(f"{result.runtime_seconds:.4f}s")

assert validate_partition(
    result.best_partition,
    K,
    M,
)

assert result.objective == "ats"

assert result.partitions_evaluated == 10 * (5 + 1)

assert result.logical_calls == (
    10 * (5 + 1) * (2 ** (K // M))
)

print("\n✅ Genetic Algorithm test passed")