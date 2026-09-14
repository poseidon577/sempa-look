from sempalab.algorithms.greedy_rss import greedy_rss
from sempalab.algorithms.sempa_look import sempa_look
from sempalab.algorithms.full_search import full_search
from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.types import Message


message = Message(
    message_id=12,
    text="a small motor bike",
    tokens=("a", "small", "motor", "bike"),
)

M = 2
P = 10
p = 0.3


def evaluate_partition(label, partition, encoder):
    ats = exact_ats(
        partition=partition,
        message=message,
        encoder=encoder,
        p=p,
    )

    print(f"\n{label}")
    print("-" * len(label))
    print("Partition:")
    print(partition)
    print(f"ATS: {ats:.10f}")

    return ats


# ============================================================
# Full Search — ground truth
# ============================================================

encoder_full = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

full_result = full_search(
    message=message,
    encoder=encoder_full,
    M=M,
    objective="ats",
    p=p,
)

full_ats = evaluate_partition(
    "FULL SEARCH",
    full_result.best_partition,
    encoder_full,
)


# ============================================================
# Greedy RSS
# ============================================================

encoder_rss = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

rss_result = greedy_rss(
    message=message,
    encoder=encoder_rss,
    M=M,
)

rss_ats = evaluate_partition(
    "GREEDY RSS",
    rss_result.best_partition,
    encoder_rss,
)


# ============================================================
# SemPA-Look, k=0
# ============================================================

encoder_k0 = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

k0_result = sempa_look(
    message=message,
    encoder=encoder_k0,
    M=M,
    P=P,
    k=0,
    seed=42,
)

k0_ats = evaluate_partition(
    "SEMPA-LOOK k=0",
    k0_result.best_partition,
    encoder_k0,
)


# ============================================================
# SemPA-Look, k=1
# ============================================================

encoder_k1 = InstrumentedEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

k1_result = sempa_look(
    message=message,
    encoder=encoder_k1,
    M=M,
    P=P,
    k=1,
    seed=42,
)

k1_ats = evaluate_partition(
    "SEMPA-LOOK k=1",
    k1_result.best_partition,
    encoder_k1,
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Full Search:     {full_ats:.10f}")
print(f"Greedy RSS:      {rss_ats:.10f}")
print(f"SemPA-Look k=0:  {k0_ats:.10f}")
print(f"SemPA-Look k=1:  {k1_ats:.10f}")

print("\nPartitions:")
print(f"Full Search:     {full_result.best_partition}")
print(f"Greedy RSS:      {rss_result.best_partition}")
print(f"SemPA-Look k=0:  {k0_result.best_partition}")
print(f"SemPA-Look k=1:  {k1_result.best_partition}")

print("\nLogical semantic evaluations:")
print(f"Greedy RSS:      {rss_result.logical_calls}")
print(f"SemPA-Look k=0:  {k0_result.logical_calls}")
print(f"SemPA-Look k=1:  {k1_result.logical_calls}")

# Basic sanity checks
assert 0.0 <= full_ats <= 1.0
assert 0.0 <= rss_ats <= 1.0
assert 0.0 <= k0_ats <= 1.0
assert 0.0 <= k1_ats <= 1.0

# K=4, M=2 => N=2 => one decision depth.
#
# k=0:
#   (N-1) * P * (0+1) = 10
#
# k=1:
#   (N-1) * P * (1+1) = 20
assert k0_result.logical_calls == 10
assert k1_result.logical_calls == 20

print("\n✅ Lookahead comparison passed")