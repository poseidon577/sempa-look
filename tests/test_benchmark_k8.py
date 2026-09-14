from sempalab.algorithms.full_search import full_search
from sempalab.algorithms.greedy_tss import greedy_tss
from sempalab.algorithms.greedy_rss import greedy_rss
from sempalab.algorithms.random_pa import random_pa
from sempalab.algorithms.sempa_look import sempa_look
from sempalab.encoder import InstrumentedEncoder
from sempalab.metrics import exact_ats
from sempalab.types import Message


# ============================================================
# Experiment configuration
# ============================================================

message = Message(
    message_id=20,
    text="a man riding a bicycle near beach",
    tokens=(
        "a",
        "man",
        "riding",
        "a",
        "bicycle",
        "near",
        "the",
        "beach",
    ),
)

K = len(message.tokens)
M = 4
P = 10
p = 0.3
seed = 42

assert K == 8
assert K % M == 0


# ============================================================
# Helper
# ============================================================

def make_encoder():
    return InstrumentedEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def evaluate(label, result, encoder):
    ats = exact_ats(
        partition=result.best_partition,
        message=message,
        encoder=encoder,
        p=p,
    )

    print(f"\n{label}")
    print("-" * len(label))
    print(f"Partition: {result.best_partition}")
    print(f"ATS:       {ats:.10f}")
    print(f"Logical:   {result.logical_calls}")
    print(f"Miss:      {result.miss_calls}")
    print(f"Cache:     {result.cache_hits}")
    print(f"Runtime:   {result.runtime_seconds:.4f}s")

    return ats


# ============================================================
# 1. Full Search
# ============================================================

encoder = make_encoder()

result_full = full_search(
    message=message,
    encoder=encoder,
    M=M,
    objective="ats",
    p=p,
)

ats_full = evaluate(
    "FULL SEARCH",
    result_full,
    encoder,
)


# ============================================================
# 2. Random PA
# ============================================================

encoder = make_encoder()

result_random = random_pa(
    message=message,
    encoder=encoder,
    M=M,
    objective="ats",
    p=p,
    rng=__import__("numpy").random.default_rng(seed),
)

ats_random = evaluate(
    "RANDOM PA",
    result_random,
    encoder,
)


# ============================================================
# 3. Greedy TSS
# ============================================================

encoder = make_encoder()

result_tss = greedy_tss(
    message=message,
    encoder=encoder,
    M=M,
)

ats_tss = evaluate(
    "GREEDY TSS",
    result_tss,
    encoder,
)


# ============================================================
# 4. Greedy RSS
# ============================================================

encoder = make_encoder()

result_rss = greedy_rss(
    message=message,
    encoder=encoder,
    M=M,
)

ats_rss = evaluate(
    "GREEDY RSS",
    result_rss,
    encoder,
)


# ============================================================
# 5. SemPA-Look k=0
# ============================================================

encoder = make_encoder()

result_k0 = sempa_look(
    message=message,
    encoder=encoder,
    M=M,
    P=P,
    k=0,
    seed=seed,
)

ats_k0 = evaluate(
    "SEMPA-LOOK k=0",
    result_k0,
    encoder,
)


# ============================================================
# 6. SemPA-Look k=1
# ============================================================

encoder = make_encoder()

result_k1 = sempa_look(
    message=message,
    encoder=encoder,
    M=M,
    P=P,
    k=1,
    seed=seed,
)

ats_k1 = evaluate(
    "SEMPA-LOOK k=1",
    result_k1,
    encoder,
)


# ============================================================
# Final comparison
# ============================================================

print("\n")
print("=" * 80)
print("K=8 / M=4 SEMPA-LAB BASELINE")
print("=" * 80)

print(f"{'Method':<20} {'ATS':>12} {'Gap vs Full':>15}")
print("-" * 80)

methods = [
    ("Full Search", ats_full),
    ("Random PA", ats_random),
    ("Greedy TSS", ats_tss),
    ("Greedy RSS", ats_rss),
    ("SemPA-Look k=0", ats_k0),
    ("SemPA-Look k=1", ats_k1),
]

for name, ats in methods:
    gap = ats_full - ats

    print(
        f"{name:<20} "
        f"{ats:>12.10f} "
        f"{gap:>15.10f}"
    )


# ============================================================
# Structural assertions
# ============================================================

for name, result in [
    ("Full Search", result_full),
    ("Random PA", result_random),
    ("Greedy TSS", result_tss),
    ("Greedy RSS", result_rss),
    ("SemPA-Look k=0", result_k0),
    ("SemPA-Look k=1", result_k1),
]:

    partition = result.best_partition

    assert len(partition) == K // M

    assert all(
        len(group) == M
        for group in partition
    )

    indices = [
        index
        for group in partition
        for index in group
    ]

    assert sorted(indices) == list(range(K))


# SemPA-Look complexity:
#
# K=8, M=4
# N=2
# N-1 = 1 decision depth
#
# k=0 -> 1 * 10 * 1 = 10
# k=1 -> 1 * 10 * 2 = 20

assert result_k0.logical_calls == 10
assert result_k1.logical_calls == 20

print("\n✅ K=8 benchmark passed")