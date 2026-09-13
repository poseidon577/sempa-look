import time
import torch

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

# Short dummy strings represented as random vectors.
# We're benchmarking tensor computation/transfer overhead,
# not the sentence-transformer yet.
batch_sizes = [1, 10, 50, 200]
vector_size = 768

devices = ["cpu"]

if torch.backends.mps.is_available():
    devices.append("mps")


for batch_size in batch_sizes:
    print(f"\n--- Batch size: {batch_size} ---")

    results = {}

    for device in devices:
        # Create data directly on the target device.
        x = torch.randn(batch_size, vector_size, device=device)
        y = torch.randn(batch_size, vector_size, device=device)

        # Warm-up
        for _ in range(5):
            _ = torch.matmul(x, y.T)

        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()

        for _ in range(100):
            _ = torch.matmul(x, y.T)

        if device == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        results[device] = elapsed

        print(f"{device:>4}: {elapsed:.6f} seconds")

    winner = min(results, key=results.get)
    print(f"Winner: {winner}")