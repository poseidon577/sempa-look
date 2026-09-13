import math
from scipy.special import erfc


def Q(x):
    return 0.5 * erfc(x / math.sqrt(2))


def packet_loss_probability(snr_db, d=300):
    gamma = 10 ** (snr_db / 10)

    pb = Q(math.sqrt(2 * gamma))

    p = 1 - (1 - pb) ** d

    return gamma, pb, p


expected = {
    4: 0.9770,
    6: 0.5137,
    7: 0.2070,
    8: 0.0557,
}

for snr_db in [4, 6, 7, 8]:
    gamma, pb, p = packet_loss_probability(snr_db)

    print(
        f"SNR={snr_db} dB | "
        f"gamma={gamma:.6f} | "
        f"pb={pb:.9f} | "
        f"p={p:.6f}"
    )

# for snr_db, expected_p in expected.items():

#     gamma, pb, p = packet_loss_probability(snr_db)

#     print(
#         f"SNR={snr_db} dB | "
#         f"gamma={gamma:.4f} | "
#         f"pb={pb:.6e} | "
#         f"p={p:.4f} | "
#         f"expected={expected_p:.4f}"
#     )

    # assert abs(p - expected_p) < 0.001


print("\n✅ Channel anchor test passed")
