from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc


DEFAULT_PACKET_BITS = 300


def q_function(x: float) -> float:
    """
    Gaussian tail probability Q(x) = P(Z > x), Z ~ N(0, 1).
    """

    return 0.5 * float(erfc(x / math.sqrt(2)))


def snr_linear(snr_db: float) -> float:
    """
    gamma = 10^(SNR_dB / 10)
    """

    return 10 ** (snr_db / 10)


def bit_error_probability(snr_db: float) -> float:
    """
    Uncoded BPSK bit-error probability over AWGN:

        p_b = Q(sqrt(2 gamma))

    where gamma is the per-bit SNR (Eb/N0).
    """

    return q_function(math.sqrt(2 * snr_linear(snr_db)))


def packet_error_probability(
    snr_db: float,
    d: int = DEFAULT_PACKET_BITS,
) -> float:
    """
    Probability that an uncoded d-bit packet contains at
    least one bit error, and is therefore erased:

        p = 1 - (1 - p_b)^d
    """

    if d <= 0:
        raise ValueError("d must be positive")

    pb = bit_error_probability(snr_db)

    return 1 - (1 - pb) ** d


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Map bits to BPSK symbols: 0 -> +1, 1 -> -1.
    """

    return 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)


def awgn(
    symbols: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add white Gaussian noise for unit-energy symbols.

    With Eb = 1 and Eb/N0 = gamma, the per-dimension noise
    variance is N0 / 2 = 1 / (2 gamma).
    """

    sigma = math.sqrt(1 / (2 * snr_linear(snr_db)))

    return symbols + sigma * rng.standard_normal(symbols.shape)


def hard_demodulate(received: np.ndarray) -> np.ndarray:
    """
    Hard decision: negative samples decode to bit 1.
    """

    return (received < 0).astype(np.uint8)


def simulate_packet_errors(
    num_packets: int,
    snr_db: float,
    d: int = DEFAULT_PACKET_BITS,
    rng: np.random.Generator | None = None,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    Send random bits through BPSK + AWGN + hard demodulation.

    This is an independent check on `packet_error_probability`:
    it never uses the Q-function.

    Returns
    -------
    np.ndarray
        Boolean array of length num_packets.
        True  = packet erased (at least one bit error)
        False = packet received
    """

    if num_packets < 0:
        raise ValueError("num_packets must be non-negative")

    if d <= 0:
        raise ValueError("d must be positive")

    if rng is None:
        rng = np.random.default_rng()

    erased = np.empty(num_packets, dtype=bool)

    # Chunked so that large simulations stay memory-bounded.
    for start in range(0, num_packets, chunk_size):

        stop = min(start + chunk_size, num_packets)

        bits = rng.integers(
            0,
            2,
            size=(stop - start, d),
            dtype=np.uint8,
        )

        received = awgn(bpsk_modulate(bits), snr_db, rng)

        decoded = hard_demodulate(received)

        erased[start:stop] = np.any(decoded != bits, axis=1)

    return erased


def empirical_packet_error_rate(
    num_packets: int,
    snr_db: float,
    d: int = DEFAULT_PACKET_BITS,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Fraction of simulated packets that are erased.
    """

    if num_packets <= 0:
        raise ValueError("num_packets must be positive")

    return float(
        simulate_packet_errors(num_packets, snr_db, d, rng).mean()
    )


def sample_packet_erasures(
    num_packets: int,
    snr_db: float,
    d: int = DEFAULT_PACKET_BITS,
    rng: np.random.Generator | None = None,
    bit_level: bool = False,
) -> np.ndarray:
    """
    Sample which packets are erased at a given SNR.

    bit_level=False draws Bernoulli(p) erasures from the
    analytical packet-error probability (fast).

    bit_level=True runs the physical BPSK/AWGN simulation.

    Returns
    -------
    np.ndarray
        Boolean array, True = erased.
    """

    if rng is None:
        rng = np.random.default_rng()

    if bit_level:
        return simulate_packet_errors(num_packets, snr_db, d, rng)

    p = packet_error_probability(snr_db, d)

    return rng.random(num_packets) < p
