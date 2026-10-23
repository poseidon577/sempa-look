"""
Packet erasure channels with memory, for channel-aware aggregation.

The paper's ATS (eq. 6) assumes every packet is erased independently
with the same probability p. Here a channel instead gives the joint
erasure law of the N packets that carry one message's subpackets.

Layout. Subpacket (slot) n of message b goes to packet b + n * stride
of a continuous packet stream (a diagonal, or convolutional,
interleaver; see `aggregation.aggregate_diagonal`). stride = 1 puts a
message's subpackets in consecutive packets, as the paper's cyclic
layout does for all but the last N - 1 messages of a batch. The
channels below are stationary, so every message sees the same law.

Pattern convention. `pattern_probabilities(N, stride)` returns a
vector Q of length 2^N. Bit n of the index is 1 when slot n survives,
the same convention as `evaluation._state_matrix`, so exact ATS under
the channel is  sum_s Q[s] * phi(union of surviving slots).

Three channels, each parameterised by its average packet erasure
probability p so that results are compared at matched PER:

    IIDErasure            the paper's model
    GilbertElliott        two-state Markov chain over packets (bursts)
    GaussMarkovRayleigh   correlated Rayleigh fading in front of the
                          paper's uncoded BPSK packet (300 bits)
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc

from sempalab.channel.awgn_bpsk import DEFAULT_PACKET_BITS


def state_matrix(N: int) -> np.ndarray:
    """
    (2^N, N) 0/1 matrix: row s marks which slots survive in pattern s.
    """

    return ((np.arange(1 << N)[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int64)


def iid_pattern_probabilities(N: int, p: float) -> np.ndarray:
    survived = state_matrix(N).sum(axis=1)
    return (1 - p) ** survived * p ** (N - survived)


def survival_moments(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Slot survival probabilities r[n] = P(slot n survives) and the joint
    survival matrix R[n, m] = P(slots n and m both survive), R[n, n] = r[n].
    """

    N = int(round(math.log2(len(Q))))
    S = state_matrix(N).astype(float)

    R = S.T @ (Q[:, None] * S)

    return np.diag(R).copy(), R


def _check_p(p: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")


def _check_layout(N: int, stride: int) -> None:
    if N < 1:
        raise ValueError("N must be positive")
    if stride < 1:
        raise ValueError("stride must be a positive integer")


# -------------------------------------------------------------
# i.i.d. erasures
# -------------------------------------------------------------


class IIDErasure:
    """
    Independent Bernoulli(p) packet erasures (the paper's channel).
    """

    def __init__(self, p: float):
        _check_p(p)
        self.p = float(p)

    @property
    def name(self) -> str:
        return "iid"

    @property
    def parameter(self) -> float:
        return 0.0

    def pattern_probabilities(self, N: int, stride: int = 1) -> np.ndarray:
        _check_layout(N, stride)
        return iid_pattern_probabilities(N, self.p)

    def sample_erasures(self, num_packets: int, rng: np.random.Generator) -> np.ndarray:
        return rng.random(num_packets) < self.p


# -------------------------------------------------------------
# Gilbert-Elliott
# -------------------------------------------------------------


class GilbertElliott:
    """
    Two-state Markov chain over the packet stream.

    The state is Good or Bad; a packet sent in state x is erased with
    probability loss[x]. p_gb = P(Good -> Bad), p_bg = P(Bad -> Good)
    per packet. With loss_good = 0 and loss_bad = 1 this is the
    Gilbert channel: erasures are the Bad state itself.

    The state chain's lag-d autocorrelation is (1 - p_gb - p_bg)^d.
    """

    def __init__(self, p_gb: float, p_bg: float, loss_good: float = 0.0, loss_bad: float = 1.0):
        if not (0.0 < p_gb <= 1.0 and 0.0 < p_bg <= 1.0):
            raise ValueError("p_gb and p_bg must be in (0, 1]")
        if not (0.0 <= loss_good <= 1.0 and 0.0 <= loss_bad <= 1.0):
            raise ValueError("loss probabilities must be in [0, 1]")

        self.p_gb = float(p_gb)
        self.p_bg = float(p_bg)
        self.loss = np.array([loss_good, loss_bad], dtype=float)

        self.stationary = np.array([p_bg, p_gb]) / (p_gb + p_bg)
        self.p = float(self.stationary @ self.loss)

    @classmethod
    def from_correlation(
        cls,
        p: float,
        rho: float,
        loss_good: float = 0.0,
        loss_bad: float = 1.0,
    ) -> "GilbertElliott":
        """
        Chain with average erasure probability p and state lag-1
        autocorrelation rho in [0, 1). rho = 0 with the Gilbert losses
        (0, 1) is the i.i.d. channel. For the Gilbert channel the mean
        burst length is 1 / ((1 - p)(1 - rho)) packets.
        """

        _check_p(p)

        if not 0.0 <= rho < 1.0:
            raise ValueError("rho must be in [0, 1)")

        if not loss_good <= p <= loss_bad or loss_good == loss_bad:
            raise ValueError("need loss_good <= p <= loss_bad with loss_good < loss_bad")

        bad_share = (p - loss_good) / (loss_bad - loss_good)

        if not 0.0 < bad_share < 1.0:
            raise ValueError("p must lie strictly between loss_good and loss_bad")

        channel = cls(
            p_gb=bad_share * (1 - rho),
            p_bg=(1 - bad_share) * (1 - rho),
            loss_good=loss_good,
            loss_bad=loss_bad,
        )
        channel.rho = float(rho)
        return channel

    @property
    def name(self) -> str:
        return "gilbert"

    @property
    def parameter(self) -> float:
        return getattr(self, "rho", 1 - self.p_gb - self.p_bg)

    @property
    def transition(self) -> np.ndarray:
        return np.array([[1 - self.p_gb, self.p_gb], [self.p_bg, 1 - self.p_bg]])

    @property
    def mean_burst_length(self) -> float:
        """
        Mean sojourn in the Bad state, in packets.
        """

        return 1 / self.p_bg

    def pattern_probabilities(self, N: int, stride: int = 1) -> np.ndarray:
        """
        Exact law of the survival pattern of slots 0..N-1 placed
        `stride` packets apart (forward algorithm of the hidden chain).
        """

        _check_layout(N, stride)

        T = np.linalg.matrix_power(self.transition, stride)
        S = state_matrix(N)

        # emit[s, x] = P(slot outcome s | state x)
        survive = 1 - self.loss

        def emit(column):
            return np.where(column[:, None] == 1, survive[None, :], self.loss[None, :])

        forward = self.stationary[None, :] * emit(S[:, 0])

        for n in range(1, N):
            forward = (forward @ T) * emit(S[:, n])

        return forward.sum(axis=1)

    def sample_erasures(self, num_packets: int, rng: np.random.Generator) -> np.ndarray:
        states = np.empty(num_packets, dtype=np.int64)
        state = int(rng.random() < self.stationary[1])
        flips = rng.random(num_packets)

        for t in range(num_packets):
            states[t] = state
            leave = self.p_gb if state == 0 else self.p_bg
            if flips[t] < leave:
                state = 1 - state

        return rng.random(num_packets) < self.loss[states]


# -------------------------------------------------------------
# Correlated Rayleigh fading + uncoded BPSK packets
# -------------------------------------------------------------


def packet_error_from_snr(gamma: np.ndarray, d: int = DEFAULT_PACKET_BITS) -> np.ndarray:
    """
    Vectorised uncoded BPSK packet error probability at linear SNR gamma:
    p = 1 - (1 - Q(sqrt(2 gamma)))^d, with Q(sqrt(2 gamma)) = erfc(sqrt(gamma)) / 2.
    """

    pb = 0.5 * erfc(np.sqrt(np.maximum(gamma, 0.0)))
    return -np.expm1(d * np.log1p(-pb))


def rayleigh_average_per(mean_snr: float, d: int = DEFAULT_PACKET_BITS, grid: int = 200_000) -> float:
    """
    E[p(mean_snr * X)] for X ~ Exp(1) (Rayleigh power), by the midpoint
    rule on u = 1 - exp(-X), which is uniform on (0, 1).
    """

    u = (np.arange(grid) + 0.5) / grid
    x = -np.log1p(-u)
    return float(packet_error_from_snr(mean_snr * x, d).mean())


def rayleigh_mean_snr_for_per(p: float, d: int = DEFAULT_PACKET_BITS) -> float:
    """
    Mean linear SNR at which Rayleigh-faded uncoded BPSK has average
    packet error probability p (bisection in dB).
    """

    _check_p(p)

    low, high = -20.0, 80.0

    for _ in range(80):
        mid = 0.5 * (low + high)
        if rayleigh_average_per(10 ** (mid / 10), d) > p:
            low = mid
        else:
            high = mid

    return 10 ** (0.5 * (low + high) / 10)


class GaussMarkovRayleigh:
    """
    Rayleigh fading with first-order Gauss-Markov (AR(1)) memory across
    packets, in front of the paper's uncoded BPSK d-bit packet:

        h_t = alpha h_{t-1} + sqrt(1 - alpha^2) w_t,   w_t ~ CN(0, 1)
        gamma_t = mean_snr * |h_t|^2
        P(packet t erased | gamma_t) = p(gamma_t)

    The fade is constant within a packet. alpha = 0 is i.i.d. fading,
    alpha -> 1 is block fading over the whole message. mean_snr is set
    so that the average erasure probability equals p.

    `pattern_probabilities` integrates the fade by Monte Carlo
    (conditioned on the fades, slot outcomes are independent, so only
    the fades are sampled); the seed makes it deterministic.
    """

    def __init__(
        self,
        p: float,
        alpha: float,
        d: int = DEFAULT_PACKET_BITS,
        samples: int = 400_000,
        seed: int = 0,
    ):
        _check_p(p)

        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")

        self.p = float(p)
        self.alpha = float(alpha)
        self.d = int(d)
        self.samples = int(samples)
        self.seed = int(seed)
        self.mean_snr = rayleigh_mean_snr_for_per(p, d)
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @property
    def name(self) -> str:
        return "rayleigh"

    @property
    def parameter(self) -> float:
        return self.alpha

    @property
    def mean_snr_db(self) -> float:
        return 10 * math.log10(self.mean_snr)

    def _fades(self, num: int, step: float, size: int, rng: np.random.Generator) -> np.ndarray:
        """
        |h|^2 for `size` independent AR(1) runs of length `num`, with
        coefficient `step` between consecutive entries.
        """

        h = (rng.standard_normal((size, num)) + 1j * rng.standard_normal((size, num))) / math.sqrt(2)

        for t in range(1, num):
            h[:, t] = step * h[:, t - 1] + math.sqrt(1 - step**2) * h[:, t]

        return np.abs(h) ** 2

    def pattern_probabilities(self, N: int, stride: int = 1) -> np.ndarray:
        _check_layout(N, stride)

        key = (N, stride)

        if key not in self._cache:
            rng = np.random.default_rng(self.seed)
            power = self._fades(N, self.alpha**stride, self.samples, rng)
            loss = packet_error_from_snr(self.mean_snr * power, self.d)  # (samples, N)

            S = state_matrix(N)
            Q = np.empty(1 << N)

            for s, row in enumerate(S):
                Q[s] = np.prod(np.where(row[None, :] == 1, 1 - loss, loss), axis=1).mean()

            self._cache[key] = Q / Q.sum()

        return self._cache[key]

    def sample_erasures(self, num_packets: int, rng: np.random.Generator) -> np.ndarray:
        power = self._fades(num_packets, self.alpha, 1, rng)[0]
        return rng.random(num_packets) < packet_error_from_snr(self.mean_snr * power, self.d)


CHANNELS = {"iid", "gilbert", "rayleigh"}


def make_channel(name: str, p: float, parameter: float = 0.0, **options):
    """
    Build a channel at average erasure probability p.

        iid        parameter ignored
        gilbert    parameter = state lag-1 correlation rho
        rayleigh   parameter = fading AR(1) coefficient alpha
    """

    if name == "iid":
        return IIDErasure(p)
    if name == "gilbert":
        return GilbertElliott.from_correlation(p, parameter, **options)
    if name == "rayleigh":
        return GaussMarkovRayleigh(p, parameter, **options)

    raise ValueError(f"channel must be one of {sorted(CHANNELS)}")
