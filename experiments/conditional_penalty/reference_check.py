"""Independent continuous-noise check of Eq. (2) on the low-health interval."""

import math
from statistics import NormalDist

import numpy as np

NORMAL = NormalDist()
LOW, HIGH = NORMAL.cdf(-3.0), NORMAL.cdf(3.0)


def tick_threshold(health):
    return -np.log1p(-np.asarray(health) / 700.0)


class LowHealthReference:
    """Enclose expected reference ticks for h<=7.5 by rounding each log increment down and up."""

    def __init__(self, spacing=2**-24):
        self.spacing = spacing
        smallest = math.log1p(0.7 / 350)
        largest = math.log1p(1.3 / 350)
        self.first = math.floor(smallest / spacing)
        last = math.floor(largest / spacing)
        edges = np.arange(self.first, last + 2) * spacing
        z = np.clip((350 * np.expm1(edges) - 1) / 0.1, -3, 3)
        cdf = (np.fromiter(map(NORMAL.cdf, z), dtype=float) - LOW) / (HIGH - LOW)
        mass = np.diff(cdf)
        assert abs(mass.sum() - 1) < 1e-12
        # 7.5 covers the entire historical window for encounters starting at h<=6.
        last_survivable_count = math.ceil(tick_threshold(7.5) / smallest) - 1
        self.distributions = []
        for n in range(1, last_survivable_count + 1):
            length = n * (len(mass) - 1) + 1
            fft_length = 1 << (length - 1).bit_length()
            power = np.fft.rfft(mass, n=fft_length) ** n
            total = np.fft.irfft(power, n=fft_length)[:length].cumsum()
            assert abs(total[-1] - 1) < 3e-12
            self.distributions.append(np.r_[0.0, np.clip(total, 0.0, 1.0), 1.0])

    def bounds(self, health):
        h = np.asarray(health, dtype=float)
        if not np.all((0 <= h) & (h <= 7.5)):
            raise ValueError("Reference check applies to health between zero and 7.5")
        bound = np.ceil(tick_threshold(h) / self.spacing).astype(int)
        lo = (h > 0).astype(float)
        hi = lo.copy()
        for n, cdf in enumerate(self.distributions, 1):
            # Computing P(sum(log increments) < threshold) for T > n.
            pos = bound - n * self.first
            lo += cdf[np.clip(pos - n, 0, len(cdf) - 1)]
            hi += cdf[np.clip(pos, 0, len(cdf) - 1)]
        return lo / 4, hi / 4

    def loss(self, health, after):
        before_lo, before_hi = self.bounds(health)
        after_lo, after_hi = self.bounds(after)
        return np.maximum(0.0, before_lo - after_hi), before_hi - after_lo
