"""A full implied-volatility surface from calibrated SABR slices, with arbitrage checks.

Between calibrated expiries the surface interpolates total implied variance
w(k, T) = sigma(k, T)^2 T linearly in T at fixed log-moneyness k = ln(K / F(T)); forwards
are interpolated log-linearly. This is the standard construction: if each slice is free of
butterfly arbitrage and w rises with T, the interpolated surface stays free of calendar
arbitrage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import black
from .black import implied_vol
from .model import SliceFit, calibrate, hagan_vol


@dataclass
class Surface:
    slices: list[SliceFit]          # sorted by maturity
    spot: float
    discount_rate: float            # continuously compounded rate used for discounting

    @property
    def maturities(self) -> np.ndarray:
        return np.array([s.t for s in self.slices])

    def forward(self, t: float) -> float:
        """Log-linear in t from spot (t = 0) through the calibrated forwards; flat carry beyond."""
        ts = np.r_[0.0, self.maturities]
        log_f = np.log(np.r_[self.spot, [s.forward for s in self.slices]])
        if t > ts[-1]:
            carry = (log_f[-1] - log_f[-2]) / (ts[-1] - ts[-2])
            return float(np.exp(log_f[-1] + carry * (t - ts[-1])))
        return float(np.exp(np.interp(t, ts, log_f)))

    def _total_variance(self, i: int, k) -> np.ndarray:
        s = self.slices[i]
        strike = s.forward * np.exp(k)
        return hagan_vol(s.forward, strike, s.t, s.params) ** 2 * s.t

    def vol(self, strike, t: float) -> np.ndarray:
        """Implied volatility at strike(s) and maturity t (in years)."""
        k = np.log(np.asarray(strike, float) / self.forward(t))
        ts = self.maturities
        if t <= ts[0]:
            return np.sqrt(self._total_variance(0, k) / ts[0])
        if t >= ts[-1]:
            return np.sqrt(self._total_variance(len(ts) - 1, k) / ts[-1])
        j = int(np.searchsorted(ts, t))
        w0, w1 = self._total_variance(j - 1, k), self._total_variance(j, k)
        lam = (t - ts[j - 1]) / (ts[j] - ts[j - 1])
        return np.sqrt(((1 - lam) * w0 + lam * w1) / t)

    def quote(self, strike: float, t: float, kind: str = "call") -> dict:
        """Price and Greeks for one European option, priced off the surface."""
        f, d = self.forward(t), float(np.exp(-self.discount_rate * t))
        sigma = float(self.vol(strike, t))
        out = {"forward": f, "vol": sigma, "price": float(black.price(f, strike, t, sigma, d, kind))}
        out.update({k: float(v) for k, v in black.greeks(f, strike, t, sigma, d, kind).items()})
        return out

    # ---------------------------------------------------------------- arbitrage checks
    def calendar_violations(self, k_grid=np.linspace(-0.3, 0.2, 51)) -> int:
        """Grid points where total variance falls between consecutive calibrated expiries."""
        w = np.array([self._total_variance(i, k_grid) for i in range(len(self.slices))])
        return int((np.diff(w, axis=0) < -1e-10).sum())

    def butterfly_violations(self, n: int = 201) -> dict:
        """Per expiry: strikes where call prices are not convex in strike (negative density)."""
        out = {}
        for s in self.slices:
            sd = float(hagan_vol(s.forward, s.forward, s.t, s.params)) * np.sqrt(s.t)
            k = s.forward * np.exp(np.linspace(-3 * sd, 3 * sd, n))
            c = black.price(s.forward, k, s.t, hagan_vol(s.forward, k, s.t, s.params))
            # Convexity on an uneven grid: the slope dC/dK must be non-decreasing in K.
            slope = np.diff(c) / np.diff(k)
            out[round(float(s.t), 4)] = int((np.diff(slope) < -1e-10).sum())
        return out


def market_vols(quotes: pd.DataFrame) -> pd.DataFrame:
    """Add Black implied vols to cleaned quotes, dropping prices outside arbitrage bounds."""
    q = quotes.copy()
    q["iv"] = [implied_vol(r.mid, r.forward, r.strike, r.t, r.discount, r.kind) for r in q.itertuples()]
    return q.dropna(subset=["iv"])


def build_surface(quotes: pd.DataFrame, spot: float, rate: float, beta: float = 1.0, max_sd: float = 4.0) -> tuple[Surface, pd.DataFrame]:
    """Calibrate one SABR slice per expiry and return the surface plus the quotes used."""
    q = market_vols(quotes)
    fits, used = [], []
    for _, g in q.groupby("expiry"):
        t = g.t.iloc[0]
        sd = np.interp(0.0, g.log_moneyness, g.iv) * np.sqrt(t)
        g = g[np.abs(g.log_moneyness) <= max_sd * sd]
        fits.append(calibrate(g.forward.iloc[0], t, g.strike.to_numpy(), g.iv.to_numpy(), beta=beta))
        used.append(g)
    fits.sort(key=lambda s: s.t)
    return Surface(fits, spot, rate), pd.concat(used, ignore_index=True)
