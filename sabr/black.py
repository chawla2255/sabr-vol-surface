"""Black-76 prices, Greeks and implied volatility on the forward.

Prices are for European options on the forward F with discount factor D:
    call = D * (F N(d1) - K N(d2)),   put = D * (K N(-d2) - F N(-d1)),
    d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T),   d2 = d1 - sigma sqrt T.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(f, k, t, sigma):
    srt = sigma * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * srt**2) / srt
    return d1, d1 - srt


def price(f, k, t, sigma, discount=1.0, kind="call"):
    """Black-76 price; `kind` is "call" or "put"."""
    d1, d2 = _d1d2(f, k, t, sigma)
    if kind == "call":
        return discount * (f * norm.cdf(d1) - k * norm.cdf(d2))
    return discount * (k * norm.cdf(-d2) - f * norm.cdf(-d1))


def greeks(f, k, t, sigma, discount=1.0, kind="call") -> dict:
    """Forward delta, gamma, vega (per 1.00 of vol) and theta per year, holding the forward fixed."""
    d1, d2 = _d1d2(f, k, t, sigma)
    pdf = norm.pdf(d1)
    delta = discount * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1)
    gamma = discount * pdf / (f * sigma * np.sqrt(t))
    vega = discount * f * pdf * np.sqrt(t)
    theta = -discount * f * pdf * sigma / (2 * np.sqrt(t))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(target, f, k, t, discount=1.0, kind="call", lo=1e-4, hi=5.0) -> float:
    """Volatility that reproduces `target`; nan if the price is outside no-arbitrage bounds."""
    intrinsic = discount * max(f - k, 0.0) if kind == "call" else discount * max(k - f, 0.0)
    upper = discount * f if kind == "call" else discount * k
    if not (intrinsic < target < upper):
        return float("nan")
    return brentq(lambda s: price(f, k, t, s, discount, kind) - target, lo, hi, xtol=1e-10)
