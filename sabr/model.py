"""SABR stochastic-volatility model: Hagan's implied-vol formula and per-expiry calibration.

    dF = sigma F^beta dW,   d sigma = nu sigma dZ,   dW dZ = rho dt,   sigma(0) = alpha

Hagan, Kumar, Lesniewski & Woodward (2002) give the Black implied volatility the model
produces at strike K as a closed-form expansion. With beta fixed, the three parameters
have clean roles: alpha sets the level, rho the skew, nu the curvature of the wings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .black import greeks


@dataclass(frozen=True)
class SabrParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def hagan_vol(f, k, t, p: SabrParams):
    """Black (lognormal) implied volatility from Hagan's SABR expansion; vectorised in k."""
    f, k = np.asarray(f, float), np.asarray(k, float)
    a, b, r, n = p.alpha, p.beta, p.rho, p.nu
    omb = 1.0 - b
    fk = f * k
    log_fk = np.log(f / k)
    fk_pow = fk ** (omb / 2)
    z = (n / a) * fk_pow * log_fk
    x = np.log((np.sqrt(1 - 2 * r * z + z * z) + z - r) / (1 - r))
    small = np.abs(z) < 1e-7
    with np.errstate(divide="ignore", invalid="ignore"):
        z_over_x = np.where(small, 1.0 - 0.5 * r * z, z / np.where(small, 1.0, x))
    denom = fk_pow * (1 + omb**2 / 24 * log_fk**2 + omb**4 / 1920 * log_fk**4)
    correction = 1 + (omb**2 / 24 * a * a / fk**omb + r * b * n * a / (4 * fk_pow) + (2 - 3 * r * r) / 24 * n * n) * t
    return a / denom * z_over_x * correction


@dataclass
class SliceFit:
    t: float
    forward: float
    params: SabrParams
    rmse_vol_points: float     # root-mean-square fit error, in volatility points (1.00 = 1%)
    n_quotes: int


def calibrate(f: float, t: float, strikes, market_vols, beta: float = 1.0, weights=None) -> SliceFit:
    """Fit alpha, rho and nu to one expiry's smile by weighted least squares in vol space.

    By default each quote is weighted by its Black vega, which down-weights the far wings
    where implied vols from tiny option prices are the least reliable.
    """
    k = np.asarray(strikes, float)
    vol = np.asarray(market_vols, float)
    if weights is None:
        weights = greeks(f, k, t, vol)["vega"]
    w = np.sqrt(np.asarray(weights, float) / np.max(weights))
    atm = float(np.interp(0.0, np.log(k / f), vol))

    def residuals(x):
        p = SabrParams(alpha=x[0], beta=beta, rho=x[1], nu=x[2])
        return w * (hagan_vol(f, k, t, p) - vol)

    x0 = np.array([atm * f ** (1 - beta), -0.5, 1.0])
    lo, hi = [1e-6, -0.999, 1e-4], [np.inf, 0.999, 10.0]
    sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale=[x0[0], 0.5, 1.0], max_nfev=2000)
    p = SabrParams(alpha=sol.x[0], beta=beta, rho=sol.x[1], nu=sol.x[2])
    rmse = float(np.sqrt(np.mean((hagan_vol(f, k, t, p) - vol) ** 2)) * 100)
    return SliceFit(t=t, forward=f, params=p, rmse_vol_points=rmse, n_quotes=len(k))
