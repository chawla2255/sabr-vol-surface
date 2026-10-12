import numpy as np
import pytest

from sabr.black import implied_vol
from sabr.model import SabrParams, calibrate, hagan_vol


def test_no_vol_of_vol_with_beta_one_is_flat_black_vol():
    p = SabrParams(alpha=0.2, beta=1.0, rho=-0.3, nu=1e-8)
    k = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(hagan_vol(100.0, k, 1.0, p), 0.2, atol=1e-6)


def test_atm_formula_is_continuous():
    p = SabrParams(alpha=0.25, beta=1.0, rho=-0.4, nu=0.8)
    atm = hagan_vol(100.0, 100.0, 0.5, p)
    near = hagan_vol(100.0, np.array([99.9999, 100.0001]), 0.5, p)
    np.testing.assert_allclose(near, atm, atol=1e-5)


def test_negative_rho_gives_a_downward_skew():
    p = SabrParams(alpha=0.2, beta=1.0, rho=-0.6, nu=0.9)
    lo, hi = hagan_vol(100.0, np.array([85.0, 115.0]), 0.5, p)
    assert lo > hi


def test_hagan_formula_matches_a_monte_carlo_of_the_sabr_process():
    """Independent check: simulate SABR (beta = 0.5), price calls, invert to Black vols."""
    p = SabrParams(alpha=0.04, beta=0.5, rho=-0.3, nu=0.4)
    f0, t, n, steps = 0.05, 1.0, 80_000, 200
    rng = np.random.default_rng(0)
    dt = t / steps
    f, s = np.full(n, f0), np.full(n, p.alpha)
    for _ in range(steps):
        z1 = rng.standard_normal(n)
        z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n)
        f = np.maximum(f + s * f**p.beta * np.sqrt(dt) * z1, 0.0)
        s = s * np.exp(p.nu * np.sqrt(dt) * z2 - 0.5 * p.nu**2 * dt)
    for k in (0.05, 0.06):
        mc = implied_vol(np.maximum(f - k, 0).mean(), f0, k, t)
        assert hagan_vol(f0, k, t, p) == pytest.approx(mc, abs=0.004)


def test_calibration_recovers_known_parameters():
    true = SabrParams(alpha=0.18, beta=1.0, rho=-0.65, nu=1.2)
    f, t = 100.0, 0.5
    k = np.linspace(70, 125, 40)
    vols = hagan_vol(f, k, t, true)
    fit = calibrate(f, t, k, vols, weights=np.ones_like(k))
    assert fit.params.alpha == pytest.approx(true.alpha, rel=1e-3)
    assert fit.params.rho == pytest.approx(true.rho, abs=1e-3)
    assert fit.params.nu == pytest.approx(true.nu, rel=1e-3)
    assert fit.rmse_vol_points < 1e-4
