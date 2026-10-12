import numpy as np
import pytest

from sabr.black import greeks, implied_vol, price


def test_put_call_parity():
    f, k, t, s, d = 100.0, 95.0, 0.5, 0.2, 0.98
    assert np.isclose(price(f, k, t, s, d, "call") - price(f, k, t, s, d, "put"), d * (f - k))


def test_atm_price_matches_the_closed_form_approximation():
    # ATM forward option: price is about 0.4 * F * sigma * sqrt(T)
    f, t, s = 100.0, 1.0, 0.2
    assert price(f, f, t, s) == pytest.approx(0.3989 * f * s * np.sqrt(t), rel=0.01)


def test_reference_value():
    # Black-76: F=100, K=100, T=1, sigma=20%, D=1 gives 7.9656
    assert price(100, 100, 1, 0.2) == pytest.approx(7.9656, abs=1e-4)


@pytest.mark.parametrize("k,kind", [(80, "put"), (100, "call"), (125, "call"), (95, "put")])
def test_implied_vol_round_trip(k, kind):
    p = price(100, k, 0.75, 0.31, 0.97, kind)
    assert implied_vol(p, 100, k, 0.75, 0.97, kind) == pytest.approx(0.31, abs=1e-8)


def test_prices_outside_arbitrage_bounds_give_nan():
    assert np.isnan(implied_vol(0.0, 100, 100, 1.0))
    assert np.isnan(implied_vol(150.0, 100, 100, 1.0))


def test_vega_matches_a_finite_difference():
    f, k, t, s = 100.0, 110.0, 0.5, 0.25
    h = 1e-5
    fd = (price(f, k, t, s + h) - price(f, k, t, s - h)) / (2 * h)
    assert greeks(f, k, t, s)["vega"] == pytest.approx(fd, rel=1e-6)
