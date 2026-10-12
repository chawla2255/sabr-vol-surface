import numpy as np
import pytest

from sabr.model import SabrParams, SliceFit
from sabr.surface import Surface


def _surface():
    slices = [
        SliceFit(0.25, 101.0, SabrParams(0.15, 1.0, -0.6, 1.5), 0.0, 0),
        SliceFit(1.00, 103.0, SabrParams(0.18, 1.0, -0.65, 0.8), 0.0, 0),
    ]
    return Surface(slices, spot=100.0, discount_rate=0.04)


def test_surface_reproduces_each_calibrated_slice():
    s = _surface()
    for sl in s.slices:
        k = np.array([90.0, 100.0, 110.0]) * sl.forward / 100
        from sabr.model import hagan_vol
        np.testing.assert_allclose(s.vol(k, sl.t), hagan_vol(sl.forward, k, sl.t, sl.params), rtol=1e-10)


def test_total_variance_interpolates_between_slices():
    s = _surface()
    t = 0.625
    f = s.forward(t)
    w = s.vol(f, t) ** 2 * t
    w0 = s.vol(s.forward(0.25), 0.25) ** 2 * 0.25
    w1 = s.vol(s.forward(1.0), 1.0) ** 2 * 1.0
    assert w0 < w < w1


def test_forward_starts_at_spot_and_hits_calibrated_forwards():
    s = _surface()
    assert s.forward(0.0) == pytest.approx(100.0)
    assert s.forward(0.25) == pytest.approx(101.0)
    assert s.forward(1.0) == pytest.approx(103.0)


def test_quote_returns_a_consistent_price():
    q = _surface().quote(100.0, 0.5, "put")
    assert q["price"] > 0 and -1 < q["delta"] < 0 and q["vega"] > 0


def test_arbitrage_checks_pass_on_a_sensible_surface():
    s = _surface()
    assert s.calendar_violations() == 0
    assert all(v == 0 for v in s.butterfly_violations().values())
