"""Option-chain snapshots from Yahoo Finance, cleaned for calibration.

The discount factor uses the 13-week US Treasury bill yield (^IRX), and each expiry's
forward comes from put-call parity, C - P = D * (F - K), as the median of K + (C - P) / D
over tight-spread near-the-money strikes (a median ignores the stale quotes that make a
regression unreliable, and SPY options are American, so parity is only approximate).
Only out-of-the-money quotes are kept (puts below the forward, calls above), since
those are the liquid ones, and quotes with no bid or a very wide spread are dropped.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TARGET_DAYS = (14, 30, 60, 90, 180, 270, 365, 540)


def pick_expiries(expiries: list[str], today: date, targets=TARGET_DAYS) -> list[str]:
    """The listed expiry closest to each target tenor, without repeats."""
    days = {e: (date.fromisoformat(e) - today).days for e in expiries}
    chosen = []
    for target in targets:
        best = min((e for e in expiries if days[e] >= 7), key=lambda e: abs(days[e] - target))
        if best not in chosen:
            chosen.append(best)
    return chosen


def download_snapshot(ticker: str = "SPY") -> Path:
    """Save today's chains for the chosen expiries to data/<ticker>_<date>.csv."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    asof = hist.index[-1].date()
    spot = float(hist["Close"].iloc[-1])
    frames = []
    for exp in pick_expiries(list(tk.options), asof):
        chain = tk.option_chain(exp)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            frames.append(df.assign(kind=kind, expiry=exp)[["expiry", "kind", "strike", "bid", "ask", "volume", "openInterest"]])
    rate = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100
    out = pd.concat(frames, ignore_index=True).assign(asof=asof.isoformat(), spot=spot, rate=rate)
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{ticker}_{asof.isoformat()}.csv"
    out.to_csv(path, index=False)
    return path


def load_snapshot(path: Path | None = None) -> pd.DataFrame:
    """The given snapshot, or the most recent one in data/."""
    if path is None:
        files = sorted(DATA.glob("*_*.csv"))
        if not files:
            path = download_snapshot()
        else:
            path = files[-1]
    return pd.read_csv(path)


def parity_forward(group: pd.DataFrame, spot: float, discount: float, band: float = 0.03) -> float:
    """Forward for one expiry: median of K + (C - P) / D over tight near-the-money pairs."""
    calls = group[group.kind == "call"].set_index("strike")
    puts = group[group.kind == "put"].set_index("strike")
    k = calls.index.intersection(puts.index)
    k = k[np.abs(k / spot - 1) < band]
    c, p = calls.loc[k], puts.loc[k]
    mid_c, mid_p = (c.bid + c.ask) / 2, (p.bid + p.ask) / 2
    spread = (c.ask - c.bid) + (p.ask - p.bid)
    ok = (c.bid > 0) & (p.bid > 0) & (spread <= spread.median() * 2)
    implied = np.asarray(k[ok], float) + (mid_c - mid_p)[ok].to_numpy() / discount
    return float(np.median(implied))


def drop_arbitrage(q: pd.DataFrame) -> pd.DataFrame:
    """Remove quotes that break monotonicity: moving away from the money, OTM prices must not rise.

    OTM calls are walked upward in strike and OTM puts downward; any quote priced above the
    previous kept quote is stale or crossed, and is dropped.
    """
    keep = []
    for kind, ascending in (("call", True), ("put", False)):
        prev = np.inf
        for i, row in q[q.kind == kind].sort_values("strike", ascending=ascending).iterrows():
            if row.mid <= prev:
                keep.append(i)
                prev = row.mid
    return q.loc[sorted(keep)]


def clean_quotes(snap: pd.DataFrame, max_rel_spread: float = 0.35, min_mid: float = 0.05) -> pd.DataFrame:
    """One row per usable out-of-the-money quote, with forward, discount factor and tenor."""
    asof = date.fromisoformat(snap["asof"].iloc[0])
    spot = float(snap["spot"].iloc[0])
    rows = []
    rate = float(snap["rate"].iloc[0])
    for exp, g in snap.groupby("expiry"):
        t = (date.fromisoformat(exp) - asof).days / 365.0
        disc = float(np.exp(-rate * t))
        fwd = parity_forward(g, spot, disc)
        q = g.assign(mid=(g.bid + g.ask) / 2)
        otm = ((q.kind == "put") & (q.strike < fwd)) | ((q.kind == "call") & (q.strike >= fwd))
        good = (q.bid > 0) & (q.mid >= min_mid) & ((q.ask - q.bid) / q.mid <= max_rel_spread)
        q = drop_arbitrage(q[otm & good]).assign(forward=fwd, discount=disc, t=t)
        rows.append(q)
    out = pd.concat(rows, ignore_index=True)
    out["log_moneyness"] = np.log(out.strike / out.forward)
    return out.sort_values(["t", "strike"]).reset_index(drop=True)
