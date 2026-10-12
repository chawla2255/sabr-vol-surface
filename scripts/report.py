"""Calibrate the SPY surface from the latest snapshot and write charts and tables to docs/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sabr.data import clean_quotes, load_snapshot
from sabr.model import hagan_vol
from sabr.surface import build_surface

DOCS = Path(__file__).resolve().parent.parent / "docs"
plt.rcParams.update({"figure.dpi": 140, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
AMBER, TEAL, SLATE = "#F5A623", "#2EC4B6", "#495A6B"


def main():
    snap = load_snapshot()
    asof, spot, rate = snap["asof"].iloc[0], float(snap["spot"].iloc[0]), float(snap["rate"].iloc[0])
    surf, used = build_surface(clean_quotes(snap), spot, rate)

    # Parameter table
    rows = []
    for s in surf.slices:
        rows.append({"days": round(s.t * 365), "forward": round(s.forward, 2),
                     "ATM vol %": round(float(hagan_vol(s.forward, s.forward, s.t, s.params)) * 100, 2),
                     "alpha": round(s.params.alpha, 4), "rho": round(s.params.rho, 3), "nu": round(s.params.nu, 3),
                     "fit RMSE (vol pts)": round(s.rmse_vol_points, 2), "quotes": s.n_quotes})
    table = pd.DataFrame(rows)
    table.to_csv(DOCS / "calibration.csv", index=False)
    print(table.to_markdown(index=False))

    # Smiles: market quotes vs SABR fit, one panel per expiry
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharey=False)
    for ax, s in zip(axes.flat, surf.slices):
        g = used[np.isclose(used.t, s.t)]
        ax.scatter(g.log_moneyness, g.iv * 100, s=6, color=SLATE, alpha=0.6, label="market")
        k = np.linspace(g.log_moneyness.min(), g.log_moneyness.max(), 200)
        ax.plot(k, hagan_vol(s.forward, s.forward * np.exp(k), s.t, s.params) * 100, color=AMBER, lw=1.6, label="SABR")
        ax.set_title(f"{round(s.t * 365)} days · RMSE {s.rmse_vol_points:.2f} pts", fontsize=8.5)
        ax.set_xlabel("ln(K / F)")
    axes.flat[0].set_ylabel("implied vol %"); axes.flat[4].set_ylabel("implied vol %")
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"SPY smiles, {asof}: market quotes vs SABR fit (beta = 1)")
    fig.tight_layout(); fig.savefig(DOCS / "smiles.png"); plt.close(fig)

    # Parameter term structure
    t = surf.maturities * 365
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, key, label, color in zip(axes, ("alpha", "rho", "nu"), ("alpha (vol level)", "rho (skew)", "nu (vol of vol)"), (AMBER, TEAL, SLATE)):
        ax.plot(t, [getattr(s.params, key) for s in surf.slices], "o-", color=color, ms=4)
        ax.set_title(label); ax.set_xlabel("days to expiry")
    fig.tight_layout(); fig.savefig(DOCS / "parameters.png"); plt.close(fig)

    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ts = np.linspace(surf.maturities[0], surf.maturities[-1], 40)
    ks = np.linspace(-0.25, 0.15, 50)
    z = np.array([surf.vol(surf.forward(tt) * np.exp(ks), tt) for tt in ts]) * 100
    K, T = np.meshgrid(ks, ts * 365)
    # Only draw strikes within 3 standard deviations of each maturity: beyond that SABR is
    # extrapolating far outside the quoted strikes, and its wings are not meaningful.
    atm = np.array([surf.vol(surf.forward(tt), tt) for tt in ts])
    z = np.where(np.abs(K) <= 3 * atm[:, None] * np.sqrt(ts[:, None]), z, np.nan)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(K, T, z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel("ln(K / F)"); ax.set_ylabel("days"); ax.set_zlabel("implied vol %")
    ax.view_init(elev=24, azim=-58)
    ax.set_title(f"SPY implied-volatility surface, {asof} (SABR)")
    fig.tight_layout(); fig.savefig(DOCS / "surface.png"); plt.close(fig)

    print("calendar violations:", surf.calendar_violations())
    print("butterfly violations:", surf.butterfly_violations())
    print("spot", spot, "rate", rate, "asof", asof)


if __name__ == "__main__":
    main()
