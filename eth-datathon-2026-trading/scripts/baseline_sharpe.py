"""Reproduce the baseline numbers in the README and draw figures/sharpe_progression.png.

- Always long and Expert 1 (mean reversion) are computed here from the training data.
- The full-blend training Sharpe (3.01) and the leaderboard score (2.95) come from
  running strategy.py during the competition (FinBERT + neural network, GPU) and are
  entered as constants.

Run from the repository root:  uv run scripts/baseline_sharpe.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ALPHA, CLIP_LO, CLIP_HI = 0.070, -5, 10
FULL_BLEND_TRAIN, LEADERBOARD = 3.01, 2.95163

seen = pd.read_parquet(DATA / "bars_seen_train.parquet").sort_values(["session", "bar_ix"])
unseen = pd.read_parquet(DATA / "bars_unseen_train.parquet").sort_values(["session", "bar_ix"])
ret = unseen.groupby("session")["close"].last() / seen.groupby("session")["close"].last() - 1

seen["log_hl_sq"] = np.log(seen["high"] / seen["low"]) ** 2
g = seen.groupby("session")
sigma = np.sqrt(g["log_hl_sq"].mean() / (4 * np.log(2))) + 1e-8
z = (g["close"].last() / g["close"].first() - 1) / sigma


def sharpe(pos):
    pnl = np.clip(pos, CLIP_LO, CLIP_HI) * ret
    return pnl.mean() / pnl.std() * 16


always_long = sharpe(np.ones(len(ret)))
expert1 = sharpe(1 - ALPHA * z)
print(f"Always long          : {always_long:.3f}")
print(f"Expert 1 (alpha={ALPHA}) : {expert1:.3f}")

labels = ["Always long (baseline)", "Expert 1: mean reversion", "Full blend: 3 experts", "Hidden test set (leaderboard)"]
values = [always_long, expert1, FULL_BLEND_TRAIN, LEADERBOARD]
colors = ["#898781", "#86b6ef", "#2a78d6", "#eb6834"]
INK, SEC, MUT, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e0"

fig, ax = plt.subplots(figsize=(9, 3.9), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
y = np.arange(len(values))[::-1]
ax.hlines(y, 2.5, 3.12, color=GRID, lw=1, zorder=1)
ax.scatter(values, y, s=150, c=colors, edgecolor="#fcfcfb", linewidth=2, zorder=3)
for yi, v in zip(y, values):
    ax.text(v + 0.015, yi, f"{v:.2f}", va="center", fontsize=10.5, color=INK)
ax.set_yticks(y, labels, color=SEC)
ax.set_xlim(2.5, 3.12)
ax.set_xlabel("Sharpe ratio (annualised)", color=SEC)
ax.set_title("Each expert adds signal; the blend holds up on unseen data", loc="left", color=INK, fontsize=12.5, pad=10)
ax.text(1.0, -0.26, "Rows 1–3: training sessions (in-sample)  ·  row 4: hidden test sessions, official leaderboard",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=MUT)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUT)
ax.tick_params(colors=SEC, length=0)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0, ls=":")
fig.tight_layout()
out = ROOT / "figures" / "sharpe_progression.png"
fig.savefig(out, facecolor=fig.get_facecolor())
print(f"saved {out}")
