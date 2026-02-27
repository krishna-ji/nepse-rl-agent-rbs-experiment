"""
PNG Plot Generators – training curves, evaluation diagnostics
==============================================================
All plots saved as high-DPI PNGs into the run's plots/ directory.
Uses matplotlib for static PNG output (Plotly HTML stays in visualize.py).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG output
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Style defaults ────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "grid.color": "#333",
    "grid.alpha": 0.5,
    "legend.facecolor": "#2a2a3e",
    "legend.edgecolor": "#555",
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "font.size": 10,
})


# ═══════════════════════════════════════════════════════════════════════════
#  Training Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_reward(
    timesteps: list[int],
    rewards: list[float],
    output_dir: str | pathlib.Path,
    filename: str = "training_reward.png",
) -> pathlib.Path:
    """Episode reward over training timesteps."""
    out = pathlib.Path(output_dir) / filename
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(timesteps, rewards, color="#00bfff", linewidth=0.8, alpha=0.5, label="Episode Reward")

    # Smoothed line
    if len(rewards) > 10:
        window = max(5, len(rewards) // 20)
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().values
        ax.plot(timesteps, smoothed, color="#ff6347", linewidth=2, label=f"MA({window})")

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training – Episode Reward Curve")
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.savefig(str(out))
    plt.close(fig)
    return out


def plot_training_loss(
    timesteps: list[int],
    policy_losses: list[float],
    value_losses: list[float],
    entropy_losses: list[float],
    output_dir: str | pathlib.Path,
    filename: str = "training_loss.png",
) -> pathlib.Path:
    """Policy loss, value loss, entropy over training."""
    out = pathlib.Path(output_dir) / filename
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Policy loss
    axes[0].plot(timesteps, policy_losses, color="#ff6347", linewidth=1.2)
    axes[0].set_ylabel("Policy Loss")
    axes[0].set_title("Training Losses")
    axes[0].grid(True)

    # Value loss
    axes[1].plot(timesteps, value_losses, color="#00bfff", linewidth=1.2)
    axes[1].set_ylabel("Value Loss")
    axes[1].grid(True)

    # Entropy
    axes[2].plot(timesteps, entropy_losses, color="#32cd32", linewidth=1.2)
    axes[2].set_ylabel("Entropy")
    axes[2].set_xlabel("Timesteps")
    axes[2].grid(True)

    for ax in axes:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.tight_layout()
    fig.savefig(str(out))
    plt.close(fig)
    return out


def plot_eval_reward(
    timesteps: list[int],
    mean_rewards: list[float],
    std_rewards: list[float],
    output_dir: str | pathlib.Path,
    filename: str = "eval_reward.png",
) -> pathlib.Path:
    """Eval callback mean reward ± std over training."""
    out = pathlib.Path(output_dir) / filename
    fig, ax = plt.subplots(figsize=(10, 5))

    mean_r = np.array(mean_rewards)
    std_r = np.array(std_rewards)
    ts = np.array(timesteps)

    ax.plot(ts, mean_r, color="#ffa500", linewidth=2, label="Mean Eval Reward")
    ax.fill_between(ts, mean_r - std_r, mean_r + std_r, alpha=0.2, color="#ffa500")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Eval Reward")
    ax.set_title("Evaluation Reward During Training")
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.savefig(str(out))
    plt.close(fig)
    return out


def plot_learning_rate(
    timesteps: list[int],
    lrs: list[float],
    output_dir: str | pathlib.Path,
    filename: str = "learning_rate.png",
) -> pathlib.Path:
    """Learning rate schedule."""
    out = pathlib.Path(output_dir) / filename
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timesteps, lrs, color="#da70d6", linewidth=1.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.savefig(str(out))
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation / Trajectory Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_portfolio_curve(
    traj: pd.DataFrame,
    ticker: str,
    output_dir: str | pathlib.Path,
    filename: str | None = None,
) -> pathlib.Path:
    """Portfolio equity curve PNG."""
    if filename is None:
        filename = f"portfolio_{ticker}.png"
    out = pathlib.Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(traj["date"])
    pv = traj["portfolio_value"].values

    ax.plot(dates, pv, color="#00bfff", linewidth=1.8, label="Portfolio Value")
    ax.axhline(y=1.0, color="#888", linestyle="--", linewidth=0.8, alpha=0.6, label="Baseline (1.0)")

    # Shade drawdown
    running_max = np.maximum.accumulate(pv)
    dd = (pv - running_max) / (running_max + 1e-10)
    ax2 = ax.twinx()
    ax2.fill_between(dates, dd * 100, 0, alpha=0.25, color="#ff4444", label="Drawdown %")
    ax2.set_ylabel("Drawdown %", color="#ff6666")
    ax2.tick_params(axis="y", labelcolor="#ff6666")

    # Buy/sell markers
    actions = traj["action"].values
    positions = traj["position"].values
    buy_mask = (positions == 0) & (actions == 1)
    sell_mask = (positions == 1) & (actions == 0)
    if buy_mask.any():
        ax.scatter(dates[buy_mask], pv[buy_mask], marker="^", color="lime",
                   s=60, zorder=5, label="BUY", edgecolors="darkgreen", linewidth=0.5)
    if sell_mask.any():
        ax.scatter(dates[sell_mask], pv[sell_mask], marker="v", color="red",
                   s=60, zorder=5, label="SELL", edgecolors="darkred", linewidth=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(f"Portfolio – {ticker}")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.autofmt_xdate()
    fig.savefig(str(out))
    plt.close(fig)
    return out


def plot_price_with_signals(
    traj: pd.DataFrame,
    feat_df: pd.DataFrame,
    ticker: str,
    output_dir: str | pathlib.Path,
    filename: str | None = None,
) -> pathlib.Path:
    """Price chart with SMA200, TSL, buy/sell signals as PNG."""
    if filename is None:
        filename = f"price_signals_{ticker}.png"
    out = pathlib.Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    dates = pd.to_datetime(traj["date"])
    close = traj["close"].values
    actions = traj["action"].values
    positions = traj["position"].values

    # ── Price panel ──
    ax = axes[0]
    ax.plot(dates, close, color="#00bfff", linewidth=1.2, label="Close")

    # SMA200
    try:
        sma200 = feat_df.loc[traj["date"].values, (ticker, "sma200")].values
        ax.plot(dates, sma200, color="#4488ff", linewidth=1, linestyle=":", alpha=0.7, label="SMA200")
    except Exception:
        pass

    # Protected swing low
    try:
        psl = feat_df.loc[traj["date"].values, (ticker, "protected_swing_low")].values
        ax.plot(dates, psl, color="#ffa500", linewidth=1, linestyle="--", alpha=0.7, label="Swing Low")
    except Exception:
        pass

    # TSL
    tsl = traj["tsl_level"].values
    valid_tsl = ~np.isnan(tsl)
    if valid_tsl.any():
        ax.plot(dates[valid_tsl], tsl[valid_tsl], color="red", linewidth=1.2,
                linestyle="--", alpha=0.9, label="TSL")

    # Buy / sell markers
    buy_mask = (positions == 0) & (actions == 1)
    sell_mask = (positions == 1) & (actions == 0)
    if buy_mask.any():
        ax.scatter(dates[buy_mask], close[buy_mask] * 0.98, marker="^", color="lime",
                   s=80, zorder=5, label="BUY", edgecolors="darkgreen", linewidth=0.5)
    if sell_mask.any():
        ax.scatter(dates[sell_mask], close[sell_mask] * 1.02, marker="v", color="red",
                   s=80, zorder=5, label="SELL", edgecolors="darkred", linewidth=0.5)

    # Forced liquidation
    if "forced_liquidation" in traj.columns:
        fl = traj["forced_liquidation"].fillna(False).astype(bool).values
        if fl.any():
            ax.scatter(dates[fl], close[fl] * 1.04, marker="x", color="magenta",
                       s=100, zorder=5, label="FORCED EXIT", linewidth=2)

    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} – RL Trading Signals")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True)

    # ── Stochastic panel ──
    ax2 = axes[1]
    try:
        pct_k = feat_df.loc[traj["date"].values, (ticker, "pct_k")].values * 100
        pct_d = feat_df.loc[traj["date"].values, (ticker, "pct_d")].values * 100
        ax2.plot(dates, pct_k, color="#00bfff", linewidth=1, label="%K")
        ax2.plot(dates, pct_d, color="#ffa500", linewidth=1, label="%D")
        ax2.axhline(20, color="#666", linestyle=":", linewidth=0.7)
        ax2.axhline(80, color="#666", linestyle=":", linewidth=0.7)
        ax2.fill_between(dates, 0, 20, alpha=0.08, color="green")
        ax2.fill_between(dates, 80, 100, alpha=0.08, color="red")
    except Exception:
        pass
    ax2.set_ylabel("Stochastic")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.set_ylim(-5, 105)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(out))
    plt.close(fig)
    return out


def plot_reward_distribution(
    rewards: list[float],
    output_dir: str | pathlib.Path,
    filename: str = "reward_distribution.png",
) -> pathlib.Path:
    """Histogram of episode rewards."""
    out = pathlib.Path(output_dir) / filename
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rewards, bins=40, color="#00bfff", edgecolor="#1e1e2e", alpha=0.8)
    ax.axvline(np.mean(rewards), color="#ff6347", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(rewards):.4f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Training Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(out))
    plt.close(fig)
    return out


def plot_metrics_summary(
    metrics: dict,
    ticker: str,
    output_dir: str | pathlib.Path,
    filename: str | None = None,
) -> pathlib.Path:
    """Summary metrics as a styled table PNG."""
    if filename is None:
        filename = f"metrics_{ticker}.png"
    out = pathlib.Path(output_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    rows = []
    for k, v in metrics.items():
        if isinstance(v, float):
            rows.append([k, f"{v:+.4f}"])
        else:
            rows.append([k, str(v)])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.55, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor("#2a2a3e")
        cell.set_text_props(color="#ccc", fontweight="bold")
        cell.set_edgecolor("#555")

    # Style data rows
    for i in range(1, len(rows) + 1):
        for j in range(2):
            cell = table[i, j]
            cell.set_facecolor("#1e1e2e" if i % 2 == 0 else "#252540")
            cell.set_text_props(color="#ccc")
            cell.set_edgecolor("#444")

    table.scale(1, 1.5)
    ax.set_title(f"Evaluation Metrics – {ticker}", fontsize=12, pad=20)
    fig.savefig(str(out))
    plt.close(fig)
    return out
