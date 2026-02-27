"""
Post-Inference Telemetry -- Tear Sheet & Trade Ledger
======================================================
Deterministic post-mortem evaluation artifacts:
  - JSON scalar metrics (tear sheet)
  - PNG equity curve (agent vs buy-and-hold baseline)
  - CSV trade ledger with explicit transition tags
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.run_manager import get_logger

# ── Dark theme (matches plots.py) ────────────────────────────────────────
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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
    "font.size": 10,
})


# ═══════════════════════════════════════════════════════════════════════════
#  Trade Ledger
# ═══════════════════════════════════════════════════════════════════════════

def build_trade_ledger(traj: pd.DataFrame) -> pd.DataFrame:
    """Build an enriched trade ledger from a trajectory DataFrame.

    Adds columns:
        transition   : "BUY (0->1)", "SELL (1->0)", "FORCED_EXIT", "HOLD_LONG",
                       "HOLD_CASH", or "END"
        exit_type    : "policy_sell" | "chandelier_exit" | None
        trade_id     : integer grouping each round-trip trade
    """
    df = traj.copy()
    n = len(df)

    transitions = []
    exit_types = []

    actions = df["action"].values
    positions = df["position"].values
    forced = df["forced_liquidation"].values if "forced_liquidation" in df.columns else np.zeros(n, dtype=bool)

    for i in range(n):
        a = int(actions[i])
        p = int(positions[i])
        fl = bool(forced[i]) if not pd.isna(forced[i]) else False

        if a == -1:
            transitions.append("END")
            exit_types.append(None)
        elif p == 0 and a == 1:
            transitions.append("BUY (0->1)")
            exit_types.append(None)
        elif p == 1 and a == 0:
            transitions.append("SELL (1->0)")
            exit_types.append("policy_sell")
        elif p == 1 and a == 1 and fl:
            transitions.append("FORCED_EXIT")
            exit_types.append("chandelier_exit")
        elif p == 1 and a == 1:
            transitions.append("HOLD_LONG")
            exit_types.append(None)
        elif p == 0 and a == 0:
            transitions.append("HOLD_CASH")
            exit_types.append(None)
        else:
            transitions.append("UNKNOWN")
            exit_types.append(None)

    df["transition"] = transitions
    df["exit_type"] = exit_types

    # Assign trade IDs -- each BUY starts a new trade
    trade_ids = []
    current_trade = 0
    in_trade = False
    for t in transitions:
        if t == "BUY (0->1)":
            current_trade += 1
            in_trade = True
        elif t in ("SELL (1->0)", "FORCED_EXIT"):
            in_trade = False
        trade_ids.append(current_trade if in_trade or t in ("SELL (1->0)", "FORCED_EXIT") else 0)

    df["trade_id"] = trade_ids
    return df


def export_trade_ledger(
    traj: pd.DataFrame,
    run_dir: str | pathlib.Path,
    ticker: str,
) -> pathlib.Path:
    """Export enriched trade ledger CSV to run_dir."""
    log = get_logger("rl_nepse.metrics")
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ledger = build_trade_ledger(traj)
    csv_path = run_dir / f"{ticker}_trade_ledger.csv"
    ledger.to_csv(csv_path, index=False)
    log.info(f"Trade ledger CSV -> {csv_path}")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════
#  Tear Sheet
# ═══════════════════════════════════════════════════════════════════════════

def generate_tear_sheet(
    traj: pd.DataFrame,
    run_dir: str | pathlib.Path,
    ticker: str,
) -> dict[str, Any]:
    """Generate a full tear sheet: JSON metrics + PNG equity curve.

    Vectorized computations:
        - Log returns & cumulative returns (agent)
        - Buy-and-hold baseline cumulative returns
        - Maximum drawdown (agent & baseline)
        - Annualized Sharpe ratio (252 periods)
        - Trade win rate (round-trip trades)
        - Profit factor, avg win / avg loss, exposure %

    Artifacts:
        {run_dir}/{ticker}_tear_sheet.json
        {run_dir}/{ticker}_equity_curve.png  (dpi=300)

    Returns
    -------
    dict of scalar metrics
    """
    log = get_logger("rl_nepse.metrics")
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pv = traj["portfolio_value"].values.astype(np.float64)
    close = traj["close"].values.astype(np.float64)
    actions = traj["action"].values
    positions = traj["position"].values
    n = len(pv)

    # ── Agent returns ──
    agent_log_rets = np.diff(np.log(pv + 1e-10))
    agent_cum_ret = pv / pv[0]  # normalized equity

    # ── Buy-and-hold baseline ──
    baseline_cum_ret = close / close[0]
    baseline_log_rets = np.diff(np.log(close + 1e-10))

    # ── Drawdowns ──
    agent_running_max = np.maximum.accumulate(agent_cum_ret)
    agent_drawdown = (agent_cum_ret - agent_running_max) / (agent_running_max + 1e-10)
    max_drawdown_agent = float(agent_drawdown.min())

    baseline_running_max = np.maximum.accumulate(baseline_cum_ret)
    baseline_drawdown = (baseline_cum_ret - baseline_running_max) / (baseline_running_max + 1e-10)
    max_drawdown_baseline = float(baseline_drawdown.min())

    # ── Sharpe (annualized, 252 periods) ──
    if np.std(agent_log_rets) > 1e-10:
        sharpe = float((np.mean(agent_log_rets) / np.std(agent_log_rets)) * np.sqrt(252))
    else:
        sharpe = 0.0

    # ── Total return ──
    total_return_agent = float(pv[-1] / pv[0] - 1.0)
    total_return_baseline = float(close[-1] / close[0] - 1.0)

    # ── Trade analysis (round-trips) ──
    ledger = build_trade_ledger(traj)
    trade_ids = ledger["trade_id"].values
    unique_trades = sorted(set(trade_ids) - {0})

    wins = 0
    losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    trade_returns: list[float] = []

    for tid in unique_trades:
        mask = trade_ids == tid
        trade_rows = ledger[mask]
        if len(trade_rows) < 2:
            continue
        entry_price = trade_rows["close"].iloc[0]
        exit_price = trade_rows["close"].iloc[-1]
        pnl = (exit_price / entry_price) - 1.0
        trade_returns.append(pnl)
        if pnl > 0:
            wins += 1
            total_win_pnl += pnl
        else:
            losses += 1
            total_loss_pnl += abs(pnl)

    num_trades = wins + losses
    win_rate = wins / num_trades if num_trades > 0 else 0.0
    avg_win = total_win_pnl / wins if wins > 0 else 0.0
    avg_loss = total_loss_pnl / losses if losses > 0 else 0.0
    profit_factor = total_win_pnl / (total_loss_pnl + 1e-10) if total_loss_pnl > 0 else float("inf") if total_win_pnl > 0 else 0.0

    # Exposure (% of time in market)
    in_market = (positions[:-1] == 1).sum() if n > 1 else 0
    exposure_pct = float(in_market / max(n - 1, 1))

    # Forced liquidation count
    forced_count = int(traj.get("forced_liquidation", pd.Series([False])).sum())

    # ── Assemble metrics ──
    metrics: dict[str, Any] = {
        "ticker": ticker,
        "total_return_agent": round(total_return_agent, 6),
        "total_return_baseline": round(total_return_baseline, 6),
        "excess_return": round(total_return_agent - total_return_baseline, 6),
        "annualized_sharpe": round(sharpe, 4),
        "max_drawdown_agent": round(max_drawdown_agent, 6),
        "max_drawdown_baseline": round(max_drawdown_baseline, 6),
        "num_trades": num_trades,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "exposure_pct": round(exposure_pct, 4),
        "forced_liquidations": forced_count,
        "episode_length": n,
        "final_portfolio_value": round(float(pv[-1]), 6),
    }

    # ── JSON artifact ──
    json_path = run_dir / f"{ticker}_tear_sheet.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Tear sheet JSON -> {json_path}")

    # ── PNG equity curve ──
    png_path = _plot_equity_curve(
        traj=traj,
        agent_cum_ret=agent_cum_ret,
        baseline_cum_ret=baseline_cum_ret,
        agent_drawdown=agent_drawdown,
        metrics=metrics,
        ticker=ticker,
        output_path=run_dir / f"{ticker}_equity_curve.png",
    )
    log.info(f"Equity curve PNG -> {png_path}")

    # ── Log summary ──
    log.info(f"--- Tear Sheet: {ticker} ---")
    for k, v in metrics.items():
        if k == "ticker":
            continue
        log.info(f"  {k:>25s}: {v}")
    log.info("-" * 40)

    return metrics


def _plot_equity_curve(
    traj: pd.DataFrame,
    agent_cum_ret: np.ndarray,
    baseline_cum_ret: np.ndarray,
    agent_drawdown: np.ndarray,
    metrics: dict,
    ticker: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Plot normalized equity curve: Agent vs Buy-and-Hold, with drawdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(traj["date"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Equity panel ──
    ax1.plot(dates, agent_cum_ret, color="#00bfff", linewidth=2, label="RL Agent")
    ax1.plot(dates, baseline_cum_ret, color="#888", linewidth=1.5,
             linestyle="--", alpha=0.7, label="Buy & Hold")
    ax1.axhline(1.0, color="#555", linewidth=0.8, linestyle=":")

    # Shade where agent > baseline
    ax1.fill_between(
        dates,
        agent_cum_ret,
        baseline_cum_ret,
        where=agent_cum_ret > baseline_cum_ret,
        alpha=0.15, color="lime", label="Agent outperforms",
    )
    ax1.fill_between(
        dates,
        agent_cum_ret,
        baseline_cum_ret,
        where=agent_cum_ret < baseline_cum_ret,
        alpha=0.15, color="red", label="Agent underperforms",
    )

    # Buy/sell markers on equity curve
    actions = traj["action"].values
    positions = traj["position"].values
    buy_mask = (positions == 0) & (actions == 1)
    sell_mask = (positions == 1) & (actions == 0)
    forced = traj["forced_liquidation"].values if "forced_liquidation" in traj.columns else np.zeros(len(traj), dtype=bool)
    forced = np.array([bool(f) if not pd.isna(f) else False for f in forced])
    forced_mask = (positions == 1) & (actions == 1) & forced

    if buy_mask.any():
        ax1.scatter(dates[buy_mask], agent_cum_ret[buy_mask],
                    marker="^", color="lime", s=50, zorder=5,
                    edgecolors="darkgreen", linewidth=0.5, label="BUY")
    if sell_mask.any():
        ax1.scatter(dates[sell_mask], agent_cum_ret[sell_mask],
                    marker="v", color="red", s=50, zorder=5,
                    edgecolors="darkred", linewidth=0.5, label="SELL")
    if forced_mask.any():
        ax1.scatter(dates[forced_mask], agent_cum_ret[forced_mask],
                    marker="x", color="magenta", s=60, zorder=5,
                    linewidth=2, label="FORCED EXIT")

    # Stats annotation box
    ret_a = metrics["total_return_agent"]
    ret_b = metrics["total_return_baseline"]
    sr = metrics["annualized_sharpe"]
    wr = metrics["win_rate"]
    mdd = metrics["max_drawdown_agent"]
    stats_text = (
        f"Agent Return: {ret_a:+.2%}\n"
        f"B&H Return:   {ret_b:+.2%}\n"
        f"Sharpe:       {sr:+.2f}\n"
        f"Win Rate:     {wr:.1%}\n"
        f"Max DD:       {mdd:.2%}"
    )
    ax1.text(
        0.02, 0.97, stats_text,
        transform=ax1.transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a3e",
                  edgecolor="#555", alpha=0.9),
        color="#ccc",
    )

    ax1.set_ylabel("Normalized Equity")
    ax1.set_title(f"{ticker} - RL Agent vs Buy & Hold", fontsize=13)
    ax1.legend(fontsize=8, ncol=3, loc="upper right")
    ax1.grid(True)

    # ── Drawdown panel ──
    ax2.fill_between(dates, agent_drawdown * 100, 0,
                     color="#ff4444", alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    ax2.set_ylim(top=5)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    return output_path
