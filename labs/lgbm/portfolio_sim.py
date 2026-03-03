#!/usr/bin/env python3
"""
Portfolio-Level Backtester for Ichimoku + LightGBM Signals
==========================================================
Simulates trading a fixed रू 10,00,000 (10 Lakh NPR) portfolio using
out-of-sample ML signals with strict position-sizing and no compounding.

Rules
-----
- 8 max concurrent slots, fixed 1,25,000 NPR per trade.
- Only top-25 % of ML predictions (by pred_pnl) are eligible.
- Friction: 1.5 % deducted from every trade's net_pnl_pct.
- Profits/losses from closed trades accumulate in a separate cash pile;
  new trades always use exactly 1,25,000 NPR from the base capital.
"""

import warnings; warnings.filterwarnings("ignore")

import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_DIRS     = sorted((PROJECT_ROOT / "runs").glob("20*"))
RUN_DIR      = RUN_DIRS[-1]

OOS_CSV      = RUN_DIR / "oos_signals_with_predictions.csv"
OUTPUT_DIR   = RUN_DIR
OUTPUT_PNG   = OUTPUT_DIR / "portfolio_equity_curve.png"
OUTPUT_CSV   = OUTPUT_DIR / "portfolio_trades.csv"

INITIAL_CAPITAL   = 1_000_000          # रू 10 Lakh
MAX_SLOTS         = 8
TRADE_SIZE        = INITIAL_CAPITAL // MAX_SLOTS   # 1,25,000
FRICTION_PCT      = 1.5                # deducted from every trade
PRED_QUANTILE     = 0.75               # top-25 % threshold

# ============================================================================
# HELPERS
# ============================================================================

def _npr(v: float) -> str:
    """Format a number as NPR with commas."""
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e7:
        return f"{sign}रू {av / 1e7:,.2f} Cr"
    if av >= 1e5:
        return f"{sign}रू {av / 1e5:,.2f} L"
    return f"{sign}रू {av:,.0f}"


def npr_formatter(x, _):
    """Matplotlib tick formatter."""
    if abs(x) >= 1e7:
        return f"रू {x / 1e7:.1f}Cr"
    if abs(x) >= 1e5:
        return f"रू {x / 1e5:.1f}L"
    return f"रू {x:,.0f}"


# ============================================================================
# LOAD & FILTER
# ============================================================================

def load_signals(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load OOS signals, parse dates, apply ML filter."""
    df = pd.read_csv(csv_path, parse_dates=["signal_date", "exit_date"])

    # Keep only filled long trades (the system is long-only Ichimoku)
    df = df[df["filled"] == True].copy()

    # Compute top-25 % threshold on pred_pnl
    threshold = df["pred_pnl"].quantile(PRED_QUANTILE)
    total_signals = len(df)
    df = df[df["pred_pnl"] >= threshold].copy()
    filtered_signals = len(df)

    print(f"[DATA] Loaded {total_signals} filled signals, "
          f"kept top-25 % (pred_pnl >= {threshold:.4f}): {filtered_signals} signals")

    df = df.sort_values("signal_date").reset_index(drop=True)
    return df, total_signals


# ============================================================================
# SIMULATION
# ============================================================================

def simulate(signals: pd.DataFrame):
    """
    Day-by-day simulation with fixed slot management.

    Returns
    -------
    trades_log : list[dict]   – executed trades with PnL
    equity_ts  : pd.Series    – daily equity indexed by date
    skipped    : int          – signals skipped due to full slots
    """

    # Group signals by signal_date for day-level iteration
    sig_by_day = signals.groupby("signal_date")
    all_dates  = sorted(signals["signal_date"].unique())

    # Build a calendar of ALL trading days (signal + exit dates)
    exit_dates = signals["exit_date"].dropna().unique()
    calendar   = sorted(set(all_dates) | set(exit_dates))

    # State
    open_trades: list[dict] = []     # currently open positions
    trades_log:  list[dict] = []     # completed trades
    cash_pile   = 0.0                # accumulated realised PnL (NPR)
    skipped     = 0

    # Daily equity tracking
    equity_dates  = []
    equity_values = []

    for day in calendar:
        # 1. Close any trades whose exit_date <= today
        still_open = []
        for t in open_trades:
            if t["exit_date"] <= day:
                # Realise PnL
                adj_pnl_pct = t["net_pnl_pct"] - FRICTION_PCT
                pnl_npr     = TRADE_SIZE * adj_pnl_pct / 100.0
                cash_pile  += pnl_npr
                trades_log.append({
                    "ticker":        t["ticker"],
                    "signal_date":   t["signal_date"],
                    "exit_date":     t["exit_date"],
                    "direction":     t["direction"],
                    "raw_pnl_pct":   t["net_pnl_pct"],
                    "friction_pct":  FRICTION_PCT,
                    "adj_pnl_pct":   adj_pnl_pct,
                    "trade_size":    TRADE_SIZE,
                    "pnl_npr":       pnl_npr,
                    "pred_pnl":      t["pred_pnl"],
                })
            else:
                still_open.append(t)
        open_trades = still_open

        # 2. Open new trades if today is a signal day and slots available
        free_slots = MAX_SLOTS - len(open_trades)
        if free_slots > 0 and day in sig_by_day.groups:
            day_signals = sig_by_day.get_group(day).copy()
            # Sort by pred_pnl descending – best predictions first
            day_signals = day_signals.sort_values("pred_pnl", ascending=False)

            for _, row in day_signals.iterrows():
                if free_slots <= 0:
                    skipped += 1
                    continue
                open_trades.append({
                    "ticker":      row["ticker"],
                    "signal_date": row["signal_date"],
                    "exit_date":   row["exit_date"],
                    "direction":   row["direction"],
                    "net_pnl_pct": row["net_pnl_pct"],
                    "pred_pnl":    row["pred_pnl"],
                })
                free_slots -= 1

            # Count remaining day signals that couldn't be placed
            if free_slots == 0:
                remaining = len(day_signals) - (MAX_SLOTS - len(open_trades) + (MAX_SLOTS - len(open_trades)))
                # already counted in the loop via `continue`
                pass
        elif day in sig_by_day.groups:
            # All slots full — every signal today is skipped
            skipped += len(sig_by_day.get_group(day))

        # 3. Compute daily equity = base capital + cash_pile + mark-to-market of open trades
        #    For mark-to-market we linearly interpolate each open trade's PnL
        #    between entry and exit (since we only have entry/exit prices).
        mtm = 0.0
        for t in open_trades:
            total_days = (t["exit_date"] - t["signal_date"]).days
            elapsed    = (day - t["signal_date"]).days
            if total_days > 0:
                frac = elapsed / total_days
            else:
                frac = 0.0
            adj_pnl_pct = t["net_pnl_pct"] - FRICTION_PCT
            mtm += TRADE_SIZE * adj_pnl_pct / 100.0 * frac

        equity = INITIAL_CAPITAL + cash_pile + mtm
        equity_dates.append(day)
        equity_values.append(equity)

    equity_ts = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), name="equity")
    return trades_log, equity_ts, skipped


# ============================================================================
# DRAWDOWN
# ============================================================================

def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """
    Compute max drawdown in NPR and as a percentage.

    Returns (mdd_npr, mdd_pct)
    """
    running_max = equity.cummax()
    drawdown    = equity - running_max
    mdd_npr     = drawdown.min()
    idx         = drawdown.idxmin()
    peak        = running_max.loc[idx]
    mdd_pct     = (mdd_npr / peak) * 100 if peak != 0 else 0.0
    return mdd_npr, mdd_pct


# ============================================================================
# PLOTTING
# ============================================================================

def plot_equity(equity: pd.Series, trades: list[dict], mdd_pct: float,
                output_path: pathlib.Path):
    """Equity curve + drawdown ribbon."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    # --- Equity curve ---
    ax1.plot(equity.index, equity.values, color="#1a73e8", linewidth=1.3,
             label="Portfolio Equity")
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8,
                label=f"Initial Capital ({_npr(INITIAL_CAPITAL)})")
    ax1.fill_between(equity.index, INITIAL_CAPITAL, equity.values,
                     where=equity.values >= INITIAL_CAPITAL,
                     color="#34a853", alpha=0.15, interpolate=True)
    ax1.fill_between(equity.index, INITIAL_CAPITAL, equity.values,
                     where=equity.values < INITIAL_CAPITAL,
                     color="#ea4335", alpha=0.15, interpolate=True)
    ax1.set_ylabel("Portfolio Value (NPR)", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(npr_formatter))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Ichimoku + LightGBM  ·  Portfolio Equity Curve", fontsize=13,
                  fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # --- Drawdown ---
    running_max = equity.cummax()
    dd_pct = ((equity - running_max) / running_max) * 100
    ax2.fill_between(dd_pct.index, 0, dd_pct.values, color="#ea4335", alpha=0.4)
    ax2.plot(dd_pct.index, dd_pct.values, color="#ea4335", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Summary box
    total_pnl = equity.iloc[-1] - INITIAL_CAPITAL
    n_trades  = len(trades)
    wins      = sum(1 for t in trades if t["adj_pnl_pct"] > 0)
    wr        = (wins / n_trades * 100) if n_trades else 0
    textstr   = (f"Trades: {n_trades}  |  Win Rate: {wr:.1f}%  |  "
                 f"Net PnL: {_npr(total_pnl)}  |  Max DD: {mdd_pct:.1f}%")
    ax1.text(0.5, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
             ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved equity curve → {output_path.relative_to(PROJECT_ROOT)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  PORTFOLIO BACKTESTER  ·  Ichimoku + LightGBM ML Filter")
    print("=" * 70)
    print(f"  Run directory : {RUN_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Capital       : {_npr(INITIAL_CAPITAL)}")
    print(f"  Max Slots     : {MAX_SLOTS}")
    print(f"  Trade Size    : {_npr(TRADE_SIZE)}")
    print(f"  Friction      : {FRICTION_PCT}%")
    print(f"  ML Filter     : Top {int((1-PRED_QUANTILE)*100)}% by pred_pnl")
    print("-" * 70)

    # Load
    signals, total_before_filter = load_signals(OOS_CSV)

    # Simulate
    trades, equity, skipped = simulate(signals)

    # Metrics
    n_exec     = len(trades)
    n_skipped  = total_before_filter - n_exec  # total signals minus executed
    total_pnl  = sum(t["pnl_npr"] for t in trades)
    wins       = sum(1 for t in trades if t["adj_pnl_pct"] > 0)
    losses     = n_exec - wins
    win_rate   = (wins / n_exec * 100) if n_exec else 0.0
    avg_win    = (np.mean([t["pnl_npr"] for t in trades if t["adj_pnl_pct"] > 0])
                  if wins else 0)
    avg_loss   = (np.mean([t["pnl_npr"] for t in trades if t["adj_pnl_pct"] <= 0])
                  if losses else 0)
    mdd_npr, mdd_pct = max_drawdown(equity)

    # Expectancy
    if n_exec > 0:
        avg_pnl = total_pnl / n_exec
        expectancy_ratio = avg_pnl / TRADE_SIZE * 100
    else:
        avg_pnl = expectancy_ratio = 0

    # Profit factor
    gross_profit = sum(t["pnl_npr"] for t in trades if t["pnl_npr"] > 0)
    gross_loss   = abs(sum(t["pnl_npr"] for t in trades if t["pnl_npr"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Print summary
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Total Signals (filled)       : {total_before_filter}")
    print(f"  After ML Filter (top 25%)    : {len(signals)}")
    print(f"  Trades Executed              : {n_exec}")
    print(f"  Trades Skipped (slots full)  : {skipped}")
    print("-" * 70)
    print(f"  Final Portfolio Value        : {_npr(INITIAL_CAPITAL + total_pnl)}")
    print(f"  Total Net PnL               : {_npr(total_pnl)}")
    print(f"  Return on Capital            : {total_pnl / INITIAL_CAPITAL * 100:+.2f}%")
    print("-" * 70)
    print(f"  Wins / Losses                : {wins} / {losses}")
    print(f"  Win Rate                     : {win_rate:.1f}%")
    print(f"  Avg Win (NPR)               : {_npr(avg_win)}")
    print(f"  Avg Loss (NPR)              : {_npr(avg_loss)}")
    print(f"  Profit Factor               : {profit_factor:.2f}")
    print(f"  Expectancy per Trade         : {_npr(avg_pnl)} ({expectancy_ratio:+.2f}%)")
    print("-" * 70)
    print(f"  Max Drawdown (NPR)          : {_npr(mdd_npr)}")
    print(f"  Max Drawdown (%)            : {mdd_pct:.2f}%")
    print("=" * 70)

    # Save trades log
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[CSV]  Saved trade log → {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")

    # Plot
    plot_equity(equity, trades, mdd_pct, OUTPUT_PNG)

    # Monthly PnL summary
    if n_exec > 0:
        trades_df["exit_month"] = pd.to_datetime(trades_df["exit_date"]).dt.to_period("M")
        monthly = trades_df.groupby("exit_month").agg(
            trades=("pnl_npr", "count"),
            pnl_npr=("pnl_npr", "sum"),
            win_rate=("adj_pnl_pct", lambda x: (x > 0).mean() * 100),
        )
        print("\n  Monthly Breakdown:")
        print("  " + "-" * 50)
        for period, row in monthly.iterrows():
            pnl_str = _npr(row["pnl_npr"])
            print(f"  {period}  |  {int(row['trades']):>3} trades  |  "
                  f"WR {row['win_rate']:5.1f}%  |  {pnl_str:>15}")
        print("  " + "-" * 50)

    print("\nDone.")


if __name__ == "__main__":
    main()
