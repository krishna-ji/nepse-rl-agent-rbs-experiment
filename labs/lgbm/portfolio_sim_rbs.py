#!/usr/bin/env python3
"""
Portfolio-Level Backtester for Raw Ichimoku RBS Signals
=======================================================
Runs BOTH Ichimoku strategies (Kumo Break + T/K Cross) and simulates
trading a fixed रू 10,00,000 (10 Lakh NPR) portfolio with strict rules.

This is the "no ML" baseline — every valid signal competes for portfolio
slots, prioritized by earliest entry date then random tiebreak.

Rules (identical to the ML version):
- 8 max concurrent slots × portfolio / 8 per trade, compounding.
- 1.5% friction deducted from every trade.
- When more signals than free slots on a given day, take them randomly
  (no ML ranking available — fair comparison baseline).

Usage::

    python labs/lgbm/portfolio_sim_rbs.py
"""

import warnings; warnings.filterwarnings("ignore")

import pathlib, sys, datetime
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
sys.path.insert(0, str(PROJECT_ROOT))

from src.rbs.kumo_break import KumoBreak
from src.rbs.tk_cross import TKCross

RUN_DIR = None  # set in main()

INITIAL_CAPITAL   = 1_000_000
MAX_SLOTS         = 8
TRADE_SIZE        = INITIAL_CAPITAL // MAX_SLOTS   # 1,25,000
FRICTION_PCT      = 1.5
SEED              = 42

# ============================================================================
# HELPERS
# ============================================================================

def _npr(v: float) -> str:
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e7:
        return f"{sign}रू {av / 1e7:,.2f} Cr"
    if av >= 1e5:
        return f"{sign}रू {av / 1e5:,.2f} L"
    return f"{sign}रू {av:,.0f}"


def npr_formatter(x, _):
    if abs(x) >= 1e7:
        return f"रू {x / 1e7:.1f}Cr"
    if abs(x) >= 1e5:
        return f"रू {x / 1e5:.1f}L"
    return f"रू {x:,.0f}"


# ============================================================================
# GENERATE TRADES FROM BOTH STRATEGIES
# ============================================================================

def generate_trades() -> pd.DataFrame:
    """
    Run both Ichimoku strategies on all tickers and return a combined
    DataFrame of trades with a 'strategy' column.
    """
    # Run Kumo Break
    print("[KUMO] Running Kumo Break backtest...")
    kumo = KumoBreak()
    kumo.run()
    df_kumo = kumo.trades_df
    print(f"[KUMO] Generated {len(df_kumo)} trades")

    # Run T/K Cross — share loaded data to avoid double I/O
    print("[TK]   Running T/K Cross backtest...")
    tk = TKCross()
    tk._frames = kumo._frames  # share data
    tk.run()
    df_tk = tk.trades_df
    print(f"[TK]   Generated {len(df_tk)} trades")

    combined = pd.concat([df_kumo, df_tk], ignore_index=True)
    combined["entry_date"] = pd.to_datetime(combined["entry_date"])
    combined["exit_date"]  = pd.to_datetime(combined["exit_date"])
    combined = combined.sort_values("entry_date").reset_index(drop=True)
    return combined


# ============================================================================
# SIMULATION
# ============================================================================

def simulate(signals: pd.DataFrame):
    """
    Day-by-day portfolio simulation with COMPOUNDING.
    Trade size = current portfolio value / MAX_SLOTS (profits reinvested).
    """
    np.random.seed(SEED)

    sig_by_day = signals.groupby("entry_date")
    all_dates  = sorted(signals["entry_date"].unique())
    exit_dates = signals["exit_date"].dropna().unique()
    calendar   = sorted(set(all_dates) | set(exit_dates))

    open_trades: list[dict] = []
    trades_log:  list[dict] = []
    portfolio_value = float(INITIAL_CAPITAL)
    skipped     = 0

    equity_dates  = []
    equity_values = []

    for day in calendar:
        # 1. Close trades whose exit_date <= today
        still_open = []
        for t in open_trades:
            if t["exit_date"] <= day:
                adj_pnl_pct = t["net_pnl_pct"] - FRICTION_PCT
                trade_sz    = t["trade_size"]
                pnl_npr     = trade_sz * adj_pnl_pct / 100.0
                portfolio_value += pnl_npr
                trades_log.append({
                    "ticker":       t["ticker"],
                    "strategy":     t["strategy"],
                    "entry_date":   t["entry_date"],
                    "exit_date":    t["exit_date"],
                    "direction":    t["direction"],
                    "raw_pnl_pct":  t["net_pnl_pct"],
                    "friction_pct": FRICTION_PCT,
                    "adj_pnl_pct":  adj_pnl_pct,
                    "trade_size":   trade_sz,
                    "pnl_npr":      pnl_npr,
                })
            else:
                still_open.append(t)
        open_trades = still_open

        # 2. Open new trades — trade size = portfolio / MAX_SLOTS (compounding)
        free_slots = MAX_SLOTS - len(open_trades)
        if free_slots > 0 and day in sig_by_day.groups:
            day_signals = sig_by_day.get_group(day).copy()
            day_signals = day_signals.sample(frac=1.0, random_state=SEED)
            trade_sz = max(portfolio_value / MAX_SLOTS, 0)

            for _, row in day_signals.iterrows():
                if free_slots <= 0:
                    skipped += 1
                    continue
                open_trades.append({
                    "ticker":      row["ticker"],
                    "strategy":    row["strategy"],
                    "entry_date":  row["entry_date"],
                    "exit_date":   row["exit_date"],
                    "direction":   row["direction"],
                    "net_pnl_pct": row["net_pnl_pct"],
                    "trade_size":  trade_sz,
                })
                free_slots -= 1
        elif day in sig_by_day.groups:
            skipped += len(sig_by_day.get_group(day))

        # 3. Daily equity with linear mark-to-market
        mtm = 0.0
        for t in open_trades:
            total_days = (t["exit_date"] - t["entry_date"]).days
            elapsed    = (day - t["entry_date"]).days
            frac = elapsed / total_days if total_days > 0 else 0.0
            adj_pnl_pct = t["net_pnl_pct"] - FRICTION_PCT
            mtm += t["trade_size"] * adj_pnl_pct / 100.0 * frac

        equity = portfolio_value + mtm
        equity_dates.append(day)
        equity_values.append(equity)

    equity_ts = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates),
                          name="equity")
    return trades_log, equity_ts, skipped


# ============================================================================
# DRAWDOWN
# ============================================================================

def max_drawdown(equity: pd.Series) -> tuple[float, float]:
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

def plot_equity_dual(eq_kumo: pd.Series, eq_tk: pd.Series, eq_combined: pd.Series,
                     trades_kumo: list, trades_tk: list, trades_combined: list,
                     output_path: pathlib.Path):
    """Three equity curves (Kumo, T/K, Combined) + drawdown panel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    ax1.plot(eq_kumo.index, eq_kumo.values, color="#e8710a", linewidth=1.3,
             label=f"Kumo Break ({len(trades_kumo)} trades)", alpha=0.9)
    ax1.plot(eq_tk.index, eq_tk.values, color="#1a73e8", linewidth=1.3,
             label=f"T/K Cross ({len(trades_tk)} trades)", alpha=0.9)
    ax1.plot(eq_combined.index, eq_combined.values, color="#34a853", linewidth=1.8,
             label=f"Combined ({len(trades_combined)} trades)", linestyle="--", alpha=0.8)
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", linewidth=0.8,
                label=f"Initial Capital ({_npr(INITIAL_CAPITAL)})")
    ax1.set_ylabel("Portfolio Value (NPR)", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(npr_formatter))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Ichimoku RBS  ·  Kumo Break vs T/K Cross  ·  Compounding (Post-2010)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    for eq, color, label in [
        (eq_kumo, "#e8710a", "Kumo Break"),
        (eq_tk, "#1a73e8", "T/K Cross"),
    ]:
        rm = eq.cummax()
        dd = ((eq - rm) / rm) * 100
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.8, label=label, alpha=0.7)
        ax2.fill_between(dd.index, 0, dd.values, color=color, alpha=0.12)

    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    def _stats(trades, eq):
        n = len(trades)
        pnl = eq.iloc[-1] - INITIAL_CAPITAL if len(eq) > 0 else 0
        w = sum(1 for t in trades if t["adj_pnl_pct"] > 0)
        wr = w / n * 100 if n else 0
        rm = eq.cummax()
        mdd = ((eq - rm) / rm * 100).min() if len(eq) > 0 else 0
        return n, pnl, wr, mdd

    n_k, pnl_k, wr_k, mdd_k = _stats(trades_kumo, eq_kumo)
    n_t, pnl_t, wr_t, mdd_t = _stats(trades_tk, eq_tk)
    textstr = (f"Kumo: {n_k} trades, WR {wr_k:.0f}%, PnL {_npr(pnl_k)}, DD {mdd_k:.1f}%\n"
               f"T/K:  {n_t} trades, WR {wr_t:.0f}%, PnL {_npr(pnl_t)}, DD {mdd_t:.1f}%")
    ax1.text(0.5, 0.03, textstr, transform=ax1.transAxes, fontsize=9,
             ha="center", va="bottom", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved dual equity curve → {output_path.relative_to(PROJECT_ROOT)}")


# ============================================================================
# PRINT STRATEGY SUMMARY
# ============================================================================

def _print_strategy_summary(name: str, trades: list, equity: pd.Series):
    n_exec = len(trades)
    if n_exec == 0:
        print(f"\n  {name}: No trades."); return
    total_pnl = sum(t["pnl_npr"] for t in trades)
    wins      = sum(1 for t in trades if t["adj_pnl_pct"] > 0)
    losses    = n_exec - wins
    win_rate  = wins / n_exec * 100
    avg_win   = np.mean([t["pnl_npr"] for t in trades if t["adj_pnl_pct"] > 0]) if wins else 0
    avg_loss  = np.mean([t["pnl_npr"] for t in trades if t["adj_pnl_pct"] <= 0]) if losses else 0
    mdd_npr, mdd_pct = max_drawdown(equity)
    avg_pnl = total_pnl / n_exec
    gross_profit = sum(t["pnl_npr"] for t in trades if t["pnl_npr"] > 0)
    gross_loss   = abs(sum(t["pnl_npr"] for t in trades if t["pnl_npr"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"\n  ── {name} ──")
    print(f"  Trades Executed   : {n_exec}")
    print(f"  Net PnL           : {_npr(total_pnl)}  ({total_pnl/INITIAL_CAPITAL*100:+.1f}%)")
    print(f"  Wins / Losses     : {wins} / {losses}  (WR {win_rate:.1f}%)")
    print(f"  Avg Win / Loss    : {_npr(avg_win)} / {_npr(avg_loss)}")
    print(f"  Profit Factor     : {pf:.2f}")
    print(f"  Expectancy        : {_npr(avg_pnl)} ({avg_pnl/TRADE_SIZE*100:+.2f}%)")
    print(f"  Max Drawdown      : {_npr(mdd_npr)} ({mdd_pct:.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global RUN_DIR
    ts = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = PROJECT_ROOT / f"runs/{ts}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PORTFOLIO BACKTESTER  ·  Ichimoku RBS — FULL HISTORY")
    print("=" * 70)
    print(f"  Run directory : runs/{ts}")
    print(f"  Capital       : {_npr(INITIAL_CAPITAL)}")
    print(f"  Max Slots     : {MAX_SLOTS}")
    print(f"  Trade Size    : portfolio / {MAX_SLOTS} (COMPOUNDING)")
    print(f"  Friction      : {FRICTION_PCT}%")
    print(f"  ML Filter     : NONE (raw signals)")
    print("-" * 70)

    # Generate trades from both strategies
    all_signals = generate_trades()
    total_signals = len(all_signals)

    # Filter: trades from 2010 onward
    oos = all_signals[all_signals["entry_date"] >= pd.Timestamp("2010-01-01")].copy()
    print(f"\n[DATA] Total trades (all time): {total_signals}")
    print(f"[DATA] After 2010 filter: {len(oos)}")

    n_kumo = (oos["strategy"] == "Kumo Break").sum()
    n_tk   = (oos["strategy"] == "T/K Cross").sum()
    print(f"       ├── Kumo Break : {n_kumo}")
    print(f"       └── T/K Cross  : {n_tk}")

    oos = oos.sort_values("entry_date").reset_index(drop=True)

    # Simulate each strategy SEPARATELY
    oos_kumo = oos[oos["strategy"] == "Kumo Break"].copy().reset_index(drop=True)
    oos_tk   = oos[oos["strategy"] == "T/K Cross"].copy().reset_index(drop=True)

    print("\n[SIM] Running Kumo Break portfolio...")
    trades_kumo, eq_kumo, skip_kumo = simulate(oos_kumo)
    print(f"       → {len(trades_kumo)} executed, {skip_kumo} skipped")

    print("[SIM] Running T/K Cross portfolio...")
    trades_tk, eq_tk, skip_tk = simulate(oos_tk)
    print(f"       → {len(trades_tk)} executed, {skip_tk} skipped")

    # Also run combined (de-duplicated) for reference
    before_dedup = len(oos)
    oos_combined = oos.drop_duplicates(subset=["ticker", "entry_date"], keep="first")
    oos_combined = oos_combined.sort_values("entry_date").reset_index(drop=True)
    dupes = before_dedup - len(oos_combined)
    if dupes > 0:
        print(f"[DEDUP] Combined: removed {dupes} duplicate ticker+entry_date pairs")
    print("[SIM] Running Combined portfolio...")
    trades_combined, eq_combined, skip_combined = simulate(oos_combined)
    print(f"       → {len(trades_combined)} executed, {skip_combined} skipped")

    # Print per-strategy summaries
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS  —  Ichimoku RBS (Post-2010)")
    print("=" * 70)
    _print_strategy_summary("KUMO BREAK", trades_kumo, eq_kumo)
    _print_strategy_summary("T/K CROSS", trades_tk, eq_tk)
    _print_strategy_summary("COMBINED (both)", trades_combined, eq_combined)
    print("\n" + "=" * 70)

    # Save trades logs
    for name, trades in [("kumo_break", trades_kumo), ("tk_cross", trades_tk),
                          ("combined", trades_combined)]:
        df = pd.DataFrame(trades)
        csv_path = RUN_DIR / f"rbs_{name}_trades.csv"
        df.to_csv(csv_path, index=False)
    print(f"\n[CSV]  Saved trade logs → {RUN_DIR.relative_to(PROJECT_ROOT)}/")

    # Plot dual equity curves
    output_png = RUN_DIR / "rbs_portfolio_equity_curves.png"
    plot_equity_dual(eq_kumo, eq_tk, eq_combined,
                     trades_kumo, trades_tk, trades_combined, output_png)

    print("\nDone.")


if __name__ == "__main__":
    main()
