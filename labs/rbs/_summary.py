#!/usr/bin/env python3
"""
Generate year-wise and stock-wise summary tables for ALL RBS strategies.
7-strategy head-to-head comparison: Ichimoku (2) + Classic (5).
"""

import pandas as pd
import numpy as np
import pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLIT_DATE = "2024-07-01"

# ──────────────────────────────────────────────────────────────────────
# Strategy registry:  short_name → (display_label, run_directory)
# ──────────────────────────────────────────────────────────────────────
STRATEGIES = {
    "kumo":      ("Ichimoku Kumo Break",      "20260303_223440"),
    "tk":        ("Ichimoku T/K Cross",        "20260303_223459"),
    "ema":       ("EMA Crossover (20/50)",     "20260303_225738"),
    "rsi":       ("RSI Mean Reversion",        "20260303_225819"),
    "boll":      ("Bollinger Breakout",        "20260303_225825"),
    "macd":      ("MACD Signal Cross",         "20260303_225852"),
    "donchian":  ("Donchian Breakout (Turtle)","20260303_225858"),
}


# ─────────────────────── helpers ──────────────────────────────────────
def _pf(s):
    """Profit factor from a Series of net_pnl_pct."""
    w = s[s > 0].sum()
    l = abs(s[s <= 0].sum())
    return w / l if l > 0 else float("inf")


def _load(run_id):
    """Load trades.csv from a run directory."""
    path = PROJECT_ROOT / "runs" / run_id / "trades.csv"
    df = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    return df


# ─────────────────────── per-strategy tables ─────────────────────────
def year_summary(df, label):
    df = df.copy()
    df["year"] = df["exit_date"].dt.year

    rows = []
    for yr, g in df.groupby("year"):
        rows.append({
            "Year":    int(yr),  "Trades": len(g),
            "Winners": int((g["net_pnl_pct"] > 0).sum()),
            "WR%":     round((g["net_pnl_pct"] > 0).mean() * 100, 1),
            "Gross%":  round(g["pnl_pct"].sum(), 2),
            "Net%":    round(g["net_pnl_pct"].sum(), 2),
            "Avg%":    round(g["net_pnl_pct"].mean(), 2),
            "Med%":    round(g["net_pnl_pct"].median(), 2),
            "PF":      round(_pf(g["net_pnl_pct"]), 2),
            "Best%":   round(g["net_pnl_pct"].max(), 2),
            "Worst%":  round(g["net_pnl_pct"].min(), 2),
            "AvgBars": round(g["bars_held"].mean(), 1),
        })
    tbl = pd.DataFrame(rows)

    # Totals
    tbl.loc[len(tbl)] = {
        "Year": "TOTAL", "Trades": len(df),
        "Winners": int((df["net_pnl_pct"] > 0).sum()),
        "WR%":     round((df["net_pnl_pct"] > 0).mean() * 100, 1),
        "Gross%":  round(df["pnl_pct"].sum(), 2),
        "Net%":    round(df["net_pnl_pct"].sum(), 2),
        "Avg%":    round(df["net_pnl_pct"].mean(), 2),
        "Med%":    round(df["net_pnl_pct"].median(), 2),
        "PF":      round(_pf(df["net_pnl_pct"]), 2),
        "Best%":   round(df["net_pnl_pct"].max(), 2),
        "Worst%":  round(df["net_pnl_pct"].min(), 2),
        "AvgBars": round(df["bars_held"].mean(), 1),
    }

    print(f"\n{'='*110}")
    print(f"  {label} — YEAR-WISE SUMMARY")
    print(f"{'='*110}")
    print(tbl.to_string(index=False))


def stock_summary(df, label, top_n=15):
    rows = []
    for tk, g in df.groupby("ticker"):
        rows.append({
            "Ticker":  tk, "Trades": len(g),
            "Winners": int((g["net_pnl_pct"] > 0).sum()),
            "WR%":     round((g["net_pnl_pct"] > 0).mean() * 100, 1),
            "Net%":    round(g["net_pnl_pct"].sum(), 2),
            "Avg%":    round(g["net_pnl_pct"].mean(), 2),
            "PF":      round(_pf(g["net_pnl_pct"]), 2),
            "Best%":   round(g["net_pnl_pct"].max(), 2),
            "Worst%":  round(g["net_pnl_pct"].min(), 2),
            "AvgBars": round(g["bars_held"].mean(), 1),
        })
    tbl = pd.DataFrame(rows).sort_values("Net%", ascending=False).reset_index(drop=True)

    print(f"\n{'='*110}")
    print(f"  {label} — TOP {top_n} STOCKS (by Net P&L)")
    print(f"{'='*110}")
    print(tbl.head(top_n).to_string(index=False))

    print(f"\n{'-'*110}")
    print(f"  {label} — BOTTOM {top_n} STOCKS (by Net P&L)")
    print(f"{'-'*110}")
    print(tbl.tail(top_n).to_string(index=False))

    profitable   = tbl[tbl["Net%"] > 0]
    unprofitable = tbl[tbl["Net%"] <= 0]
    print(f"\n  Profitable tickers: {len(profitable)}/{len(tbl)}  "
          f"(total Net: {profitable['Net%'].sum():+.2f}%)")
    print(f"  Unprofitable tickers: {len(unprofitable)}/{len(tbl)}  "
          f"(total Net: {unprofitable['Net%'].sum():+.2f}%)")
    return tbl


def _strat_col(df, stock_tbl):
    """Build one column of metrics for the head-to-head table."""
    oos = df[df["entry_date"] >= SPLIT_DATE]
    prof = (stock_tbl["Net%"] > 0).sum()
    unpr = (stock_tbl["Net%"] <= 0).sum()
    return [
        len(df),
        f"{(df['net_pnl_pct']>0).mean()*100:.1f}",
        f"{df.loc[df['net_pnl_pct']>0,'net_pnl_pct'].mean():+.2f}" if (df['net_pnl_pct']>0).any() else "N/A",
        f"{df.loc[df['net_pnl_pct']<=0,'net_pnl_pct'].mean():+.2f}" if (df['net_pnl_pct']<=0).any() else "N/A",
        f"{_pf(df['net_pnl_pct']):.2f}",
        f"{df['net_pnl_pct'].mean():+.3f}",
        f"{df['net_pnl_pct'].median():+.2f}",
        f"{df['net_pnl_pct'].max():+.2f}",
        f"{df['net_pnl_pct'].min():+.2f}",
        f"{df['bars_held'].mean():.1f}",
        f"{df['net_pnl_pct'].sum():+.2f}",
        f"{prof}/{len(stock_tbl)}",
        f"{unpr}/{len(stock_tbl)}",
        # OOS section
        len(oos),
        f"{(oos['net_pnl_pct']>0).mean()*100:.1f}" if len(oos) else "N/A",
        f"{_pf(oos['net_pnl_pct']):.2f}" if len(oos) else "N/A",
        f"{oos['net_pnl_pct'].mean():+.3f}" if len(oos) else "N/A",
        f"{oos['net_pnl_pct'].sum():+.2f}" if len(oos) else "N/A",
    ]


# ─────────────────────── main ─────────────────────────────────────────
def main():
    # Load all strategies
    data = {}    # key → DataFrame
    stocks = {}  # key → stock summary table
    for key, (label, run_id) in STRATEGIES.items():
        data[key] = _load(run_id)

    # ── Per-strategy year-wise summaries ──
    for key, (label, _) in STRATEGIES.items():
        year_summary(data[key], label)

    # ── Per-strategy stock-wise summaries ──
    for key, (label, _) in STRATEGIES.items():
        stocks[key] = stock_summary(data[key], label, top_n=15)

    # ══════════════════════════════════════════════════════════════════
    #  7-STRATEGY HEAD-TO-HEAD COMPARISON
    # ══════════════════════════════════════════════════════════════════
    metrics = [
        "Total Trades", "Win Rate %", "Avg Win %", "Avg Loss %",
        "Profit Factor", "Expectancy %", "Median PnL %",
        "Best Trade %", "Worst Trade %", "Avg Bars Held",
        "Net P&L (all) %", "Profitable Tickers", "Unprofitable Tickers",
        "--- OOS Trades ---", "OOS Win Rate %", "OOS Profit Factor",
        "OOS Expectancy %", "OOS Net P&L %",
    ]

    comp = {"Metric": metrics}
    for key, (label, _) in STRATEGIES.items():
        # Use a short column name for readability
        short = label.split("(")[0].strip()
        if len(short) > 16:
            short = short[:16]
        comp[short] = _strat_col(data[key], stocks[key])

    comp_df = pd.DataFrame(comp)

    print(f"\n{'='*160}")
    print("  ALL RBS STRATEGIES — HEAD-TO-HEAD COMPARISON")
    print(f"{'='*160}")
    print(comp_df.to_string(index=False))

    # ── Rank table (sorted by OOS Expectancy) ──
    rank_rows = []
    for key, (label, _) in STRATEGIES.items():
        df = data[key]
        oos = df[df["entry_date"] >= SPLIT_DATE]
        rank_rows.append({
            "Strategy":       label,
            "Trades":         len(df),
            "WR%":            round((df["net_pnl_pct"] > 0).mean() * 100, 1),
            "PF":             round(_pf(df["net_pnl_pct"]), 2),
            "Expectancy%":    round(df["net_pnl_pct"].mean(), 3),
            "OOS_Trades":     len(oos),
            "OOS_WR%":        round((oos["net_pnl_pct"] > 0).mean() * 100, 1) if len(oos) else 0,
            "OOS_PF":         round(_pf(oos["net_pnl_pct"]), 2) if len(oos) else 0,
            "OOS_Expect%":    round(oos["net_pnl_pct"].mean(), 3) if len(oos) else 0,
            "OOS_Net%":       round(oos["net_pnl_pct"].sum(), 2) if len(oos) else 0,
            "AvgBars":        round(df["bars_held"].mean(), 1),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values("OOS_Expect%", ascending=False).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df.index.name = "Rank"

    print(f"\n{'='*160}")
    print("  STRATEGY RANKING (sorted by OOS Expectancy)")
    print(f"{'='*160}")
    print(rank_df.to_string())
    print()


if __name__ == "__main__":
    main()
