#!/usr/bin/env python3
"""Generate year-wise and stock-wise summary tables for Ichimoku strategies."""

import pandas as pd
import numpy as np
import pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Latest runs (filtered: no promoter shares, mutual funds, corp debentures)
KUMO_RUN = PROJECT_ROOT / "runs/20260303_223440"
TK_RUN   = PROJECT_ROOT / "runs/20260303_223459"


def _pf(s):
    """Profit factor from a Series of net_pnl_pct."""
    w = s[s > 0].sum()
    l = abs(s[s <= 0].sum())
    return w / l if l > 0 else float("inf")


def year_summary(df, label):
    df = df.copy()
    df["year"] = df["exit_date"].dt.year

    rows = []
    for yr, g in df.groupby("year"):
        rows.append({
            "Year":       int(yr),
            "Trades":     len(g),
            "Winners":    (g["net_pnl_pct"] > 0).sum(),
            "WR%":        round((g["net_pnl_pct"] > 0).mean() * 100, 1),
            "Gross%":     round(g["pnl_pct"].sum(), 2),
            "Net%":       round(g["net_pnl_pct"].sum(), 2),
            "Avg%":       round(g["net_pnl_pct"].mean(), 2),
            "Med%":       round(g["net_pnl_pct"].median(), 2),
            "PF":         round(_pf(g["net_pnl_pct"]), 2),
            "Best%":      round(g["net_pnl_pct"].max(), 2),
            "Worst%":     round(g["net_pnl_pct"].min(), 2),
            "AvgBars":    round(g["bars_held"].mean(), 1),
        })
    tbl = pd.DataFrame(rows)

    # Totals row
    tbl.loc[len(tbl)] = {
        "Year":    "TOTAL",
        "Trades":  len(df),
        "Winners": (df["net_pnl_pct"] > 0).sum(),
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

    print(f"\n{'='*100}")
    print(f"  {label} — YEAR-WISE SUMMARY")
    print(f"{'='*100}")
    print(tbl.to_string(index=False))


def stock_summary(df, label, top_n=20):
    rows = []
    for tk, g in df.groupby("ticker"):
        rows.append({
            "Ticker":   tk,
            "Trades":   len(g),
            "Winners":  (g["net_pnl_pct"] > 0).sum(),
            "WR%":      round((g["net_pnl_pct"] > 0).mean() * 100, 1),
            "Net%":     round(g["net_pnl_pct"].sum(), 2),
            "Avg%":     round(g["net_pnl_pct"].mean(), 2),
            "PF":       round(_pf(g["net_pnl_pct"]), 2),
            "Best%":    round(g["net_pnl_pct"].max(), 2),
            "Worst%":   round(g["net_pnl_pct"].min(), 2),
            "AvgBars":  round(g["bars_held"].mean(), 1),
        })
    tbl = pd.DataFrame(rows).sort_values("Net%", ascending=False).reset_index(drop=True)

    print(f"\n{'='*100}")
    print(f"  {label} — TOP {top_n} STOCKS (by Net P&L)")
    print(f"{'='*100}")
    print(tbl.head(top_n).to_string(index=False))

    print(f"\n{'-'*100}")
    print(f"  {label} — BOTTOM {top_n} STOCKS (by Net P&L)")
    print(f"{'-'*100}")
    print(tbl.tail(top_n).to_string(index=False))

    # Overall bucket summary
    profitable = tbl[tbl["Net%"] > 0]
    unprofitable = tbl[tbl["Net%"] <= 0]
    print(f"\n  Profitable tickers: {len(profitable)}/{len(tbl)}  "
          f"(total Net: {profitable['Net%'].sum():+.2f}%)")
    print(f"  Unprofitable tickers: {len(unprofitable)}/{len(tbl)}  "
          f"(total Net: {unprofitable['Net%'].sum():+.2f}%)")
    print(f"  Net across all: {tbl['Net%'].sum():+.2f}%")

    return tbl


def main():
    kb = pd.read_csv(KUMO_RUN / "trades.csv", parse_dates=["entry_date", "exit_date"])
    tk = pd.read_csv(TK_RUN / "trades.csv", parse_dates=["entry_date", "exit_date"])

    # ── Year-wise ──
    year_summary(kb, "KUMO BREAK STRATEGY")
    year_summary(tk, "T/K CROSS STRATEGY")

    # ── Stock-wise ──
    kb_stocks = stock_summary(kb, "KUMO BREAK STRATEGY", top_n=20)
    tk_stocks = stock_summary(tk, "T/K CROSS STRATEGY", top_n=20)

    # ── Side-by-side strategy comparison ──
    print(f"\n{'='*100}")
    print("  STRATEGY COMPARISON (Head-to-Head)")
    print(f"{'='*100}")
    comp = pd.DataFrame({
        "Metric": ["Total Trades", "Win Rate %", "Avg Win %", "Avg Loss %",
                    "Profit Factor", "Expectancy %", "Median PnL %",
                    "Best Trade %", "Worst Trade %", "Avg Bars Held",
                    "Net P&L (all) %", "Profitable Tickers", "Unprofitable Tickers"],
        "Kumo Break": [
            len(kb),
            f"{(kb['net_pnl_pct']>0).mean()*100:.1f}",
            f"{kb.loc[kb['net_pnl_pct']>0,'net_pnl_pct'].mean():+.2f}",
            f"{kb.loc[kb['net_pnl_pct']<=0,'net_pnl_pct'].mean():+.2f}",
            f"{_pf(kb['net_pnl_pct']):.2f}",
            f"{kb['net_pnl_pct'].mean():+.3f}",
            f"{kb['net_pnl_pct'].median():+.2f}",
            f"{kb['net_pnl_pct'].max():+.2f}",
            f"{kb['net_pnl_pct'].min():+.2f}",
            f"{kb['bars_held'].mean():.1f}",
            f"{kb['net_pnl_pct'].sum():+.2f}",
            f"{(kb_stocks['Net%']>0).sum()}/{len(kb_stocks)}",
            f"{(kb_stocks['Net%']<=0).sum()}/{len(kb_stocks)}",
        ],
        "T/K Cross": [
            len(tk),
            f"{(tk['net_pnl_pct']>0).mean()*100:.1f}",
            f"{tk.loc[tk['net_pnl_pct']>0,'net_pnl_pct'].mean():+.2f}",
            f"{tk.loc[tk['net_pnl_pct']<=0,'net_pnl_pct'].mean():+.2f}",
            f"{_pf(tk['net_pnl_pct']):.2f}",
            f"{tk['net_pnl_pct'].mean():+.3f}",
            f"{tk['net_pnl_pct'].median():+.2f}",
            f"{tk['net_pnl_pct'].max():+.2f}",
            f"{tk['net_pnl_pct'].min():+.2f}",
            f"{tk['bars_held'].mean():.1f}",
            f"{tk['net_pnl_pct'].sum():+.2f}",
            f"{(tk_stocks['Net%']>0).sum()}/{len(tk_stocks)}",
            f"{(tk_stocks['Net%']<=0).sum()}/{len(tk_stocks)}",
        ],
    })
    print(comp.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
