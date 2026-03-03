#!/usr/bin/env python3
"""
RBS Strategy Audit — Addressing 3 Critical Backtest Loopholes
=============================================================
1. POST-2024 REGIME GAP   — Year-by-year profit decomposition (is 2020–21 masking death?)
2. SURVIVORSHIP BIAS      — How many trades came from stocks that later died/suspended?
3. EXECUTION SLIPPAGE     — Friction sensitivity analysis (0.5% → 1.0% → 1.5% → 2.0%)

Run:  python labs/rbs/_audit.py
"""

import pandas as pd
import numpy as np
import pathlib, json, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data/ohlcv/1D/stocks"

STRATEGIES = {
    "Kumo Break":       "20260303_223440",
    "T/K Cross":        "20260303_223459",
    "EMA Cross":        "20260303_225738",
    "RSI Reversion":    "20260303_225819",
    "Bollinger":        "20260303_225825",
    "MACD Cross":       "20260303_225852",
    "Donchian":         "20260303_225858",
}

FRICTION_LEVELS = [0.5, 1.0, 1.5, 2.0]   # % round-trip cost


# ──────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────
def _pf(s):
    w = s[s > 0].sum(); l = abs(s[s <= 0].sum())
    return w / l if l > 0 else float("inf")


def _load(run_id):
    path = PROJECT_ROOT / "runs" / run_id / "trades.csv"
    return pd.read_csv(path, parse_dates=["entry_date", "exit_date"])


def _build_zombie_set():
    """Build set of tickers whose data ends before 2026 (suspended / delisted / merged)."""
    zombies = {}
    for csv in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv, parse_dates=["Timestamp"], usecols=["Timestamp"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
            last = df["Timestamp"].max()
            if pd.isna(last):
                zombies[csv.stem] = "empty_file"
            elif last < pd.Timestamp("2025-07-01"):
                zombies[csv.stem] = str(last.date())
        except:
            zombies[csv.stem] = "read_error"
    return zombies


# ══════════════════════════════════════════════════════════════════════
#  AUDIT 1: REGIME GAP — Year-by-year decomposition (2019–2026)
# ══════════════════════════════════════════════════════════════════════
def audit_regime_gap():
    print("\n" + "█" * 120)
    print("  AUDIT 1: POST-2024 REGIME GAP — Is 2020–21 masking a dead strategy?")
    print("█" * 120)
    print("  If Net% is negative for 2023–2026, the strategy is back-fitting to the bull run.\n")

    years = list(range(2019, 2027))

    # Build a matrix: strategy × year
    header = f"{'Strategy':>16s}"
    for y in years:
        header += f"  {y:>10d}"
    header += f"  {'2025+2026':>10s}"
    print(header)
    print("─" * len(header))

    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        df["year"] = df["exit_date"].dt.year
        row = f"{name:>16s}"
        for y in years:
            sub = df[df["year"] == y]
            if len(sub) > 0:
                net = sub["net_pnl_pct"].sum()
                wr  = (sub["net_pnl_pct"] > 0).mean() * 100
                row += f"  {net:>+8.1f}%"
            else:
                row += f"  {'—':>9s}"
        # Combined 2025+2026
        recent = df[df["year"] >= 2025]
        if len(recent) > 0:
            net = recent["net_pnl_pct"].sum()
            row += f"  {net:>+8.1f}%"
        else:
            row += f"  {'—':>9s}"
        print(row)

    # Detail table: trade count + WR + expectancy per year per strategy
    print(f"\n{'─'*120}")
    print("  Detailed breakdown (Trades | WR% | Expectancy%) per year per strategy")
    print(f"{'─'*120}")

    header2 = f"{'Strategy':>16s}"
    for y in years:
        header2 += f"  {'T':>4s} {'WR':>5s} {'E%':>7s}"
    print(header2)
    print("─" * len(header2))

    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        df["year"] = df["exit_date"].dt.year
        row = f"{name:>16s}"
        for y in years:
            sub = df[df["year"] == y]
            if len(sub) > 0:
                n = len(sub)
                wr = (sub["net_pnl_pct"] > 0).mean() * 100
                exp = sub["net_pnl_pct"].mean()
                row += f"  {n:>4d} {wr:>4.0f}% {exp:>+6.2f}"
            else:
                row += f"  {'—':>4s} {'—':>5s} {'—':>7s}"
        print(row)

    # Verdict
    print(f"\n{'─'*120}")
    print("  VERDICT: Strategy 'alive' if 2025+2026 combined expectancy > 0")
    print(f"{'─'*120}")
    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        df["year"] = df["exit_date"].dt.year
        recent = df[df["year"] >= 2025]
        if len(recent) == 0:
            status = "NO DATA"
            exp = 0
        else:
            exp = recent["net_pnl_pct"].mean()
            status = "ALIVE ✓" if exp > 0 else "DEAD ✗"
        print(f"  {name:>16s}:  {status}  (2025-26 expectancy: {exp:+.3f}%, {len(recent)} trades)")


# ══════════════════════════════════════════════════════════════════════
#  AUDIT 2: SURVIVORSHIP BIAS
# ══════════════════════════════════════════════════════════════════════
def audit_survivorship():
    print("\n\n" + "█" * 120)
    print("  AUDIT 2: SURVIVORSHIP BIAS — Trades on stocks that later died / got suspended")
    print("█" * 120)
    print("  A 'zombie' is any ticker whose data ends before 2025-07-01 (likely delisted/suspended/merged).\n")

    zombies = _build_zombie_set()
    print(f"  Total zombie tickers found: {len(zombies)}")
    print(f"  (out of {len(list(DATA_DIR.glob('*.csv')))} CSVs in data folder)\n")

    header = f"{'Strategy':>16s}  {'Total':>6s}  {'Zombie':>6s}  {'%Zombie':>7s}  {'All E%':>8s}  {'Zombie E%':>9s}  {'Clean E%':>8s}  {'Zombie WR%':>10s}  {'Clean WR%':>10s}  {'Zombie PF':>9s}  {'Clean PF':>9s}"
    print(header)
    print("─" * len(header))

    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        is_zombie = df["ticker"].isin(zombies)
        z = df[is_zombie]
        c = df[~is_zombie]

        pct = len(z) / len(df) * 100 if len(df) > 0 else 0
        all_e = df["net_pnl_pct"].mean()
        z_e = z["net_pnl_pct"].mean() if len(z) > 0 else 0
        c_e = c["net_pnl_pct"].mean() if len(c) > 0 else 0
        z_wr = (z["net_pnl_pct"] > 0).mean() * 100 if len(z) > 0 else 0
        c_wr = (c["net_pnl_pct"] > 0).mean() * 100 if len(c) > 0 else 0
        z_pf = _pf(z["net_pnl_pct"]) if len(z) > 0 else 0
        c_pf = _pf(c["net_pnl_pct"]) if len(c) > 0 else 0

        print(f"{name:>16s}  {len(df):>6d}  {len(z):>6d}  {pct:>6.1f}%  {all_e:>+7.3f}  {z_e:>+8.3f}  {c_e:>+7.3f}  {z_wr:>9.1f}%  {c_wr:>9.1f}%  {z_pf:>9.2f}  {c_pf:>9.2f}")

    # Show which zombie tickers contributed the most trades
    print(f"\n{'─'*120}")
    print("  Top 15 zombie tickers by trade count (across all strategies combined)")
    print(f"{'─'*120}")
    combined = pd.concat([_load(r) for r in STRATEGIES.values()], ignore_index=True)
    z_trades = combined[combined["ticker"].isin(zombies)]
    if len(z_trades) > 0:
        top_z = z_trades.groupby("ticker").agg(
            n=("net_pnl_pct", "count"),
            net=("net_pnl_pct", "sum"),
            avg=("net_pnl_pct", "mean"),
            wr=("net_pnl_pct", lambda x: (x > 0).mean() * 100),
        ).sort_values("n", ascending=False).head(15)
        top_z["last_trade"] = [zombies.get(t, "?") for t in top_z.index]
        print(top_z.to_string())
    else:
        print("  No zombie trades found.")


# ══════════════════════════════════════════════════════════════════════
#  AUDIT 3: EXECUTION SLIPPAGE / FRICTION SENSITIVITY
# ══════════════════════════════════════════════════════════════════════
def audit_slippage():
    print("\n\n" + "█" * 120)
    print("  AUDIT 3: FRICTION SENSITIVITY — What happens as transaction costs increase?")
    print("█" * 120)
    print("  Current backtest uses 0.5% (SEBON fee + broker). Real-world slippage on thin NEPSE")
    print("  stocks may add 0.5–1.0% more. This tests expectancy at multiple friction levels.\n")

    # For each strategy, recalculate expectancy at different friction levels
    # Original net_pnl_pct = pnl_pct - 0.5%
    # At X% friction:     adjusted = pnl_pct - X% = net_pnl_pct + 0.5% - X%
    # i.e.  adjusted = net_pnl_pct - (X% - 0.5%)

    header = f"{'Strategy':>16s}"
    for tc in FRICTION_LEVELS:
        header += f"  {'E@'+f'{tc:.1f}%':>10s}  {'PF@'+f'{tc:.1f}%':>10s}  {'WR@'+f'{tc:.1f}%':>10s}"
    header += f"  {'Break-even%':>11s}"
    print(header)
    print("─" * len(header))

    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        pnl_raw = df["pnl_pct"]  # gross pnl (before any friction)
        row = f"{name:>16s}"
        for tc in FRICTION_LEVELS:
            adj = pnl_raw - tc
            exp = adj.mean()
            pf  = _pf(adj)
            wr  = (adj > 0).mean() * 100
            row += f"  {exp:>+9.3f}%  {pf:>10.2f}  {wr:>9.1f}%"

        # Break-even friction: what TC makes expectancy = 0?
        # E[pnl_raw - tc] = 0  →  tc = E[pnl_raw]
        be = pnl_raw.mean()
        row += f"  {be:>10.2f}%"
        print(row)

    # OOS-specific friction analysis
    print(f"\n{'─'*120}")
    print("  OOS-ONLY (post 2024-07-01) friction sensitivity")
    print(f"{'─'*120}")

    header2 = f"{'Strategy':>16s}"
    for tc in FRICTION_LEVELS:
        header2 += f"  {'E@'+f'{tc:.1f}%':>10s}  {'PF@'+f'{tc:.1f}%':>10s}"
    header2 += f"  {'OOS Brk-even':>12s}"
    print(header2)
    print("─" * len(header2))

    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        oos = df[df["entry_date"] >= "2024-07-01"]
        if len(oos) == 0:
            print(f"{name:>16s}  NO OOS DATA")
            continue
        pnl_raw = oos["pnl_pct"]
        row = f"{name:>16s}"
        for tc in FRICTION_LEVELS:
            adj = pnl_raw - tc
            exp = adj.mean()
            pf  = _pf(adj)
            row += f"  {exp:>+9.3f}%  {pf:>10.2f}"
        be = pnl_raw.mean()
        row += f"  {be:>11.2f}%"
        print(row)

    # Practical recommendation
    print(f"\n{'─'*120}")
    print("  PRACTICAL SLIPPAGE ESTIMATE FOR NEPSE")
    print(f"{'─'*120}")
    print("  SEBON fees + broker commission  : ~0.40% (regulatory)")
    print("  DP charge (demat transfer)      : ~0.015%")
    print("  Bid-ask spread (liquid stocks)  : ~0.2–0.5%")
    print("  Bid-ask spread (illiquid hydro) : ~0.5–2.0%")
    print("  Market impact (>10L order)      : ~0.3–1.0%")
    print("  → Realistic all-in cost         : 1.0–1.5% round-trip for most stocks")
    print("  → For micro-cap / low-vol       : 1.5–2.5% round-trip")
    print()
    print("  RULE OF THUMB: If a strategy's expectancy < 1.5% at 1.0% friction, avoid it.")


# ══════════════════════════════════════════════════════════════════════
#  BONUS: Bull-run concentration ratio
# ══════════════════════════════════════════════════════════════════════
def audit_bull_concentration():
    print("\n\n" + "█" * 120)
    print("  BONUS: BULL-RUN CONCENTRATION — How much profit comes from 2020–2021 alone?")
    print("█" * 120)

    header = f"{'Strategy':>16s}  {'Total Net%':>10s}  {'2020-21 Net%':>12s}  {'% from Bull':>11s}  {'Ex-Bull Net%':>12s}  {'Ex-Bull E%':>10s}"
    print(header)
    print("─" * len(header))

    for name, run_id in STRATEGIES.items():
        df = _load(run_id)
        df["year"] = df["exit_date"].dt.year
        total = df["net_pnl_pct"].sum()
        bull = df[df["year"].isin([2020, 2021])]["net_pnl_pct"].sum()
        ex_bull = df[~df["year"].isin([2020, 2021])]
        ex_net = ex_bull["net_pnl_pct"].sum()
        ex_exp = ex_bull["net_pnl_pct"].mean() if len(ex_bull) > 0 else 0
        pct_bull = bull / total * 100 if total != 0 else 0
        print(f"{name:>16s}  {total:>+9.1f}%  {bull:>+11.1f}%  {pct_bull:>10.1f}%  {ex_net:>+11.1f}%  {ex_exp:>+9.3f}%")

    print("\n  INTERPRETATION: If >50% of net profit is from 2020–21, the strategy")
    print("  may be 'riding the tide'. Check if Ex-Bull expectancy is still positive.\n")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 120)
    print("  RBS STRATEGY AUDIT — 3 Critical Loophole Checks")
    print("=" * 120)

    audit_regime_gap()
    audit_survivorship()
    audit_slippage()
    audit_bull_concentration()

    print("\n" + "=" * 120)
    print("  AUDIT COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
