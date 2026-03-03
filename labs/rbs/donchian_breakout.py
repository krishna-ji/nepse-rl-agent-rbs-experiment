#!/usr/bin/env python3
"""
NEPSE Donchian Channel Breakout Strategy — Rule-Based Backtest
==============================================================
Turtle-trading inspired system using N-period highs/lows for entries
and a shorter channel for exits.

Entry Logic (Long):
  1. Price breaks above 20-period high (Donchian upper channel)  — breakout
  2. Previous breakout was NOT a winner (Turtle S1 filter)        — avoids chasing
     (If previous trade was profitable, skip this signal. Take the next one.)
  3. ATR(20) confirms volatility is non-trivial                  — liquid enough

Entry Execution:
  Buy-stop at 20-period high. Fills at that level or gaps above at open.

Stop / Exit:
  - Initial stop: entry − 2 × ATR(20)
  - Trail: max(stop, 10-period low − 0.5 × ATR)                 — Donchian exit channel
  - Exit: close below 10-period low
  - Hard stop if price gaps below

The "skip after winner" filter is the classic Turtle System 1 rule.
It avoids piling into the same breakout direction after taking profits,
waiting for one failed breakout to "reset" the system.

Lookahead Bias Prevention:
  - Donchian channels use data up to bar t
  - Signal at close of t → buy-stop fills on bar t+1
  - Trail uses EOD values
"""

import warnings; warnings.filterwarnings("ignore")
import json, logging, pathlib, datetime, sys
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT     = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR         = PROJECT_ROOT / "data/ohlcv/1D/stocks"

# Donchian parameters
ENTRY_PERIOD     = 20          # N-period high for entry (upper channel)
EXIT_PERIOD      = 10          # N-period low for exit (lower channel)
ATR_PERIOD_DC    = 20          # ATR period for stop sizing
ATR_STOP_MULT_DC = 2.0         # initial stop = entry − 2 × ATR
USE_S1_FILTER    = True        # skip signal after a winning trade

# Backtest parameters
MIN_ROWS         = 250
WARMUP           = 60
SPLIT_DATE       = "2024-07-01"
LONG_ONLY        = True
TRANSACTION_COST = 0.005
ORDER_TIMEOUT    = 3           # buy-stop valid for 3 bars
SEED             = 42

# ============================================================================
# SETUP
# ============================================================================

def setup():
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = PROJECT_ROOT / f"runs/{RUN_TS}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("donchian_breakout")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(RUN_DIR / "donchian_breakout.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); log.addHandler(sh)

    log.info("=" * 70)
    log.info("NEPSE Donchian Channel Breakout Strategy — Rule-Based Backtest")
    log.info("=" * 70)
    log.info(f"Run directory  : {RUN_DIR.resolve()}")
    log.info(f"Entry channel  : {ENTRY_PERIOD}-period high")
    log.info(f"Exit channel   : {EXIT_PERIOD}-period low")
    log.info(f"ATR stop       : {ATR_STOP_MULT_DC} × ATR({ATR_PERIOD_DC})")
    log.info(f"S1 filter      : {USE_S1_FILTER}")
    log.info(f"Split date     : {SPLIT_DATE}")
    log.info(f"Transaction    : {TRANSACTION_COST*100:.1f}%")
    return log, RUN_DIR

# ============================================================================
# DATA LOADING
# ============================================================================

def _build_exclude_set():
    excl = set()
    stocks_json = PROJECT_ROOT / "data/stocks.json"
    if stocks_json.exists():
        for s in json.load(open(stocks_json, encoding="utf-8")):
            if s.get("sector") == "PROMOTSHARE":
                excl.add(s["script"])
    for fname in ("mutual.json", "corpdeben.json"):
        p = PROJECT_ROOT / "data" / fname
        if p.exists():
            for s in json.load(open(p, encoding="utf-8")):
                excl.add(s["script"])
    return excl

EXCLUDE_TICKERS = _build_exclude_set()

def load_ohlcv(log):
    log.info("Loading OHLCV data...")
    log.info(f"Excluding {len(EXCLUDE_TICKERS)} tickers")
    frames, skipped, excluded = {}, 0, 0
    for csv in sorted(DATA_DIR.glob("*.csv")):
        if csv.stem in EXCLUDE_TICKERS:
            excluded += 1; continue
        try:
            df = pd.read_csv(csv, parse_dates=["Timestamp"])
            if df.empty or len(df) < MIN_ROWS:
                skipped += 1; continue
            df = df.rename(columns={"Timestamp": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                skipped += 1; continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < MIN_ROWS:
                skipped += 1; continue
            frames[csv.stem] = df
        except Exception as e:
            log.warning(f"Skip {csv.stem}: {e}"); skipped += 1
    log.info(f"Loaded {len(frames)} tickers, skipped {skipped}, excluded {excluded}")
    return frames

# ============================================================================
# INDICATOR COMPUTATION
# ============================================================================

def compute_indicators(df):
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    # Donchian channels
    upper = h.rolling(ENTRY_PERIOD).max()     # 20-period high
    lower = l.rolling(EXIT_PERIOD).min()       # 10-period low

    # ATR
    tr = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD_DC).mean()

    return {
        "upper": upper.values,
        "lower": lower.values,
        "atr":   atr.values,
    }

# ============================================================================
# SINGLE-TICKER BACKTEST
# ============================================================================

def backtest_ticker(ticker, df):
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index
    n = len(df)

    if n < WARMUP + 10:
        return []

    ind = compute_indicators(df)
    dc_upper = ind["upper"]
    dc_lower = ind["lower"]
    atr      = ind["atr"]

    trades = []
    state = "FLAT"
    direction = None
    last_trade_won = False       # S1 filter state

    pend_level = 0.0
    pend_stop  = 0.0
    sig_bar    = 0

    entry_price = 0.0
    stop_px     = 0.0
    entry_bar   = 0
    entry_date  = None

    for t in range(WARMUP, n):
        if any(np.isnan(x[t]) for x in [dc_upper, dc_lower, atr]):
            continue

        # ── PENDING ───────────────────────────────────────────────────
        if state == "PENDING":
            filled = False
            if direction == "LONG":
                if o[t] >= pend_level:
                    entry_price = o[t]; filled = True
                elif h[t] >= pend_level:
                    entry_price = pend_level; filled = True

            if filled:
                stop_px = entry_price - ATR_STOP_MULT_DC * atr[t]
                entry_bar = t; entry_date = dates[t]
                state = "POSITION"; continue

            if t - sig_bar >= ORDER_TIMEOUT:
                state = "FLAT"; direction = None
            continue

        # ── POSITION ──────────────────────────────────────────────────
        if state == "POSITION":
            exit_px = None; exit_rsn = None

            if direction == "LONG":
                if o[t] <= stop_px:
                    exit_px = o[t]; exit_rsn = "gap_stop"
                elif l[t] <= stop_px:
                    exit_px = stop_px; exit_rsn = "hard_stop"
                elif c[t] < dc_lower[t]:
                    exit_px = c[t]; exit_rsn = "channel_exit"
                else:
                    # Trail stop: 10-period low − 0.5 × ATR
                    trail = dc_lower[t] - 0.5 * atr[t]
                    if not np.isnan(trail):
                        stop_px = max(stop_px, trail)

            if exit_px is not None:
                pnl = (exit_px / entry_price - 1.0) * 100.0
                net = pnl - TRANSACTION_COST * 100.0
                last_trade_won = (net > 0)
                trades.append({
                    "ticker": ticker, "direction": "LONG",
                    "entry_date": entry_date, "exit_date": dates[t],
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_px, 2),
                    "bars_held": t - entry_bar,
                    "pnl_pct": round(pnl, 4),
                    "net_pnl_pct": round(net, 4),
                    "exit_reason": exit_rsn,
                })
                state = "FLAT"; direction = None
            continue

        # ── FLAT — scan for Donchian breakout ─────────────────────────
        if t > 0 and not np.isnan(dc_upper[t-1]):
            # Price breaks above the 20-period high
            breakout = c[t] > dc_upper[t-1]   # close > yesterday's 20-bar high

            if breakout:
                # S1 filter: skip if previous trade was a winner
                if USE_S1_FILTER and last_trade_won:
                    last_trade_won = False       # reset → take the next one
                    continue

                pend_level = dc_upper[t-1]       # buy-stop at channel high
                pend_stop  = c[t] - ATR_STOP_MULT_DC * atr[t]
                sig_bar    = t
                direction  = "LONG"
                state      = "PENDING"

    # Close open position
    if state == "POSITION":
        last_c = c[-1]
        pnl = (last_c / entry_price - 1.0) * 100.0
        net = pnl - TRANSACTION_COST * 100.0
        trades.append({
            "ticker": ticker, "direction": "LONG",
            "entry_date": entry_date, "exit_date": dates[-1],
            "entry_price": round(entry_price, 2),
            "exit_price": round(last_c, 2),
            "bars_held": n - 1 - entry_bar,
            "pnl_pct": round(pnl, 4),
            "net_pnl_pct": round(net, 4),
            "exit_reason": "data_end",
        })

    return trades

# ============================================================================
# REPORTING
# ============================================================================

def _compute_stats(df):
    n = len(df)
    if n == 0: return {}
    wins = df[df["net_pnl_pct"] > 0]
    losses = df[df["net_pnl_pct"] <= 0]
    g_win = wins["net_pnl_pct"].sum() if len(wins) else 0.0
    g_loss = abs(losses["net_pnl_pct"].sum()) if len(losses) else 0.0
    return {
        "n_trades": n, "n_tickers": df["ticker"].nunique(),
        "win_rate_pct": len(wins) / n * 100,
        "avg_win_pct": wins["net_pnl_pct"].mean() if len(wins) else 0.0,
        "avg_loss_pct": losses["net_pnl_pct"].mean() if len(losses) else 0.0,
        "profit_factor": g_win / g_loss if g_loss > 0 else float("inf"),
        "expectancy_pct": df["net_pnl_pct"].mean(),
        "median_pnl_pct": df["net_pnl_pct"].median(),
        "max_win_pct": df["net_pnl_pct"].max(),
        "max_loss_pct": df["net_pnl_pct"].min(),
        "avg_bars_held": df["bars_held"].mean(),
    }

def _log_stats(df, label, log):
    s = _compute_stats(df)
    if not s: log.info(f"\n{label}: no trades"); return
    log.info(f"\n{'─'*50}")
    log.info(f"{label} ({s['n_trades']} trades, {s['n_tickers']} tickers)")
    log.info(f"  Win Rate          : {s['win_rate_pct']:.1f}%")
    log.info(f"  Avg Win           : {s['avg_win_pct']:+.2f}%")
    log.info(f"  Avg Loss          : {s['avg_loss_pct']:+.2f}%")
    log.info(f"  Profit Factor     : {s['profit_factor']:.2f}")
    log.info(f"  Expectancy        : {s['expectancy_pct']:+.3f}%")
    log.info(f"  Median PnL        : {s['median_pnl_pct']:+.2f}%")
    log.info(f"  Best Trade        : {s['max_win_pct']:+.2f}%")
    log.info(f"  Worst Trade       : {s['max_loss_pct']:+.2f}%")
    log.info(f"  Avg Bars Held     : {s['avg_bars_held']:.1f}")

def report_results(all_trades, log, run_dir):
    if not all_trades:
        log.info("No trades generated."); return
    df = pd.DataFrame(all_trades)
    df.to_csv(run_dir / "trades.csv", index=False)
    log.info(f"\n{'='*70}")
    log.info("DONCHIAN CHANNEL BREAKOUT BACKTEST RESULTS")
    log.info(f"{'='*70}")
    _log_stats(df, "ALL TRADES", log)
    oos = df[df["entry_date"] >= pd.Timestamp(SPLIT_DATE)]
    if len(oos) > 0:
        _log_stats(oos, f"OUT-OF-SAMPLE (entry >= {SPLIT_DATE})", log)
    log.info(f"\n{'─'*50}")
    log.info("BY EXIT REASON:")
    for rsn, grp in df.groupby("exit_reason"):
        wr = (grp["net_pnl_pct"] > 0).mean() * 100
        log.info(f"  {rsn:15s}: {len(grp):5d} trades, WR {wr:5.1f}%, avg {grp['net_pnl_pct'].mean():+.2f}%")

    per_tk = df.groupby("ticker").agg(
        n_trades=("net_pnl_pct", "count"), total_pnl=("net_pnl_pct", "sum"),
        avg_pnl=("net_pnl_pct", "mean"),
        win_rate=("net_pnl_pct", lambda x: (x > 0).mean() * 100),
    ).sort_values("total_pnl", ascending=False)
    per_tk.to_csv(run_dir / "per_ticker.csv")

    log.info(f"\n{'─'*50}")
    log.info("TOP 10 TICKERS:")
    for tk, row in per_tk.head(10).iterrows():
        log.info(f"  {tk:10s}: {int(row['n_trades']):3d} trades, total {row['total_pnl']:+8.2f}%, avg {row['avg_pnl']:+6.2f}%, WR {row['win_rate']:.0f}%")
    log.info(f"\nBOTTOM 10 TICKERS:")
    for tk, row in per_tk.tail(10).iterrows():
        log.info(f"  {tk:10s}: {int(row['n_trades']):3d} trades, total {row['total_pnl']:+8.2f}%, avg {row['avg_pnl']:+6.2f}%, WR {row['win_rate']:.0f}%")

    df_sorted = df.sort_values("exit_date").reset_index(drop=True)
    cum_pnl = df_sorted["net_pnl_pct"].cumsum().values
    equity = 100.0 + cum_pnl
    pd.DataFrame({"trade_num": np.arange(1, len(equity)+1), "exit_date": df_sorted["exit_date"].values, "equity": equity}).to_csv(run_dir / "equity_curve.csv", index=False)
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100
    max_dd = dd_pct.min()
    total_return = (equity[-1] / 100.0 - 1) * 100
    log.info(f"\n{'='*70}")
    log.info(f"EQUITY (1-unit per trade): 100.00 -> {equity[-1]:.2f}  (total P&L {total_return:+.2f}%)")
    log.info(f"MAX DRAWDOWN: {max_dd:.2f}%")
    log.info(f"{'='*70}")
    summary = _compute_stats(df)
    summary.update({"max_drawdown_pct": max_dd, "final_equity": equity[-1], "split_date": SPLIT_DATE})
    pd.DataFrame([summary]).to_csv(run_dir / "summary_metrics.csv", index=False)
    log.info(f"\nAll outputs saved to {run_dir.resolve()}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log, run_dir = setup()
    frames = load_ohlcv(log)
    log.info(f"\nBacktesting Donchian Breakout on {len(frames)} tickers...")
    all_trades, active = [], 0
    for i, (tk, df) in enumerate(sorted(frames.items())):
        trades = backtest_ticker(tk, df)
        if trades: all_trades.extend(trades); active += 1
        if (i + 1) % 100 == 0:
            log.info(f"  ... processed {i+1}/{len(frames)} tickers ({len(all_trades)} trades so far)")
    log.info(f"Backtest complete: {len(all_trades)} trades from {active}/{len(frames)} tickers")
    report_results(all_trades, log, run_dir)
    log.info("\nDone.")

if __name__ == "__main__":
    main()
