#!/usr/bin/env python3
"""
NEPSE RSI Mean Reversion Strategy — Rule-Based Backtest
=======================================================
Buys oversold bounces (RSI < 30 then crossing back above) and sells
when price reaches overbought territory or reverts to mean.

Entry Logic (Long):
  1. RSI(14) was below OVERSOLD (30) within the last 3 bars
  2. RSI(14) crosses back above OVERSOLD                        — bounce confirm
  3. Close > SMA(200)                                           — long-term uptrend filter
  4. Close is within 1.5× ATR of a support (20-period low)      — near support

Entry Execution:
  Market buy at next bar's open (mean reversion = immediate action).

Stop / Exit:
  - Initial stop: recent 20-bar low − ATR_STOP_MULT × ATR
  - Profit target: entry + REWARD_MULT × risk (risk = entry − stop)
  - Exit also if RSI > OVERBOUGHT (70)                          — target hit
  - Time stop: exit if not profitable after MAX_HOLD bars

Lookahead Bias Prevention:
  - RSI, SMA, ATR computed from data up to bar t
  - Signal at close of bar t → fill at open of bar t+1
  - Stop/target computed from entry bar values only
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

# RSI parameters
RSI_PERIOD       = 14
OVERSOLD         = 30
OVERBOUGHT       = 70
LOOKBACK_OVERSOLD = 3          # RSI must have been < 30 within last N bars

# Trend filter
SMA_TREND_PERIOD = 200

# Position sizing
REWARD_MULT      = 2.0         # risk-reward ratio for profit target
SUPPORT_PERIOD   = 20          # recent swing low period
MAX_HOLD         = 40          # max bars to hold before time stop

# Backtest parameters
MIN_ROWS         = 350         # need 200 for SMA + warmup
WARMUP           = 220
SPLIT_DATE       = "2024-07-01"
LONG_ONLY        = True
ATR_PERIOD       = 14
ATR_STOP_MULT    = 1.0
TRANSACTION_COST = 0.005
SEED             = 42

# ============================================================================
# SETUP
# ============================================================================

def setup():
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = PROJECT_ROOT / f"runs/{RUN_TS}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("rsi_mean_reversion")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(RUN_DIR / "rsi_mean_reversion.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); log.addHandler(sh)

    log.info("=" * 70)
    log.info("NEPSE RSI Mean Reversion Strategy — Rule-Based Backtest")
    log.info("=" * 70)
    log.info(f"Run directory  : {RUN_DIR.resolve()}")
    log.info(f"RSI period     : {RSI_PERIOD}")
    log.info(f"Oversold/OB    : {OVERSOLD}/{OVERBOUGHT}")
    log.info(f"SMA trend      : {SMA_TREND_PERIOD}")
    log.info(f"Reward mult    : {REWARD_MULT}R")
    log.info(f"Max hold       : {MAX_HOLD} bars")
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
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    # RSI
    delta = c.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # SMA trend filter
    sma_trend = c.rolling(SMA_TREND_PERIOD).mean()

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    # Support (20-bar low)
    support = l.rolling(SUPPORT_PERIOD).min()

    return {
        "rsi":       rsi.values,
        "sma_trend": sma_trend.values,
        "atr":       atr.values,
        "support":   support.values,
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
    rsi   = ind["rsi"]
    sma   = ind["sma_trend"]
    atr   = ind["atr"]
    supp  = ind["support"]

    trades = []
    state = "FLAT"

    # Signal bar (bar where signal fires, fill at t+1 open)
    sig_bar = -1

    # Position state
    entry_price = 0.0
    stop_px     = 0.0
    target_px   = 0.0
    entry_bar   = 0
    entry_date  = None

    for t in range(WARMUP, n):
        if any(np.isnan(x[t]) for x in [rsi, sma, atr, supp]):
            continue

        # ── STATE: PENDING (fill at open of t if sig_bar == t-1) ──────
        if state == "PENDING" and t == sig_bar + 1:
            entry_price = o[t]
            stop_level  = supp[sig_bar] - ATR_STOP_MULT * atr[sig_bar]
            risk = entry_price - stop_level
            if risk <= 0:
                state = "FLAT"; continue
            stop_px    = stop_level
            target_px  = entry_price + REWARD_MULT * risk
            entry_bar  = t
            entry_date = dates[t]
            state = "POSITION"
            continue

        if state == "PENDING":
            state = "FLAT"
            continue

        # ── STATE: POSITION ───────────────────────────────────────────
        if state == "POSITION":
            exit_px  = None
            exit_rsn = None

            if o[t] <= stop_px:
                exit_px = o[t]; exit_rsn = "gap_stop"
            elif l[t] <= stop_px:
                exit_px = stop_px; exit_rsn = "hard_stop"
            elif h[t] >= target_px:
                exit_px = target_px; exit_rsn = "target"
            elif rsi[t] > OVERBOUGHT:
                exit_px = c[t]; exit_rsn = "rsi_overbought"
            elif (t - entry_bar) >= MAX_HOLD:
                exit_px = c[t]; exit_rsn = "time_stop"

            if exit_px is not None:
                pnl = (exit_px / entry_price - 1.0) * 100.0
                net = pnl - TRANSACTION_COST * 100.0
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
                state = "FLAT"
            continue

        # ── STATE: FLAT — entry scan ──────────────────────────────────
        # Check: RSI was below oversold recently and just crossed above
        rsi_was_oversold = any(
            rsi[max(WARMUP, t - k)] < OVERSOLD
            for k in range(1, LOOKBACK_OVERSOLD + 1) if t - k >= 0
        )
        rsi_cross_up = rsi[t] >= OVERSOLD and t > 0 and rsi[t-1] < OVERSOLD

        if rsi_was_oversold and rsi_cross_up:
            cond_trend = c[t] > sma[t]                     # above 200 SMA
            cond_support = (c[t] - supp[t]) / max(atr[t], 1e-10) < 1.5  # near support

            if cond_trend and cond_support:
                sig_bar = t
                state   = "PENDING"

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
    log.info("RSI MEAN REVERSION BACKTEST RESULTS")
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
    log.info(f"\nBacktesting RSI Mean Reversion on {len(frames)} tickers...")
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
