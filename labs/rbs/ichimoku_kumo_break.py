#!/usr/bin/env python3
"""
NEPSE Ichimoku Kumo Break Strategy — Rule-Based Backtest
========================================================
Rule-based trading system implementing the Kumo Break strategy
from "How to Make Money Trading the Ichimoku System" by B.M. Sadekar (Ch.4).

5-Point Entry Checklist (ALL must confirm for Long):
  1. Price closes above the Kumo (Senkou A / Senkou B cloud)
  2. Future Kumo is bullish (Senkou A > Senkou B, unshifted)
  3. Chikou Span is free and clear above price from 26 periods ago
  4. Tenkan-sen > Kijun-sen
  5. Price > both Tenkan-sen and Kijun-sen (not over-extended)

Entry: Buy-stop at 9-period high (ensures Tenkan is pulled upward).
       Uses 26-period high instead if close to 9-period high (pulls Kijun too).
Stop:  Kijun-sen minus ATR buffer (trailed with Kijun each bar).
Exit:  Close below Kijun-sen, or hard stop hit intrabar.

Lookahead Bias Prevention:
  - Current Kumo at chart position t = Senkou A/B computed at t-26 → .shift(26)
  - Future Kumo visible at t+26 = Senkou A/B computed at t (unshifted) → safe
  - Chikou check = close[t] vs historical highs/lows around t-26 → safe
  - Tenkan/Kijun = rolling high/low up to t → safe
  - Entry: signal at close of bar t, stop order fills at bar t+1 onward → safe
  - Stop/trail: uses Kijun/ATR computed at EOD → safe
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

# Ichimoku periods (standard)
TENKAN_PERIOD    = 9
KIJUN_PERIOD     = 26
SENKOU_B_PERIOD  = 52
DISPLACEMENT     = 26

# Backtest parameters
MIN_ROWS         = 250
ICHIMOKU_WARMUP  = 80          # 52 + 26 + buffer for Senkou B shift + ATR
SPLIT_DATE       = "2024-07-01"
LONG_ONLY        = True        # NEPSE does not support short selling
ATR_PERIOD       = 14
ATR_STOP_MULT    = 1.5         # stop = kijun ± mult × ATR
TRANSACTION_COST = 0.005       # 0.5% round-trip (broker + SEBON fees)
ORDER_TIMEOUT    = 5           # cancel pending stop order after N bars
CHIKOU_FREE_HALF = 2           # ± bars around t-26 for Chikou clearance window
MAX_KIJUN_DIST   = 5.0         # reject entry if price > N × ATR from Kijun
HH_PROXIMITY     = 1.5         # use HH26 if HH26−HH9 < N × ATR
FLAT_SB_LOOKBACK = 15          # periods to check Senkou B flatness
FLAT_SB_TOL      = 0.001       # relative tolerance for "flat" (0.1%)
FLAT_SB_STRONG   = 0.5         # min candle body as fraction of ATR near flat SB
SEED             = 42

# ============================================================================
# SETUP
# ============================================================================

def setup():
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = PROJECT_ROOT / f"runs/{RUN_TS}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("ichimoku_kumo_break")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(RUN_DIR / "ichimoku_kumo_break.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); log.addHandler(sh)

    log.info("=" * 70)
    log.info("NEPSE Ichimoku Kumo Break Strategy — Rule-Based Backtest")
    log.info("=" * 70)
    log.info(f"Run directory  : {RUN_DIR.resolve()}")
    log.info(f"Data dir       : {DATA_DIR.resolve()}")
    log.info(f"Split date     : {SPLIT_DATE}")
    log.info(f"Long only      : {LONG_ONLY}")
    log.info(f"ATR stop mult  : {ATR_STOP_MULT}")
    log.info(f"Transaction    : {TRANSACTION_COST*100:.1f}%")
    log.info(f"Order timeout  : {ORDER_TIMEOUT} bars")
    log.info(f"Chikou window  : ±{CHIKOU_FREE_HALF} bars around t-{DISPLACEMENT}")
    log.info(f"Max Kijun dist : {MAX_KIJUN_DIST} × ATR")
    return log, RUN_DIR

# ============================================================================
# DATA LOADING
# ============================================================================

def _build_exclude_set():
    """Tickers to exclude: PROMOTSHARE sector + mutual funds + corporate debentures."""
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
    """Load OHLCV CSVs → dict of per-ticker DataFrames."""
    log.info("Loading OHLCV data...")
    log.info(f"Excluding {len(EXCLUDE_TICKERS)} tickers (promoter shares, mutual funds, corp debentures)")
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
# ICHIMOKU COMPUTATION
# ============================================================================

def compute_ichimoku(df):
    """
    Compute all Ichimoku components for a single ticker.

    Lookahead-free design:
      - Current Kumo at chart pos t = Senkou A/B computed at t-26 → .shift(26)
      - Future Kumo (leading edge at t+26) = Senkou A/B computed at t (unshifted)
      - All rolling windows use only data up to and including t

    Returns dict of numpy arrays aligned with df.index.
    """
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    # Tenkan-sen (Conversion Line): (9H + 9L) / 2
    tenkan = (h.rolling(TENKAN_PERIOD).max() + l.rolling(TENKAN_PERIOD).min()) / 2

    # Kijun-sen (Base Line): (26H + 26L) / 2
    kijun = (h.rolling(KIJUN_PERIOD).max() + l.rolling(KIJUN_PERIOD).min()) / 2

    # Senkou Span A raw (before displacement): (Tenkan + Kijun) / 2
    senkou_a_raw = (tenkan + kijun) / 2

    # Senkou Span B raw (before displacement): (52H + 52L) / 2
    senkou_b_raw = (h.rolling(SENKOU_B_PERIOD).max() + l.rolling(SENKOU_B_PERIOD).min()) / 2

    # Current Kumo at chart position t (computed at t − DISPLACEMENT)
    senkou_a = senkou_a_raw.shift(DISPLACEMENT)
    senkou_b = senkou_b_raw.shift(DISPLACEMENT)

    # Future Kumo: computed at t, will be plotted at t + DISPLACEMENT
    # Uses only data available at t → NO lookahead
    future_senkou_a = senkou_a_raw   # unshifted
    future_senkou_b = senkou_b_raw   # unshifted

    # Kumo boundaries at current chart position
    kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    kumo_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    # ATR (Average True Range) for stop-loss buffer
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    return {
        "tenkan":     tenkan.values,
        "kijun":      kijun.values,
        "senkou_a":   senkou_a.values,
        "senkou_b":   senkou_b.values,
        "kumo_top":   kumo_top.values,
        "kumo_bot":   kumo_bottom.values,
        "future_sa":  future_senkou_a.values,
        "future_sb":  future_senkou_b.values,
        "atr":        atr.values,
    }

# ============================================================================
# CHIKOU FREEDOM CHECK
# ============================================================================

def chikou_is_free(c_arr, h_arr, l_arr, t, direction="long",
                   sa_arr=None, sb_arr=None):
    """
    Check if Chikou Span is 'free and clear' of BOTH historical price
    action AND the historical Kumo at that point.

    Chikou at chart position t-26 has value close[t].
    For it to be 'free' (Long):
      close[t]  >  max(high  in ±window around t-26,
                       Senkou_A[t-26], Senkou_B[t-26])
    This ensures the Chikou is above candles AND above the cloud
    at the historical point — Sadekar's full Chikou freedom rule.
    """
    center = t - DISPLACEMENT
    lo = max(0, center - CHIKOU_FREE_HALF)
    hi = min(len(c_arr), center + CHIKOU_FREE_HALF + 1)

    if lo >= hi or center < 0:
        return False

    if direction == "long":
        barrier = np.nanmax(h_arr[lo:hi])
        if np.isnan(barrier) or c_arr[t] <= barrier:
            return False
        # Chikou must also clear the historical Kumo at t-26
        if sa_arr is not None and sb_arr is not None:
            kumo_top_hist = max(sa_arr[center], sb_arr[center])
            if not np.isnan(kumo_top_hist) and c_arr[t] <= kumo_top_hist:
                return False
        return True
    else:
        barrier = np.nanmin(l_arr[lo:hi])
        if np.isnan(barrier) or c_arr[t] >= barrier:
            return False
        # Chikou must also be below the historical Kumo at t-26
        if sa_arr is not None and sb_arr is not None:
            kumo_bot_hist = min(sa_arr[center], sb_arr[center])
            if not np.isnan(kumo_bot_hist) and c_arr[t] >= kumo_bot_hist:
                return False
        return True


def is_senkou_b_flat(sb_arr, t, lookback=FLAT_SB_LOOKBACK, tol=FLAT_SB_TOL):
    """
    Detect a flat Senkou B — the 52-period equilibrium magnet.

    Returns True if Senkou B has not moved more than `tol` (relative)
    over the last `lookback` periods. A flat Senkou B acts as a
    gravity field that pulls price back toward it.
    """
    if t < lookback:
        return False
    recent = sb_arr[t - lookback:t + 1]
    if np.any(np.isnan(recent)):
        return False
    return (np.nanmax(recent) - np.nanmin(recent)) / max(np.nanmean(recent), 1e-10) < tol


def flat_sb_strong_candle(o_t, c_t, atr_t, direction="long"):
    """
    When near a flat Senkou B, require a strong candle body
    (body >= FLAT_SB_STRONG × ATR) to confirm the breakout.
    Weak closes near flat SB get pulled back like a magnet.
    """
    body = (c_t - o_t) if direction == "long" else (o_t - c_t)
    return body >= FLAT_SB_STRONG * atr_t

# ============================================================================
# SINGLE-TICKER BACKTEST
# ============================================================================

def backtest_ticker(ticker, df):
    """
    Run Kumo Break strategy on a single ticker.

    State machine:
      FLAT → (break + 5-point checklist) → PENDING → (stop order fills) → POSITION → (exit) → FLAT

    After each trade exit, a new Kumo break event is required before re-entry.
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index
    n = len(df)

    if n < ICHIMOKU_WARMUP + 10:
        return []

    ich = compute_ichimoku(df)
    tenkan   = ich["tenkan"]
    kijun    = ich["kijun"]
    senkou_a = ich["senkou_a"]             # current Kumo components (for Chikou)
    senkou_b = ich["senkou_b"]
    kumo_top = ich["kumo_top"]
    kumo_bot = ich["kumo_bot"]
    fut_sa   = ich["future_sa"]
    fut_sb   = ich["future_sb"]
    atr      = ich["atr"]

    trades = []
    state = "FLAT"
    direction = None
    kumo_brk_long  = False
    kumo_brk_short = False

    # Pending order state
    pend_level = 0.0
    pend_stop  = 0.0
    sig_bar    = 0

    # Position state
    entry_price = 0.0
    stop_px     = 0.0
    entry_bar   = 0
    entry_date  = None

    for t in range(ICHIMOKU_WARMUP, n):
        # Skip bars with any NaN in critical Ichimoku values
        if any(np.isnan(x[t]) for x in [tenkan, kijun, kumo_top, kumo_bot,
                                         fut_sa, fut_sb, atr]):
            continue

        # ── Kumo break detection (runs every bar) ──────────────────────
        if t > 0 and not np.isnan(kumo_top[t - 1]) and not np.isnan(kumo_bot[t - 1]):
            # Long break: close crosses above Kumo top from at/below
            if c[t] > kumo_top[t] and c[t - 1] <= kumo_top[t - 1]:
                kumo_brk_long = True
            # Short break: close crosses below Kumo bottom from at/above
            if not LONG_ONLY:
                if c[t] < kumo_bot[t] and c[t - 1] >= kumo_bot[t - 1]:
                    kumo_brk_short = True

        # Invalidate breaks if price retreats
        if c[t] <= kumo_top[t]:
            kumo_brk_long = False
        if c[t] >= kumo_bot[t]:
            kumo_brk_short = False

        # ── STATE: PENDING — check if stop order fills ────────────────
        if state == "PENDING":
            filled = False
            if direction == "LONG":
                if o[t] >= pend_level:                     # gap up → fill at open
                    entry_price = o[t]; filled = True
                elif h[t] >= pend_level:                   # trades through level
                    entry_price = pend_level; filled = True
            elif direction == "SHORT":
                if o[t] <= pend_level:                     # gap down → fill at open
                    entry_price = o[t]; filled = True
                elif l[t] <= pend_level:                   # trades through level
                    entry_price = pend_level; filled = True

            if filled:
                stop_px    = pend_stop
                entry_bar  = t
                entry_date = dates[t]
                state = "POSITION"
                kumo_brk_long = False                       # consumed
                kumo_brk_short = False
                continue                                    # don't evaluate exit on entry bar

            # Timeout or condition invalidation
            cancel = (t - sig_bar >= ORDER_TIMEOUT)
            if direction == "LONG":
                cancel = cancel or c[t] < kumo_bot[t] or tenkan[t] < kijun[t]
            elif direction == "SHORT":
                cancel = cancel or c[t] > kumo_top[t] or tenkan[t] > kijun[t]
            if cancel:
                state = "FLAT"; direction = None
            continue

        # ── STATE: POSITION — check exit conditions ───────────────────
        if state == "POSITION":
            exit_px  = None
            exit_rsn = None

            if direction == "LONG":
                if o[t] <= stop_px:                        # gap down past stop
                    exit_px = o[t]; exit_rsn = "gap_stop"
                elif l[t] <= stop_px:                      # intrabar hard stop
                    exit_px = stop_px; exit_rsn = "hard_stop"
                elif c[t] < kijun[t]:                      # close below Kijun
                    exit_px = c[t]; exit_rsn = "kijun_close"
                else:                                      # trail stop with Kijun
                    trail = kijun[t] - ATR_STOP_MULT * atr[t]
                    if not np.isnan(trail):
                        stop_px = max(stop_px, trail)

            elif direction == "SHORT":
                if o[t] >= stop_px:
                    exit_px = o[t]; exit_rsn = "gap_stop"
                elif h[t] >= stop_px:
                    exit_px = stop_px; exit_rsn = "hard_stop"
                elif c[t] > kijun[t]:
                    exit_px = c[t]; exit_rsn = "kijun_close"
                else:
                    trail = kijun[t] + ATR_STOP_MULT * atr[t]
                    if not np.isnan(trail):
                        stop_px = min(stop_px, trail)

            if exit_px is not None:
                if direction == "LONG":
                    pnl = (exit_px / entry_price - 1.0) * 100.0
                else:
                    pnl = (1.0 - exit_px / entry_price) * 100.0
                net = pnl - TRANSACTION_COST * 100.0
                trades.append({
                    "ticker":      ticker,
                    "direction":   direction,
                    "entry_date":  entry_date,
                    "exit_date":   dates[t],
                    "entry_price": round(entry_price, 2),
                    "exit_price":  round(exit_px, 2),
                    "bars_held":   t - entry_bar,
                    "pnl_pct":     round(pnl, 4),
                    "net_pnl_pct": round(net, 4),
                    "exit_reason": exit_rsn,
                })
                state = "FLAT"; direction = None
            continue

        # ── STATE: FLAT — scan for entry signals ──────────────────────
        # Long signal
        if kumo_brk_long:
            cond1 = c[t] > kumo_top[t]                                      # price above Kumo
            cond2 = fut_sa[t] > fut_sb[t]                                   # future Kumo bullish
            cond3 = chikou_is_free(c, h, l, t, "long",                      # Chikou free
                                   sa_arr=senkou_a, sb_arr=senkou_b)         # incl. hist Kumo
            cond4 = tenkan[t] > kijun[t]                                    # Tenkan > Kijun
            cond5 = (c[t] > tenkan[t]) and (c[t] > kijun[t])               # price above both
            cond6 = (c[t] - kijun[t]) / max(atr[t], 1e-10) < MAX_KIJUN_DIST  # not stretched

            # Expert filter: flat Senkou B magnet — require strong candle
            cond7 = True
            if is_senkou_b_flat(fut_sb, t):
                if not flat_sb_strong_candle(o[t], c[t], atr[t], "long"):
                    cond7 = False

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7:
                hh9  = np.nanmax(h[max(0, t - TENKAN_PERIOD + 1):t + 1])
                hh26 = np.nanmax(h[max(0, t - KIJUN_PERIOD  + 1):t + 1])
                # Use HH26 when close to HH9 — pulls Kijun in trade direction
                if (hh26 - hh9) / max(atr[t], 1e-10) < HH_PROXIMITY:
                    pend_level = hh26
                else:
                    pend_level = hh9
                pend_stop = kijun[t] - ATR_STOP_MULT * atr[t]
                sig_bar   = t
                direction = "LONG"
                state     = "PENDING"

        # Short signal
        elif not LONG_ONLY and kumo_brk_short:
            cond1 = c[t] < kumo_bot[t]
            cond2 = fut_sa[t] < fut_sb[t]
            cond3 = chikou_is_free(c, h, l, t, "short",
                                   sa_arr=senkou_a, sb_arr=senkou_b)
            cond4 = tenkan[t] < kijun[t]
            cond5 = (c[t] < tenkan[t]) and (c[t] < kijun[t])
            cond6 = (kijun[t] - c[t]) / max(atr[t], 1e-10) < MAX_KIJUN_DIST

            # Expert filter: flat Senkou B magnet — require strong candle
            cond7 = True
            if is_senkou_b_flat(fut_sb, t):
                if not flat_sb_strong_candle(o[t], c[t], atr[t], "short"):
                    cond7 = False

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7:
                ll9  = np.nanmin(l[max(0, t - TENKAN_PERIOD + 1):t + 1])
                ll26 = np.nanmin(l[max(0, t - KIJUN_PERIOD  + 1):t + 1])
                if (ll9 - ll26) / max(atr[t], 1e-10) < HH_PROXIMITY:
                    pend_level = ll26
                else:
                    pend_level = ll9
                pend_stop = kijun[t] + ATR_STOP_MULT * atr[t]
                sig_bar   = t
                direction = "SHORT"
                state     = "PENDING"

    # Close any open position at last bar
    if state == "POSITION":
        last_c = c[-1]
        if direction == "LONG":
            pnl = (last_c / entry_price - 1.0) * 100.0
        else:
            pnl = (1.0 - last_c / entry_price) * 100.0
        net = pnl - TRANSACTION_COST * 100.0
        trades.append({
            "ticker":      ticker,
            "direction":   direction,
            "entry_date":  entry_date,
            "exit_date":   dates[-1],
            "entry_price": round(entry_price, 2),
            "exit_price":  round(last_c, 2),
            "bars_held":   n - 1 - entry_bar,
            "pnl_pct":     round(pnl, 4),
            "net_pnl_pct": round(net, 4),
            "exit_reason": "data_end",
        })

    return trades

# ============================================================================
# AGGREGATION & REPORTING
# ============================================================================

def _compute_stats(df):
    """Compute summary statistics from a DataFrame of trades."""
    n = len(df)
    if n == 0:
        return {}
    wins   = df[df["net_pnl_pct"] > 0]
    losses = df[df["net_pnl_pct"] <= 0]
    g_win  = wins["net_pnl_pct"].sum()   if len(wins)   else 0.0
    g_loss = abs(losses["net_pnl_pct"].sum()) if len(losses) else 0.0
    return {
        "n_trades":       n,
        "n_tickers":      df["ticker"].nunique(),
        "win_rate_pct":   len(wins) / n * 100,
        "avg_win_pct":    wins["net_pnl_pct"].mean()   if len(wins)   else 0.0,
        "avg_loss_pct":   losses["net_pnl_pct"].mean() if len(losses) else 0.0,
        "profit_factor":  g_win / g_loss if g_loss > 0 else float("inf"),
        "expectancy_pct": df["net_pnl_pct"].mean(),
        "median_pnl_pct": df["net_pnl_pct"].median(),
        "max_win_pct":    df["net_pnl_pct"].max(),
        "max_loss_pct":   df["net_pnl_pct"].min(),
        "avg_bars_held":  df["bars_held"].mean(),
    }


def _log_stats(df, label, log):
    """Log summary statistics for a set of trades."""
    s = _compute_stats(df)
    if not s:
        log.info(f"\n{label}: no trades"); return
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
    """Aggregate all trades, compute stats, export CSVs."""
    if not all_trades:
        log.info("No trades generated.")
        return

    df = pd.DataFrame(all_trades)
    df.to_csv(run_dir / "trades.csv", index=False)

    log.info(f"\n{'='*70}")
    log.info("KUMO BREAK BACKTEST RESULTS")
    log.info(f"{'='*70}")

    # ── Overall stats ──
    _log_stats(df, "ALL TRADES", log)

    # ── OOS stats ──
    oos = df[df["entry_date"] >= pd.Timestamp(SPLIT_DATE)]
    if len(oos) > 0:
        _log_stats(oos, f"OUT-OF-SAMPLE (entry >= {SPLIT_DATE})", log)

    # ── By direction ──
    for d in df["direction"].unique():
        _log_stats(df[df["direction"] == d], f"DIRECTION: {d}", log)

    # ── By exit reason ──
    log.info(f"\n{'─'*50}")
    log.info("BY EXIT REASON:")
    for rsn, grp in df.groupby("exit_reason"):
        wr = (grp["net_pnl_pct"] > 0).mean() * 100
        log.info(f"  {rsn:15s}: {len(grp):5d} trades, "
                 f"WR {wr:5.1f}%, avg {grp['net_pnl_pct'].mean():+.2f}%")

    # ── Per-ticker summary ──
    per_tk = df.groupby("ticker").agg(
        n_trades=("net_pnl_pct", "count"),
        total_pnl=("net_pnl_pct", "sum"),
        avg_pnl=("net_pnl_pct", "mean"),
        win_rate=("net_pnl_pct", lambda x: (x > 0).mean() * 100),
    ).sort_values("total_pnl", ascending=False)
    per_tk.to_csv(run_dir / "per_ticker.csv")

    log.info(f"\n{'─'*50}")
    log.info("TOP 10 TICKERS (by total PnL):")
    for tk, row in per_tk.head(10).iterrows():
        log.info(f"  {tk:10s}: {int(row['n_trades']):3d} trades, "
                 f"total {row['total_pnl']:+8.2f}%, "
                 f"avg {row['avg_pnl']:+6.2f}%, WR {row['win_rate']:.0f}%")

    log.info(f"\nBOTTOM 10 TICKERS:")
    for tk, row in per_tk.tail(10).iterrows():
        log.info(f"  {tk:10s}: {int(row['n_trades']):3d} trades, "
                 f"total {row['total_pnl']:+8.2f}%, "
                 f"avg {row['avg_pnl']:+6.2f}%, WR {row['win_rate']:.0f}%")

    # ── Equity curve (additive, 1 unit of capital per trade) ──────────
    # Additive P&L is the correct aggregation for a multi-stock universe
    # where trades run concurrently. Each trade risks 1 unit.
    df_sorted = df.sort_values("exit_date").reset_index(drop=True)
    cum_pnl = df_sorted["net_pnl_pct"].cumsum().values
    equity = 100.0 + cum_pnl          # start at 100 units

    eq_df = pd.DataFrame({
        "trade_num": np.arange(1, len(equity) + 1),
        "exit_date": df_sorted["exit_date"].values,
        "equity":    equity,
    })
    eq_df.to_csv(run_dir / "equity_curve.csv", index=False)

    # Max drawdown on the additive equity curve
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100
    max_dd = dd_pct.min()
    total_return = (equity[-1] / 100.0 - 1) * 100

    log.info(f"\n{'='*70}")
    log.info(f"EQUITY (1-unit per trade): 100.00 -> {equity[-1]:.2f}  "
             f"(total P&L {total_return:+.2f}%)")
    log.info(f"MAX DRAWDOWN: {max_dd:.2f}%")
    log.info(f"{'='*70}")

    # ── Summary CSV ──
    summary = _compute_stats(df)
    summary.update({
        "max_drawdown_pct":  max_dd,
        "final_equity":      equity[-1],
        "split_date":        SPLIT_DATE,
        "long_only":         LONG_ONLY,
        "atr_stop_mult":     ATR_STOP_MULT,
        "transaction_cost":  TRANSACTION_COST,
        "order_timeout":     ORDER_TIMEOUT,
        "chikou_window":     CHIKOU_FREE_HALF,
        "max_kijun_dist":    MAX_KIJUN_DIST,
    })
    pd.DataFrame([summary]).to_csv(run_dir / "summary_metrics.csv", index=False)

    log.info(f"\nAll outputs saved to {run_dir.resolve()}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log, run_dir = setup()
    frames = load_ohlcv(log)

    log.info(f"\nBacktesting Kumo Break strategy on {len(frames)} tickers...")
    all_trades = []
    active = 0
    for i, (tk, df) in enumerate(sorted(frames.items())):
        trades = backtest_ticker(tk, df)
        if trades:
            all_trades.extend(trades)
            active += 1
        if (i + 1) % 100 == 0:
            log.info(f"  ... processed {i + 1}/{len(frames)} tickers "
                     f"({len(all_trades)} trades so far)")

    log.info(f"Backtest complete: {len(all_trades)} trades from "
             f"{active}/{len(frames)} tickers")
    report_results(all_trades, log, run_dir)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
