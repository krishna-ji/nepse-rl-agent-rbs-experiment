"""
Phase 2: Feature Engineering – The Continuous Tensor Matrix
============================================================
All indicators are computed vectorised across the entire master DataFrame
**before** environment initialisation.  No look-ahead bias.

Features produced per ticker:
    %K, %D           – Stochastic Oscillator (14, 3)
    delta_k           – Stochastic velocity
    natr              – Normalised ATR(14)
    bbw               – Bollinger Bandwidth (20, 2)
    d_low             – Distance to protected swing low
    sma50, sma200     – Simple Moving Averages
    macro_trend       – 1 if SMA50 > SMA200 else 0
    atr14             – Raw ATR(14) used by the environment for TSL
    protected_swing_low – rolling structural floor
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.run_manager import get_logger


# ── Indicator primitives ──────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, min_periods=n, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(span=n, min_periods=n, adjust=False).mean()


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest + 1e-10)
    pct_k = raw_k.rolling(d_period, min_periods=d_period).mean()  # smoothed %K
    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean()  # %D
    return pct_k, pct_d


def _bollinger_bandwidth(close: pd.Series, n: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = _sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (upper - lower) / (sma + 1e-10)


def _protected_swing_low(low: pd.Series, window: int = 60) -> pd.Series:
    """Rolling min of the last `window` lows – structural support floor."""
    return low.rolling(window, min_periods=window).min()


# ── Main pipeline ─────────────────────────────────────────────────────────

def compute_features(
    master_df: pd.DataFrame,
    valid_start_dates: Dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Return a MultiIndex DataFrame (Ticker, Feature) with all engineered
    features aligned to the same universal date index as *master_df*.

    Features that are undefined before the warm-up period remain NaN.
    """
    tickers = master_df.columns.get_level_values("Ticker").unique()
    result_pieces: dict[tuple[str, str], pd.Series] = {}

    for ticker in tickers:
        o = master_df[(ticker, "Open")]
        h = master_df[(ticker, "High")]
        l = master_df[(ticker, "Low")]
        c = master_df[(ticker, "Close")]
        v = master_df[(ticker, "Volume")]

        # ── Raw OHLCV pass-through (needed by env / viz) ──
        result_pieces[(ticker, "open")]   = o
        result_pieces[(ticker, "high")]   = h
        result_pieces[(ticker, "low")]    = l
        result_pieces[(ticker, "close")]  = c
        result_pieces[(ticker, "volume")] = v

        # ── Trend & Structure ──
        sma50  = _sma(c, 50)
        sma200 = _sma(c, 200)
        psl    = _protected_swing_low(l, window=60)

        result_pieces[(ticker, "sma50")]  = sma50
        result_pieces[(ticker, "sma200")] = sma200
        result_pieces[(ticker, "macro_trend")] = (sma50 > sma200).astype(np.float32)
        result_pieces[(ticker, "protected_swing_low")] = psl
        result_pieces[(ticker, "d_low")] = (c - psl) / (c + 1e-10)

        # ── Momentum (Pullback Vector) ──
        pct_k, pct_d = _stochastic(h, l, c)
        result_pieces[(ticker, "pct_k")]   = pct_k / 100.0   # scale to [0,1]
        result_pieces[(ticker, "pct_d")]   = pct_d / 100.0
        result_pieces[(ticker, "delta_k")] = (pct_k - pct_k.shift(1)) / 100.0

        # ── Volatility (Regime Filter) ──
        atr14 = _atr(h, l, c, 14)
        result_pieces[(ticker, "atr14")] = atr14
        result_pieces[(ticker, "natr")]  = atr14 / (c + 1e-10)
        result_pieces[(ticker, "bbw")]   = _bollinger_bandwidth(c, 20, 2.0)

    feat_df = pd.DataFrame(result_pieces)
    feat_df.columns = pd.MultiIndex.from_tuples(feat_df.columns, names=["Ticker", "Feature"])
    feat_df = feat_df.sort_index()

    # ── Rolling 252-day Z-score normalization ─────────────────────────
    # Normalise cross-asset heterogeneous indicators so the NN sees a
    # standardised state space regardless of underlying price domain.
    # Do NOT Z-score %K / %D (already bounded [0, 1]).
    ZSCORE_COLS = ["natr", "bbw", "d_low"]
    ZSCORE_WINDOW = 252
    ZSCORE_CLIP = 3.0

    for ticker in tickers:
        for col in ZSCORE_COLS:
            key = (ticker, col)
            if key not in feat_df.columns:
                continue
            raw = feat_df[key]
            roll_mean = raw.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
            roll_std = raw.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std()
            z = (raw - roll_mean) / (roll_std + 1e-8)
            feat_df[key] = z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)

    log = get_logger("rl_nepse.features")
    log.info(f"Computed {len(feat_df.columns.get_level_values('Feature').unique())} "
             f"features x {len(tickers)} tickers | shape {feat_df.shape}")
    log.info(f"Z-scored {ZSCORE_COLS} with {ZSCORE_WINDOW}-day rolling window, clipped to [-{ZSCORE_CLIP}, {ZSCORE_CLIP}]")
    return feat_df


# ── Quick smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib
    from data_loader import load_universe

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    DATA = ROOT / "data" / "stocks"
    master, vstarts = load_universe(DATA)
    feat = compute_features(master, vstarts)
    print(feat.head())
