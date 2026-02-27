"""
Phase 1: Universal Data Architecture & Temporal Alignment
=========================================================
Loads all NEPSE OHLCV CSVs into a master MultiIndex DataFrame,
aligns on a universal date axis, and computes the warm-up padding
matrix (valid_start_date per ticker = first-valid + 200 trading days).
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.run_manager import get_logger


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_single_csv(path: pathlib.Path) -> pd.DataFrame | None:
    """Return an OHLCV DataFrame indexed by date, or None if empty."""
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    if df.empty or len(df) < 2:
        return None
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    # Remove duplicate dates (keep last observation)
    df = df[~df.index.duplicated(keep="last")]
    # Guarantee required columns exist
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return None
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_universe(
    data_dir: str | pathlib.Path,
    min_rows: int = 250,
) -> Tuple[pd.DataFrame, Dict[str, pd.Timestamp]]:
    """Load all valid stock CSVs and return (master_df, valid_start_dates).

    Parameters
    ----------
    data_dir : path to the directory containing per-ticker CSV files.
    min_rows : minimum number of rows for a ticker to be included
               (must have enough history for 200-day SMA warm-up).

    Returns
    -------
    master_df : pd.DataFrame
        MultiIndex columns = (Ticker, Feature) where Feature ∈
        {Open, High, Low, Close, Volume}.  Index = universal DatetimeIndex.
    valid_start_dates : dict
        {ticker: first date after 200-day warm-up}.
    """
    data_dir = pathlib.Path(data_dir)
    frames: dict[str, pd.DataFrame] = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        ticker = csv_path.stem
        df = _load_single_csv(csv_path)
        if df is not None and len(df) >= min_rows:
            frames[ticker] = df

    if not frames:
        raise RuntimeError(f"No valid CSVs with >= {min_rows} rows found in {data_dir}")

    # Build MultiIndex DataFrame aligned on a universal date axis
    pieces = {}
    for ticker, df in frames.items():
        for col in df.columns:
            pieces[(ticker, col)] = df[col]

    master_df = pd.DataFrame(pieces)
    master_df.columns = pd.MultiIndex.from_tuples(
        master_df.columns, names=["Ticker", "Feature"]
    )
    master_df = master_df.sort_index()

    # ---------- Warm-up padding (200 trading-day lookback) ----------
    WARMUP = 200
    valid_start_dates: Dict[str, pd.Timestamp] = {}
    tickers = master_df.columns.get_level_values("Ticker").unique()

    for ticker in tickers:
        close = master_df[(ticker, "Close")]
        first_valid = close.first_valid_index()
        if first_valid is None:
            continue
        # Shift forward by WARMUP valid (non-NaN) trading days
        valid_idx = close.dropna().index
        if len(valid_idx) < WARMUP + 252:
            continue                       # not enough data even after warm-up
        valid_start_dates[ticker] = valid_idx[WARMUP]

    # Keep only tickers that survive the warm-up filter
    surviving = sorted(valid_start_dates.keys())
    master_df = master_df[surviving]

    log = get_logger("rl_nepse.data")
    log.info(f"Loaded {len(surviving)} tickers | "
             f"Date range: {master_df.index.min().date()} -> {master_df.index.max().date()}")

    return master_df, valid_start_dates


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).resolve().parents[1]
    DATA = ROOT / "data" / "stocks"
    master, vstarts = load_universe(DATA)
    print(f"Master shape: {master.shape}")
    print(f"Sample valid_start_dates:")
    for t in list(vstarts)[:5]:
        print(f"  {t}: {vstarts[t].date()}")
