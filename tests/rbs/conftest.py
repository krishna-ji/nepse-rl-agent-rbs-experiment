"""
Shared test fixtures for src.rbs monolith tests.
"""

from __future__ import annotations

import pathlib
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import IchimokuParams


# ── Synthetic OHLCV generator ───────────────────────────────────────

def make_ohlcv(
    n: int = 400,
    start: str = "2015-01-01",
    seed: int = 42,
    trend: float = 0.0005,
    vol: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic OHLCV with a controllable trend + volatility."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=n)
    close = np.empty(n)
    close[0] = 100.0

    for i in range(1, n):
        ret = trend + vol * rng.randn()
        close[i] = close[i - 1] * (1 + ret)

    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    opn = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.randint(1000, 50000, n).astype(float)

    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def write_ticker_csvs(
    tmpdir: pathlib.Path,
    tickers: Dict[str, pd.DataFrame],
) -> pathlib.Path:
    """Write DataFrames as CSV files matching the loader's expected format."""
    stocks_dir = tmpdir / "data" / "ohlcv" / "1D" / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in tickers.items():
        csv_df = df.copy()
        csv_df.index.name = "Timestamp"
        csv_df.to_csv(stocks_dir / f"{ticker}.csv")

    # Empty exclusion files so loader doesn't fail
    data_dir = tmpdir / "data"
    (data_dir / "stocks.json").write_text("[]", encoding="utf-8")
    (data_dir / "mutual.json").write_text("[]", encoding="utf-8")
    (data_dir / "corpdeben.json").write_text("[]", encoding="utf-8")

    return stocks_dir


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def tmp_project(tmp_path):
    """Create a temp project directory with 3 synthetic tickers.
    Returns (stocks_dir, project_root)."""
    tickers = {
        "AAA": make_ohlcv(400, seed=1, trend=0.001),
        "BBB": make_ohlcv(400, seed=2, trend=-0.0003),
        "CCC": make_ohlcv(400, seed=3, trend=0.0005),
    }
    stocks_dir = write_ticker_csvs(tmp_path, tickers)
    return stocks_dir, tmp_path


@pytest.fixture
def default_params() -> IchimokuParams:
    return IchimokuParams()


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    """400-bar bullish synthetic OHLCV for unit tests."""
    return make_ohlcv(400, seed=42, trend=0.001)


@pytest.fixture
def tiny_ohlcv() -> pd.DataFrame:
    """50-bar OHLCV — below warmup threshold."""
    return make_ohlcv(50, seed=99, trend=0.0)
