"""
Tests for portfolio.metrics module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import (
    StrategyMetrics,
    compute_metrics,
    monthly_returns,
    ticker_metrics,
    yearly_metrics,
    _compute_streaks,
)


def _make_trades(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trades DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2015-01-01", periods=n * 5)

    rows = []
    for i in range(n):
        entry = dates[i * 5]
        exit_ = dates[i * 5 + rng.randint(1, 4)]
        pnl = rng.uniform(-8, 12)
        rows.append({
            "ticker": f"T{i % 5}",
            "strategy": "TestStrat" if i % 2 == 0 else "OtherStrat",
            "direction": "LONG",
            "entry_date": entry,
            "exit_date": exit_,
            "net_pnl_pct": round(pnl, 4),
            "pnl_pct": round(pnl + 0.5, 4),
            "bars_held": rng.randint(1, 20),
        })
    return pd.DataFrame(rows)


class TestStrategyMetrics:
    def test_defaults(self):
        m = StrategyMetrics()
        assert m.n_trades == 0
        assert m.win_rate == 0.0

    def test_to_dict(self):
        m = StrategyMetrics(n_trades=10, win_rate=50.0)
        d = m.to_dict()
        assert d["n_trades"] == 10
        assert d["win_rate"] == 50.0


class TestComputeMetrics:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["net_pnl_pct", "bars_held", "ticker"])
        m = compute_metrics(df)
        assert m.n_trades == 0

    def test_basic_counts(self):
        df = _make_trades(50)
        m = compute_metrics(df)
        assert m.n_trades == 50
        assert m.n_wins + m.n_losses == 50
        assert m.n_tickers == 5

    def test_win_rate_range(self):
        df = _make_trades(50)
        m = compute_metrics(df)
        assert 0 <= m.win_rate <= 100

    def test_all_winners(self):
        df = pd.DataFrame({
            "ticker": ["A"] * 5,
            "net_pnl_pct": [3.0, 5.0, 1.0, 2.0, 4.0],
            "bars_held": [3, 4, 5, 2, 6],
        })
        m = compute_metrics(df)
        assert m.win_rate == 100.0
        assert m.n_losses == 0
        assert m.profit_factor == float("inf")
        assert m.avg_loss == 0

    def test_all_losers(self):
        df = pd.DataFrame({
            "ticker": ["A"] * 5,
            "net_pnl_pct": [-3.0, -5.0, -1.0, -2.0, -4.0],
            "bars_held": [3, 4, 5, 2, 6],
        })
        m = compute_metrics(df)
        assert m.win_rate == 0.0
        assert m.n_wins == 0
        assert m.profit_factor == 0.0

    def test_stats_consistency(self):
        df = _make_trades(100, seed=77)
        m = compute_metrics(df)
        assert m.best_trade >= m.avg_pnl
        assert m.worst_trade <= m.avg_pnl
        assert m.std_pnl >= 0
        assert m.avg_bars_held > 0

    def test_expectancy_equals_avg(self):
        df = _make_trades(30)
        m = compute_metrics(df)
        assert m.expectancy == m.avg_pnl

    def test_profit_factor_positive(self):
        df = _make_trades(50)
        m = compute_metrics(df)
        assert m.profit_factor >= 0

    def test_custom_pnl_col(self):
        df = _make_trades(20)
        m = compute_metrics(df, pnl_col="pnl_pct")
        assert m.n_trades == 20
        # pnl_pct has +0.5 offset so avg should be slightly different
        m2 = compute_metrics(df, pnl_col="net_pnl_pct")
        assert abs(m.avg_pnl - m2.avg_pnl - 0.5) < 0.01


class TestComputeStreaks:
    def test_empty(self):
        ws, ls = _compute_streaks(np.array([]))
        assert ws == []
        assert ls == []

    def test_all_wins(self):
        ws, ls = _compute_streaks(np.array([1, 2, 3]))
        assert ws == [3]
        assert ls == []

    def test_all_losses(self):
        ws, ls = _compute_streaks(np.array([-1, -2, -3]))
        assert ws == []
        assert ls == [3]

    def test_alternating(self):
        ws, ls = _compute_streaks(np.array([1, -1, 1, -1]))
        assert ws == [1, 1]
        assert ls == [1, 1]

    def test_mixed(self):
        ws, ls = _compute_streaks(np.array([1, 2, -1, -2, -3, 1]))
        assert ws == [2, 1]
        assert ls == [3]


class TestYearlyMetrics:
    def test_empty(self):
        df = pd.DataFrame(columns=["entry_date", "net_pnl_pct"])
        result = yearly_metrics(df)
        assert result.empty

    def test_basic(self):
        df = _make_trades(50)
        result = yearly_metrics(df)
        assert not result.empty
        assert "trades" in result.columns
        assert "total_pnl" in result.columns
        assert "win_rate" in result.columns

    def test_year_index(self):
        df = _make_trades(50)
        result = yearly_metrics(df)
        assert result.index.name == "year"
        for yr in result.index:
            assert isinstance(yr, (int, np.integer))


class TestTickerMetrics:
    def test_empty(self):
        df = pd.DataFrame(columns=["ticker", "net_pnl_pct"])
        result = ticker_metrics(df)
        assert result.empty

    def test_basic(self):
        df = _make_trades(50)
        result = ticker_metrics(df)
        assert not result.empty
        assert "trades" in result.columns
        assert "total_pnl" in result.columns
        assert result.index.name == "ticker"

    def test_min_trades_filter(self):
        df = _make_trades(50)
        full = ticker_metrics(df, min_trades=1)
        filtered = ticker_metrics(df, min_trades=15)
        assert len(filtered) <= len(full)

    def test_sorted_descending(self):
        df = _make_trades(50)
        result = ticker_metrics(df)
        pnl_vals = result["total_pnl"].values
        assert all(pnl_vals[i] >= pnl_vals[i + 1] for i in range(len(pnl_vals) - 1))


class TestMonthlyReturns:
    def test_empty(self):
        df = pd.DataFrame(columns=["entry_date", "net_pnl_pct"])
        result = monthly_returns(df)
        assert result.empty

    def test_basic(self):
        df = _make_trades(50)
        result = monthly_returns(df)
        assert not result.empty
        assert result.index.name == "year"
        # Columns should be month numbers
        for col in result.columns:
            assert 1 <= col <= 12
