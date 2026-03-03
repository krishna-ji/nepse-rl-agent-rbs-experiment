"""
Tests for simulate_portfolio function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import simulate_portfolio


def _make_signals(n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic signals DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n * 3)

    rows = []
    for i in range(n):
        entry = dates[i * 3]
        exit_ = dates[i * 3 + 2]
        pnl = rng.uniform(-5, 10)
        rows.append({
            "ticker": f"T{i % 3}",
            "strategy": "TestStrat",
            "direction": "LONG",
            "entry_date": entry,
            "exit_date": exit_,
            "net_pnl_pct": round(pnl, 2),
            "bars_held": 2,
        })
    return pd.DataFrame(rows)


class TestSimulatePortfolio:
    def test_run_returns_dict(self):
        sig = _make_signals()
        result = simulate_portfolio(sig, filter_after=None)
        assert isinstance(result, dict)
        assert "trades_df" in result
        assert "equity" in result
        assert "total_return_pct" in result

    def test_trades_executed(self):
        sig = _make_signals(10)
        result = simulate_portfolio(sig, max_slots=10, filter_after=None)
        assert result["n_trades"] > 0

    def test_equity_length(self):
        sig = _make_signals(10)
        result = simulate_portfolio(sig, filter_after=None)
        assert len(result["equity"]) > 0

    def test_equity_starts_near_capital(self):
        sig = _make_signals(5)
        result = simulate_portfolio(sig, initial_capital=1_000_000, filter_after=None)
        assert abs(result["equity"].iloc[0] - 1_000_000) < 200_000

    def test_total_return_type(self):
        sig = _make_signals()
        result = simulate_portfolio(sig, filter_after=None)
        assert isinstance(result["total_return_pct"], float)

    def test_max_drawdown_type(self):
        sig = _make_signals()
        result = simulate_portfolio(sig, filter_after=None)
        assert isinstance(result["max_drawdown_pct"], (float, np.floating))
        assert result["max_drawdown_pct"] <= 0

    def test_win_rate_range(self):
        sig = _make_signals()
        result = simulate_portfolio(sig, filter_after=None)
        assert 0 <= result["win_rate"] <= 100

    def test_filter_after(self):
        sig = _make_signals(10)
        result = simulate_portfolio(sig, filter_after="2020-01-20")
        assert result["n_trades"] <= 10

    def test_fixed_sizing(self):
        sig = _make_signals(5)
        result = simulate_portfolio(sig, sizing="fixed", max_slots=5, filter_after=None)
        tdf = result["trades_df"]
        if not tdf.empty:
            expected_size = 1_000_000 / 5
            for _, row in tdf.iterrows():
                assert row["trade_size"] == expected_size

    def test_compounding_sizing(self):
        sig = _make_signals(5)
        result = simulate_portfolio(sig, sizing="compounding", max_slots=5, filter_after=None)
        assert result["n_trades"] > 0

    def test_empty_signals(self):
        sig = pd.DataFrame(columns=[
            "ticker", "strategy", "direction",
            "entry_date", "exit_date", "net_pnl_pct",
        ])
        result = simulate_portfolio(sig, filter_after=None)
        assert result["n_trades"] == 0

    def test_slot_limit(self):
        """Shouldn't open more trades than max_slots simultaneously."""
        rows = []
        entry = pd.Timestamp("2020-01-01")
        exit_ = pd.Timestamp("2020-01-15")
        for i in range(20):
            rows.append({
                "ticker": f"T{i}", "strategy": "Test", "direction": "LONG",
                "entry_date": entry, "exit_date": exit_,
                "net_pnl_pct": 5.0, "bars_held": 10,
            })
        sig = pd.DataFrame(rows)
        result = simulate_portfolio(sig, max_slots=5, filter_after=None)
        assert result["n_trades"] == 5
        assert result["skipped"] == 15

    def test_trades_df_columns(self):
        sig = _make_signals(5)
        result = simulate_portfolio(sig, filter_after=None)
        tdf = result["trades_df"]
        if not tdf.empty:
            assert "adj_pnl_pct" in tdf.columns
            assert "pnl_npr" in tdf.columns
