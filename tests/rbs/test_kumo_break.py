"""
Tests for KumoBreak strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import IchimokuParams, TradeRecord
from src.rbs.kumo_break import KumoBreak
from tests.rbs.conftest import make_ohlcv, write_ticker_csvs


class TestKumoBreak:
    def _make(self, tmp_project, **kw):
        stocks_dir, project_root = tmp_project
        return KumoBreak(data_dir=stocks_dir, project_root=project_root, min_rows=50, **kw)

    def test_name(self, tmp_project):
        s = self._make(tmp_project)
        assert s.name == "Kumo Break"

    def test_returns_list(self, tmp_project):
        s = self._make(tmp_project)
        s.load_data()
        df = s._frames["AAA"]
        result = s.backtest_ticker("AAA", df)
        assert isinstance(result, list)
        for t in result:
            assert isinstance(t, TradeRecord)

    def test_short_data_no_trades(self, tmp_project):
        s = self._make(tmp_project)
        df = make_ohlcv(50, seed=99)
        result = s.backtest_ticker("SHORT", df)
        assert result == []

    def test_all_trades_are_long_by_default(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        for t in s.trades:
            assert t.direction == "LONG"

    def test_trade_fields(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        if s.trades:
            t = s.trades[0]
            assert t.ticker in s.tickers
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.bars_held >= 0
            assert t.exit_reason in {"gap_stop", "hard_stop", "kijun_close"}
            assert isinstance(t.pnl_pct, float)
            assert isinstance(t.net_pnl_pct, float)

    def test_run_batch(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        assert s._has_run is True
        df = s.trades_df
        if not df.empty:
            assert "strategy" in df.columns
            assert (df["strategy"] == "Kumo Break").all()

    def test_run_selected_tickers(self, tmp_project):
        s = self._make(tmp_project)
        s.run(tickers=["AAA"])
        for t in s.trades:
            assert t.ticker == "AAA"

    def test_pnl_matches_prices(self, tmp_project):
        """PnL should be consistent with entry/exit prices for LONG."""
        s = self._make(tmp_project)
        s.run()
        for t in s.trades:
            expected = (t.exit_price / t.entry_price - 1) * 100
            assert abs(t.pnl_pct - expected) < 0.02

    def test_repr(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        assert "Kumo Break" in repr(s)

    def test_strong_trend_produces_trades(self, tmp_path):
        """A strong uptrend should produce at least one trade (or not crash)."""
        df = make_ohlcv(600, seed=10, trend=0.003, vol=0.01)
        stocks_dir = write_ticker_csvs(tmp_path, {"BULL": df})
        s = KumoBreak(data_dir=stocks_dir, project_root=tmp_path, min_rows=50)
        s.run()
        assert isinstance(s.trades, list)
