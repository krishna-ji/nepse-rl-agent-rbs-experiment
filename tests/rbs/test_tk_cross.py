"""
Tests for TKCross strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import IchimokuParams, TradeRecord
from src.rbs.tk_cross import TKCross, classify_cross_strength
from tests.rbs.conftest import make_ohlcv


class TestClassifyCrossStrength:
    def test_above_kumo_long(self):
        assert classify_cross_strength(110, 100, 90, "LONG") == "strong"

    def test_above_kumo_short(self):
        assert classify_cross_strength(110, 100, 90, "SHORT") == "weak"

    def test_below_kumo_long(self):
        assert classify_cross_strength(80, 100, 90, "LONG") == "weak"

    def test_below_kumo_short(self):
        assert classify_cross_strength(80, 100, 90, "SHORT") == "strong"

    def test_inside_kumo(self):
        assert classify_cross_strength(95, 100, 90, "LONG") == "neutral"
        assert classify_cross_strength(95, 100, 90, "SHORT") == "neutral"


class TestTKCross:
    def _make(self, tmp_project, **kw):
        stocks_dir, project_root = tmp_project
        return TKCross(data_dir=stocks_dir, project_root=project_root, min_rows=50, **kw)

    def test_name(self, tmp_project):
        s = self._make(tmp_project)
        assert s.name == "T/K Cross"

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

    def test_all_trades_long_by_default(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        for t in s.trades:
            assert t.direction == "LONG"

    def test_cross_strength_in_extra(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        for t in s.trades:
            assert "cross_strength" in t.extra
            assert t.extra["cross_strength"] in {"strong", "weak", "neutral"}

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

    def test_run_batch(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        df = s.trades_df
        if not df.empty:
            assert (df["strategy"] == "T/K Cross").all()

    def test_run_selected_tickers(self, tmp_project):
        s = self._make(tmp_project)
        s.run(tickers=["BBB"])
        for t in s.trades:
            assert t.ticker == "BBB"

    def test_pnl_consistent(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        for t in s.trades:
            expected = (t.exit_price / t.entry_price - 1) * 100
            assert abs(t.pnl_pct - expected) < 0.02

    def test_repr(self, tmp_project):
        s = self._make(tmp_project)
        s.run()
        assert "T/K Cross" in repr(s)
