"""
Tests for base strategy: IchimokuParams, TradeRecord, helper functions, ABC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import (
    BacktestStrategy,
    IchimokuParams,
    TradeRecord,
    chikou_is_free,
    compute_ichimoku,
    flat_sb_strong_candle,
    is_senkou_b_flat,
)


class TestIchimokuParams:
    def test_defaults(self):
        p = IchimokuParams()
        assert p.tenkan_period == 9
        assert p.kijun_period == 26
        assert p.senkou_b_period == 52
        assert p.displacement == 26
        assert p.long_only is True
        assert p.warmup == 80

    def test_override(self):
        p = IchimokuParams(tenkan_period=7, seed=123)
        assert p.tenkan_period == 7
        assert p.seed == 123


class TestTradeRecord:
    def test_to_dict(self):
        tr = TradeRecord(
            ticker="AAA", direction="LONG",
            entry_date=pd.Timestamp("2020-01-01"),
            exit_date=pd.Timestamp("2020-01-10"),
            entry_price=100.0, exit_price=110.0,
            bars_held=7, pnl_pct=10.0, net_pnl_pct=9.0,
            exit_reason="kijun_close", strategy="Kumo Break",
        )
        d = tr.to_dict()
        assert d["ticker"] == "AAA"
        assert d["pnl_pct"] == 10.0
        assert d["strategy"] == "Kumo Break"

    def test_extra_fields(self):
        tr = TradeRecord(
            ticker="BBB", direction="LONG",
            entry_date=pd.Timestamp("2020-01-01"),
            exit_date=pd.Timestamp("2020-01-10"),
            entry_price=100.0, exit_price=90.0,
            bars_held=5, pnl_pct=-10.0, net_pnl_pct=-11.0,
            exit_reason="hard_stop",
            extra={"cross_strength": "strong"},
        )
        d = tr.to_dict()
        assert d["cross_strength"] == "strong"


class TestComputeIchimoku:
    def test_output_keys(self, small_ohlcv):
        p = IchimokuParams()
        result = compute_ichimoku(small_ohlcv, p)
        expected_keys = {"tenkan", "kijun", "senkou_a", "senkou_b",
                         "kumo_top", "kumo_bot", "future_sa", "future_sb", "atr"}
        assert set(result.keys()) == expected_keys

    def test_output_lengths(self, small_ohlcv):
        p = IchimokuParams()
        result = compute_ichimoku(small_ohlcv, p)
        n = len(small_ohlcv)
        for key, arr in result.items():
            assert len(arr) == n, f"{key} length mismatch"

    def test_tenkan_kijun_relationship(self, small_ohlcv):
        """Tenkan is faster MA, should have NaN for fewer initial bars than kijun."""
        p = IchimokuParams()
        ich = compute_ichimoku(small_ohlcv, p)
        first_valid_tenkan = np.argmax(~np.isnan(ich["tenkan"]))
        first_valid_kijun = np.argmax(~np.isnan(ich["kijun"]))
        assert first_valid_tenkan <= first_valid_kijun

    def test_kumo_top_gte_bot(self, small_ohlcv):
        """kumo_top >= kumo_bot by definition."""
        p = IchimokuParams()
        ich = compute_ichimoku(small_ohlcv, p)
        valid = ~np.isnan(ich["kumo_top"]) & ~np.isnan(ich["kumo_bot"])
        assert np.all(ich["kumo_top"][valid] >= ich["kumo_bot"][valid])


class TestChikouIsFree:
    def _make_arrays(self, n=200, price=100.0):
        c = np.full(n, price)
        h = np.full(n, price + 1)
        l = np.full(n, price - 1)
        sa = np.full(n, price - 5)
        sb = np.full(n, price - 10)
        return c, h, l, sa, sb

    def test_free_long(self):
        c, h, l, sa, sb = self._make_arrays(200, 100)
        c[50] = 200
        assert chikou_is_free(c, h, l, 50, "long", sa, sb, 26, 2) is True

    def test_not_free_long(self):
        c, h, l, sa, sb = self._make_arrays(200, 100)
        assert chikou_is_free(c, h, l, 50, "long", sa, sb, 26, 2) is False

    def test_edge_early_bar(self):
        c, h, l, sa, sb = self._make_arrays(200, 100)
        assert chikou_is_free(c, h, l, 5, "long", sa, sb, 26, 2) is False


class TestSenkouBFlat:
    def test_flat(self):
        sb = np.full(100, 50.0)
        assert is_senkou_b_flat(sb, 99, 15, 0.001) == True

    def test_not_flat(self):
        sb = np.linspace(40, 60, 100)
        assert is_senkou_b_flat(sb, 99, 15, 0.001) == False

    def test_short_lookback(self):
        sb = np.full(100, 50.0)
        assert is_senkou_b_flat(sb, 5, 15, 0.001) is False


class TestFlatSbStrongCandle:
    def test_strong_long(self):
        assert flat_sb_strong_candle(100, 110, 5, "long", 0.5) is True

    def test_weak_long(self):
        assert flat_sb_strong_candle(100, 101, 5, "long", 0.5) is False

    def test_strong_short(self):
        assert flat_sb_strong_candle(100, 90, 5, "short", 0.5) is True


class TestBacktestStrategyABC:
    def test_cannot_instantiate(self, tmp_project):
        stocks_dir, project_root = tmp_project
        with pytest.raises(TypeError):
            BacktestStrategy(data_dir=stocks_dir, project_root=project_root)

    def test_run_before_access_raises(self, tmp_project):
        stocks_dir, project_root = tmp_project

        class DummyStrat(BacktestStrategy):
            @property
            def name(self):
                return "Dummy"
            def backtest_ticker(self, ticker, df):
                return []

        s = DummyStrat(data_dir=stocks_dir, project_root=project_root, min_rows=50)
        with pytest.raises(RuntimeError):
            _ = s.trades

    def test_dummy_strategy_runs(self, tmp_project):
        stocks_dir, project_root = tmp_project

        class DummyStrat(BacktestStrategy):
            @property
            def name(self):
                return "Dummy"
            def backtest_ticker(self, ticker, df):
                return [
                    TradeRecord(
                        ticker=ticker, direction="LONG",
                        entry_date=df.index[0], exit_date=df.index[-1],
                        entry_price=100, exit_price=110,
                        bars_held=len(df), pnl_pct=10, net_pnl_pct=9,
                        exit_reason="test",
                    )
                ]

        s = DummyStrat(data_dir=stocks_dir, project_root=project_root, min_rows=50)
        s.run()
        assert len(s.trades) == 3
        assert all(t.strategy == "Dummy" for t in s.trades)
        df = s.trades_df
        assert len(df) == 3
        assert "entry_date" in df.columns
