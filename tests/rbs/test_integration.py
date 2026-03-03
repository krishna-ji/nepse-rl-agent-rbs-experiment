"""
Integration test: end-to-end monolith pipeline with synthetic data.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import IchimokuParams, compute_metrics, simulate_portfolio
from src.rbs.kumo_break import KumoBreak
from src.rbs.tk_cross import TKCross

from tests.rbs.conftest import make_ohlcv, write_ticker_csvs


class TestE2EPipeline:
    """End-to-end test: load data → run strategies → portfolio sim → metrics."""

    @pytest.fixture
    def e2e_project(self, tmp_path):
        """5 synthetic tickers. Returns (stocks_dir, project_root)."""
        tickers = {
            f"SYN{i}": make_ohlcv(500, seed=i * 10, trend=0.0005 * (i - 2))
            for i in range(5)
        }
        stocks_dir = write_ticker_csvs(tmp_path, tickers)
        return stocks_dir, tmp_path

    def test_full_pipeline(self, e2e_project, tmp_path):
        """Run both strategies → combine → portfolio sim → check metrics."""
        stocks_dir, project_root = e2e_project
        params = IchimokuParams(seed=42)

        kumo = KumoBreak(data_dir=stocks_dir, project_root=project_root,
                         params=params, min_rows=50)
        kumo.run()

        tkc = TKCross(data_dir=stocks_dir, project_root=project_root,
                       params=params, min_rows=50)
        tkc.run()

        dfs = []
        if not kumo.trades_df.empty:
            dfs.append(kumo.trades_df)
        if not tkc.trades_df.empty:
            dfs.append(tkc.trades_df)

        if not dfs:
            pytest.skip("No trades generated with synthetic data")

        combined = pd.concat(dfs, ignore_index=True)
        assert len(combined) > 0
        assert "strategy" in combined.columns

        result = simulate_portfolio(combined, max_slots=5, filter_after=None)
        assert result["n_trades"] > 0
        assert len(result["equity"]) > 0

        m = compute_metrics(result["trades_df"], pnl_col="adj_pnl_pct")
        assert m.n_trades == result["n_trades"]
        assert 0 <= m.win_rate <= 100

    def test_each_strategy_independent(self, e2e_project):
        """Each strategy should produce independent trade sets."""
        stocks_dir, project_root = e2e_project
        params = IchimokuParams(seed=42)

        kumo = KumoBreak(data_dir=stocks_dir, project_root=project_root,
                         params=params, min_rows=50)
        kumo.run()

        tkc = TKCross(data_dir=stocks_dir, project_root=project_root,
                       params=params, min_rows=50)
        tkc.run()

        if kumo.trades and tkc.trades:
            kumo_names = {t.strategy for t in kumo.trades}
            tkc_names = {t.strategy for t in tkc.trades}
            assert kumo_names == {"Kumo Break"}
            assert tkc_names == {"T/K Cross"}

    def test_rerun_same_results(self, e2e_project):
        """Same seed should produce identical results."""
        stocks_dir, project_root = e2e_project
        params = IchimokuParams(seed=42)

        s1 = KumoBreak(data_dir=stocks_dir, project_root=project_root,
                        params=params, min_rows=50)
        s1.run()

        s2 = KumoBreak(data_dir=stocks_dir, project_root=project_root,
                        params=params, min_rows=50)
        s2.run()

        assert len(s1.trades) == len(s2.trades)
        for t1, t2 in zip(s1.trades, s2.trades):
            assert t1.ticker == t2.ticker
            assert t1.pnl_pct == t2.pnl_pct
