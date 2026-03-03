"""
Tests for generate_report function.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.rbs.base import generate_report


def _make_report_trades(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic trades DataFrame for report generation."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2015-01-01", periods=n * 5)

    rows = []
    for i in range(n):
        entry = dates[i * 5]
        exit_ = dates[i * 5 + rng.randint(1, 4)]
        pnl = rng.uniform(-8, 12)
        rows.append({
            "ticker": f"T{i % 8}",
            "strategy": "Kumo Break" if i % 2 == 0 else "T/K Cross",
            "direction": "LONG",
            "entry_date": entry,
            "exit_date": exit_,
            "net_pnl_pct": round(pnl, 4),
            "pnl_pct": round(pnl + 0.5, 4),
            "bars_held": rng.randint(1, 30),
            "exit_reason": rng.choice(["kijun_close", "hard_stop", "gap_stop"]),
        })
    return pd.DataFrame(rows)


def _make_equity(n: int = 200, seed: int = 7) -> pd.Series:
    """Create a synthetic equity curve."""
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.RandomState(seed)
    eq_vals = 1_000_000 + np.cumsum(rng.uniform(-5000, 8000, n))
    return pd.Series(eq_vals, index=dates, name="equity")


class TestGenerateReport:
    def test_creates_pdf(self, tmp_path):
        df = _make_report_trades(80)
        out = tmp_path / "test_report.pdf"
        result_path = generate_report(df, out)
        assert result_path.exists()
        assert result_path.suffix == ".pdf"
        assert result_path.stat().st_size > 0

    def test_with_equity(self, tmp_path):
        df = _make_report_trades(80)
        eq = _make_equity()
        out = tmp_path / "test_full.pdf"
        result_path = generate_report(df, out, equity=eq)
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_nested_output_dir(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "dir" / "report.pdf"
        df = _make_report_trades(30)
        result_path = generate_report(df, out)
        assert result_path.exists()

    def test_small_dataset(self, tmp_path):
        df = _make_report_trades(5, seed=99)
        out = tmp_path / "small.pdf"
        result_path = generate_report(df, out)
        assert result_path.exists()
