"""
Tests for data loading functions in base.py
"""

import json
import pathlib

import pandas as pd
import pytest

from src.rbs.base import load_ohlcv, _build_exclude_set
from tests.rbs.conftest import make_ohlcv, write_ticker_csvs


class TestBuildExcludeSet:
    def test_empty_json(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "stocks.json").write_text("[]", encoding="utf-8")
        (tmp_path / "data" / "mutual.json").write_text("[]", encoding="utf-8")
        (tmp_path / "data" / "corpdeben.json").write_text("[]", encoding="utf-8")
        excl = _build_exclude_set(tmp_path)
        assert excl == set()

    def test_promotshare_excluded(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "stocks.json").write_text(
            json.dumps([{"script": "EXCL", "sector": "PROMOTSHARE"}]),
            encoding="utf-8",
        )
        (tmp_path / "data" / "mutual.json").write_text("[]", encoding="utf-8")
        (tmp_path / "data" / "corpdeben.json").write_text("[]", encoding="utf-8")
        excl = _build_exclude_set(tmp_path)
        assert "EXCL" in excl

    def test_extra_exclude(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "stocks.json").write_text("[]", encoding="utf-8")
        (tmp_path / "data" / "mutual.json").write_text("[]", encoding="utf-8")
        (tmp_path / "data" / "corpdeben.json").write_text("[]", encoding="utf-8")
        excl = _build_exclude_set(tmp_path, extra={"CUSTOM"})
        assert "CUSTOM" in excl


class TestLoadOhlcv:
    def test_load_basic(self, tmp_project):
        stocks_dir, project_root = tmp_project
        frames = load_ohlcv(stocks_dir, project_root, min_rows=50)
        assert len(frames) == 3
        assert sorted(frames.keys()) == ["AAA", "BBB", "CCC"]

    def test_get_existing(self, tmp_project):
        stocks_dir, project_root = tmp_project
        frames = load_ohlcv(stocks_dir, project_root, min_rows=50)
        assert "AAA" in frames
        df = frames["AAA"]
        assert "Close" in df.columns

    def test_get_missing(self, tmp_project):
        stocks_dir, project_root = tmp_project
        frames = load_ohlcv(stocks_dir, project_root, min_rows=50)
        assert "NONEXISTENT" not in frames

    def test_exclusions(self, tmp_path):
        """Tickers in PROMOTSHARE sector should be excluded."""
        tickers = {
            "KEEP": make_ohlcv(300, seed=1),
            "EXCLUDE": make_ohlcv(300, seed=2),
        }
        stocks_dir = write_ticker_csvs(tmp_path, tickers)
        stocks_json = tmp_path / "data" / "stocks.json"
        stocks_json.write_text(
            json.dumps([{"script": "EXCLUDE", "sector": "PROMOTSHARE"}]),
            encoding="utf-8",
        )
        frames = load_ohlcv(stocks_dir, tmp_path, min_rows=50)
        assert len(frames) == 1
        assert "KEEP" in frames
        assert "EXCLUDE" not in frames

    def test_min_rows_filter(self, tmp_path):
        """Files with too few rows should be skipped."""
        tickers = {
            "SHORT": make_ohlcv(30, seed=1),
            "LONG": make_ohlcv(300, seed=2),
        }
        stocks_dir = write_ticker_csvs(tmp_path, tickers)
        frames = load_ohlcv(stocks_dir, tmp_path, min_rows=100)
        assert len(frames) == 1
        assert "LONG" in frames

    def test_dataframe_structure(self, tmp_project):
        stocks_dir, project_root = tmp_project
        frames = load_ohlcv(stocks_dir, project_root, min_rows=50)
        df = frames["AAA"]
        assert df.index.name == "Date"
        assert not df.index.duplicated().any()
        assert df.index.is_monotonic_increasing
        assert df.index.tz is None
