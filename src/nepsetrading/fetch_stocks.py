"""
Fetch all stock symbols from NepseTrading API.

Usage:
    try:
        from src.nepsetrading.fetch_stocks import fetch_nepsetrading_stocks
    except ImportError:
        from fetch_stocks import fetch_nepsetrading_stocks
    fetch_nepsetrading_stocks()
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Handle imports when run directly vs when imported
try:
    from src.nepsetrading.scraper import NepseTradingScraper
except ImportError:
    # Running directly - add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.nepsetrading.scraper import NepseTradingScraper


def setup_logging() -> logging.Logger:
    """Set up comprehensive logging with file and console handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Create outputs directory structure for logs only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("outputs") / f"log_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler for logs
    log_file = log_dir / "nepsetrading_stocks.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized - Log file: {log_file}")
    return logger


logger = setup_logging()


class NepseTradingSymbolFetcher(NepseTradingScraper):
    """Fetch all listed NEPSE symbols from the sectors endpoint.

    Saves JSON files to ``{out_dir}/stocks.json``, ``mutual.json``, ``corpdeben.json``.
    API URL: https://api.nepsetrading.com/historical-chart/sectors
    """

    EXCLUDED_SECTORS = {"CORPDEBEN", "MUTUAL"}

    def __init__(
        self,
        timeout: int = 30,
        out_dir: Path | None = None,
    ) -> None:
        super().__init__(timeout=timeout, out_dir=out_dir or Path("data"))
        self._out_file = self.out_dir / "stocks.json"
        self._mutual_file = self.out_dir / "mutual.json"
        self._corpdeben_file = self.out_dir / "corpdeben.json"

    def _fetch_symbols(self) -> list[dict]:
        path = "/historical-chart/sectors"
        data = self.get(path)
        if not isinstance(data, list):
            raise RuntimeError(f"Expected list response, got {type(data)}")
        return data

    @staticmethod
    def _parse(raw: list[dict]) -> list[dict]:
        result: list[dict] = []
        for item in raw:
            script = item.get("symbol", "") or item.get("ticker", "")
            name = item.get("description", "") or item.get("full_name", "")
            sector = item.get("sector", "")
            result.append(
                {
                    "script": script.strip().upper(),
                    "name": name.strip(),
                    "sector": sector.strip().upper(),
                }
            )
        return result

    def _save(self, records: list[dict]) -> None:
        self._out_file.parent.mkdir(parents=True, exist_ok=True)
        stocks = [r for r in records if r["sector"] not in self.EXCLUDED_SECTORS]
        mutual = [r for r in records if r["sector"] == "MUTUAL"]
        corpdeben = [r for r in records if r["sector"] == "CORPDEBEN"]

        for path, data in [
            (self._out_file, stocks),
            (self._mutual_file, mutual),
            (self._corpdeben_file, corpdeben),
        ]:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(data)} symbols to {path}")

    def run(self) -> None:
        logger.info("[NepseTrading] Starting stock symbols fetch from sectors endpoint")
        raw = self._fetch_symbols()
        records = self._parse(raw)
        logger.info(f"Retrieved {len(records)} total symbols from NepseTrading")
        self._save(records)


def fetch_nepsetrading_stocks(
    timeout: int = 30,
    out_dir: Path | None = None,
) -> None:
    """Fetch all stock symbols from NepseTrading API."""
    fetcher = NepseTradingSymbolFetcher(timeout=timeout, out_dir=out_dir or Path("data"))
    fetcher.run()
    logger.info("Successfully fetched all stock symbols from NepseTrading")


if __name__ == "__main__":
    fetch_nepsetrading_stocks()
