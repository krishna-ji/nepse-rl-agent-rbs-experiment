"""
Fetch OHLCV data from NepseTrading API.

Usage:
    python -m src.dataScrap.nepsetrading.fetch_ohlcv NABIL GBBL
    python -m src.dataScrap.nepsetrading.fetch_ohlcv          # all symbols
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import sys

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
    log_file = log_dir / "nepsetrading_ohlcv.log"
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

# ---------------------------------------------------------------------------
# Multiprocessing worker (top-level for pickling)
# ---------------------------------------------------------------------------


def _fetch_and_save_symbol(
    symbol: str,
    from_date: str,
    to_date: str,
    timeout: int,
    out_dir_str: str,
    category: str = "stocks",
    resolution: str = "1D",
) -> tuple[int, str]:
    """Worker function for multiprocessing."""
    import requests as _requests

    session = _requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Referer": "https://nepsetrading.com/",
            "Origin": "https://nepsetrading.com",
        }
    )

    url = "https://api.nepsetrading.com/historical-chart/daily/adjusted"
    params = {"code": symbol, "from": from_date, "to": to_date}
    resp = session.get(url, params=params, timeout=timeout)
    resp.raise_for_status()

    if not resp.text.strip():
        raise RuntimeError(f"Empty response from {url}")

    data = resp.json()

    if isinstance(data, dict):
        if data.get("s") != "ok":
            raise RuntimeError(f"{symbol}: non-ok status: {data.get('s')}")

        required_keys = ("o", "h", "l", "c", "v", "t")
        for k in required_keys:
            if k not in data or not isinstance(data[k], list):
                raise RuntimeError(f"{symbol}: missing/invalid key '{k}'")

        timestamps = pd.to_datetime(pd.Series(data["t"], dtype="int64"), unit="s", utc=True)
        df = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Open": pd.array(data["o"], dtype="float32"),
                "High": pd.array(data["h"], dtype="float32"),
                "Low": pd.array(data["l"], dtype="float32"),
                "Close": pd.array(data["c"], dtype="float32"),
                "Volume": pd.array(data["v"], dtype="float32"),
            }
        )
    else:
        if not data:
            raise RuntimeError(f"{symbol}: No data returned")
        df = pd.DataFrame(data)
        df = df.rename(
            columns={
                "date": "Timestamp",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    ohlcv_dir = Path(out_dir_str) / "ohlcv" / resolution / category
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    path = ohlcv_dir / f"{symbol}.csv"
    df.to_csv(path, index=False)

    return len(df), str(path)


# ---------------------------------------------------------------------------
# Downloader class
# ---------------------------------------------------------------------------


class NepseTradingOHLCVDownloader(NepseTradingScraper):
    """Fetch and persist OHLCV CSV files from NepseTrading API.

    Files are stored under ``{out_dir}/ohlcv/{resolution}/{category}/{SYMBOL}.csv``.
    """

    def __init__(
        self,
        symbols: list[str],
        from_date: str = "1970-01-01",
        to_date: str = "2026-03-01",
        timeout: int = 30,
        out_dir: Path | None = None,
        max_workers: int | None = None,
        category: str = "stocks",
        resolution: str = "1D",
    ) -> None:
        super().__init__(timeout=timeout, out_dir=out_dir or Path("data"))
        self.symbols = [s.strip().upper() for s in symbols]
        self.from_date = from_date
        self.to_date = to_date
        self.max_workers = max_workers or min(len(symbols), 8)
        self.category = category
        self.resolution = resolution

    def run(self) -> None:
        logger.info(
            f"[NepseTrading] Starting OHLCV download for {len(self.symbols)} {self.category} symbols"
        )
        logger.info(
            f"Date range: {self.from_date} → {self.to_date}  |  Workers: {self.max_workers}"
        )

        failed, success = [], []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    _fetch_and_save_symbol,
                    sym,
                    self.from_date,
                    self.to_date,
                    self.timeout,
                    str(self.out_dir),
                    self.category,
                    self.resolution,
                ): sym
                for sym in self.symbols
            }

            done = 0
            for fut in as_completed(futures):
                sym = futures[fut]
                done += 1
                try:
                    rows, path = fut.result()
                    success.append(sym)
                    logger.info(f"✅ [{done}/{len(self.symbols)}] {sym}: {rows} rows saved")
                except Exception as exc:
                    failed.append(sym)
                    logger.error(f"❌ [{done}/{len(self.symbols)}] {sym}: {exc}")

        # Final summary
        logger.info("OHLCV download completed:")
        logger.info(
            f"  - Success: {len(success)}  |  Failed: {len(failed)}  |  Total: {len(self.symbols)}"
        )
        if failed:
            logger.warning(f"  - Failed symbols: {', '.join(failed)}")

    def run_sequential(self) -> None:
        """Sequential variant useful for debugging."""
        logger.info(
            f"[NepseTrading] Starting sequential OHLCV download for {len(self.symbols)} symbols"
        )
        failed, success = [], []
        for i, sym in enumerate(self.symbols, 1):
            try:
                rows, path = _fetch_and_save_symbol(
                    sym,
                    self.from_date,
                    self.to_date,
                    self.timeout,
                    str(self.out_dir),
                    self.category,
                    self.resolution,
                )
                success.append(sym)
                logger.info(f"✅ [{i}/{len(self.symbols)}] {sym}: {rows} rows saved")
            except Exception as exc:
                failed.append(sym)
                logger.error(f"❌ [{i}/{len(self.symbols)}] {sym}: {exc}")

        logger.info(
            f"Sequential download completed - Success: {len(success)}  |  Failed: {len(failed)}"
        )


CATEGORIES = {"stocks": "stocks.json", "mutual": "mutual.json", "corpdeben": "corpdeben.json"}


def fetch_nepsetrading_ohlcv(
    symbols: list[str] | None = None,
    resolution: str = "1D",
    from_date: str = "1970-01-01",
    to_date: str = "2026-03-01",
    max_workers: int = 8,
    out_dir: Path | None = None,
    sequential: bool = False,
) -> None:
    """Fetch historical OHLCV from NepseTrading for all categories."""
    out = out_dir or Path("data")

    if symbols:
        categories_to_run = {"stocks": [s.upper() for s in symbols]}
    else:
        categories_to_run = {}
        for cat, jf in CATEGORIES.items():
            jp = out / jf
            try:
                with open(jp) as f:
                    data = json.load(f)
                categories_to_run[cat] = [s["script"] for s in data]
            except FileNotFoundError:
                try:
                    from src.nepsetrading.fetch_stocks import NepseTradingSymbolFetcher
                except ImportError:
                    from fetch_stocks import NepseTradingSymbolFetcher

                NepseTradingSymbolFetcher().run()
                with open(jp) as f:
                    data = json.load(f)
                categories_to_run[cat] = [s["script"] for s in data]

    for cat, syms in categories_to_run.items():
        if not syms:
            continue
        dl = NepseTradingOHLCVDownloader(
            symbols=syms,
            from_date=from_date,
            to_date=to_date,
            max_workers=max_workers,
            out_dir=out,
            category=cat,
            resolution=resolution,
        )
        if sequential:
            dl.run_sequential()
        else:
            dl.run()


if __name__ == "__main__":
    fetch_nepsetrading_ohlcv()
