"""
FastAPI serving layer for NEPSE RL artifacts.
Strictly targets outputs/latest_eval/ for deterministic artifact delivery.
Serves raw OHLCV data, trade ledgers, per-ticker tear sheets,
and the macro-portfolio system_tear_sheet.json.
"""

import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NEPSE RL API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE_DIR / "outputs" / "latest_eval"
DATA_DIR = BASE_DIR / "data" / "stocks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv_as_dicts(path: Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/system")
def get_system_tear_sheet():
    """Serve the macro-portfolio system_tear_sheet.json."""
    ts_path = EVAL_DIR / "system_tear_sheet.json"
    if not ts_path.is_file():
        raise HTTPException(status_code=404, detail="system_tear_sheet.json not found. Run eval_agent.py first.")
    with open(ts_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/tickers")
def list_tickers():
    """Return tickers available in latest_eval/."""
    if not EVAL_DIR.is_dir():
        return []

    # Prefer aggregate_tear_sheet.csv
    tear_sheet = EVAL_DIR / "aggregate_tear_sheet.csv"
    if tear_sheet.is_file():
        rows = _read_csv_as_dicts(tear_sheet)
        return [r["ticker"] for r in rows if "ticker" in r]

    # Fallback: parse *_trade_ledger.csv filenames
    tickers = sorted(
        f.stem.replace("_trade_ledger", "")
        for f in EVAL_DIR.glob("*_trade_ledger.csv")
    )
    return tickers


@app.get("/api/metrics/{ticker}")
def get_metrics(ticker: str):
    """Serve the tear-sheet metrics for a single ticker."""
    tear_sheet = EVAL_DIR / "aggregate_tear_sheet.csv"
    if tear_sheet.is_file():
        rows = _read_csv_as_dicts(tear_sheet)
        for r in rows:
            if r.get("ticker") == ticker:
                for k, v in r.items():
                    if k == "ticker":
                        continue
                    try:
                        r[k] = float(v) if v else None
                    except (ValueError, TypeError):
                        pass
                return r
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found in tear sheet")

    raise HTTPException(status_code=404, detail="No tear-sheet data available")


@app.get("/api/ledger/{ticker}")
def get_ledger(ticker: str):
    """Serve the trade ledger CSV as JSON array."""
    ledger_path = EVAL_DIR / f"{ticker}_trade_ledger.csv"
    rows = _read_csv_as_dicts(ledger_path)

    # Light type coercion
    for r in rows:
        for k in ("close", "tsl_level", "portfolio_value"):
            if k in r:
                try:
                    r[k] = float(r[k]) if r[k] else None
                except (ValueError, TypeError):
                    r[k] = None
        for k in ("action", "position", "trade_id"):
            if k in r:
                try:
                    r[k] = int(r[k]) if r[k] else None
                except (ValueError, TypeError):
                    r[k] = None
        if "forced_liquidation" in r:
            r["forced_liquidation"] = r["forced_liquidation"] in ("True", "true", "1")
    return rows


@app.get("/api/data/{ticker}")
def get_ohlcv(ticker: str):
    """Serve raw OHLCV CSV from data/stocks/ as JSON array."""
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"No data file for ticker '{ticker}'")
    rows = _read_csv_as_dicts(csv_path)

    out: list[dict] = []
    for r in rows:
        ts = r.get("Timestamp", "")
        date_str = ts[:10] if ts else ""
        try:
            out.append({
                "time": date_str,
                "open": float(r["Open"]) if r.get("Open") else None,
                "high": float(r["High"]) if r.get("High") else None,
                "low": float(r["Low"]) if r.get("Low") else None,
                "close": float(r["Close"]) if r.get("Close") else None,
                "volume": float(r["Volume"]) if r.get("Volume") else None,
            })
        except (ValueError, KeyError):
            continue
    return out
