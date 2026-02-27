"""
Eval Agent – Deterministic evaluation + macro-aggregation
==========================================================
Standalone, Code Runner–compatible.
Loads weights from outputs/latest_model/, evaluates on 10 NEPSE tickers,
dumps per-ticker ledgers + system_tear_sheet.json → outputs/latest_eval/
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_manager import setup_logging
from src.data_loader import load_universe
from src.features import compute_features
from src.trainer import load_model
from src.environment import UniversalNepseEnv, run_deterministic_episode
from src.metrics import export_trade_ledger, generate_tear_sheet, build_trade_ledger
from src.visualize import evaluate_and_plot, compute_metrics


# ── Macro aggregation engine ──────────────────────────────────────────────

def aggregate_system_tear_sheet(
    per_ticker: Dict[str, dict],
) -> dict[str, Any]:
    """Compute portfolio-level Risk / Reward from per-ticker tear sheets."""
    total_trades = 0
    total_wins = 0
    total_closed = 0
    gross_profit = 0.0
    gross_loss = 0.0
    total_forced = 0

    for ticker, ts in per_ticker.items():
        nt = ts.get("num_trades", 0)
        wr = ts.get("win_rate", 0.0)
        total_trades += nt
        wins = int(round(nt * wr))
        total_wins += wins
        total_closed += nt
        total_forced += ts.get("forced_liquidations", 0)

        avg_w = ts.get("avg_win", 0.0) or 0.0
        avg_l = abs(ts.get("avg_loss", 0.0) or 0.0)
        gross_profit += avg_w * wins
        gross_loss += avg_l * (nt - wins)

    win_rate = total_wins / total_closed if total_closed else 0.0
    profit_factor = gross_profit / (gross_loss + 1e-10)
    expectancy = (gross_profit - gross_loss) / (total_closed + 1e-10)

    return {
        "total_system_trades": total_trades,
        "total_closed": total_closed,
        "total_wins": total_wins,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "system_expectancy": round(expectancy, 6),
        "gross_profit": round(gross_profit, 6),
        "gross_loss": round(gross_loss, 6),
        "forced_liquidations": total_forced,
        "tickers_evaluated": list(per_ticker.keys()),
    }


if __name__ == "__main__":
    # ── Config ───────────────────────────────────────────────────────
    DATA_DIR = str(ROOT / "data" / "stocks")
    MODEL_DIR = ROOT / "outputs" / "latest_model"
    MODEL_PATH = MODEL_DIR / "ppo_nepse_final"
    EPISODE_LENGTH = 252

    # Heterogeneous 10-ticker eval universe (banks, hydro, insurance, mfg, dev)
    EVAL_TICKERS = [
        "NABIL", "CHCL", "AHPC", "CBBL", "SHIVM",
        "ADBL", "AKPL", "ALICL", "API", "BOKL",
    ]

    # ── Deterministic output directory ───────────────────────────────
    EVAL_DIR = ROOT / "outputs" / "latest_eval"
    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = EVAL_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log = setup_logging(EVAL_DIR)
    log.info("=" * 60)
    log.info("  EVAL AGENT  –  %d tickers", len(EVAL_TICKERS))
    log.info("=" * 60)

    # ── Data & Features ──────────────────────────────────────────────
    t0 = time.time()
    master_df, vstarts = load_universe(DATA_DIR)
    feat_df = compute_features(master_df, vstarts)
    log.info("Data + features ready in %.1fs", time.time() - t0)

    # ── Load model ───────────────────────────────────────────────────
    model = load_model(str(MODEL_PATH))
    log.info("Model loaded from %s", MODEL_PATH)

    # ── Filter eval tickers to those present in data ─────────────────
    eval_tickers = [t for t in EVAL_TICKERS if t in vstarts]
    missing = set(EVAL_TICKERS) - set(eval_tickers)
    if missing:
        log.warning("Tickers not in universe (skipped): %s", missing)
    log.info("Evaluating: %s", eval_tickers)

    # ── Per-ticker evaluation ────────────────────────────────────────
    per_ticker_ts: Dict[str, dict] = {}
    all_tear_sheets: List[dict] = []

    for ticker in eval_tickers:
        log.info("\n%s  %s  %s", "=" * 20, ticker, "=" * 20)

        traj = evaluate_and_plot(
            model=model,
            feat_df=feat_df,
            valid_start_dates=vstarts,
            ticker=ticker,
            output_dir=str(EVAL_DIR),
            episode_length=EPISODE_LENGTH,
            plots_dir=str(PLOTS_DIR),
        )

        # Trade ledger CSV → outputs/latest_eval/{ticker}_trade_ledger.csv
        export_trade_ledger(traj, str(EVAL_DIR), ticker)

        # Tear sheet JSON → outputs/latest_eval/plots/{ticker}_tear_sheet.json
        ts = generate_tear_sheet(traj, str(PLOTS_DIR), ticker)
        per_ticker_ts[ticker] = ts
        all_tear_sheets.append(ts)

        # Action distribution log
        actions = traj["action"]
        n_act1 = int((actions == 1).sum())
        n_act0 = int((actions == 0).sum())
        log.info("  Actions: 0=%d  1=%d  (%.1f%% in-market)",
                 n_act0, n_act1, 100 * n_act1 / (n_act0 + n_act1 + 1e-10))
        log.info("  Final PV: %.4f", traj["portfolio_value"].iloc[-1])

    # ── Aggregate tear sheet CSV ─────────────────────────────────────
    if all_tear_sheets:
        tsdf = pd.DataFrame(all_tear_sheets)
        if "ticker" in tsdf.columns:
            tsdf = tsdf.set_index("ticker")
        tsdf.to_csv(EVAL_DIR / "aggregate_tear_sheet.csv")
        log.info("Aggregate tear sheet → %s", EVAL_DIR / "aggregate_tear_sheet.csv")

    # ── System tear sheet (macro-portfolio) ──────────────────────────
    system_ts = aggregate_system_tear_sheet(per_ticker_ts)
    system_ts_path = EVAL_DIR / "system_tear_sheet.json"
    with open(system_ts_path, "w") as f:
        json.dump(system_ts, f, indent=2)
    log.info("System tear sheet → %s", system_ts_path)
    log.info("\n%s", json.dumps(system_ts, indent=2))

    total = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info("  EVAL COMPLETE  –  %.1f min", total / 60)
    log.info("  Artifacts → %s", EVAL_DIR)
    log.info("=" * 60)
