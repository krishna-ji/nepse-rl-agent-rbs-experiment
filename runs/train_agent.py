"""
Train Agent – 500k PPO on UniversalNepseEnv
=============================================
Standalone, Code Runner–compatible.
Outputs model .zip + training_losses.csv → outputs/latest_model/
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_manager import setup_logging
from src.data_loader import load_universe
from src.features import compute_features
from src.trainer import train_ppo


if __name__ == "__main__":
    # ── Hyperparameters ──────────────────────────────────────────────
    TIMESTEPS = 500_000
    DEVICE = "auto"
    N_ENVS = 4
    EPISODE_LENGTH = 252
    SEED = 42
    DATA_DIR = str(ROOT / "data" / "stocks")

    # ── Deterministic output directory ───────────────────────────────
    OUT_DIR = ROOT / "outputs" / "latest_model"
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR = OUT_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    log = setup_logging(OUT_DIR)
    log.info("=" * 60)
    log.info("  TRAIN AGENT  –  PPO %s timesteps", f"{TIMESTEPS:,}")
    log.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────
    t0 = time.time()
    master_df, vstarts = load_universe(DATA_DIR)
    log.info("Data loaded (%d tickers) in %.1fs", len(vstarts), time.time() - t0)

    # ── Features ─────────────────────────────────────────────────────
    t1 = time.time()
    feat_df = compute_features(master_df, vstarts)
    log.info("Features computed in %.1fs", time.time() - t1)

    # ── PPO Training ─────────────────────────────────────────────────
    t2 = time.time()
    model = train_ppo(
        feat_df=feat_df,
        valid_start_dates=vstarts,
        total_timesteps=TIMESTEPS,
        episode_length=EPISODE_LENGTH,
        n_envs=N_ENVS,
        save_dir=str(OUT_DIR),
        seed=SEED,
        device=DEVICE,
        plots_dir=str(PLOTS_DIR),
    )
    log.info("Training complete in %.1fs", time.time() - t2)

    # ── Verify artifacts ─────────────────────────────────────────────
    model_zip = OUT_DIR / "ppo_nepse_final.zip"
    losses_csv = PLOTS_DIR / "training_losses.csv"
    for p in (model_zip, losses_csv):
        status = "OK" if p.exists() else "MISSING"
        log.info("  [%s] %s", status, p.name)

    log.info("=" * 60)
    log.info("  DONE  –  %.1f min total", (time.time() - t0) / 60)
    log.info("  Artifacts → %s", OUT_DIR)
    log.info("=" * 60)
