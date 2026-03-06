#!/usr/bin/env python3
"""
NEPSE Mean-Reversion Decision Support System (DSS)
===================================================

Production-grade algorithmic trading DSS for Nepal Stock Exchange (NEPSE).
Evaluates mean-reverting regimes and outputs **buy-the-dip** signals using a
hybrid statistical (Ornstein-Uhlenbeck, Z-score) and machine-learning
(LightGBM / PPO-RL) pipeline.

Architecture
------------
*   **Statistical layer** – Ornstein-Uhlenbeck parameter estimation, rolling
    Z-score, Hurst exponent regime detection.
*   **ML layer** – LightGBM binary classifier for signal confirmation;
    PPO-based RL agent (``stable-baselines3``) for position sizing.
*   **Observability** – TensorBoard experiment tracking (training mode);
    Python ``logging`` with dual handlers (inference mode); rich console
    tables; ``matplotlib`` diagnostic plots.

Usage
-----
::

    # Training — fits LightGBM + RL, logs to TensorBoard, saves models/
    python expt/mean_reversion.py --mode train

    # Inference — loads models, generates signals CSV + plots
    python expt/mean_reversion.py --mode inference

    # With filters
    python expt/mean_reversion.py --mode inference --symbols ADBL,NABIL
    python expt/mean_reversion.py --mode train --sectors BANKING,HYDROPOWER
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, TypeAlias

import gymnasium as gym
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # headless backend — must precede pyplot import

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402
from gymnasium import spaces  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Constants & Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
OHLCV_STOCKS_DIR: Final[Path] = DATA_DIR / "ohlcv" / "1D" / "stocks"
STOCKS_META_PATH: Final[Path] = DATA_DIR / "stocks.json"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
OUTPUT_BASE_DIR: Final[Path] = PROJECT_ROOT / "output"
TB_LOG_DIR: Final[Path] = PROJECT_ROOT / "runs"

LGBM_MODEL_NAME: Final[str] = "lgbm_mean_reversion.txt"
RL_MODEL_NAME: Final[str] = "ppo_position_sizer"
SCALER_NAME: Final[str] = "scaler.pkl"

# ── Type aliases ──────────────────────────────────────────────────────────────

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]

# ── Module-level state ────────────────────────────────────────────────────────

console: Final[Console] = Console()
logger: logging.Logger = logging.getLogger("nepse_dss")

# ── Feature column registry (order matters for model I/O) ────────────────────

FEATURE_COLS: Final[list[str]] = [
    "zscore",
    "hurst",
    "ou_theta",
    "ou_sigma",
    "ou_halflife",
    "rsi_14",
    "bb_width",
    "vol_ratio",
    "volatility_20",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "close_to_sma50",
    "close_to_sma200",
]


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable runtime configuration built from CLI args."""

    mode: str = "inference"
    symbols: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)

    # ── statistical ───────────────────────────────────────────────────────
    zscore_window: int = 20
    hurst_window: int = 252
    hurst_max_lag: int = 20
    ou_window: int = 60
    min_history_days: int = 300
    zscore_entry: float = -2.0
    zscore_exit: float = 0.0

    # ── LightGBM ──────────────────────────────────────────────────────────
    lgbm_rounds: int = 200
    lgbm_early_stop: int = 20
    forward_days: int = 10
    forward_return_threshold: float = 0.02

    # ── RL (PPO) ──────────────────────────────────────────────────────────
    rl_timesteps: int = 50_000
    rl_lr: float = 3e-4
    rl_initial_capital: float = 1_000_000.0

    # ── general ───────────────────────────────────────────────────────────
    n_cv_splits: int = 5
    seed: int = 42


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Logging Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOG_FORMAT: Final[str] = (
    "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] - %(message)s"
)


def setup_logging(run_dir: Path | None = None) -> Path | None:
    """Configure the root logger with stream + optional file handler.

    Parameters
    ----------
    run_dir:
        When provided a ``DEBUG``-level :class:`logging.FileHandler` is
        attached, writing to ``run_dir/execution.log``.

    Returns
    -------
    Path | None
        Path to the log file, or ``None`` if no file handler was created.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Stream → stdout, INFO
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_path: Path | None = None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "execution.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        logger.info("Log file → %s", log_path)

    return log_path


def make_run_dir() -> Path:
    """Create and return ``output/YYYYMMDD_HHMMSS/`` directory."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_stock_metadata() -> pd.DataFrame:
    """Load ``data/stocks.json`` into a DataFrame with *script*, *name*, *sector*."""
    with open(STOCKS_META_PATH, encoding="utf-8") as fh:
        raw: list[dict[str, str]] = json.load(fh)
    return pd.DataFrame(raw)


def load_ohlcv(symbol: str, directory: Path = OHLCV_STOCKS_DIR) -> pd.DataFrame | None:
    """Load a single symbol's daily OHLCV CSV.  Returns ``None`` on failure."""
    path = directory / f"{symbol}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["Timestamp"], index_col="Timestamp")
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df = df.sort_index()
        df = df.loc[df["Volume"] > 0].dropna(subset=["Close"])
        return df if len(df) >= 10 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", symbol, exc)
        return None


def load_universe(cfg: Config) -> dict[str, pd.DataFrame]:
    """Load OHLCV for the requested universe, respecting ``--symbols`` / ``--sectors``."""
    meta = load_stock_metadata()
    all_symbols: list[str] = meta["script"].tolist()

    if cfg.symbols:
        target = [s.upper() for s in cfg.symbols]
    elif cfg.sectors:
        mask = meta["sector"].str.upper().isin([s.upper() for s in cfg.sectors])
        target = meta.loc[mask, "script"].tolist()
    else:
        target = all_symbols

    universe: dict[str, pd.DataFrame] = {}
    for sym in tqdm(target, desc="Loading OHLCV", unit="sym", leave=False):
        df = load_ohlcv(sym)
        if df is not None and len(df) >= cfg.min_history_days:
            universe[sym] = df
        else:
            logger.debug("Skipped %s (insufficient history or missing)", sym)

    logger.info("Loaded %d / %d requested symbols", len(universe), len(target))
    return universe


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Statistical Features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Z-score of *close* price."""
    mu = close.rolling(window, min_periods=window).mean()
    sigma = close.rolling(window, min_periods=window).std(ddof=1)
    return (close - mu) / sigma.replace(0.0, np.nan)


# ── Hurst exponent (R/S method) ──────────────────────────────────────────────


def _hurst_rs(data: FloatArray, max_lag: int = 20) -> float:
    """Compute the Hurst exponent of *data* (returns) via R/S analysis."""
    n = len(data)
    if n < max_lag * 2:
        return 0.5

    lags = np.arange(2, min(max_lag + 1, n // 2))
    log_lags: list[float] = []
    log_rs: list[float] = []

    for lag in lags:
        n_segs = n // lag
        if n_segs < 1:
            continue
        trimmed = data[: n_segs * lag].reshape(n_segs, lag)
        means = trimmed.mean(axis=1, keepdims=True)
        cum_dev = np.cumsum(trimmed - means, axis=1)
        R = cum_dev.max(axis=1) - cum_dev.min(axis=1)
        S = trimmed.std(axis=1, ddof=1)
        valid = S > 1e-12
        if valid.any():
            log_lags.append(float(np.log(lag)))
            log_rs.append(float(np.log(np.mean(R[valid] / S[valid]))))

    if len(log_lags) < 2:
        return 0.5
    return float(np.polyfit(log_lags, log_rs, 1)[0])


def compute_rolling_hurst(
    returns: pd.Series,
    window: int = 252,
    max_lag: int = 20,
) -> pd.Series:
    """Rolling Hurst exponent computed on log-return windows."""
    out = pd.Series(np.nan, index=returns.index, dtype=np.float64)
    vals = returns.values.astype(np.float64)
    for i in range(window, len(vals)):
        seg = vals[i - window : i]
        if np.isnan(seg).any():
            continue
        out.iloc[i] = _hurst_rs(seg, max_lag)
    return out.ffill()


# ── Ornstein-Uhlenbeck parameter estimation ──────────────────────────────────


def estimate_ou_params(close: pd.Series) -> tuple[float, float, float, float]:
    r"""Estimate OU parameters via OLS on discretised log-prices.

    Model: :math:`dX_t = \theta\,(\mu - X_t)\,dt + \sigma\,dW_t`

    Returns
    -------
    tuple[float, float, float, float]
        ``(theta, mu, sigma_annual, half_life_days)``
    """
    lp = np.log(close.values.astype(np.float64))
    n = len(lp)
    if n < 20:
        return 0.0, float(np.mean(lp)), 0.0, np.inf

    y = lp[1:] - lp[:-1]  # ΔX
    x = lp[:-1]  # X_{t-1}

    slope, intercept, _, _, _ = stats.linregress(x, y)
    theta = max(-slope, 1e-10)
    mu = intercept / theta
    residuals = y - (intercept + slope * x)
    sigma = float(np.std(residuals, ddof=1)) * np.sqrt(252.0)
    half_life = np.log(2.0) / theta if theta > 1e-10 else np.inf
    return theta, mu, sigma, half_life


def compute_rolling_ou(close: pd.Series, window: int = 60) -> pd.DataFrame:
    """Rolling OU parameter estimation (θ, σ, half-life)."""
    n = len(close)
    theta_arr = np.full(n, np.nan)
    sigma_arr = np.full(n, np.nan)
    hl_arr = np.full(n, np.nan)

    for i in range(window, n):
        th, _, sig, hl = estimate_ou_params(close.iloc[i - window : i])
        theta_arr[i] = th
        sigma_arr[i] = sig
        hl_arr[i] = min(hl, 1_000.0)

    return pd.DataFrame(
        {"ou_theta": theta_arr, "ou_sigma": sigma_arr, "ou_halflife": hl_arr},
        index=close.index,
    )


# ── RSI ───────────────────────────────────────────────────────────────────────


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Exponential-weighted Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feature Engineering Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_features(ohlcv: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute the full feature matrix for a single symbol.

    Rows with insufficient look-back are ``NaN`` and should be dropped by
    the caller.
    """
    close: pd.Series = ohlcv["Close"]
    volume: pd.Series = ohlcv["Volume"]
    returns: pd.Series = close.pct_change()

    feat = pd.DataFrame(index=ohlcv.index)

    # Z-score
    feat["zscore"] = compute_zscore(close, cfg.zscore_window)

    # Hurst exponent (on log-returns)
    log_ret = np.log(close / close.shift(1))
    feat["hurst"] = compute_rolling_hurst(log_ret, cfg.hurst_window, cfg.hurst_max_lag)

    # OU parameters
    ou = compute_rolling_ou(close, cfg.ou_window)
    feat["ou_theta"] = ou["ou_theta"]
    feat["ou_sigma"] = ou["ou_sigma"]
    feat["ou_halflife"] = ou["ou_halflife"]

    # RSI-14
    feat["rsi_14"] = compute_rsi(close)

    # Bollinger-Band width (normalised)
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=1)
    feat["bb_width"] = 2.0 * std20 / sma20.replace(0.0, np.nan)

    # Volume ratio
    vol_avg = volume.rolling(20, min_periods=20).mean()
    feat["vol_ratio"] = volume / vol_avg.replace(0.0, np.nan)

    # Annualised 20-day volatility
    feat["volatility_20"] = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)

    # Momentum (return look-backs)
    feat["ret_5d"] = close.pct_change(5)
    feat["ret_20d"] = close.pct_change(20)
    feat["ret_60d"] = close.pct_change(60)

    # Distance from SMAs
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    feat["close_to_sma50"] = close / sma50 - 1.0
    feat["close_to_sma200"] = close / sma200 - 1.0

    return feat[FEATURE_COLS]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Label Generation (supervised learning)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_labels(
    close: pd.Series,
    forward_days: int = 10,
    threshold: float = 0.02,
) -> pd.Series:
    """Binary label: **1** if the forward *N*-day return exceeds *threshold*."""
    fwd = close.shift(-forward_days) / close - 1.0
    return (fwd > threshold).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LightGBM Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _lgbm_base_params(cfg: Config) -> dict[str, Any]:
    """Return the base LightGBM parameter dict."""
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": cfg.seed,
    }


def _make_lgbm_tb_callback(writer: SummaryWriter) -> Any:
    """LightGBM callback that streams eval metrics to TensorBoard."""

    def _callback(env: Any) -> None:
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            writer.add_scalar(
                f"lgbm/{data_name}_{eval_name}",
                result,
                env.iteration,
            )

    _callback.order = 10  # type: ignore[attr-defined]
    return _callback


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Config,
    writer: SummaryWriter | None = None,
) -> lgb.Booster:
    """Train a LightGBM binary classifier with optional TensorBoard logging."""
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks: list[Any] = [
        lgb.log_evaluation(period=20),
        lgb.early_stopping(stopping_rounds=cfg.lgbm_early_stop),
    ]
    if writer is not None:
        callbacks.append(_make_lgbm_tb_callback(writer))

    model = lgb.train(
        _lgbm_base_params(cfg),
        dtrain,
        num_boost_round=cfg.lgbm_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model


def log_feature_importance(
    model: lgb.Booster,
    writer: SummaryWriter,
    feature_names: list[str],
) -> None:
    """Write a bar chart of LightGBM feature importances to TensorBoard."""
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in order],
        imp[order],
        color="#2196F3",
    )
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM Feature Importance (Gain)")
    ax.invert_yaxis()
    fig.tight_layout()
    writer.add_figure("lgbm/feature_importance", fig, global_step=0)
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reinforcement Learning — Gymnasium Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MeanReversionTradingEnv(gym.Env[FloatArray, int]):
    """Custom Gymnasium env for mean-reversion position management.

    Observation
        Feature vector concatenated with ``[position_pct, unrealised_ret]``.
    Action
        ``Discrete(3)`` → ``{0: hold, 1: buy_10pct_cash, 2: sell_50pct_pos}``.
    Reward
        Per-step portfolio return in basis points with a drawdown penalty.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        assert len(features) == len(prices), "features / prices length mismatch"
        self.features: np.ndarray = features.astype(np.float32)
        self.prices: np.ndarray = prices.astype(np.float64)
        self.initial_capital: float = initial_capital
        self.n_steps: int = len(features)

        n_obs = features.shape[1] + 2  # + position_pct + unrealised_ret
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs,),
            dtype=np.float32,
        )
        self.action_space: spaces.Discrete = spaces.Discrete(3)

        # mutable state — reset in reset()
        self._step_idx: int = 0
        self._cash: float = initial_capital
        self._position: float = 0.0
        self._peak_value: float = initial_capital

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_idx = 0
        self._cash = self.initial_capital
        self._position = 0.0
        self._peak_value = self.initial_capital
        return self._obs(), {}

    def step(
        self,
        action: int,
    ) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        price = self.prices[self._step_idx]
        prev_val = self._cash + self._position * price

        # ── execute action ────────────────────────────────────────────────
        if action == 1 and self._cash > 0:
            qty = (self._cash * 0.10) / price
            self._position += qty
            self._cash -= qty * price
        elif action == 2 and self._position > 0:
            qty = self._position * 0.50
            self._position -= qty
            self._cash += qty * price

        self._step_idx += 1
        truncated = self._step_idx >= self.n_steps - 1

        new_price = self.prices[min(self._step_idx, self.n_steps - 1)]
        new_val = self._cash + self._position * new_price
        self._peak_value = max(self._peak_value, new_val)

        # reward: bps return minus drawdown penalty
        ret_bps = (new_val - prev_val) / self.initial_capital * 10_000.0
        drawdown = (self._peak_value - new_val) / self._peak_value
        reward = float(ret_bps - 50.0 * drawdown)

        info: dict[str, Any] = {"portfolio_value": new_val}
        return self._obs(), reward, False, truncated, info

    # ── helpers ───────────────────────────────────────────────────────────

    def _obs(self) -> FloatArray:
        idx = min(self._step_idx, self.n_steps - 1)
        feat = self.features[idx]
        price = self.prices[idx]
        pv = self._cash + self._position * price
        pos_pct = np.float32(self._position * price / pv if pv > 0 else 0.0)
        unreal = np.float32((pv - self.initial_capital) / self.initial_capital)
        return np.concatenate([feat, [pos_pct, unreal]]).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RL Training Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class _RLRewardCallback(BaseCallback):
    """Logs per-episode reward & length to TensorBoard."""

    def __init__(self, writer: SummaryWriter, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._writer = writer
        self._ep_count: int = 0

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_count += 1
                ep = info["episode"]
                self._writer.add_scalar("rl/episode_reward", ep["r"], self._ep_count)
                self._writer.add_scalar("rl/episode_length", ep["l"], self._ep_count)
        return True


def train_rl_agent(
    features: np.ndarray,
    prices: np.ndarray,
    cfg: Config,
    writer: SummaryWriter,
) -> PPO:
    """Train a PPO agent on :class:`MeanReversionTradingEnv`."""

    def _make_env() -> MeanReversionTradingEnv:
        return MeanReversionTradingEnv(features, prices, cfg.rl_initial_capital)

    env = DummyVecEnv([_make_env])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.rl_lr,
        n_steps=min(2048, len(features) - 2),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        tensorboard_log=str(TB_LOG_DIR / "rl"),
        seed=cfg.seed,
    )
    callback = _RLRewardCallback(writer)
    logger.info("Training RL agent for %s timesteps …", f"{cfg.rl_timesteps:,}")
    model.learn(total_timesteps=cfg.rl_timesteps, callback=callback, progress_bar=True)
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Regime Classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def classify_regime(hurst: float) -> str:
    """Map a scalar Hurst exponent to a human-readable regime label."""
    if hurst < 0.45:
        return "mean_reverting"
    if hurst > 0.55:
        return "trending"
    return "random_walk"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Signal Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_signals(
    universe: dict[str, pd.DataFrame],
    cfg: Config,
    lgbm_model: lgb.Booster | None = None,
    rl_model: PPO | None = None,
    scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """Produce the composite signal :class:`~pandas.DataFrame`.

    For every symbol the **latest non-NaN row** is evaluated across all
    three signal layers (statistical, ML, RL) and a weighted composite
    score is emitted.
    """
    w_stat: float = 0.30
    w_ml: float = 0.50
    w_rl: float = 0.20

    rows: list[dict[str, Any]] = []

    for sym, ohlcv in tqdm(universe.items(), desc="Generating signals", leave=False):
        feat = build_features(ohlcv, cfg)
        valid_feat = feat.dropna()
        if valid_feat.empty:
            logger.debug("No valid features for %s — skipped", sym)
            continue

        latest_idx = valid_feat.index[-1]
        latest = valid_feat.loc[latest_idx]
        close_val = float(ohlcv.loc[latest_idx, "Close"])

        # ── statistical signal ────────────────────────────────────────────
        stat_signal = int(
            latest["zscore"] <= cfg.zscore_entry and latest["hurst"] < 0.5
        )

        # ── ML signal ─────────────────────────────────────────────────────
        ml_prob: float = 0.5
        if lgbm_model is not None:
            feat_row = latest[FEATURE_COLS].values.reshape(1, -1).astype(np.float64)
            if scaler is not None:
                feat_row = scaler.transform(feat_row)
            ml_prob = float(lgbm_model.predict(feat_row)[0])

        # ── RL signal ─────────────────────────────────────────────────────
        rl_action: int = 0
        if rl_model is not None:
            feat_vec = latest[FEATURE_COLS].values.astype(np.float32)
            if scaler is not None:
                feat_vec = (
                    scaler.transform(feat_vec.reshape(1, -1)).flatten().astype(np.float32)
                )
            obs = np.concatenate([feat_vec, np.array([0.0, 0.0], dtype=np.float32)])
            rl_action = int(rl_model.predict(obs, deterministic=True)[0])

        # ── composite score ───────────────────────────────────────────────
        rl_score = {0: 0.5, 1: 1.0, 2: 0.0}.get(rl_action, 0.5)
        composite = w_stat * float(stat_signal) + w_ml * ml_prob + w_rl * rl_score

        rows.append(
            {
                "symbol": sym,
                "date": str(latest_idx.date()),
                "close": round(close_val, 2),
                "zscore": round(float(latest["zscore"]), 4),
                "hurst": round(float(latest["hurst"]), 4),
                "ou_halflife": round(float(latest["ou_halflife"]), 2),
                "stat_signal": stat_signal,
                "ml_probability": round(ml_prob, 4),
                "rl_action": rl_action,
                "composite_score": round(composite, 4),
                "regime": classify_regime(float(latest["hurst"])),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plotting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_COLOUR_PRICE: Final[str] = "#1976D2"
_COLOUR_ZSCORE: Final[str] = "#E53935"
_COLOUR_HURST: Final[str] = "#6A1B9A"
_COLOUR_MR: Final[str] = "#C8E6C9"
_COLOUR_TREND: Final[str] = "#FFCDD2"
_COLOUR_RW: Final[str] = "#FFF9C4"


def plot_zscore_overlay(
    ohlcv: pd.DataFrame,
    feat: pd.DataFrame,
    symbol: str,
    save_dir: Path,
    trailing_days: int = 500,
) -> Path:
    """Price action with rolling Z-score overlay (dual Y-axis).

    Returns the path to the saved PNG.
    """
    df = ohlcv.join(feat[["zscore"]]).dropna(subset=["zscore"]).tail(trailing_days)
    if df.empty:
        raise ValueError(f"No plottable data for {symbol}")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_title(f"{symbol} — Price & Z-Score Overlay", fontsize=13, fontweight="bold")

    ax1.plot(df.index, df["Close"], color=_COLOUR_PRICE, linewidth=1.2, label="Close")
    ax1.set_ylabel("Close Price (NPR)", color=_COLOUR_PRICE)
    ax1.tick_params(axis="y", labelcolor=_COLOUR_PRICE)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    ax2 = ax1.twinx()
    ax2.plot(
        df.index,
        df["zscore"],
        color=_COLOUR_ZSCORE,
        linewidth=0.9,
        alpha=0.85,
        label="Z-Score",
    )
    ax2.axhline(-2.0, color=_COLOUR_ZSCORE, ls="--", alpha=0.5, lw=0.7)
    ax2.axhline(0.0, color="grey", ls=":", alpha=0.4, lw=0.7)
    ax2.axhline(2.0, color="#43A047", ls="--", alpha=0.5, lw=0.7)
    ax2.fill_between(
        df.index,
        df["zscore"],
        -2.0,
        where=df["zscore"] <= -2.0,
        color=_COLOUR_ZSCORE,
        alpha=0.15,
        label="Buy-the-Dip Zone",
    )
    ax2.set_ylabel("Z-Score", color=_COLOUR_ZSCORE)
    ax2.tick_params(axis="y", labelcolor=_COLOUR_ZSCORE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    path = save_dir / f"{symbol}_zscore_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_regime_hurst(
    ohlcv: pd.DataFrame,
    feat: pd.DataFrame,
    symbol: str,
    save_dir: Path,
    trailing_days: int = 500,
) -> Path:
    """Hurst exponent time-series with regime-coloured background.

    Two-panel chart: upper = price with shaded regime background; lower =
    Hurst exponent with threshold bands and annotated current state.

    Returns the saved PNG path.
    """
    df = ohlcv.join(feat[["hurst"]]).dropna(subset=["hurst"]).tail(trailing_days)
    if df.empty:
        raise ValueError(f"No Hurst data for {symbol}")

    fig, (ax_p, ax_h) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # ── upper panel: price + regime background ────────────────────────────
    ax_p.set_title(
        f"{symbol} — Regime Detection (Hurst Exponent)",
        fontsize=13,
        fontweight="bold",
    )
    ax_p.plot(df.index, df["Close"], color=_COLOUR_PRICE, lw=1.1)
    ax_p.set_ylabel("Close Price (NPR)")

    for i in range(len(df) - 1):
        h = df["hurst"].iloc[i]
        if h < 0.45:
            bg = _COLOUR_MR
        elif h > 0.55:
            bg = _COLOUR_TREND
        else:
            bg = _COLOUR_RW
        ax_p.axvspan(df.index[i], df.index[i + 1], alpha=0.35, color=bg, lw=0)

    # ── lower panel: Hurst exponent ───────────────────────────────────────
    ax_h.plot(df.index, df["hurst"], color=_COLOUR_HURST, lw=1.0)
    ax_h.axhline(0.50, color="grey", ls="--", lw=0.7, label="H = 0.50 (Random Walk)")
    ax_h.axhline(0.45, color="#43A047", ls=":", lw=0.7, label="Mean-Reverting Threshold")
    ax_h.axhline(0.55, color=_COLOUR_ZSCORE, ls=":", lw=0.7, label="Trending Threshold")
    ax_h.fill_between(
        df.index, df["hurst"], 0.45, where=df["hurst"] < 0.45, color="#43A047", alpha=0.15
    )
    ax_h.fill_between(
        df.index, df["hurst"], 0.55, where=df["hurst"] > 0.55, color=_COLOUR_ZSCORE, alpha=0.15
    )
    ax_h.set_ylabel("Hurst Exponent")
    ax_h.set_ylim(0.0, 1.0)
    ax_h.legend(loc="upper right", fontsize=8)

    # annotate current state
    cur_h = float(df["hurst"].iloc[-1])
    regime_now = classify_regime(cur_h)
    ax_h.annotate(
        f"Current: H = {cur_h:.3f} ({regime_now})",
        xy=(df.index[-1], cur_h),
        xytext=(-140, 25),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": _COLOUR_HURST},
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "white",
            "ec": _COLOUR_HURST,
            "alpha": 0.9,
        },
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    path = save_dir / f"{symbol}_regime_hurst.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Training Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_training(cfg: Config) -> None:
    """End-to-end training pipeline: LightGBM + PPO-RL with TensorBoard."""
    setup_logging(run_dir=None)  # console-only during training
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    tb_dir = TB_LOG_DIR / f"train_{ts}"
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info("TensorBoard logs → %s", tb_dir)
    logger.info("Launch TensorBoard:  tensorboard --logdir %s", TB_LOG_DIR)

    # persist hyper-params
    cfg_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    hparams = {k: str(v) for k, v in cfg_dict.items()}
    writer.add_text("config/hyperparams", json.dumps(hparams, indent=2), 0)

    # ── 1. Load data ──────────────────────────────────────────────────────
    universe = load_universe(cfg)
    if not universe:
        logger.error("No symbols loaded — aborting training.")
        writer.close()
        return

    # ── 2. Build pooled feature matrix ────────────────────────────────────
    logger.info("Building features for %d symbols …", len(universe))
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series] = []
    rl_candidates: list[tuple[np.ndarray, np.ndarray]] = []  # (features, prices)

    for sym, ohlcv in tqdm(universe.items(), desc="Feature engineering"):
        feat = build_features(ohlcv, cfg)
        labels = generate_labels(
            ohlcv["Close"], cfg.forward_days, cfg.forward_return_threshold
        )
        combined = feat.join(labels.rename("label")).dropna()
        if len(combined) < 100:
            logger.debug("Skipped %s (< 100 clean rows)", sym)
            continue
        all_X.append(combined[FEATURE_COLS])
        all_y.append(combined["label"])
        if len(combined) > 500:
            rl_candidates.append(
                (
                    combined[FEATURE_COLS].values,
                    ohlcv.loc[combined.index, "Close"].values,
                )
            )

    if not all_X:
        logger.error("No valid training rows after feature engineering — aborting.")
        writer.close()
        return

    X_pool = pd.concat(all_X, axis=0)
    y_pool = pd.concat(all_y, axis=0)
    logger.info(
        "Pooled dataset: %s samples × %d features (%.1f%% positive)",
        f"{len(X_pool):,}",
        X_pool.shape[1],
        y_pool.mean() * 100,
    )

    # ── 3. Scale features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_pool),
        columns=FEATURE_COLS,
        index=X_pool.index,
    )

    # ── 4. Train LightGBM with time-series CV ────────────────────────────
    logger.info("Training LightGBM (max %d rounds, %d folds) …", cfg.lgbm_rounds, cfg.n_cv_splits)
    tscv = TimeSeriesSplit(n_splits=cfg.n_cv_splits)
    best_model: lgb.Booster | None = None
    best_val_loss: float = float("inf")

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_scaled)):
        logger.info("  Fold %d / %d", fold_idx + 1, cfg.n_cv_splits)
        model = train_lgbm(
            X_scaled.iloc[tr_idx],
            y_pool.iloc[tr_idx],
            X_scaled.iloc[va_idx],
            y_pool.iloc[va_idx],
            cfg,
            writer,
        )
        val_loss: float = model.best_score["valid"]["binary_logloss"]
        writer.add_scalar("lgbm/fold_val_logloss", val_loss, fold_idx)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    assert best_model is not None, "No model was trained"
    lgbm_path = MODELS_DIR / LGBM_MODEL_NAME
    best_model.save_model(str(lgbm_path))
    logger.info(
        "LightGBM saved → %s  (best val logloss: %.5f)",
        lgbm_path,
        best_val_loss,
    )
    log_feature_importance(best_model, writer, FEATURE_COLS)

    # Save scaler
    scaler_path = MODELS_DIR / SCALER_NAME
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    logger.info("Scaler saved → %s", scaler_path)

    # ── 5. Train RL agent ─────────────────────────────────────────────────
    if rl_candidates:
        longest_idx = int(np.argmax([len(p) for _, p in rl_candidates]))
        rl_feat_raw, rl_prices = rl_candidates[longest_idx]
        rl_feat = scaler.transform(rl_feat_raw).astype(np.float32)

        rl_model = train_rl_agent(rl_feat, rl_prices, cfg, writer)
        rl_path = MODELS_DIR / RL_MODEL_NAME
        rl_model.save(str(rl_path))
        logger.info("RL model saved → %s", rl_path)
    else:
        logger.warning("No series long enough for RL training — skipped.")

    writer.close()
    console.print("\n[bold green]✓ Training complete.[/bold green]")
    console.print(f"  Models       → {MODELS_DIR}")
    console.print(f"  TensorBoard  → {tb_dir}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Inference Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_inference(cfg: Config) -> None:
    """Daily DSS execution: load models → generate signals → CSV + plots."""
    run_dir = make_run_dir()
    setup_logging(run_dir)

    logger.info("=" * 64)
    logger.info("  NEPSE Mean-Reversion DSS — Inference Run")
    logger.info("=" * 64)
    logger.info("Run directory : %s", run_dir)
    cfg_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    logger.info("Config        : %s", json.dumps(cfg_dict, default=str, indent=2))

    # ── 1. Load trained artefacts ─────────────────────────────────────────
    lgbm_model: lgb.Booster | None = None
    rl_model: PPO | None = None
    scaler: StandardScaler | None = None

    lgbm_path = MODELS_DIR / LGBM_MODEL_NAME
    if lgbm_path.exists():
        lgbm_model = lgb.Booster(model_file=str(lgbm_path))
        logger.info("Loaded LightGBM model from %s", lgbm_path)
    else:
        logger.warning(
            "LightGBM model not found at %s — statistical-only mode", lgbm_path
        )

    rl_zip = MODELS_DIR / (RL_MODEL_NAME + ".zip")
    if rl_zip.exists():
        rl_model = PPO.load(str(MODELS_DIR / RL_MODEL_NAME))
        logger.info("Loaded RL model from %s", rl_zip)
    else:
        logger.warning("RL model not found at %s — RL signals disabled", rl_zip)

    scaler_path = MODELS_DIR / SCALER_NAME
    if scaler_path.exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)  # noqa: S301
        logger.info("Loaded scaler from %s", scaler_path)

    # ── 2. Load universe ──────────────────────────────────────────────────
    universe = load_universe(cfg)
    if not universe:
        logger.error("No symbols loaded — aborting inference.")
        return

    # ── 3. Generate signals ───────────────────────────────────────────────
    signals_df = generate_signals(universe, cfg, lgbm_model, rl_model, scaler)
    if signals_df.empty:
        logger.warning("Signal generation returned zero rows.")
        return

    today_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    csv_path = run_dir / f"signals_{today_str}.csv"
    signals_df.to_csv(csv_path, index=False)
    logger.info("Signals CSV → %s  (%d rows)", csv_path, len(signals_df))

    # ── 4. Rich summary table ─────────────────────────────────────────────
    table = Table(title="NEPSE Mean-Reversion Signals (Top 20)", show_lines=True)
    table.add_column("Symbol", style="bold cyan")
    table.add_column("Close", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("Hurst", justify="right")
    table.add_column("OU HL", justify="right")
    table.add_column("ML Prob", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Regime", style="italic")

    for _, row in signals_df.head(20).iterrows():
        z_style = "bold red" if row["zscore"] <= -2.0 else ""
        h_style = "bold green" if row["hurst"] < 0.45 else ""
        table.add_row(
            str(row["symbol"]),
            f"{row['close']:.2f}",
            f"[{z_style}]{row['zscore']:.3f}[/{z_style}]" if z_style else f"{row['zscore']:.3f}",
            f"[{h_style}]{row['hurst']:.3f}[/{h_style}]" if h_style else f"{row['hurst']:.3f}",
            f"{row['ou_halflife']:.1f}d",
            f"{row['ml_probability']:.3f}",
            f"{row['composite_score']:.3f}",
            str(row["regime"]),
        )
    console.print(table)

    # ── 5. Diagnostic plots ───────────────────────────────────────────────
    plot_syms: list[str] = list(signals_df.head(5)["symbol"])
    strong_mr = signals_df.loc[
        (signals_df["regime"] == "mean_reverting") & (signals_df["zscore"] <= -1.5)
    ]["symbol"].tolist()
    plot_syms = list(dict.fromkeys(plot_syms + strong_mr))[:10]

    for sym in plot_syms:
        ohlcv = universe[sym]
        feat = build_features(ohlcv, cfg)
        try:
            p1 = plot_zscore_overlay(ohlcv, feat, sym, run_dir)
            logger.info("Plot saved → %s", p1.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Z-score plot failed for %s: %s", sym, exc)
        try:
            p2 = plot_regime_hurst(ohlcv, feat, sym, run_dir)
            logger.info("Plot saved → %s", p2.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Regime plot failed for %s: %s", sym, exc)

    logger.info("=" * 64)
    logger.info("  Inference complete.  Artefacts → %s", run_dir)
    logger.info("=" * 64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI Argument Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_args(argv: list[str] | None = None) -> Config:
    """Parse CLI arguments into an immutable :class:`Config`."""
    p = argparse.ArgumentParser(
        description="NEPSE Mean-Reversion DSS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["train", "inference"],
        help="Execution mode.",
    )
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbol filter (e.g. ADBL,NABIL).",
    )
    p.add_argument(
        "--sectors",
        type=str,
        default="",
        help="Comma-separated sector filter (e.g. BANKING,HYDROPOWER).",
    )
    p.add_argument("--epochs", type=int, default=200, help="LightGBM boosting rounds.")
    p.add_argument("--rl-timesteps", type=int, default=50_000, help="PPO total timesteps.")
    p.add_argument("--zscore-window", type=int, default=20, help="Z-score rolling window.")
    p.add_argument("--hurst-window", type=int, default=252, help="Hurst rolling window.")
    p.add_argument("--forward-days", type=int, default=10, help="Label look-ahead days.")
    p.add_argument("--min-history", type=int, default=300, help="Min bars per symbol.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")

    args = p.parse_args(argv)

    return Config(
        mode=args.mode,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        sectors=[s.strip() for s in args.sectors.split(",") if s.strip()],
        zscore_window=args.zscore_window,
        hurst_window=args.hurst_window,
        lgbm_rounds=args.epochs,
        rl_timesteps=args.rl_timesteps,
        forward_days=args.forward_days,
        min_history_days=args.min_history,
        seed=args.seed,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main(argv: list[str] | None = None) -> None:
    """Main entry point — dispatches to training or inference."""
    cfg = parse_args(argv)
    np.random.seed(cfg.seed)

    if cfg.mode == "train":
        run_training(cfg)
    else:
        run_inference(cfg)


if __name__ == "__main__":
    main()
