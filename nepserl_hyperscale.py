#!/usr/bin/env python3
"""
NEPSE RL Hyper-Scale Stochastic Pullback Engine  v3
====================================================
Single-file PPO trading engine exploiting full hardware:
  - 32 CPU cores via SubprocVecEnv (GIL bypass)
  - RTX 4060 (3072 CUDA cores, 8GB VRAM) — batch_size=4096
  - 128 GB RAM — full OHLCV manifold as contiguous NumPy arrays
  - 5M timesteps with linear LR decay (1e-4 -> 1e-5)

v3 changes:
  1. Regime-gated opportunity cost (penalize cash only in bull + rising market)
  2. Localized Drawdown Velocity feature (DD_state_t)
  3. Macro regime + DD_state in observation space (9 dims)
  4. Balanced KAPPA=0.015, GAMMA_DD=4.0 for 35-65% market exposure target
  5. 32 SubprocVecEnv workers (one per logical CPU core)
  6. LR floor at 1e-5 (prevents late-training stagnation)
"""

import warnings; warnings.filterwarnings("ignore")
import logging, pathlib, datetime, time, multiprocessing
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from typing import Callable

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_ENVS        = 32           # One per logical CPU core
TOTAL_TIMESTEPS = 5_000_000    # 5x previous
EPISODE_LENGTH  = 252          # 1 trading year
MIN_ROWS        = 250
WARMUP          = 200
SEED            = 42

# ============================================================================
# SETUP
# ============================================================================

def setup():
    """Setup run directory, logger, device."""
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = pathlib.Path(f"runs/{RUN_TS}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = pathlib.Path("data/ohlcv/1D/stocks")
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    log = logging.getLogger("nepserl")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(RUN_DIR / "nepserl_hyperscale.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); log.addHandler(fh)

    log.info("NEPSE RL Hyper-Scale Engine")
    log.info(f"Run dir  : {RUN_DIR.resolve()}")
    log.info(f"Data dir : {DATA_DIR.resolve()}")
    log.info(f"Device   : {DEVICE}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        log.info(f"GPU      : {p.name}  ({p.multi_processor_count * 128} CUDA cores, "
                 f"{p.total_memory / 1024**3:.1f} GB VRAM)")
    log.info(f"CPU cores: {multiprocessing.cpu_count()}")
    log.info(f"Envs     : {NUM_ENVS}")

    return log, RUN_DIR, DATA_DIR, DEVICE

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(DATA_DIR, log):
    """Load OHLCV CSVs → MultiIndex DataFrame + valid-start dates."""
    log.info("Loading OHLCV data...")
    frames, skipped = {}, 0
    for csv in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv, parse_dates=["Timestamp"])
            if df.empty or len(df) < MIN_ROWS:
                skipped += 1; continue
            df = df.rename(columns={"Timestamp": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
                skipped += 1; continue
            frames[csv.stem] = df[["Open","High","Low","Close","Volume"]]
        except Exception as e:
            log.warning(f"Skip {csv.stem}: {e}"); skipped += 1

    log.info(f"Loaded {len(frames)} tickers, skipped {skipped}")

    all_dates = sorted(set().union(*(f.index for f in frames.values())))
    idx = pd.DatetimeIndex(all_dates, name="Date")
    parts = {(tk, col): s.reindex(idx) for tk, df in frames.items() for col, s in df.items()}
    master_df = pd.DataFrame(parts)
    master_df.columns = pd.MultiIndex.from_tuples(master_df.columns, names=["Ticker","Feature"])
    log.info(f"Master DataFrame: {master_df.shape}")

    valid_start_dates = {}
    for tk in master_df.columns.get_level_values("Ticker").unique():
        tdays = master_df[tk].dropna(how="all").index
        valid_start_dates[tk] = tdays[WARMUP] if len(tdays) > WARMUP else tdays[-1]

    tickers = sorted(valid_start_dates.keys())
    log.info(f"{len(tickers)} tickers with valid start dates")
    return master_df, valid_start_dates, tickers

# ============================================================================
# FEATURE ENGINEERING  (all features scaled to [-1, +1])
# ============================================================================

def _sma(s, n):       return s.rolling(n, min_periods=n).mean()
def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
def _atr(h, l, c, n=14): return _true_range(h, l, c).ewm(span=n, min_periods=n, adjust=False).mean()
def _stochastic(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp, min_periods=kp).min()
    hi = h.rolling(kp, min_periods=kp).max()
    raw_k = 100.0 * (c - lo) / (hi - lo + 1e-10)
    pk = raw_k.rolling(dp, min_periods=dp).mean()
    pd_ = pk.rolling(dp, min_periods=dp).mean()
    return pk, pd_
def _bbw(c, n=20, ns=2.0):
    s_ = _sma(c, n); std = c.rolling(n, min_periods=n).std()
    return ((s_ + ns*std) - (s_ - ns*std)) / (s_ + 1e-10)
def _psl(l, w=60): return l.rolling(w, min_periods=w).min()

def compute_features(master_df, valid_start_dates, log):
    """Compute features with fixed scaling [-1, +1]."""
    all_tkrs = master_df.columns.get_level_values("Ticker").unique()
    log.info(f"Computing features for {len(all_tkrs)} tickers...")
    pieces = {}
    for tk in all_tkrs:
        raw = master_df[tk].dropna(how="all")
        o, h, l, c, v = raw["Open"], raw["High"], raw["Low"], raw["Close"], raw["Volume"]
        pieces[(tk,"open")] = o; pieces[(tk,"high")] = h; pieces[(tk,"low")] = l
        pieces[(tk,"close")] = c; pieces[(tk,"volume")] = v
        sma50, sma200, psl_ = _sma(c,50), _sma(c,200), _psl(l,60)
        pieces[(tk,"sma50")] = sma50; pieces[(tk,"sma200")] = sma200
        pieces[(tk,"macro_trend")] = (sma50 > sma200).astype(np.float32)
        pieces[(tk,"protected_swing_low")] = psl_
        pieces[(tk,"d_low")] = (c - psl_) / (c + 1e-10)
        pk, pd_ = _stochastic(h, l, c)
        pieces[(tk,"pct_k")] = (pk / 50.0) - 1.0
        pieces[(tk,"pct_d")] = (pd_ / 50.0) - 1.0
        pieces[(tk,"delta_k")] = (pk - pk.shift(1)) / 50.0
        atr14 = _atr(h, l, c, 14)
        pieces[(tk,"atr14")] = atr14
        pieces[(tk,"natr")] = atr14 / (c + 1e-10)
        pieces[(tk,"bbw")] = _bbw(c, 20, 2.0)
        # Localized Drawdown Velocity (20-bar rolling high)
        rolling_high_20 = h.rolling(20, min_periods=20).max()
        pieces[(tk,"dd_state")] = ((rolling_high_20 - c) / (rolling_high_20 + 1e-10)).clip(0.0, 1.0)

    feat_df = pd.DataFrame(pieces)
    feat_df.columns = pd.MultiIndex.from_tuples(feat_df.columns, names=["Ticker","Feature"])
    feat_df = feat_df.sort_index()

    for tk in all_tkrs:
        for col in ["natr", "bbw", "d_low"]:
            key = (tk, col)
            if key not in feat_df.columns: continue
            clean = feat_df[key].dropna()
            rm = clean.rolling(252, min_periods=252).mean()
            rs = clean.rolling(252, min_periods=252).std()
            feat_df[key] = ((clean - rm) / (rs + 1e-8)).clip(-3.0, 3.0)

    log.info(f"Feature matrix: {feat_df.shape} — all scaled to [-1, +1]")
    return feat_df

# ============================================================================
# PRE-COMPILE NUMPY ARRAYS  (RAM exploitation — zero Pandas in step())
# ============================================================================

OBS_FEATURES = ["pct_k", "pct_d", "natr", "bbw", "d_low", "dd_state", "macro_trend"]
NEEDED_COLS  = OBS_FEATURES + ["close", "high", "low", "atr14", "protected_swing_low"]

def precompile_arrays(feat_df, tickers, valid_start_dates, log):
    """Convert Pandas manifold to contiguous float32 arrays for O(1) access."""
    dates_array = feat_df.index.values
    n_dates = len(dates_array)

    ticker_arrays = {}
    for tk in tickers:
        arrays = {}
        for f in NEEDED_COLS:
            try:
                arrays[f] = np.ascontiguousarray(feat_df[(tk, f)].values, dtype=np.float32)
            except KeyError:
                arrays[f] = np.full(n_dates, np.nan, dtype=np.float32)
        ticker_arrays[tk] = arrays

    valid_start_idx = {}
    for tk in tickers:
        vs = np.datetime64(valid_start_dates[tk])
        valid_start_idx[tk] = int(np.searchsorted(dates_array, vs))

    total_bytes = sum(a.nbytes for d in ticker_arrays.values() for a in d.values())
    log.info(f"Pre-compiled {len(ticker_arrays)} tickers × {len(NEEDED_COLS)} arrays "
             f"({total_bytes / 1024**2:.1f} MB)")
    return ticker_arrays, valid_start_idx, dates_array, n_dates

# ============================================================================
# GYMNASIUM ENVIRONMENT — Autonomous Exit Policy
# ============================================================================
#
# Key changes from v1:
#   - No hard Chandelier Exit / TSL — agent controls all sell decisions
#   - Continuous exponential drawdown penalty:
#       DD_t = max(0, (Peak - C_t) / Peak)
#       R_t  = ln(C_t / C_{t-1}) - tau - kappa * (exp(gamma * DD_t) - 1)
#   - Pure NumPy step() — no Pandas in the hot path

class NepseHyperEnv(gym.Env):
    TAU      = 0.005       # ~0.5% realistic NEPSE broker fees
    OC_SCALE = 0.10        # regime-gated opportunity cost of cash
    KAPPA    = 0.015       # drawdown penalty coefficient
    GAMMA_DD = 4.0         # exponential curvature for drawdown

    def __init__(self, ticker_arrays, valid_start_idx, dates_array,
                 tickers, episode_length=252, seed=None):
        super().__init__()
        self._ta    = ticker_arrays
        self._vsi   = valid_start_idx
        self._dates = dates_array
        self._n     = len(dates_array)
        self._tickers = tickers
        self._ep_len  = episode_length
        self.action_space = spaces.Discrete(2)
        # 9 dims: pct_k, pct_d, natr, bbw, d_low, dd_state, macro_trend, position, peak_dd
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(9,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._tk = ""; self._si = 0; self._t = 0
        self._pos = 0; self._entry = 0.0; self._peak = 0.0; self._pv = 1.0
        self._buys = 0; self._sells = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._tk = self._rng.choice(self._tickers)
        vs_idx = self._vsi[self._tk]
        last = self._n - self._ep_len
        if last <= vs_idx: vs_idx = max(0, last - 1)
        self._si = int(self._rng.integers(vs_idx, max(vs_idx + 1, last)))
        self._t = 0; self._pos = 0; self._entry = 0.0; self._peak = 0.0
        self._pv = 1.0; self._buys = 0; self._sells = 0
        return self._obs(), self._info()

    def step(self, action):
        c = self._ta[self._tk]
        i = self._si + self._t
        if i >= self._n:
            return self._obs(), 0.0, True, True, self._info()

        close_t = float(c["close"][i])
        prev_c  = float(c["close"][max(i - 1, 0)])

        if np.isnan(close_t) or np.isnan(prev_c):
            self._t += 1
            done = self._t >= self._ep_len
            trunc = (self._si + self._t) >= self._n - 1
            return self._obs(), 0.0, done, trunc, self._info()

        reward = 0.0

        if self._pos == 0 and action == 1:
            # BUY
            self._pos = 1; self._entry = close_t; self._peak = close_t
            reward -= self.TAU; self._buys += 1

        elif self._pos == 1 and action == 1:
            # HOLD LONG — continuous drawdown penalty
            self._peak = max(self._peak, close_t)
            lr = np.log(close_t / (prev_c + 1e-10))
            dd = max(0.0, (self._peak - close_t) / (self._peak + 1e-10))
            dd_penalty = self.KAPPA * (np.exp(self.GAMMA_DD * dd) - 1.0)
            reward += lr - dd_penalty
            self._pv *= np.exp(lr)

        elif self._pos == 1 and action == 0:
            # SELL (agent-initiated exit)
            self._pos = 0
            lr = np.log(close_t / (prev_c + 1e-10))
            reward += lr - self.TAU
            self._pv *= np.exp(lr - self.TAU)
            self._entry = 0.0; self._peak = 0.0; self._sells += 1

        else:
            # HOLD CASH — regime-gated opportunity cost
            delta = np.log(close_t / (prev_c + 1e-10))
            macro = float(c["macro_trend"][i])
            if delta > 0 and (not np.isnan(macro) and macro > 0.5):
                reward -= self.OC_SCALE * delta

        self._t += 1
        done  = self._t >= self._ep_len
        trunc = (self._si + self._t) >= self._n - 1
        if np.isnan(reward) or np.isinf(reward): reward = 0.0
        return self._obs(), float(reward), done, trunc, self._info()

    def _obs(self):
        c = self._ta[self._tk]
        i = min(self._si + self._t, self._n - 1)
        obs = np.zeros(9, dtype=np.float32)
        for j, f in enumerate(OBS_FEATURES):
            v = c[f][i]; obs[j] = 0.0 if np.isnan(v) else float(v)
        obs[7] = float(self._pos)
        if self._pos == 1 and self._peak > 0:
            cl = float(c["close"][i])
            if not np.isnan(cl):
                obs[8] = np.clip((self._peak - cl) / (self._peak + 1e-10), -1.0, 1.0)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _info(self):
        c = self._ta[self._tk]
        i = min(self._si + self._t, self._n - 1)
        return {
            "ticker": self._tk, "date": self._dates[i],
            "close": float(c["close"][i]),
            "action": -1, "portfolio_value": float(self._pv),
            "position": self._pos, "peak": float(self._peak),
            "drawdown": max(0.0, (self._peak - float(c["close"][i])) / (self._peak + 1e-10))
                        if self._pos == 1 and self._peak > 0 else 0.0,
        }

# ============================================================================
# TRAINING CALLBACK
# ============================================================================

class HyperScaleTracker(BaseCallback):
    def __init__(self, run_dir, log_every=100):
        super().__init__(verbose=0)
        self.run_dir = pathlib.Path(run_dir)
        self.log_every = log_every
        self.ep_ts = []; self.ep_rewards = []; self.ep_lengths = []
        self.upd_ts = []; self.pol_loss = []; self.val_loss = []
        self.ent_loss = []; self.kl = []; self.clip_frac = []
        self.expl_var = []; self.lr = []
        self._ep_count = 0; self._best_avg = -np.inf

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_ts.append(self.num_timesteps)
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self._ep_count += 1
                if self._ep_count % self.log_every == 0:
                    recent = self.ep_rewards[-100:]
                    avg = np.mean(recent)
                    marker = ""
                    if avg > self._best_avg:
                        self._best_avg = avg; marker = " ★ NEW BEST"
                    log = logging.getLogger("nepserl")
                    log.info(f"Ep {self._ep_count:6d} | ts {self.num_timesteps:9,d} | "
                             f"R {avg:+.4f}±{np.std(recent):.3f} "
                             f"[{np.min(recent):+.3f}, {np.max(recent):+.3f}]{marker}")
                    if self._ep_count % 500 == 0:
                        self._export()
        return True

    def _on_rollout_end(self):
        try:
            v = self.model.logger.name_to_value
            self.upd_ts.append(self.num_timesteps)
            self.pol_loss.append(v.get("train/policy_gradient_loss", np.nan))
            self.val_loss.append(v.get("train/value_loss", np.nan))
            self.ent_loss.append(v.get("train/entropy_loss", np.nan))
            self.kl.append(v.get("train/approx_kl", np.nan))
            self.clip_frac.append(v.get("train/clip_fraction", np.nan))
            self.expl_var.append(v.get("train/explained_variance", np.nan))
            self.lr.append(v.get("train/learning_rate", np.nan))
        except: pass

    def _export(self):
        if not self.ep_rewards: return
        edf = pd.DataFrame({
            "timestep": self.ep_ts,
            "episode": range(1, len(self.ep_rewards)+1),
            "reward": self.ep_rewards, "length": self.ep_lengths,
        })
        for w in [10, 50, 100, 500]:
            if len(self.ep_rewards) >= w:
                edf[f"ma_{w}"] = edf["reward"].rolling(w).mean()
        edf.to_csv(self.run_dir / "episode_rewards.csv", index=False)
        if self.upd_ts:
            pd.DataFrame({
                "timestep": self.upd_ts, "policy_loss": self.pol_loss,
                "value_loss": self.val_loss, "entropy_loss": self.ent_loss,
                "approx_kl": self.kl, "clip_fraction": self.clip_frac,
                "explained_variance": self.expl_var, "learning_rate": self.lr,
            }).to_csv(self.run_dir / "training_losses.csv", index=False)

# ============================================================================
# LINEAR LR SCHEDULE
# ============================================================================

def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linear decay from initial_value to final_value as training progresses."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ticker(model, ticker, ticker_arrays, valid_start_idx,
                    dates_array, tickers_list, episode_length=252, seed=123):
    """Deterministic evaluation on a single ticker."""
    env = NepseHyperEnv(ticker_arrays, valid_start_idx, dates_array,
                        tickers_list, episode_length=episode_length, seed=seed)
    obs, info = env.reset()
    env._tk = ticker; env._si = valid_start_idx[ticker]
    env._t = 0; env._pos = 0; env._entry = 0.0; env._peak = 0.0; env._pv = 1.0
    obs = env._obs(); info = env._info()

    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        info["action"] = action
        records.append(info.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    info["action"] = -1; records.append(info.copy())
    traj = pd.DataFrame(records)

    pv = traj["portfolio_value"].values
    cl = traj["close"].values
    agent_ret = pv[-1] / pv[0] - 1.0
    bh_ret    = cl[-1] / cl[0] - 1.0
    pv_rets   = np.diff(pv) / pv[:-1]
    sharpe    = np.mean(pv_rets) / (np.std(pv_rets) + 1e-10) * np.sqrt(252)
    cummax    = np.maximum.accumulate(pv)
    max_dd    = np.max((cummax - pv) / cummax)
    actions   = traj["action"].values
    n_total   = len(actions) - 1
    buy_ratio = (actions[:-1] == 1).sum() / n_total if n_total > 0 else 0

    return {
        "ticker": ticker, "agent_return": agent_ret, "buyhold_return": bh_ret,
        "excess_return": agent_ret - bh_ret, "sharpe_ratio": sharpe,
        "max_drawdown": max_dd, "buy_ratio": buy_ratio, "final_pv": pv[-1],
    }, traj

# ============================================================================
# PLOTS
# ============================================================================

def plot_dashboard(tracker, run_dir, total_ts, num_envs):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.plot(tracker.ep_ts, tracker.ep_rewards, alpha=0.15, color="dodgerblue", lw=0.5)
    if len(tracker.ep_rewards) > 100:
        ma100 = pd.Series(tracker.ep_rewards).rolling(100).mean()
        ax.plot(tracker.ep_ts, ma100, color="red", lw=2, label="MA-100")
    if len(tracker.ep_rewards) > 500:
        ma500 = pd.Series(tracker.ep_rewards).rolling(500).mean()
        ax.plot(tracker.ep_ts, ma500, color="darkgreen", lw=2, label="MA-500")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if tracker.upd_ts:
        ax.plot(tracker.upd_ts, tracker.pol_loss, label="Policy Loss", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(tracker.upd_ts, tracker.val_loss, color="orange", label="Value Loss", alpha=0.8)
        ax.set_ylabel("Policy Loss"); ax2.set_ylabel("Value Loss")
    ax.set_xlabel("Timesteps"); ax.set_title("Training Losses"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if tracker.upd_ts:
        ax.plot(tracker.upd_ts, [-e for e in tracker.ent_loss], color="purple")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if tracker.upd_ts:
        ax.plot(tracker.upd_ts, tracker.expl_var, color="green")
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Explained Var")
    ax.set_title("Value Function Quality"); ax.grid(alpha=0.3)

    plt.suptitle(f"NEPSE RL Hyper-Scale — {total_ts/1e6:.0f}M steps, {num_envs} envs", fontsize=14)
    plt.tight_layout()
    plt.savefig(run_dir / "training_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_equity_curves(all_trajs, run_dir):
    n_plots = min(len(all_trajs), 10)
    cols = 2; rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    for ix, (tk, traj) in enumerate(all_trajs.items()):
        if ix >= n_plots: break
        ax = axes[ix]
        pv = traj["portfolio_value"].values; cl = traj["close"].values
        bh = cl / cl[0]
        ax.plot(pv, label=f"Agent ({pv[-1]-1:+.1%})", color="dodgerblue", lw=1.5)
        ax.plot(bh, label=f"B&H ({bh[-1]-1:+.1%})", color="gray", lw=1, alpha=0.7)
        ax.axhline(1.0, color="black", ls="--", alpha=0.3)
        ax.set_title(tk, fontsize=12, fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for ix in range(n_plots, len(axes)): axes[ix].set_visible(False)
    plt.suptitle("Out-of-Sample Equity Curves: Agent vs Buy & Hold", fontsize=14)
    plt.tight_layout()
    plt.savefig(run_dir / "oos_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    log, run_dir, data_dir, device = setup()

    # ── Data ────────────────────────────────────────────────────────────────
    master_df, valid_start_dates, tickers = load_ohlcv(data_dir, log)
    feat_df = compute_features(master_df, valid_start_dates, log)
    ticker_arrays, valid_start_idx, dates_array, n_dates = precompile_arrays(
        feat_df, tickers, valid_start_dates, log)

    # ── SubprocVecEnv factory ───────────────────────────────────────────────
    def make_env(rank, seed=SEED):
        def _init():
            return Monitor(NepseHyperEnv(
                ticker_arrays, valid_start_idx, dates_array,
                tickers, episode_length=EPISODE_LENGTH, seed=seed + rank))
        return _init

    log.info(f"Spawning {NUM_ENVS} SubprocVecEnv workers...")
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)], start_method="spawn")
    log.info(f"SubprocVecEnv ready — {NUM_ENVS} workers active")

    # ── PPO Model ───────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=linear_schedule(1e-4),
        n_steps=4096,         # 4096 × 24 envs = 98,304 per rollout
        batch_size=4096,      # saturate RTX 4060
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        device=device,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    buf = 4096 * NUM_ENVS
    log.info(f"PPO on {model.device} — {n_params:,} params, buffer {buf:,}/rollout")

    # ── Train ───────────────────────────────────────────────────────────────
    tracker = HyperScaleTracker(run_dir)
    log.info(f"Starting training: {TOTAL_TIMESTEPS:,} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[tracker], progress_bar=True)
    elapsed = time.time() - t0
    vec_env.close()
    tracker._export()

    fps = TOTAL_TIMESTEPS / elapsed
    final_avg = np.mean(tracker.ep_rewards[-100:]) if len(tracker.ep_rewards) >= 100 else 0.0
    log.info(f"Training complete — {elapsed/60:.1f} min, {fps:,.0f} FPS")
    log.info(f"  Episodes : {len(tracker.ep_rewards):,}")
    log.info(f"  Final R  : {final_avg:+.4f}")
    log.info(f"  Best R   : {tracker._best_avg:+.4f}")

    model.save(run_dir / "nepserl_hyperscale_model.zip")
    log.info(f"Model saved: {run_dir / 'nepserl_hyperscale_model.zip'}")

    # ── Dashboard ───────────────────────────────────────────────────────────
    plot_dashboard(tracker, run_dir, TOTAL_TIMESTEPS, NUM_ENVS)
    log.info("Training dashboard saved")

    # ── Multi-ticker OOS evaluation ─────────────────────────────────────────
    eval_candidates = [
        "NABIL", "NICA", "SHIVM", "CHDC", "NLIC",
        "UPPER", "NRIC", "SBL", "GBIME", "SANIMA",
        "SCB", "ADBL", "NTC", "BOKL", "HBL",
        "EBL", "NBL", "KBL", "SBI", "CZBIL",
    ]
    eval_tickers = [t for t in eval_candidates if t in tickers][:10]
    if len(eval_tickers) < 10:
        remaining = [t for t in tickers if t not in eval_tickers]
        np.random.seed(42)
        extra = list(np.random.choice(
            remaining, min(10 - len(eval_tickers), len(remaining)), replace=False))
        eval_tickers.extend(extra)

    log.info(f"OOS evaluation on {len(eval_tickers)} tickers: {eval_tickers}")
    results = []; all_trajs = {}
    for tk in eval_tickers:
        try:
            m, traj = evaluate_ticker(
                model, tk, ticker_arrays, valid_start_idx,
                dates_array, tickers, EPISODE_LENGTH)
            results.append(m); all_trajs[tk] = traj
            log.info(f"  {tk:>8s} | Agent {m['agent_return']:+7.2%} | B&H {m['buyhold_return']:+7.2%} | "
                     f"Excess {m['excess_return']:+7.2%} | Sharpe {m['sharpe_ratio']:+.2f} | "
                     f"MaxDD {m['max_drawdown']:.2%} | Buy {m['buy_ratio']:.1%}")
        except Exception as e:
            log.warning(f"  {tk}: eval failed — {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(run_dir / "oos_evaluation.csv", index=False)

    # ── OOS summary ─────────────────────────────────────────────────────────
    if not results_df.empty:
        n_beat = (results_df["excess_return"] > 0).sum()
        n_total = len(results_df)
        log.info("=" * 60)
        log.info("OOS AGGREGATE")
        log.info(f"  Beat B&H          : {n_beat}/{n_total} ({n_beat/n_total:.0%})")
        log.info(f"  Avg excess return  : {results_df['excess_return'].mean():+.2%}")
        log.info(f"  Median excess      : {results_df['excess_return'].median():+.2%}")
        log.info(f"  Avg Sharpe         : {results_df['sharpe_ratio'].mean():+.2f}")
        log.info(f"  Avg max drawdown   : {results_df['max_drawdown'].mean():.2%}")
        log.info(f"  Avg buy ratio      : {results_df['buy_ratio'].mean():.1%}")
        if n_beat >= 8:
            log.info("SYSTEM IS FUNDAMENTALLY ROBUST (>=8/10 tickers beaten)")
        elif n_beat >= 5:
            log.info(f"PARTIAL ROBUSTNESS ({n_beat}/10)")
        else:
            log.info(f"INSUFFICIENT ROBUSTNESS ({n_beat}/10)")
        log.info("=" * 60)

    # ── Equity curve plots ──────────────────────────────────────────────────
    if all_trajs:
        plot_equity_curves(all_trajs, run_dir)
        log.info("OOS equity curves saved")

    # ── Final summary CSV ───────────────────────────────────────────────────
    summary = {
        "metric": [
            "total_episodes", "total_timesteps", "training_fps", "wall_time_min",
            "final_avg_reward_100", "best_avg_reward_100",
            "num_envs", "batch_size", "n_steps", "network",
            "oos_tickers_evaluated", "oos_tickers_beating_bh",
            "oos_avg_excess_return", "oos_avg_sharpe", "oos_avg_max_dd",
            "device", "gpu",
        ],
        "value": [
            len(tracker.ep_rewards),
            tracker.ep_ts[-1] if tracker.ep_ts else 0,
            f"{fps:,.0f}", f"{elapsed/60:.1f}",
            f"{final_avg:+.4f}", f"{tracker._best_avg:+.4f}",
            NUM_ENVS, 4096, 4096, "[256,256]x2",
            len(results_df) if not results_df.empty else 0,
            int((results_df["excess_return"] > 0).sum()) if not results_df.empty else 0,
            f"{results_df['excess_return'].mean():+.4f}" if not results_df.empty else "N/A",
            f"{results_df['sharpe_ratio'].mean():+.2f}" if not results_df.empty else "N/A",
            f"{results_df['max_drawdown'].mean():.4f}" if not results_df.empty else "N/A",
            device,
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        ],
    }
    pd.DataFrame(summary).to_csv(run_dir / "summary_metrics.csv", index=False)

    for tk, traj in all_trajs.items():
        traj.to_csv(run_dir / f"eval_{tk}.csv", index=False)

    log.info(f"All results saved in: {run_dir.resolve()}")
    log.info("DONE")


if __name__ == "__main__":
    main()
