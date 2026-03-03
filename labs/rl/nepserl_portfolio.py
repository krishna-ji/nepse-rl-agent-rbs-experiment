#!/usr/bin/env python3
"""
NEPSE RL Portfolio Engine v12 — Action Lock + Macro Veto + Relative Momentum
======================================================================
Single-file PPO engine trading 10 randomly-sampled NEPSE assets per
episode, with a strict temporal firewall between training and evaluation.

v12 innovations (over v11):
  1. T+3 ACTION LOCK: Once capital is allocated to an asset, the weight
     is locked for LOCK_DAYS=3 trading days. Agent cannot sell/reduce.
     Kills high-frequency whipsawing; forces swing-trade conviction.
  2. MACRO VETO: Extra action dim (slot 12) acts as binary regime gate.
     If veto fires (action[11] < 0), entire portfolio forced to 100% Cash.
     Agent learns to sit out bear/choppy markets entirely.
  3. RELATIVE MOMENTUM REWARD: Agent is rewarded purely on its distance
     from the Risk Parity benchmark. If RP is down -2% and agent is
     down -0.5%, the reward is highly positive. No absolute return in reward.

Retained from v11:
  - WALK-FORWARD EXPANDING WINDOW: 7 chronological folds (2019-2025)
  - TRANSFER LEARNING: Neural network weights carry forward across folds
  - RISK PARITY BENCHMARK: Inverse Volatility Weighting
  - REALISTIC FRICTION: TAU=0.0045
  - TEMPORAL FIREWALL, DYNAMIC UNIVERSE, TICKER-AGNOSTIC

Architecture:
  - State: (10×11 features) + (11 weights) + (11 lock timers) = 132 dims
  - Action: Box(-3, 3, shape=(12,)) → [0:11] Softmax → weights, [11] macro veto
  - Reward: ALPHA_SCALE × (port_ret - rp_ret) - friction

Hardware: 16 CPU SubprocVecEnv, RTX 4060, 128 GB RAM
Data: 325 NEPSE tickers, 6833 daily bars (1997-2026)
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

NUM_ENVS        = 16          # SubprocVecEnv workers
EPISODE_LENGTH  = 252         # 1 trading year
MIN_ROWS        = 250         # minimum rows to load a ticker
WARMUP          = 200         # feature warmup rows
SEED            = 42
N_ASSETS        = 10          # portfolio size (slots per episode)

# Walk-Forward Expanding Window fold dates
FOLD_DATES = ["2019-01-01", "2020-01-01", "2021-01-01",
              "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]
FOLD_1_TIMESTEPS = 5_000_000    # initial long training (first fold)
FOLD_N_TIMESTEPS = 1_000_000    # adaptation per subsequent fold

# Fixed 10-asset evaluation universe (locked during OOS eval)
EVAL_UNIVERSE = ["NABIL", "NICA", "SHIVM", "CHDC", "NLIC",
                 "UPPER", "NRIC", "SBL", "GBIME", "SANIMA"]

# ============================================================================
# SETUP
# ============================================================================

def setup():
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = PROJECT_ROOT / f"runs/{RUN_TS}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR = PROJECT_ROOT / "data/ohlcv/1D/stocks"
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

    log = logging.getLogger("nepserl")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(RUN_DIR / "nepserl_portfolio.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); log.addHandler(fh)

    log.info("NEPSE RL Portfolio Engine v12 — Action Lock + Macro Veto + Relative Momentum")
    log.info(f"Run dir    : {RUN_DIR.resolve()}")
    log.info(f"Device     : {DEVICE}")
    log.info(f"Fold dates : {FOLD_DATES}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        log.info(f"GPU        : {p.name}  ({p.multi_processor_count * 128} CUDA cores, "
                 f"{p.total_memory / 1024**3:.1f} GB VRAM)")
    log.info(f"CPU cores  : {multiprocessing.cpu_count()}")
    log.info(f"Eval universe: {EVAL_UNIVERSE}")
    return log, RUN_DIR, DATA_DIR, DEVICE

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(DATA_DIR, log):
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

    # Forward-fill OHLCV to eliminate NaN gaps (e.g. GBIME has 59% NaN after
    # reindex to master dates). If a stock didn't trade, last known price persists.
    # Only fill within each ticker's actual date range (don't fill before IPO).
    for tk in frames:
        first_valid = frames[tk].index[0]
        last_valid  = frames[tk].index[-1]
        mask = (master_df.index >= first_valid) & (master_df.index <= last_valid)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            key = (tk, col)
            if key in master_df.columns:
                master_df.loc[mask, key] = master_df.loc[mask, key].ffill()
    ffill_nans = master_df.isna().sum().sum()
    log.info(f"Master DataFrame: {master_df.shape} (remaining NaN after ffill: {ffill_nans:,})")

    valid_start_dates = {}
    for tk in master_df.columns.get_level_values("Ticker").unique():
        tdays = master_df[tk].dropna(how="all").index
        valid_start_dates[tk] = tdays[WARMUP] if len(tdays) > WARMUP else tdays[-1]

    tickers = sorted(valid_start_dates.keys())
    log.info(f"{len(tickers)} tickers with valid start dates")
    return master_df, valid_start_dates, tickers

# ============================================================================
# FEATURE ENGINEERING
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
def _bb_pctb(c, n=20, ns=2.0):
    s_ = _sma(c, n); std = c.rolling(n, min_periods=n).std()
    upper = s_ + ns*std; lower = s_ - ns*std
    return ((c - lower) / (upper - lower + 1e-10) * 2.0 - 1.0).clip(-3.0, 3.0)
def _psl(l, w=60): return l.rolling(w, min_periods=w).min()

def compute_features(master_df, valid_start_dates, log):
    all_tkrs = master_df.columns.get_level_values("Ticker").unique()
    log.info(f"Computing features for {len(all_tkrs)} tickers...")
    pieces = {}
    for tk in all_tkrs:
        raw = master_df[tk].dropna(how="all")
        o, h, l, c, v = raw["Open"], raw["High"], raw["Low"], raw["Close"], raw["Volume"]
        pieces[(tk,"open")] = o; pieces[(tk,"high")] = h; pieces[(tk,"low")] = l
        pieces[(tk,"close")] = c; pieces[(tk,"volume")] = v
        atr14 = _atr(h, l, c, 14)
        pieces[(tk,"atr14")] = atr14
        psl_ = _psl(l, 60)
        pieces[(tk,"protected_swing_low")] = psl_
        sma10, sma20, sma50 = _sma(c, 10), _sma(c, 20), _sma(c, 50)
        sma100, sma200 = _sma(c, 100), _sma(c, 200)
        pieces[(tk,"clv")] = ((c - l) - (h - c)) / (h - l + 1e-10)
        pieces[(tk,"lower_wick")] = ((c - l) / (atr14 + 1e-10)).clip(0.0, 3.0)
        pk, pd_ = _stochastic(h, l, c)
        pieces[(tk,"pct_k")] = (pk / 50.0) - 1.0
        pieces[(tk,"pct_d")] = (pd_ / 50.0) - 1.0
        pieces[(tk,"natr")] = atr14 / (c + 1e-10)
        pieces[(tk,"bb_pctb")] = _bb_pctb(c, 20, 2.0)
        mf_mult = ((c - l) - (h - c)) / (h - l + 1e-10)
        mf_vol  = mf_mult * v
        cmf_20  = mf_vol.rolling(20, min_periods=20).sum() / (v.rolling(20, min_periods=20).sum() + 1e-10)
        pieces[(tk,"cmf")] = cmf_20.clip(-1.0, 1.0)
        pieces[(tk,"d_low")] = ((c - psl_) / (atr14 + 1e-10)).clip(-3.0, 3.0)
        rolling_high_20 = h.rolling(20, min_periods=20).max()
        pieces[(tk,"dd_state")] = ((rolling_high_20 - c) / (atr14 + 1e-10)).clip(0.0, 5.0)
        bull_count = ((sma10 > sma20).astype(float) + (sma20 > sma50).astype(float) +
                      (sma50 > sma100).astype(float) + (sma100 > sma200).astype(float))
        pieces[(tk,"ribbon_align")] = bull_count / 4.0 * 2.0 - 1.0
        pieces[(tk,"ribbon_disp")] = ((sma10 - sma200) / (sma200 + 1e-10) * 10.0).clip(-3.0, 3.0)

        # 20-day rolling std of log returns (for Risk Parity weighting)
        log_ret = np.log(c / c.shift(1))
        pieces[(tk, "rstd20")] = log_ret.rolling(20, min_periods=20).std()

    feat_df = pd.DataFrame(pieces)
    feat_df.columns = pd.MultiIndex.from_tuples(feat_df.columns, names=["Ticker","Feature"])
    feat_df = feat_df.sort_index()

    for tk in all_tkrs:
        key = (tk, "natr")
        if key not in feat_df.columns: continue
        clean = feat_df[key].dropna()
        rm = clean.rolling(100, min_periods=100).mean()
        rs = clean.rolling(100, min_periods=100).std()
        feat_df[key] = ((clean - rm) / (rs + 1e-8)).clip(-3.0, 3.0)

    log.info(f"Feature matrix: {feat_df.shape}")
    return feat_df

# ============================================================================
# PRE-COMPILE NUMPY ARRAYS
# ============================================================================

OBS_FEATURES = ["clv", "lower_wick", "pct_k", "pct_d", "natr", "bb_pctb", "cmf",
                "d_low", "dd_state", "ribbon_align", "ribbon_disp"]
N_FEATURES   = len(OBS_FEATURES)      # 11 per asset
NEEDED_COLS  = OBS_FEATURES + ["close", "high", "low", "atr14", "protected_swing_low", "rstd20"]

def precompile_arrays(feat_df, tickers, valid_start_dates, log):
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
# PORTFOLIO ENVIRONMENT — T+3 Lock + Macro Veto + Relative Momentum
# ============================================================================
#
# State:  [asset_1_features(11)×10, w_cash, w_1..w_10, lock_cash, lock_1..lock_10]
#         = 110 + 11 + 11 = 132 dims
# Action: Box(-3, 3, shape=(12,)) → [0:11] Softmax → weights, [11] = macro veto
# Reward: ALPHA × (port_ret - rp_ret) - friction   (pure relative momentum)

class NepsePortfolioEnv(gym.Env):
    TAU         = 0.0045   # 0.45% per unit of turnover (NEPSE broker+SEBON+slippage)
    ALPHA_SCALE = 30.0     # Alpha Gradient: strong conviction signal over RP benchmark
    TEMPERATURE = 2.0      # Softmax temperature: sharper conviction
    DEADBAND    = 0.01     # Ignore weight shifts < 1% (silence exploration noise)
    LOCK_DAYS   = 3        # Minimum holding period (T+3 lock)

    def __init__(self, ticker_arrays, valid_start_idx, dates_array,
                 all_tickers, n_assets=10, episode_length=252,
                 split_index=None, mode="train", eval_universe=None,
                 seed=None):
        super().__init__()
        self._ta    = ticker_arrays
        self._vsi   = valid_start_idx       # {ticker: first_valid_idx}
        self._dates = dates_array
        self._n     = len(dates_array)
        self._all_tickers = list(all_tickers)
        self._n_assets = n_assets
        self._ep_len  = episode_length
        self._mode    = mode                # "train" or "eval"

        # Temporal firewall
        if split_index is not None:
            self._split = split_index
        else:
            self._split = self._n           # no split → use everything

        # For eval mode: fixed universe
        self._eval_universe = eval_universe

        # Precompute min training start: earliest index with ≥ n_assets valid
        if mode == "train":
            counts = np.zeros(self._n, dtype=np.int32)
            for si in self._vsi.values():
                if si < self._n:
                    counts[si:] += 1
            valid_idxs = np.where(counts >= n_assets)[0]
            self._min_train_start = int(valid_idxs[0]) if len(valid_idxs) > 0 else 0
        else:
            self._min_train_start = 0

        # obs = features(110) + weights(11) + lock_timers(11) = 132
        obs_dim = n_assets * N_FEATURES + (n_assets + 1) + (n_assets + 1)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(obs_dim,), dtype=np.float32)
        # action = portfolio_logits(11) + macro_veto(1) = 12
        self.action_space = spaces.Box(-3.0, 3.0, shape=(n_assets + 2,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._current_universe = []
        self._weights = np.zeros(n_assets + 1, dtype=np.float32)
        self._lock_timer = np.zeros(n_assets + 1, dtype=np.int32)  # T+3 countdown
        self._si = 0; self._t = 0; self._pv = 1.0
        self._macro_veto_active = False

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._mode == "eval":
            # ── OOS Evaluation: locked universe, start at split ────────────
            self._current_universe = list(self._eval_universe)
            self._si = self._split
        else:
            # ── Training: Date First, Assets Second ────────────────────────
            # 1. Pick a random start date strictly BEFORE the split
            max_start = self._split - self._ep_len
            if max_start <= self._min_train_start:
                max_start = self._min_train_start + 1
            self._si = int(self._rng.integers(
                self._min_train_start, max_start))

            # 2. Build valid pool: tickers alive at this start date
            valid_pool = [tk for tk in self._all_tickers
                          if self._vsi.get(tk, 99999) <= self._si]

            # 3. Sample N_ASSETS random tickers from pool
            if len(valid_pool) < self._n_assets:
                # Extremely rare fallback: shift to later date
                self._si = self._split - self._ep_len - 1
                valid_pool = [tk for tk in self._all_tickers
                              if self._vsi.get(tk, 99999) <= self._si]
            pool_arr = np.array(valid_pool)
            chosen = self._rng.choice(
                pool_arr, size=self._n_assets, replace=False)
            self._current_universe = chosen.tolist()

        self._t = 0; self._pv = 1.0
        self._weights = np.zeros(self._n_assets + 1, dtype=np.float32)
        self._weights[0] = 1.0  # start 100% cash
        self._lock_timer = np.zeros(self._n_assets + 1, dtype=np.int32)
        self._macro_veto_active = False
        return self._obs(), self._info()

    def step(self, action):
        i = self._si + self._t
        if i >= self._n - 1:
            return self._obs(), 0.0, True, True, self._info()

        # ── Decrement lock timers ──────────────────────────────────────────
        self._lock_timer = np.maximum(self._lock_timer - 1, 0)

        # ── Macro Veto: action[11] < 0 → force 100% Cash ──────────────────
        veto_signal = float(action[self._n_assets + 1])  # slot 12 (idx 11)
        self._macro_veto_active = (veto_signal < 0.0)

        if self._macro_veto_active:
            # Force liquidation to 100% cash
            target_weights = np.zeros(self._n_assets + 1, dtype=np.float32)
            target_weights[0] = 1.0
        else:
            # ── Softmax Temperature on portfolio logits [0:11] ─────────────
            portfolio_logits = action[:self._n_assets + 1]
            sharpened = portfolio_logits.astype(np.float64) * self.TEMPERATURE
            target_weights = self._softmax(sharpened).astype(np.float32)

        # ── T+3 Action Lock: prevent reducing locked positions ─────────────
        for j in range(self._n_assets + 1):
            if self._lock_timer[j] > 0:
                # Position is locked — clamp target to at least current weight
                if target_weights[j] < self._weights[j]:
                    target_weights[j] = self._weights[j]
        # Re-normalize after lock enforcement
        target_weights = target_weights / (target_weights.sum() + 1e-10)

        # ── Deadband Filter: silence exploration noise micro-churning ──────
        weight_delta = target_weights - self._weights
        weight_delta = np.where(np.abs(weight_delta) < self.DEADBAND, 0.0, weight_delta)
        executed_weights = self._weights + weight_delta
        executed_weights = executed_weights / (executed_weights.sum() + 1e-10)

        # ── Start new locks on positions that increased ────────────────────
        for j in range(self._n_assets + 1):
            if executed_weights[j] > self._weights[j] + self.DEADBAND:
                self._lock_timer[j] = self.LOCK_DAYS

        # ── Turnover: computed on actually-executed weight changes ──────────
        turnover = float(np.sum(np.abs(executed_weights - self._weights)))
        friction = self.TAU * turnover

        # ── Daily log returns per asset ────────────────────────────────────
        returns = np.zeros(self._n_assets, dtype=np.float64)
        for j, tk in enumerate(self._current_universe):
            c_arr = self._ta[tk]["close"]
            c_t   = float(c_arr[i + 1])
            c_prev = float(c_arr[i])
            if np.isnan(c_t) or np.isnan(c_prev) or c_prev < 1e-6:
                returns[j] = 0.0
            else:
                returns[j] = np.log(c_t / c_prev)

        # ── Portfolio return (weighted sum, cash earns 0) ──────────────────
        port_return = 0.0
        for j in range(self._n_assets):
            port_return += self._weights[j + 1] * returns[j]

        # ── Risk Parity Benchmark (Inverse Volatility Weighting) ──────────
        vols = np.empty(self._n_assets, dtype=np.float64)
        for j, tk in enumerate(self._current_universe):
            v = float(self._ta[tk]["rstd20"][i])
            vols[j] = v if (not np.isnan(v) and v > 1e-8) else 1e-4
        inv_vols = 1.0 / vols
        rp_weights = inv_vols / inv_vols.sum()
        rp_return = float(np.dot(rp_weights, returns))
        active_return = port_return - rp_return

        # ── Reward: Pure Relative Momentum ─────────────────────────────────
        #  Agent is rewarded ONLY on its distance from the RP benchmark.
        #  If RP is down -2% and agent is down -0.5%, reward is positive.
        #  No absolute return component — purely relative alpha.
        reward = self.ALPHA_SCALE * active_return - friction

        # ── Update portfolio value (actual PnL, no alpha bonus) ────────────
        self._pv *= np.exp(port_return - friction)
        self._weights = executed_weights
        self._t += 1

        done  = self._t >= self._ep_len
        trunc = (self._si + self._t) >= self._n - 1
        if np.isnan(reward) or np.isinf(reward): reward = 0.0
        return self._obs(), float(reward), done, trunc, self._info()

    def _obs(self):
        i = min(self._si + self._t, self._n - 1)
        obs_dim = self._n_assets * N_FEATURES + (self._n_assets + 1) + (self._n_assets + 1)
        obs = np.zeros(obs_dim, dtype=np.float32)

        # Fill 11 features per asset
        for j, tk in enumerate(self._current_universe):
            c = self._ta[tk]
            off = j * N_FEATURES
            for k, f in enumerate(OBS_FEATURES):
                v = c[f][i]
                obs[off + k] = 0.0 if np.isnan(v) else float(v)

        # Append current weights
        off = self._n_assets * N_FEATURES
        obs[off:off + self._n_assets + 1] = self._weights

        # Append lock timers (normalized: 0=unlocked, 1=fully locked)
        off2 = off + self._n_assets + 1
        obs[off2:off2 + self._n_assets + 1] = (
            self._lock_timer.astype(np.float32) / max(self.LOCK_DAYS, 1))

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _info(self):
        i = min(self._si + self._t, self._n - 1)
        info = {
            "date": self._dates[i],
            "portfolio_value": float(self._pv),
            "cash_weight": float(self._weights[0]),
            "macro_veto": int(self._macro_veto_active),
            "locked_slots": int(np.sum(self._lock_timer > 0)),
        }
        for j, tk in enumerate(self._current_universe):
            info[f"w_{tk}"] = float(self._weights[j + 1])
            c_val = float(self._ta[tk]["close"][i])
            info[f"close_{tk}"] = c_val if not np.isnan(c_val) else 0.0
        return info

# ============================================================================
# TRAINING CALLBACK
# ============================================================================

class PortfolioTracker(BaseCallback):
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
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func

# ============================================================================
# OOS EVALUATION — locked universe, starts at SPLIT_INDEX
# ============================================================================

def evaluate_portfolio(model, ticker_arrays, valid_start_idx, dates_array,
                       all_tickers, split_index, eval_universe,
                       n_assets=10, episode_length=252):
    """Deterministic evaluation on unseen data (post-split)."""
    env = NepsePortfolioEnv(
        ticker_arrays, valid_start_idx, dates_array,
        all_tickers=all_tickers, n_assets=n_assets,
        episode_length=episode_length,
        split_index=split_index, mode="eval",
        eval_universe=eval_universe, seed=123)
    obs, info = env.reset()

    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        records.append(info.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    records.append(info.copy())
    return pd.DataFrame(records)

# ============================================================================
# RISK PARITY BENCHMARK
# ============================================================================

def compute_risk_parity_bh(traj, eval_universe, ticker_arrays, dates_array,
                           split_index):
    """Compute Risk Parity (inverse volatility) benchmark equity curve."""
    n = len(traj)
    rp_pv = np.ones(n)
    for t in range(1, n):
        day_idx = split_index + t
        if day_idx >= len(dates_array):
            break
        # Inverse-vol weights from previous day's 20-day return std
        vols = np.empty(len(eval_universe), dtype=np.float64)
        for j, tk in enumerate(eval_universe):
            v = float(ticker_arrays[tk]["rstd20"][day_idx - 1])
            vols[j] = v if (not np.isnan(v) and v > 1e-8) else 1e-4
        inv_vols = 1.0 / vols
        rp_w = inv_vols / inv_vols.sum()
        # Daily return weighted by inv-vol
        daily_ret = 0.0
        for j, tk in enumerate(eval_universe):
            c_now = float(ticker_arrays[tk]["close"][day_idx])
            c_prev = float(ticker_arrays[tk]["close"][day_idx - 1])
            if not np.isnan(c_now) and not np.isnan(c_prev) and c_prev > 1e-6:
                daily_ret += rp_w[j] * (c_now / c_prev - 1.0)
        rp_pv[t] = rp_pv[t - 1] * (1.0 + daily_ret)
    return rp_pv

# ============================================================================
# PLOTS
# ============================================================================

def plot_dashboard(tracker, run_dir, total_ts, num_envs, split_date):
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

    plt.suptitle(f"NEPSE Portfolio v12 — {total_ts/1e6:.0f}M steps, "
                 f"{num_envs} envs, {split_date}", fontsize=14)
    plt.tight_layout()
    plt.savefig(run_dir / "training_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_portfolio_equity(traj, universe, run_dir, split_date,
                          rp_pv=None, fold_num=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Top: Portfolio equity vs Risk Parity & EW benchmarks
    ax = axes[0]
    pv = traj["portfolio_value"].values
    ax.plot(pv, label=f"Agent ({pv[-1]-1:+.1%})", color="dodgerblue", lw=2)

    if rp_pv is not None:
        ax.plot(rp_pv, label=f"RP B&H ({rp_pv[-1]-1:+.1%})", color="darkorange", lw=2)

    # EW B&H for reference
    n = len(traj)
    bh_pv = np.ones(n)
    for tk in universe:
        col = f"close_{tk}"
        if col in traj.columns:
            cl = traj[col].values
            if cl[0] > 0 and not np.isnan(cl[0]):
                bh_pv += (cl / cl[0] - 1.0) / len(universe)
    ax.plot(bh_pv, label=f"EW B&H ({bh_pv[-1]-1:+.1%})", color="gray", lw=1, ls="--", alpha=0.5)
    ax.axhline(1.0, color="black", ls="--", alpha=0.3)
    ax.set_ylabel("Portfolio Value")
    fold_str = f" Fold {fold_num}" if fold_num else ""
    ax.set_title(f"OOS{fold_str} (post {split_date}): Agent vs Risk Parity B&H")
    ax.legend(); ax.grid(alpha=0.3)

    # Bottom: Weight allocation over time (stacked area)
    ax = axes[1]
    weight_cols = ["cash_weight"] + [f"w_{tk}" for tk in universe]
    labels = ["Cash"] + list(universe)
    wdata = np.zeros((n, len(weight_cols)))
    for j, col in enumerate(weight_cols):
        if col in traj.columns:
            wdata[:, j] = traj[col].values
    ax.stackplot(range(n), wdata.T, labels=labels, alpha=0.8)
    ax.set_ylabel("Weight"); ax.set_xlabel("Day")
    ax.set_title("Portfolio Weight Allocation Over Time")
    ax.legend(loc="upper left", fontsize=7, ncol=4); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    suffix = f"_fold{fold_num}" if fold_num else ""
    plt.savefig(run_dir / f"portfolio_equity{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ============================================================================
# MAIN — Walk-Forward Expanding Window with Transfer Learning
# ============================================================================

def main():
    log, run_dir, data_dir, device = setup()

    # ── Data ────────────────────────────────────────────────────────────────
    master_df, valid_start_dates, tickers = load_ohlcv(data_dir, log)
    feat_df = compute_features(master_df, valid_start_dates, log)
    ticker_arrays, valid_start_idx, dates_array, n_dates = precompile_arrays(
        feat_df, tickers, valid_start_dates, log)

    # ── Compute fold indices from FOLD_DATES ────────────────────────────────
    fold_indices = []
    for d in FOLD_DATES:
        dt = np.datetime64(d)
        idx = int(np.searchsorted(dates_array, dt))
        fold_indices.append(idx)
    fold_indices.append(n_dates - 1)  # end-of-data sentinel
    n_folds = len(FOLD_DATES)

    log.info(f"Walk-Forward: {n_folds} folds")
    for fi in range(n_folds):
        s_date = pd.Timestamp(dates_array[fold_indices[fi]]).strftime("%Y-%m-%d")
        e_date = pd.Timestamp(dates_array[fold_indices[fi + 1]]).strftime("%Y-%m-%d")
        e_days = fold_indices[fi + 1] - fold_indices[fi]
        log.info(f"  Fold {fi+1}: train < {s_date}, "
                 f"eval [{s_date} → {e_date}] = {e_days} days")

    # ── Verify all eval tickers valid before earliest fold ──────────────────
    earliest_split = fold_indices[0]
    for tk in EVAL_UNIVERSE:
        si = valid_start_idx.get(tk, 99999)
        if si >= earliest_split:
            log.warning(f"  WARNING {tk} valid_start={si} >= "
                        f"earliest split={earliest_split}!")
        else:
            log.info(f"  OK {tk} valid_start={si} "
                     f"({pd.Timestamp(dates_array[si]).strftime('%Y-%m-%d')})")

    obs_dim = N_ASSETS * N_FEATURES + (N_ASSETS + 1) + (N_ASSETS + 1)  # 132
    act_dim = N_ASSETS + 2  # 11 portfolio + 1 macro veto = 12
    log.info(f"Obs dim: {obs_dim} (feats+weights+locks), Act dim: {act_dim} (weights+veto)")

    # ── Env factory (parameterized by split_index) ──────────────────────────
    def make_env(rank, split_index, seed=SEED):
        def _init():
            return Monitor(NepsePortfolioEnv(
                ticker_arrays, valid_start_idx, dates_array,
                all_tickers=tickers,
                n_assets=N_ASSETS,
                episode_length=EPISODE_LENGTH,
                split_index=split_index,
                mode="train",
                seed=seed + rank))
        return _init

    # ── Initialize PPO model on first fold's env ────────────────────────────
    first_split = fold_indices[0]
    log.info(f"Spawning {NUM_ENVS} SubprocVecEnv workers "
             f"(fold 1, split={first_split})...")
    vec_env = SubprocVecEnv(
        [make_env(i, first_split) for i in range(NUM_ENVS)],
        start_method="spawn")

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=linear_schedule(2e-4),
        n_steps=4096,
        batch_size=4096,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        device=device,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
            log_std_init=-1.0,
        ),
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    log.info(f"PPO on {model.device} — {n_params:,} params")

    # ── Walk-Forward Training + Evaluation Loop ─────────────────────────────
    tracker = PortfolioTracker(run_dir)
    fold_results = []
    t0_total = time.time()

    for fold_i in range(n_folds):
        split_index = fold_indices[fold_i]
        eval_end_idx = fold_indices[fold_i + 1]
        eval_days = eval_end_idx - split_index
        split_actual = pd.Timestamp(
            dates_array[split_index]).strftime("%Y-%m-%d")
        eval_end_actual = pd.Timestamp(
            dates_array[eval_end_idx]).strftime("%Y-%m-%d")

        timesteps = FOLD_1_TIMESTEPS if fold_i == 0 else FOLD_N_TIMESTEPS

        log.info("=" * 70)
        log.info(f"FOLD {fold_i+1}/{n_folds}: train < {split_actual}, "
                 f"eval [{split_actual} → {eval_end_actual}], "
                 f"{timesteps:,} steps")
        log.info("=" * 70)

        # Recreate envs with updated split_index (except fold 1)
        if fold_i > 0:
            vec_env.close()
            log.info(f"Spawning {NUM_ENVS} workers "
                     f"(fold {fold_i+1}, split={split_index})...")
            vec_env = SubprocVecEnv(
                [make_env(i, split_index) for i in range(NUM_ENVS)],
                start_method="spawn")
            model.set_env(vec_env)

        # ── Train (transfer learning: weights carry forward) ────────────
        t0 = time.time()
        model.learn(total_timesteps=timesteps,
                    callback=[tracker], progress_bar=True,
                    reset_num_timesteps=(fold_i == 0))
        elapsed = time.time() - t0
        fps = timesteps / elapsed if elapsed > 0 else 0
        log.info(f"Fold {fold_i+1} training: {elapsed/60:.1f} min, "
                 f"{fps:,.0f} FPS")

        if tracker.ep_rewards:
            recent = tracker.ep_rewards[-100:]
            log.info(f"  Reward (last 100 eps): "
                     f"{np.mean(recent):+.4f}±{np.std(recent):.3f}")

        model.save(run_dir / f"model_fold_{fold_i+1}.zip")

        # ── OOS Evaluation on this fold's window ────────────────────────
        eval_length = min(eval_days - 1, n_dates - split_index - 1)
        log.info(f"Evaluating fold {fold_i+1}: {EVAL_UNIVERSE}, "
                 f"{eval_length} days post {split_actual}")

        traj = evaluate_portfolio(
            model, ticker_arrays, valid_start_idx, dates_array,
            all_tickers=tickers, split_index=split_index,
            eval_universe=EVAL_UNIVERSE,
            n_assets=N_ASSETS, episode_length=eval_length)
        traj.to_csv(run_dir / f"eval_fold_{fold_i+1}.csv", index=False)

        # Agent metrics
        pv = traj["portfolio_value"].values
        agent_ret = pv[-1] / pv[0] - 1.0
        pv_rets = np.diff(pv) / (pv[:-1] + 1e-10)
        sharpe = (np.mean(pv_rets) / (np.std(pv_rets) + 1e-10)
                  * np.sqrt(252))
        cummax = np.maximum.accumulate(pv)
        max_dd = np.max((cummax - pv) / (cummax + 1e-10))

        # Risk Parity benchmark
        rp_pv = compute_risk_parity_bh(
            traj, EVAL_UNIVERSE, ticker_arrays, dates_array, split_index)
        rp_ret = rp_pv[-1] / rp_pv[0] - 1.0

        # EW B&H for reference
        n_traj = len(traj)
        ew_pv = np.ones(n_traj)
        for tk in EVAL_UNIVERSE:
            col = f"close_{tk}"
            if col in traj.columns:
                cl = traj[col].values
                if cl[0] > 0 and not np.isnan(cl[0]):
                    ew_pv += (cl / cl[0] - 1.0) / len(EVAL_UNIVERSE)
        ew_ret = ew_pv[-1] / ew_pv[0] - 1.0

        fold_results.append({
            "fold": fold_i + 1,
            "split_date": split_actual,
            "eval_end": eval_end_actual,
            "eval_days": len(traj),
            "agent_return": agent_ret,
            "rp_bh_return": rp_ret,
            "ew_bh_return": ew_ret,
            "excess_vs_rp": agent_ret - rp_ret,
            "excess_vs_ew": agent_ret - ew_ret,
            "sharpe": sharpe,
            "max_dd": max_dd,
        })

        log.info(f"  Agent Return     : {agent_ret:+.2%}")
        log.info(f"  RP B&H Return    : {rp_ret:+.2%}")
        log.info(f"  EW B&H Return    : {ew_ret:+.2%}")
        log.info(f"  Excess (vs RP)   : {agent_ret - rp_ret:+.2%}")
        log.info(f"  Sharpe           : {sharpe:+.2f}")
        log.info(f"  Max Drawdown     : {max_dd:.2%}")

        # Per-fold equity plot
        plot_portfolio_equity(traj, EVAL_UNIVERSE, run_dir, split_actual,
                              rp_pv=rp_pv, fold_num=fold_i + 1)

    vec_env.close()
    tracker._export()
    total_elapsed = time.time() - t0_total

    # ── Export fold matrix ──────────────────────────────────────────────────
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(run_dir / "fold_matrix.csv", index=False)

    log.info("=" * 70)
    log.info("WALK-FORWARD FOLD MATRIX")
    log.info("=" * 70)
    log.info(f"{'Fold':>4} | {'Split Date':>11} | {'Eval End':>11} | "
             f"{'Agent':>8} | {'RP B&H':>8} | {'Excess':>8} | "
             f"{'Sharpe':>7} | {'MaxDD':>7}")
    log.info("-" * 85)
    for r in fold_results:
        log.info(
            f"{r['fold']:4d} | {r['split_date']:>11s} | "
            f"{r['eval_end']:>11s} | "
            f"{r['agent_return']:+7.2%} | {r['rp_bh_return']:+7.2%} | "
            f"{r['excess_vs_rp']:+7.2%} | {r['sharpe']:+6.2f} | "
            f"{r['max_dd']:6.2%}")
    log.info("-" * 85)

    # Aggregates
    avg_excess = np.mean([r["excess_vs_rp"] for r in fold_results])
    avg_sharpe = np.mean([r["sharpe"] for r in fold_results])
    wins = sum(1 for r in fold_results if r["excess_vs_rp"] > 0)
    log.info(f"Avg Excess vs RP: {avg_excess:+.2%} | "
             f"Avg Sharpe: {avg_sharpe:+.2f} | "
             f"Win Rate: {wins}/{n_folds}")
    log.info(f"Total wall time: {total_elapsed/60:.1f} min")

    # ── Training dashboard ──────────────────────────────────────────────────
    total_ts = FOLD_1_TIMESTEPS + FOLD_N_TIMESTEPS * (n_folds - 1)
    plot_dashboard(tracker, run_dir, total_ts, NUM_ENVS,
                   f"{n_folds} folds")
    log.info("Training dashboard saved")

    # ── Summary CSV ─────────────────────────────────────────────────────────
    summary = {
        "metric": [
            "version", "n_folds", "fold_dates",
            "fold_1_timesteps", "fold_n_timesteps", "total_timesteps",
            "eval_universe", "n_assets", "obs_dim", "act_dim",
            "network", "tau", "alpha_scale",
            "avg_agent_return", "avg_rp_bh_return", "avg_excess_vs_rp",
            "avg_ew_bh_return", "avg_excess_vs_ew",
            "avg_sharpe", "win_rate_vs_rp",
            "total_episodes", "wall_time_min",
            "device", "gpu",
        ],
        "value": [
            "v12_actionlock_macroveto_relmomentum",
            n_folds, ",".join(FOLD_DATES),
            FOLD_1_TIMESTEPS, FOLD_N_TIMESTEPS, total_ts,
            ",".join(EVAL_UNIVERSE), N_ASSETS, obs_dim, act_dim,
            "[512,512,256]x2", NepsePortfolioEnv.TAU,
            NepsePortfolioEnv.ALPHA_SCALE,
            f"{np.mean([r['agent_return'] for r in fold_results]):+.4f}",
            f"{np.mean([r['rp_bh_return'] for r in fold_results]):+.4f}",
            f"{avg_excess:+.4f}",
            f"{np.mean([r['ew_bh_return'] for r in fold_results]):+.4f}",
            f"{np.mean([r['excess_vs_ew'] for r in fold_results]):+.4f}",
            f"{avg_sharpe:+.2f}",
            f"{wins}/{n_folds}",
            len(tracker.ep_rewards),
            f"{total_elapsed/60:.1f}",
            device,
            (torch.cuda.get_device_name(0)
             if torch.cuda.is_available() else "N/A"),
        ],
    }
    pd.DataFrame(summary).to_csv(
        run_dir / "summary_metrics.csv", index=False)
    log.info(f"All results saved in: {run_dir.resolve()}")
    log.info("DONE")


if __name__ == "__main__":
    main()
