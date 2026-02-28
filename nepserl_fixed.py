#!/usr/bin/env python3
"""
NEPSE RL Stochastic Pullback Engine - FIXED VERSION
==================================================
Consolidated single-file version with critical MDP fixes:
1. Removed FORCED_EXIT_PENALTY (reward topology fix)
2. Standardized feature scaling (gradient stability fix) 
3. Fixed PPO hyperparameters (convergence fix)
4. Widened ATR multiplier for NEPSE volatility (3.5 vs 2.5)
"""

import warnings; warnings.filterwarnings("ignore")
import logging, pathlib, datetime, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

def setup_logging_and_directories():
    """Setup timestamped run directory and logging"""
    RUN_TS = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = pathlib.Path(f"runs/{RUN_TS}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Logger setup
    log = logging.getLogger("nepserl")
    log.setLevel(logging.INFO)  # Set to INFO for cleaner output
    log.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    
    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    
    # File handler
    LOG_FILE = RUN_DIR / "nepserl_fixed.log"
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    
    DATA_DIR = pathlib.Path("data/stocks")
    
    log.info(f"🚀 NEPSE RL FIXED VERSION STARTING")
    log.info(f"Run dir : {RUN_DIR.resolve()}")
    log.info(f"Data dir: {DATA_DIR.resolve()}")
    
    return log, RUN_DIR, DATA_DIR

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv_data(DATA_DIR, log, MIN_ROWS=250, WARMUP=200):
    """Load all OHLCV CSVs and create MultiIndex DataFrame"""
    log.info("📈 Loading OHLCV data...")
    
    frames, skipped = {}, 0
    for csv in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv, parse_dates=["Timestamp"])
            if df.empty or len(df) < MIN_ROWS:
                skipped += 1
                continue
            
            df = df.rename(columns={"Timestamp": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            
            if not {"Open","High","Low","Close","Volume"}.issubset(df.columns):
                skipped += 1
                continue
                
            frames[csv.stem] = df[["Open","High","Low","Close","Volume"]]
            
        except Exception as e:
            log.warning(f"Skipping {csv.stem}: {e}")
            skipped += 1
    
    log.info(f"Loaded {len(frames)} tickers, skipped {skipped} (< {MIN_ROWS} rows)")
    
    # Build MultiIndex DataFrame
    all_dates = sorted(set().union(*(f.index for f in frames.values())))
    idx = pd.DatetimeIndex(all_dates, name="Date")
    parts = {(tk, col): s.reindex(idx) for tk, df in frames.items() for col, s in df.items()}
    master_df = pd.DataFrame(parts)
    master_df.columns = pd.MultiIndex.from_tuples(master_df.columns, names=["Ticker","Feature"])
    
    log.info(f"Master DataFrame: {master_df.shape}, date range {master_df.index.min().date()} → {master_df.index.max().date()}")
    
    # Calculate valid start dates
    valid_start_dates = {}
    for tk in master_df.columns.get_level_values("Ticker").unique():
        trading_days = master_df[tk].dropna(how="all").index
        if len(trading_days) > WARMUP:
            valid_start_dates[tk] = trading_days[WARMUP]
        else:
            valid_start_dates[tk] = trading_days[-1]
    
    tickers = sorted(valid_start_dates.keys())
    log.info(f"{len(tickers)} tickers with valid start dates (warmup={WARMUP} trading days)")
    
    return master_df, valid_start_dates, tickers

# ============================================================================
# TECHNICAL INDICATORS & FEATURE ENGINEERING  
# ============================================================================

def _sma(s, n):
    """Simple Moving Average"""
    return s.rolling(n, min_periods=n).mean()

def _true_range(high, low, close):
    """True Range calculation"""
    prev_c = close.shift(1)
    return pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)

def _atr(high, low, close, n=14):
    """Average True Range"""
    return _true_range(high, low, close).ewm(span=n, min_periods=n, adjust=False).mean()

def _stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator %K and %D"""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest + 1e-10)
    pct_k = raw_k.rolling(d_period, min_periods=d_period).mean()
    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean()
    return pct_k, pct_d

def _bollinger_bandwidth(close, n=20, num_std=2.0):
    """Bollinger Bandwidth"""
    sma = _sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    return ((sma + num_std * std) - (sma - num_std * std)) / (sma + 1e-10)

def _protected_swing_low(low, window=60):
    """Protected swing low - rolling minimum"""
    return low.rolling(window, min_periods=window).min()

def compute_features_fixed(master_df, valid_start_dates, log):
    """
    FIXED: Compute features with standardized scaling [-1, +1] range for all features
    This fixes the gradient fracturing issue from mixed [0,1] and [-3,+3] ranges
    """
    all_tickers = master_df.columns.get_level_values("Ticker").unique()
    log.info(f"🔧 Computing features with FIXED SCALING for {len(all_tickers)} tickers...")
    pieces = {}
    
    for ticker in all_tickers:
        # Drop non-trading days for clean rolling windows
        raw = master_df[ticker].dropna(how="all")
        o, h, l, c, v = raw["Open"], raw["High"], raw["Low"], raw["Close"], raw["Volume"]
        
        # Pass-through raw OHLCV
        pieces[(ticker, "open")] = o
        pieces[(ticker, "high")] = h
        pieces[(ticker, "low")] = l
        pieces[(ticker, "close")] = c
        pieces[(ticker, "volume")] = v
        
        # Trend & structure indicators
        sma50 = _sma(c, 50)
        sma200 = _sma(c, 200)
        psl = _protected_swing_low(l, 60)
        pieces[(ticker, "sma50")] = sma50
        pieces[(ticker, "sma200")] = sma200
        pieces[(ticker, "macro_trend")] = (sma50 > sma200).astype(np.float32)
        pieces[(ticker, "protected_swing_low")] = psl
        
        # Raw distance to swing low (will be normalized later)
        pieces[(ticker, "d_low")] = (c - psl) / (c + 1e-10)
        
        # FIXED: Stochastic oscillators centered to [-1, +1] range instead of [0, 1]
        pct_k, pct_d = _stochastic(h, l, c)
        pieces[(ticker, "pct_k")] = (pct_k / 50.0) - 1.0  # [0,100] -> [-1,+1]
        pieces[(ticker, "pct_d")] = (pct_d / 50.0) - 1.0  # [0,100] -> [-1,+1]
        pieces[(ticker, "delta_k")] = (pct_k - pct_k.shift(1)) / 50.0  # Centered derivative
        
        # Volatility indicators (raw values, will be Z-scored)
        atr14 = _atr(h, l, c, 14)
        pieces[(ticker, "atr14")] = atr14
        pieces[(ticker, "natr")] = atr14 / (c + 1e-10)
        pieces[(ticker, "bbw")] = _bollinger_bandwidth(c, 20, 2.0)
    
    # Assemble DataFrame
    feat_df = pd.DataFrame(pieces)
    feat_df.columns = pd.MultiIndex.from_tuples(feat_df.columns, names=["Ticker", "Feature"])
    feat_df = feat_df.sort_index()
    
    # FIXED: Apply strict Z-scoring with clipping to [-1, +1] range
    # This aligns all features to the same scale as the centered stochastic oscillators
    for ticker in all_tickers:
        for col in ["natr", "bbw", "d_low"]:
            key = (ticker, col)
            if key not in feat_df.columns:
                continue
            clean = feat_df[key].dropna()
            rm = clean.rolling(252, min_periods=252).mean()
            rs = clean.rolling(252, min_periods=252).std()
            # FIXED: Clip to [-1, +1] instead of [-3, +3] for consistency
            z_scored = ((clean - rm) / (rs + 1e-8)).clip(-1.0, 1.0)
            feat_df[key] = z_scored
    
    n_features = feat_df.columns.get_level_values("Feature").nunique()
    log.info(f"✅ FIXED feature engineering complete — shape {feat_df.shape}, {n_features} features per ticker")
    log.info(f"✅ ALL FEATURES NOW SCALED TO [-1, +1] RANGE")
    
    return feat_df

# ============================================================================
# FIXED GYMNASIUM ENVIRONMENT
# ============================================================================

class NepseEnvFixed(gym.Env):
    """
    FIXED NEPSE Environment with corrected reward topology:
    1. REMOVED FORCED_EXIT_PENALTY (was creating gradient cliff)
    2. WIDENED ATR_MULT from 2.5 -> 3.5 for NEPSE volatility
    3. REDUCED cash opportunity cost scale
    """
    
    OBS_FEATURES = ["pct_k", "pct_d", "natr", "bbw", "d_low"]
    TAU = 0.015              # Transaction friction (unchanged)
    ATR_MULT = 3.5           # FIXED: Widened from 2.5 to 3.5 for NEPSE microstructure
    # FORCED_EXIT_PENALTY = 0.0  # FIXED: REMOVED - let PnL speak for itself
    OC_SCALE = 0.1           # FIXED: Reduced from 0.5 to 0.1 (less cash punishment)
    CASH_FRICTION = 0.0      # FIXED: Removed cash friction entirely
    
    def __init__(self, feat_df, valid_start_dates, episode_length=252, seed=None):
        super().__init__()
        self.feat_df = feat_df
        self.valid_start_dates = valid_start_dates
        self.tickers = sorted(valid_start_dates.keys())
        self.episode_length = episode_length
        self.dates = feat_df.index
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(7,), dtype=np.float32)  # Wider range
        self._rng = np.random.default_rng(seed)
        
        # Episode state
        self._ticker = ""
        self._start_idx = 0
        self._step = 0
        self._position = 0
        self._entry_price = 0.0
        self._hh = 0.0
        self._tsl = 0.0
        self._pv = 1.0
        self._buys = 0
        self._sells = 0
        self._forced = 0
        
        # Pre-cache numpy arrays
        self._cache = {}
        needed = self.OBS_FEATURES + ["close", "high", "low", "atr14", "protected_swing_low"]
        for tk in self.tickers:
            c = {}
            for f in needed:
                try:
                    c[f] = feat_df[(tk, f)].values.astype(np.float64)
                except KeyError:
                    c[f] = np.full(len(self.dates), np.nan)
            self._cache[tk] = c
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self._ticker = self._rng.choice(self.tickers)
        vs = self.valid_start_dates[self._ticker]
        vs_idx = int(np.searchsorted(self.dates, vs))
        last = len(self.dates) - self.episode_length
        if last <= vs_idx:
            vs_idx = max(0, last - 1)
        self._start_idx = int(self._rng.integers(vs_idx, max(vs_idx + 1, last)))
        
        # Reset episode state
        self._step = 0
        self._position = 0
        self._entry_price = 0.0
        self._hh = 0.0
        self._tsl = 0.0
        self._pv = 1.0
        self._buys = 0
        self._sells = 0
        self._forced = 0
        
        return self._obs(), self._info()
    
    def step(self, action):
        """
        FIXED step function with corrected reward topology:
        - Removed FORCED_EXIT_PENALTY 
        - Reduced opportunity cost punishment
        - Let natural PnL and friction drive learning
        """
        c = self._cache[self._ticker]
        idx = self._start_idx + self._step
        
        # Handle index bounds
        if idx >= len(self.dates):
            return self._obs(), 0.0, True, True, self._info()
        
        close_t = c["close"][idx]
        high_t = c["high"][idx]
        low_t = c["low"][idx]
        atr_t = c["atr14"][idx]
        psl_t = c["protected_swing_low"][idx]
        prev_c = c["close"][max(idx - 1, 0)]
        
        # Handle NaN values
        if np.isnan(close_t) or np.isnan(prev_c):
            self._step += 1
            terminated = self._step >= self.episode_length
            truncated = (self._start_idx + self._step) >= len(self.dates) - 1
            return self._obs(), 0.0, terminated, truncated, self._info()
        
        # NaN handling
        if np.isnan(atr_t): atr_t = 0.0
        if np.isnan(psl_t): psl_t = 0.0
        if np.isnan(high_t): high_t = close_t
        if np.isnan(low_t): low_t = close_t
        
        reward = 0.0
        forced = False
        
        if self._position == 0 and action == 1:
            # BUY (Cash -> Long)
            self._position = 1
            self._entry_price = close_t
            self._hh = close_t
            self._tsl = self._hh - self.ATR_MULT * atr_t  # Using widened 3.5x multiplier
            reward -= self.TAU  # Transaction friction only
            self._buys += 1
            
        elif self._position == 1 and action == 1:
            # HOLD LONG
            self._hh = max(self._hh, high_t)
            self._tsl = max(self._tsl, self._hh - self.ATR_MULT * atr_t)
            
            if low_t <= self._tsl or low_t <= psl_t:
                # FORCED EXIT - FIXED: No additional penalty!
                forced = True
                self._position = 0
                exit_p = min(max(self._tsl, psl_t), prev_c)
                lr = np.log(exit_p / (prev_c + 1e-10))
                # FIXED: Only log return + friction, NO FORCED_EXIT_PENALTY
                reward += lr - self.TAU  
                self._pv *= np.exp(lr - self.TAU)
                self._entry_price = 0.0
                self._forced += 1
            else:
                # Continue holding
                lr = np.log(close_t / (prev_c + 1e-10))
                reward += lr
                self._pv *= np.exp(lr)
                
        elif self._position == 1 and action == 0:
            # SELL (Long -> Cash)
            self._position = 0
            lr = np.log(close_t / (prev_c + 1e-10))
            reward += lr - self.TAU  # Log return minus friction
            self._pv *= np.exp(lr - self.TAU)
            self._entry_price = 0.0
            self._sells += 1
            
        else:
            # HOLD CASH - FIXED: Minimal opportunity cost
            delta = np.log(close_t / (prev_c + 1e-10))
            # FIXED: Much smaller opportunity cost, no cash friction
            if delta > 0:
                reward -= delta * self.OC_SCALE  # Only 0.1x instead of 0.5x
        
        self._step += 1
        terminated = self._step >= self.episode_length
        truncated = (self._start_idx + self._step) >= len(self.dates) - 1
        
        info = self._info()
        info["forced_liquidation"] = forced
        
        # Sanitize reward
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
            
        return self._obs(), float(reward), terminated, truncated, info
    
    def _obs(self):
        """Observation vector with all features in [-1, +1] range"""
        c = self._cache[self._ticker]
        idx = min(self._start_idx + self._step, len(self.dates) - 1)
        obs = np.zeros(7, dtype=np.float32)
        
        # Technical features (all now in [-1, +1] range)
        for i, f in enumerate(self.OBS_FEATURES):
            v = c[f][idx]
            obs[i] = 0.0 if np.isnan(v) else float(v)
        
        # Position indicator
        obs[5] = float(self._position)
        
        # Distance to TSL (normalized)
        if self._position == 1:
            cl = c["close"][idx]
            if not np.isnan(cl) and cl != 0:
                obs[6] = np.clip((cl - self._tsl) / (cl + 1e-10), -1.0, 1.0)
        
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    def _info(self):
        """Episode info dictionary"""
        c = self._cache[self._ticker]
        idx = min(self._start_idx + self._step, len(self.dates) - 1)
        return {
            "ticker": self._ticker,
            "date": self.dates[idx],
            "close": float(c["close"][idx]),
            "action": -1,
            "tsl_level": float(self._tsl) if self._position == 1 else np.nan,
            "portfolio_value": float(self._pv),
            "position": self._position,
        }

# ============================================================================
# FIXED PPO TRAINING CONFIGURATION
# ============================================================================

class EnhancedRewardTracker(BaseCallback):
    """Enhanced callback with CSV export capability"""
    def __init__(self, run_dir):
        super().__init__(verbose=0)
        self.run_dir = pathlib.Path(run_dir)
        self.ep_timesteps = []
        self.ep_rewards = []
        self.update_ts = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self._ep_count = 0
        
    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_timesteps.append(self.num_timesteps)
                self.ep_rewards.append(info["episode"]["r"])
                self._ep_count += 1
                
                if self._ep_count % 50 == 0:
                    recent_avg = np.mean(self.ep_rewards[-50:])
                    recent_std = np.std(self.ep_rewards[-50:])
                    # Log with more detail
                    log = logging.getLogger("nepserl")
                    log.info(f"Episode {self._ep_count:4d} | Steps: {self.num_timesteps:7d} | "
                           f"Reward: {recent_avg:+.4f}±{recent_std:.3f}")
                    
                    # Export intermediate CSV every 100 episodes  
                    if self._ep_count % 100 == 0:
                        self._export_metrics()
        return True
    
    def _on_rollout_end(self):
        try:
            vals = self.model.logger.name_to_value
            self.update_ts.append(self.num_timesteps)
            self.policy_losses.append(vals.get("train/policy_gradient_loss", 0.0))
            self.value_losses.append(vals.get("train/value_loss", 0.0))
            self.entropy_losses.append(vals.get("train/entropy_loss", 0.0))
        except Exception:
            pass
    
    def _export_metrics(self):
        """Export training metrics to CSV"""
        if not self.ep_rewards:
            return
            
        # Episode rewards CSV
        episode_df = pd.DataFrame({
            'timestep': self.ep_timesteps,
            'episode_reward': self.ep_rewards,
            'episode_number': range(1, len(self.ep_rewards) + 1)
        })
        
        # Add moving averages
        for window in [10, 50]:
            if len(self.ep_rewards) >= window:
                episode_df[f'reward_ma_{window}'] = episode_df['episode_reward'].rolling(window).mean()
        
        episode_path = self.run_dir / "episode_rewards.csv"
        episode_df.to_csv(episode_path, index=False)
        
        # Training losses CSV
        if self.update_ts:
            loss_df = pd.DataFrame({
                'timestep': self.update_ts,
                'policy_loss': self.policy_losses,
                'value_loss': self.value_losses,
                'entropy_loss': self.entropy_losses
            })
            loss_path = self.run_dir / "training_losses.csv"  
            loss_df.to_csv(loss_path, index=False)

def train_fixed_ppo(feat_df, valid_start_dates, run_dir, log):
    """
    FIXED PPO training with corrected hyperparameters:
    - Reduced entropy coefficient from 0.05 -> 0.005 (10x reduction)
    - Lowered learning rate from 3e-4 -> 1e-4
    - Increased batch size from 256 -> 512
    - Extended training to 1M timesteps
    """
    
    # FIXED Training configuration
    TOTAL_TIMESTEPS = 1_000_000  # Extended from 500k
    N_ENVS = 4  
    SEED = 42
    
    log.info("🔧 FIXED PPO Training Configuration:")
    log.info(f"   Timesteps: {TOTAL_TIMESTEPS:,} (extended)")
    log.info(f"   Learning Rate: 1e-4 (reduced from 3e-4)")
    log.info(f"   Entropy Coef: 0.005 (reduced from 0.05)")
    log.info(f"   Batch Size: 512 (increased from 256)")
    log.info(f"   ATR Multiplier: 3.5 (increased from 2.5)")
    log.info(f"   Forced Exit Penalty: REMOVED")
    
    def make_env(seed):
        def _init():
            return Monitor(NepseEnvFixed(feat_df, valid_start_dates, episode_length=252, seed=seed))
        return _init
    
    vec_env = DummyVecEnv([make_env(SEED + i) for i in range(N_ENVS)])
    
    # FIXED PPO model with corrected hyperparameters
    model = PPO(
        "MlpPolicy", 
        vec_env,
        learning_rate=1e-4,      # FIXED: Reduced from 3e-4 for stable convergence
        n_steps=2048,            # Unchanged
        batch_size=512,          # FIXED: Increased from 256 for smoother GAE
        n_epochs=10,             # Unchanged
        gamma=0.99,              # Unchanged  
        gae_lambda=0.95,         # Unchanged
        clip_range=0.2,          # Unchanged
        ent_coef=0.005,          # FIXED: Reduced from 0.05 to stop exploration bleed
        vf_coef=0.5,             # Unchanged
        max_grad_norm=0.5,       # Unchanged
        seed=SEED,
        device="auto",
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )
    
    log.info(f"✅ FIXED PPO model created — device={model.device}")
    
    # Enhanced callback with CSV export
    tracker = EnhancedRewardTracker(run_dir)
    
    log.info("🚀 Starting FIXED training...")
    
    # Train the model  
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[tracker],
        progress_bar=True,
    )
    
    vec_env.close()
    
    # Export final metrics
    tracker._export_metrics()
    
    final_avg = np.mean(tracker.ep_rewards[-20:]) if len(tracker.ep_rewards) >= 20 else 0.0
    log.info(f"✅ FIXED training complete!")
    log.info(f"   Episodes: {len(tracker.ep_rewards)}")
    log.info(f"   Final avg reward (20 ep): {final_avg:+.4f}")
    
    # Save model
    model_path = run_dir / "nepserl_fixed_model.zip"
    model.save(model_path)
    log.info(f"💾 Model saved: {model_path}")
    
    return model, tracker

# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================

def run_episode_evaluation(model, feat_df, valid_start_dates, ticker=None, episode_length=252, seed=123):
    """Run a single episode evaluation"""
    env = NepseEnvFixed(feat_df, valid_start_dates, episode_length=episode_length, seed=seed)
    obs, info = env.reset()
    
    if ticker is not None:
        # Force specific ticker
        env._ticker = ticker
        vs = valid_start_dates[ticker]
        env._start_idx = int(np.searchsorted(env.dates, vs))
        env._step = 0
        env._position = 0
        env._entry_price = 0.0
        env._hh = 0.0
        env._tsl = 0.0
        env._pv = 1.0
        obs = env._obs()
        info = env._info()
    
    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        info["action"] = action
        records.append(info.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Final record
    info["action"] = -1
    records.append(info.copy())
    
    traj = pd.DataFrame(records)
    return traj

def compute_basic_metrics(traj):
    """Compute basic performance metrics"""
    pv = traj["portfolio_value"].values
    close = traj["close"].values
    
    total_ret_agent = pv[-1] / pv[0] - 1.0
    total_ret_bh = close[-1] / close[0] - 1.0
    
    actions = traj["action"].values
    n_buys = (actions == 1).sum()
    n_total = len(actions) - 1  # Exclude END action
    
    metrics = {
        "total_return_agent": total_ret_agent,
        "total_return_buyhold": total_ret_bh,
        "excess_return": total_ret_agent - total_ret_bh,
        "final_pv": pv[-1],
        "num_buys": n_buys,
        "buy_ratio": n_buys / n_total if n_total > 0 else 0,
        "cash_ratio": 1 - (n_buys / n_total) if n_total > 0 else 1,
    }
    
    return metrics

def create_summary_plots(tracker, run_dir, log):
    """Create training summary plots"""
    if not tracker.ep_rewards:
        log.warning("No episode rewards to plot")
        return
    
    # Episode rewards plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(tracker.ep_timesteps, tracker.ep_rewards, alpha=0.3, color="dodgerblue", label="Episode reward")
    
    if len(tracker.ep_rewards) > 50:
        # Moving average
        rewards_series = pd.Series(tracker.ep_rewards)
        ma_50 = rewards_series.rolling(50, min_periods=1).mean()
        ax.plot(tracker.ep_timesteps, ma_50, color="red", linewidth=2, label="MA(50)")
    
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")  
    ax.set_title("NEPSE RL Fixed Training - Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    reward_plot_path = run_dir / "training_rewards_fixed.png"
    plt.savefig(reward_plot_path, dpi=150, bbox_inches="tight")
    log.info(f"📊 Saved reward plot: {reward_plot_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Setup
    log, run_dir, data_dir = setup_logging_and_directories()
    
    try:
        # Load data
        master_df, valid_start_dates, tickers = load_ohlcv_data(data_dir, log)
        
        # FIXED feature engineering  
        feat_df = compute_features_fixed(master_df, valid_start_dates, log)
        
        # Sample one row to verify fixed scaling
        sample_ticker = tickers[0]  
        sample_row = feat_df[sample_ticker].dropna(how='all').iloc[500]
        
        log.info("🔍 FEATURE SCALING VERIFICATION:")
        log.info("=" * 50)
        for col in ["pct_k", "pct_d", "natr", "bbw", "d_low"]:
            if col in sample_row.index:
                val = sample_row[col]
                log.info(f"   {col:>10s}: {val:+.6f}")
        log.info("✅ All features should be in [-1, +1] range!")
        log.info("=" * 50)
        
        # FIXED PPO Training
        model, tracker = train_fixed_ppo(feat_df, valid_start_dates, run_dir, log)
        
        # Create summary plots
        create_summary_plots(tracker, run_dir, log)
        
        # Quick evaluation on sample ticker
        log.info("📊 Running sample evaluation...")
        eval_ticker = tickers[0]
        traj = run_episode_evaluation(model, feat_df, valid_start_dates, ticker=eval_ticker)
        metrics = compute_basic_metrics(traj)
        
        log.info("🎯 SAMPLE EVALUATION RESULTS:")
        log.info("=" * 50)
        log.info(f"   Ticker: {eval_ticker}")
        log.info(f"   Agent Return: {metrics['total_return_agent']:+.2%}")  
        log.info(f"   Buy&Hold Return: {metrics['total_return_buyhold']:+.2%}")
        log.info(f"   Excess Return: {metrics['excess_return']:+.2%}")
        log.info(f"   Final PV: {metrics['final_pv']:.4f}")
        log.info(f"   Buy Ratio: {metrics['buy_ratio']:.1%}")
        log.info(f"   Cash Ratio: {metrics['cash_ratio']:.1%}")
        log.info("=" * 50)
        
        # Action distribution analysis
        actions = traj["action"].value_counts().sort_index()
        log.info("📊 Action Distribution:")
        for action, count in actions.items():
            action_name = "Cash" if action == 0 else "Long" if action == 1 else "End"
            log.info(f"   Action {action} ({action_name}): {count} ({count/len(traj):.1%})")
        
        log.info("🎉 NEPSE RL FIXED VERSION COMPLETE!")
        log.info(f"📁 Results saved in: {run_dir}")
        log.info("")
        log.info("🔧 FIXES APPLIED:")
        log.info("   ✅ Removed FORCED_EXIT_PENALTY (reward topology fix)")
        log.info("   ✅ Standardized all features to [-1, +1] range (gradient fix)")  
        log.info("   ✅ Fixed PPO hyperparameters (convergence fix)")
        log.info("   ✅ Widened ATR multiplier 2.5 -> 3.5 (volatility fix)")
        log.info("")
        log.info("Expected improvements:")
        log.info("   - Higher final rewards (target: -0.1 to +0.2)")
        log.info("   - More balanced action distribution") 
        log.info("   - Stable convergence without oscillation")
        log.info("   - Reduced forced liquidations")
        
    except Exception as e:
        log.error(f"❌ Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()