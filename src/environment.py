"""
Phase 3 – 5: Custom Gymnasium Environment
==========================================
UniversalNepseEnv
  • Discrete(2) action space:  0 → target Cash, 1 → target Long
  • Observation = [%K, %D, NATR, BBW, D_low, current_position, dist_to_TSL]
  • Chandelier-Exit TSL hard-override inside step()
  • Reward:  log-return while long, −0.015 friction on transitions,
            −2.0 catastrophic penalty on forced liquidation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.run_manager import get_logger


class UniversalNepseEnv(gym.Env):
    """Multi-asset stochastic pullback RL environment for NEPSE."""

    metadata = {"render_modes": []}

    # Observation features (order matters – matches NN input)
    OBS_FEATURES = ["pct_k", "pct_d", "natr", "bbw", "d_low"]

    # Transaction friction (1.5 % round-trip brokerage + taxes)
    TAU = 0.015
    # Chandelier-Exit ATR multiplier
    ATR_MULT = 2.5
    # Structural penalty for forced liquidation (scaled to daily vol)
    FORCED_EXIT_PENALTY = 0.05
    # Asymmetric opportunity-cost parameters
    # When sitting in cash during an up-move, penalise proportionally to
    # the missed log-return (scaled by OC_SCALE).  On down/flat days the
    # agent receives only a small static friction so it isn't rewarded
    # for correct inaction but still feels pressure to search.
    OC_SCALE = 0.5       # fraction of missed Δ charged as penalty
    CASH_FRICTION = 0.001  # baseline daily cost when market is flat/down
    # Minimum episode length in trading days (~1 year)
    MIN_EPISODE_LEN = 252

    def __init__(
        self,
        feat_df: pd.DataFrame,
        valid_start_dates: Dict[str, pd.Timestamp],
        episode_length: int = 252,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.feat_df = feat_df
        self.valid_start_dates = valid_start_dates
        self.tickers = sorted(valid_start_dates.keys())
        self.episode_length = episode_length
        self.dates = feat_df.index  # universal date axis

        # Spaces
        self.action_space = spaces.Discrete(2)
        # 7-dim: %K, %D, NATR, BBW, D_low, position, dist_to_tsl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._log = get_logger("rl_nepse.env")

        # Episode state (set in reset)
        self._ticker: str = ""
        self._start_idx: int = 0
        self._current_step: int = 0
        self._position: int = 0  # 0 = cash, 1 = long
        self._entry_price: float = 0.0
        self._highest_high: float = 0.0
        self._tsl_price: float = 0.0
        self._portfolio_value: float = 1.0

        # Per-ticker caches (numpy arrays for speed)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._build_caches()

    # ------------------------------------------------------------------ #
    #  Cache building – avoids repeated DataFrame indexing inside step()  #
    # ------------------------------------------------------------------ #
    def _build_caches(self) -> None:
        """Pre-extract numpy arrays for every ticker for O(1) indexing."""
        needed = self.OBS_FEATURES + [
            "close", "high", "low", "atr14", "protected_swing_low",
            "open", "sma200",
        ]
        for ticker in self.tickers:
            cache: Dict[str, np.ndarray] = {}
            for feat in needed:
                try:
                    cache[feat] = self.feat_df[(ticker, feat)].values.astype(np.float64)
                except KeyError:
                    cache[feat] = np.full(len(self.dates), np.nan)
            self._cache[ticker] = cache

    # ------------------------------------------------------------------ #
    #  reset()                                                            #
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random ticker
        self._ticker = self._rng.choice(self.tickers)
        cache = self._cache[self._ticker]
        close = cache["close"]

        # Valid date window
        vs = self.valid_start_dates[self._ticker]
        vs_idx = np.searchsorted(self.dates, vs)

        # Last possible start so there are episode_length days remaining
        last_possible = len(self.dates) - self.episode_length
        if last_possible <= vs_idx:
            vs_idx = max(0, last_possible - 1)

        self._start_idx = int(self._rng.integers(vs_idx, max(vs_idx + 1, last_possible)))
        self._current_step = 0

        # State
        self._position = 0
        self._entry_price = 0.0
        self._highest_high = 0.0
        self._tsl_price = 0.0
        self._portfolio_value = 1.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------ #
    #  step()                                                             #
    # ------------------------------------------------------------------ #
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        cache = self._cache[self._ticker]
        idx = self._start_idx + self._current_step

        close_t   = cache["close"][idx]
        high_t    = cache["high"][idx]
        low_t     = cache["low"][idx]
        atr_t     = cache["atr14"][idx]
        psl_t     = cache["protected_swing_low"][idx]
        prev_close = cache["close"][max(idx - 1, 0)]

        # Guard against NaN prices (should not happen after warm-up but be safe)
        if np.isnan(close_t) or np.isnan(prev_close):
            self._current_step += 1
            terminated = self._current_step >= self.episode_length
            truncated = (self._start_idx + self._current_step) >= len(self.dates) - 1
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()

        if np.isnan(atr_t):
            atr_t = 0.0
        if np.isnan(psl_t):
            psl_t = 0.0
        if np.isnan(high_t):
            high_t = close_t
        if np.isnan(low_t):
            low_t = close_t

        reward = 0.0
        forced_liquidation = False

        # ------ Execution routing (strict Markov: t-1 → t only) ------
        if self._position == 0 and action == 1:
            # ── BUY ──
            self._position = 1
            self._entry_price = close_t
            self._highest_high = close_t
            self._tsl_price = self._highest_high - (self.ATR_MULT * atr_t)
            reward -= self.TAU  # Apply entry friction
            self._log.debug(
                f"BUY  {self._ticker} @ {close_t:.2f} | "
                f"TSL={self._tsl_price:.2f}")

        elif self._position == 1 and action == 1:
            # ── HOLD LONG ──
            self._highest_high = max(self._highest_high, high_t)
            new_tsl = self._highest_high - (self.ATR_MULT * atr_t)
            self._tsl_price = max(self._tsl_price, new_tsl)

            # Check Hard Override (Low breaches TSL or PSL)
            if low_t <= self._tsl_price or low_t <= psl_t:
                forced_liquidation = True
                self._position = 0

                # Calculate the exact exit price based on the breach
                exit_price = max(self._tsl_price, psl_t)
                exit_price = min(exit_price, prev_close)  # Cap at prev_close if gap down

                # Reward is only the return from yesterday to the stop price
                log_ret = np.log(exit_price / (prev_close + 1e-10))

                # The penalty is scaled to daily volatility, not a massive scalar
                reward += log_ret - self.TAU - self.FORCED_EXIT_PENALTY
                self._portfolio_value *= np.exp(log_ret - self.TAU)
                self._entry_price = 0.0
                self._log.debug(
                    f"FORCED EXIT {self._ticker} @ {exit_price:.2f} | "
                    f"log_ret={log_ret:+.4f} | pv={self._portfolio_value:.4f}")
            else:
                # Normal Hold: Reward is strictly yesterday to today
                log_ret = np.log(close_t / (prev_close + 1e-10))
                reward += log_ret
                self._portfolio_value *= np.exp(log_ret)

        elif self._position == 1 and action == 0:
            # ── SELL (Natural) ──
            self._position = 0
            # Reward is only yesterday to today, minus exit friction
            log_ret = np.log(close_t / (prev_close + 1e-10))
            reward += log_ret - self.TAU
            self._portfolio_value *= np.exp(log_ret - self.TAU)
            self._entry_price = 0.0
            self._log.debug(
                f"SELL {self._ticker} @ {close_t:.2f} | "
                f"log_ret={log_ret:+.4f} | pv={self._portfolio_value:.4f}")

        # action == 0 and position == 0 → HOLD CASH → asymmetric opportunity cost
        elif self._position == 0 and action == 0:
            delta = np.log(close_t / (prev_close + 1e-10))
            if delta > 0:
                # Missed a profitable up-move: penalise proportionally
                reward -= delta * self.OC_SCALE
            else:
                # Market flat/down: small static friction
                reward -= self.CASH_FRICTION

        # ------ Advance ------
        self._current_step += 1
        terminated = self._current_step >= self.episode_length
        truncated = (self._start_idx + self._current_step) >= len(self.dates) - 1

        obs = self._get_obs()
        info = self._get_info()
        info["forced_liquidation"] = forced_liquidation

        # Sanitise reward
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Observation & info                                                 #
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        cache = self._cache[self._ticker]
        idx = self._start_idx + self._current_step
        idx = min(idx, len(self.dates) - 1)

        obs = np.zeros(7, dtype=np.float32)

        for i, feat in enumerate(self.OBS_FEATURES):
            val = cache[feat][idx]
            obs[i] = 0.0 if np.isnan(val) else float(val)

        obs[5] = float(self._position)

        # Distance to TSL  (only meaningful when holding long)
        if self._position == 1:
            c = cache["close"][idx]
            if np.isnan(c) or c == 0:
                obs[6] = 0.0
            else:
                obs[6] = float((c - self._tsl_price) / (c + 1e-10))
        else:
            obs[6] = 0.0

        # Final guard – replace any remaining NaN / Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _get_info(self) -> dict:
        cache = self._cache[self._ticker]
        idx = self._start_idx + self._current_step
        idx = min(idx, len(self.dates) - 1)
        return {
            "ticker": self._ticker,
            "date": self.dates[idx],
            "close": float(cache["close"][idx]),
            "action": -1,  # will be filled by caller
            "tsl_level": float(self._tsl_price) if self._position == 1 else np.nan,
            "portfolio_value": float(self._portfolio_value),
            "position": self._position,
        }


# ── Deterministic evaluation wrapper ──

def run_deterministic_episode(
    env: UniversalNepseEnv,
    model,
    ticker: str | None = None,
    start_idx: int | None = None,
) -> pd.DataFrame:
    """Roll-out with frozen weights.  Returns a trajectory DataFrame."""
    obs, info = env.reset()

    # Optionally force ticker / start
    if ticker is not None:
        env._ticker = ticker
        cache = env._cache[ticker]
        vs = env.valid_start_dates[ticker]
        vs_idx = int(np.searchsorted(env.dates, vs))
        env._start_idx = start_idx if start_idx is not None else vs_idx
        env._current_step = 0
        env._position = 0
        env._entry_price = 0.0
        env._highest_high = 0.0
        env._tsl_price = 0.0
        env._portfolio_value = 1.0
        obs = env._get_obs()
        info = env._get_info()

    records = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        info["action"] = action
        records.append(info.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Final step info
    info["action"] = -1
    records.append(info.copy())
    return pd.DataFrame(records)
