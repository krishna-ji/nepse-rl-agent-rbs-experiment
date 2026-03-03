# ================================================================
# NEPSE RL MDP Debugging - Exact Code Contexts
# ================================================================

# Context 1: Reward & Transition Physics from step() function
print("="*60)
print("CONTEXT 1: REWARD & TRANSITION PHYSICS - step() function")
print("="*60)

step_function_code = '''
def step(self, action):
    c = self._cache[self._ticker]
    idx = self._start_idx + self._step
    close_t  = c["close"][idx]
    high_t   = c["high"][idx]
    low_t    = c["low"][idx]
    atr_t    = c["atr14"][idx]
    psl_t    = c["protected_swing_low"][idx]
    prev_c   = c["close"][max(idx - 1, 0)]

    if np.isnan(close_t) or np.isnan(prev_c):
        self._step += 1
        t = self._step >= self.episode_length
        tr = (self._start_idx + self._step) >= len(self.dates) - 1
        return self._obs(), 0.0, t, tr, self._info()

    for v in [atr_t, psl_t, high_t, low_t]:
        if np.isnan(atr_t): atr_t = 0.0
        if np.isnan(psl_t): psl_t = 0.0
        if np.isnan(high_t): high_t = close_t
        if np.isnan(low_t): low_t = close_t

    reward = 0.0
    forced = False

    if self._position == 0 and action == 1:          # BUY (Cash -> Long)
        self._position = 1
        self._entry_price = close_t
        self._hh = close_t
        self._tsl = self._hh - self.ATR_MULT * atr_t
        reward -= self.TAU                            # -0.015 transaction friction
        self._buys += 1

    elif self._position == 1 and action == 1:        # HOLD LONG
        self._hh = max(self._hh, high_t)
        self._tsl = max(self._tsl, self._hh - self.ATR_MULT * atr_t)
        if low_t <= self._tsl or low_t <= psl_t:     # CHANDELIER EXIT FORCED LIQUIDATION
            forced = True
            self._position = 0
            exit_p = min(max(self._tsl, psl_t), prev_c)
            lr = np.log(exit_p / (prev_c + 1e-10))
            reward += lr - self.TAU - self.FORCED_EXIT_PENALTY  # Log return - friction - PENALTY
            self._pv *= np.exp(lr - self.TAU)
            self._entry_price = 0.0
            self._forced += 1
        else:
            lr = np.log(close_t / (prev_c + 1e-10))
            reward += lr                              # Pure log return while holding
            self._pv *= np.exp(lr)

    elif self._position == 1 and action == 0:        # SELL (Long -> Cash)
        self._position = 0
        lr = np.log(close_t / (prev_c + 1e-10))
        reward += lr - self.TAU                       # Log return - friction
        self._pv *= np.exp(lr - self.TAU)
        self._entry_price = 0.0
        self._sells += 1

    else:                                            # HOLD CASH (position=0, action=0)
        delta = np.log(close_t / (prev_c + 1e-10))
        reward -= delta * self.OC_SCALE if delta > 0 else self.CASH_FRICTION  # Opportunity cost or cash friction

    # CRITICAL CONSTANTS:
    # TAU = 0.015              (transaction friction)
    # FORCED_EXIT_PENALTY = 0.05  (additional penalty on forced exit)
    # OC_SCALE = 0.5          (opportunity cost scale)
    # CASH_FRICTION = 0.001   (cash holding friction)
    
    self._step += 1
    terminated = self._step >= self.episode_length
    truncated  = (self._start_idx + self._step) >= len(self.dates) - 1
    return self._obs(), float(reward), terminated, truncated, self._info()
'''

print(step_function_code)

# Context 2: Feature Normalization Pipeline
print("\n" + "="*60)
print("CONTEXT 2: FEATURE NORMALIZATION PIPELINE")
print("="*60)

feature_code = '''
def compute_features(master_df, valid_start_dates):
    for ticker in all_tickers:
        raw = master_df[ticker].dropna(how="all")
        o, h, l, c, v = raw["Open"], raw["High"], raw["Low"], raw["Close"], raw["Volume"]

        # Momentum (pullback vector)
        pct_k, pct_d = _stochastic(h, l, c)
        pieces[(ticker, "pct_k")]   = pct_k / 100.0         # SCALED TO [0,1]
        pieces[(ticker, "pct_d")]   = pct_d / 100.0         # SCALED TO [0,1]
        
        # Volatility (regime filter)  
        atr14 = _atr(h, l, c, 14)
        pieces[(ticker, "natr")]  = atr14 / (c + 1e-10)     # NORMALIZED ATR [~0.005-0.050]
        pieces[(ticker, "bbw")]   = _bollinger_bandwidth(c, 20, 2.0)  # BOLLINGER WIDTH [~0.02-0.15]
        
        # Structure
        psl = _protected_swing_low(l, 60)
        pieces[(ticker, "d_low")] = (c - psl) / (c + 1e-10) # DISTANCE TO SWING LOW [0.0-0.3+]
        
    # Z-SCORE NORMALIZATION on 252-day rolling window
    for ticker in all_tickers:
        for col in ["natr", "bbw", "d_low"]:
            clean = feat_df[key].dropna()
            rm = clean.rolling(252, min_periods=252).mean()
            rs = clean.rolling(252, min_periods=252).std()
            feat_df[key] = ((clean - rm) / (rs + 1e-8)).clip(-3, 3)  # Z-SCORE CLIPPED [-3, +3]

# Final state vector construction in _obs():
def _obs(self):
    obs = np.zeros(7, dtype=np.float32)
    # obs[0] = pct_k      # [0, 1] after /100.0
    # obs[1] = pct_d      # [0, 1] after /100.0  
    # obs[2] = natr       # Z-scored [-3, 3]
    # obs[3] = bbw        # Z-scored [-3, 3]
    # obs[4] = d_low      # Z-scored [-3, 3]
    # obs[5] = position   # {0, 1}
    # obs[6] = dist_to_tsl # [-X, +X] normalized distance
    return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
'''

print(feature_code)

# Context 3: PPO Hyperparameters
print("\n" + "="*60)
print("CONTEXT 3: PPO HYPERPARAMETER MATRIX")
print("="*60)

ppo_config = '''
model = PPO(
    "MlpPolicy", vec_env,
    learning_rate=3e-4,      # DEFAULT - MAY BE TOO HIGH FOR CONVERGENCE
    n_steps=2048,           # ROLLOUT BUFFER SIZE
    batch_size=256,         # MINIBATCH SIZE  
    n_epochs=10,            # GRADIENT UPDATES PER ROLLOUT
    gamma=0.99,             # DISCOUNT FACTOR
    gae_lambda=0.95,        # GAE PARAMETER
    clip_range=0.2,         # PPO CLIP RANGE
    ent_coef=0.05,          # ENTROPY COEFFICIENT ⚠️ CRITICAL FOR EXPLORATION
    vf_coef=0.5,            # VALUE FUNCTION COEFFICIENT
    max_grad_norm=0.5,      # GRADIENT CLIPPING
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),  # NETWORK ARCHITECTURE
)

# TRAINING CONFIG:
# TOTAL_TIMESTEPS = 500_000
# N_ENVS = 4  
# SEED = 42
'''

print(ppo_config)

# Context 4: Expected Action Distribution Analysis
print("\n" + "="*60) 
print("CONTEXT 4: ACTION DISTRIBUTION DIAGNOSTICS")
print("="*60)

print('''
From the logs, the final evaluation shows:
Actions — Cash(0):204, Long(1):48, End(-1):1

This translates to:
- Cash (Action 0): 204/252 = 81% of episode
- Long (Action 1): 48/252 = 19% of episode  

PROBLEM DIAGNOSED:
The agent has learned that taking positions (Action 1) leads to:
1. Immediate -0.015 transaction friction on entry
2. High probability of forced liquidation (-0.05 penalty + -0.015 friction = -0.065 total)
3. Opportunity cost while holding during downturns

The reward function is PUNISHING market exposure, so the agent rationally 
converges to minimizing harm rather than maximizing returns.

EXPECTED SCALAR MAGNITUDES (typical values):
- pct_k, pct_d: [0.0, 1.0] (normalized stochastic)
- natr: Z-scored [-3.0, +3.0] (but raw values ~0.005-0.050)
- bbw: Z-scored [-3.0, +3.0] (but raw values ~0.02-0.15)  
- d_low: Z-scored [-3.0, +3.0] (but raw values 0.0-0.3+)
- position: {0, 1}
- dist_to_tsl: Unbounded, typically [-1.0, +1.0] range

GRADIENT FRACTURING ISSUE:
The stochastic features [0,1] vs Z-scored features [-3,+3] have different 
scales, potentially causing unstable training dynamics.
''')

print("\n" + "="*60)
print("ROOT CAUSE ANALYSIS COMPLETE")
print("="*60)
print("The agent is trapped in a pathological equilibrium where:")
print("1. Market exposure guarantees friction penalties")  
print("2. Chandelier exits trigger additional -0.05 penalties")
print("3. Cash holding minimizes catastrophic losses")
print("4. Feature normalization inconsistencies destabilize learning")
print("5. entropy_coef=0.05 may be too high, causing excessive exploration noise")