# Interview Questions & Deep-Dive — nepserl

Comprehensive breakdown of every algorithm, methodology, logic, and rationale in this project.
Organized by topic with likely interview questions and strong answers.

---

## Table of Contents

1. [General / Portfolio-Level Questions](#1-general--portfolio-level-questions)
2. [Data Pipeline & Feature Engineering](#2-data-pipeline--feature-engineering)
3. [Rule-Based Strategies — Framework Design](#3-rule-based-strategies--framework-design)
4. [Ichimoku Kumo Break Strategy](#4-ichimoku-kumo-break-strategy)
5. [Ichimoku T/K Cross Strategy](#5-ichimoku-tk-cross-strategy)
6. [EMA Crossover Strategy](#6-ema-crossover-strategy)
7. [RSI Mean Reversion Strategy](#7-rsi-mean-reversion-strategy)
8. [Bollinger Breakout Strategy](#8-bollinger-breakout-strategy)
9. [MACD Signal Cross Strategy](#9-macd-signal-cross-strategy)
10. [Donchian Breakout Strategy](#10-donchian-breakout-strategy)
11. [LightGBM Signal Quality Filter](#11-lightgbm-signal-quality-filter)
12. [Cross-Sectional Ranking Engine](#12-cross-sectional-ranking-engine)
13. [Reinforcement Learning — PPO Portfolio Agent](#13-reinforcement-learning--ppo-portfolio-agent)
14. [RL — Hyperscale Single-Asset Agent](#14-rl--hyperscale-single-asset-agent)
15. [MDP Debugging & Reward Shaping](#15-mdp-debugging--reward-shaping)
16. [Backtesting Methodology & Bias Prevention](#16-backtesting-methodology--bias-prevention)
17. [Risk Management & Position Sizing](#17-risk-management--position-sizing)
18. [Testing & Software Engineering](#18-testing--software-engineering)
19. [Market Microstructure — NEPSE Specifics](#19-market-microstructure--nepse-specifics)
20. [Statistics & Evaluation Metrics](#20-statistics--evaluation-metrics)

---

## 1. General / Portfolio-Level Questions

### Q: Walk me through your project at a high level

**A:** This is a quantitative trading research platform for the Nepal Stock Exchange (NEPSE) — a frontier market with 441 listed securities. It covers the full pipeline:

1. **Data layer** — parallel OHLCV scraper pulling daily bars from nepsetrading.com (130k+ bars across 112 tickers, 2012–2026).
2. **Rule-based strategies** — 7 systematic strategies (Ichimoku ×2, EMA, RSI, Bollinger, MACD, Donchian) sharing a common backtesting framework with realistic friction.
3. **ML signal filtering** — a 38-feature LightGBM model that scores each rule-based signal's quality before execution, improving win rate from 36% → 41%.
4. **Deep RL agents** — PPO-based agents (portfolio-level and single-asset) trained with walk-forward validation, T+3 settlement locks, and regime-conditioned rewards.

The measured results: Ichimoku T/K Cross produces a 2.34 profit factor over 991 trades with +4.51% expectancy per trade, and this holds in out-of-sample (PF 1.65 post July 2024).

### Q: Why NEPSE? Why not a more liquid market?

**A:** Three reasons:

1. **Alpha availability** — frontier markets are inefficient. Institutional coverage is minimal, so systematic approaches have genuine edge.
2. **Unique constraints force better engineering** — T+3 settlement, 10% circuit breakers, low liquidity, and leptokurtic returns mean you can't use off-the-shelf approaches. Every design decision (ATR multipliers, stop mechanics, action locks) had to be adapted.
3. **Full-stack ownership** — no Bloomberg/Reuters API exists for NEPSE, so I built the entire pipeline from scraper to portfolio simulation.

### Q: What's your best-performing strategy and why?

**A:** Ichimoku T/K Cross: 991 trades, 42.1% win rate, profit factor 2.34, +4.51% expectancy per trade, -18.2% max drawdown. The key is asymmetry — average winners (+18.7%) are 3.2× larger than average losers (-5.8%). The Kijun-based trailing stop cuts losers fast while letting winners ride in trending regimes. Out-of-sample (post 2024-07-01), profit factor holds at 1.65 — the edge doesn't collapse.

### Q: Your win rate is only 42%. Isn't that bad?

**A:** No — this is a trend-following system, and sub-50% win rates are the norm. What matters is the *expectancy*: `E = (WR × AvgWin) - ((1-WR) × AvgLoss)` = `(0.421 × 18.7) - (0.579 × 5.8)` = +4.51% per trade. The profit factor of 2.34 means for every Rs 1 lost, you make Rs 2.34 in gross profits. The system works because losses are small and tightly controlled (Kijun + ATR trailing stop), while winners have no upside cap.

### Q: How do you know this isn't curve-fitted?

**A:** Five safeguards:

1. **Out-of-sample validation** — all strategies split at 2024-07-01; OOS profit factor stays positive (1.57–1.65).
2. **Walk-forward ML training** — LightGBM models are retrained every 3 months with 30-day purged gaps.
3. **No parameter optimization on test set** — Ichimoku parameters (9/26/52) are standard from Hosoda's original work, not optimized on NEPSE data.
4. **Realistic friction** — 0.5% per trade is conservative for NEPSE's actual brokerage + SEBON fees.
5. **Cross-strategy consistency** — all 7 strategies share the same framework; the edge pattern (high PF, sub-50% WR, asymmetric payoff) is consistent across different entry logics.

---

## 2. Data Pipeline & Feature Engineering

### Q: How do you handle data quality for a frontier market?

**A:** The scraper (`src/nepsetrading/fetch_ohlcv.py`) uses `ProcessPoolExecutor` for parallel downloads from nepsetrading.com's API. Quality checks include:

- Minimum 250 bars per ticker (skip thin histories)
- Exclude PROMOTSHARE, mutual funds, and corporate debentures (illiquid instruments)
- 80-bar warmup period for indicator computation (Ichimoku needs 52+26 bars minimum)
- NaN-safe feature computation — tickers with sparse early history don't crash the pipeline

### Q: You say "no TA-Lib." Why implement indicators from scratch?

**A:** Three reasons:

1. **Portability** — TA-Lib requires C compilation which fails on many deployment environments.
2. **Auditability** — I can verify every indicator is lookahead-free. With a black-box library, you trust their implementation.
3. **Customization** — NEPSE-specific adjustments (like the Senkou B flatness filter or Chikou ±2 window check) aren't available in standard libraries.

### Q: Walk me through your 38 LightGBM features

**A:** Six groups, all computed with data available at bar t only:

| Group | Count | Key Features | Rationale |
|-------|-------|-------------|-----------|
| **Ichimoku** | 11 | Cloud thickness, TK spread, Chikou clearance, SB flatness, Kijun/Tenkan slopes | Captures cloud structure, trend momentum, support/resistance topology |
| **Trend** | 6 | Price vs SMA(20/50/200), SMA slope, ADX, trend alignment score | Multi-timeframe trend context |
| **Momentum** | 6 | RSI, RSI slope, MACD histogram + slope, ROC(5/20) | Rate of change across timeframes |
| **Volatility** | 5 | ATR%, BB width, BB position, vol ratio (5d/20d), ATR expansion | Regime detection — contraction vs expansion |
| **Volume** | 4 | Surge, trend, OBV slope, volume-price confirmation | Participation conviction |
| **Candlestick** | 4 | Body ratio, shadow ratio, CLV, gap% | Microstructure / bar-level sentiment |

The `trend_alignment` feature is notable — it's a composite score: `(c>sma20) + (c>sma50) + (c>sma200) + (sma20>sma50) + (sma50>sma200)` normalized to [0,1]. This captures multi-timeframe trend agreement in a single dimension.

### Q: How do you prevent lookahead bias in features?

**A:** Every feature uses only data available at or before bar t:

- Rolling windows use `rolling(n)` which only looks backward
- Ichimoku displacement: current Kumo at chart position t = Senkou A/B computed at t-26 (shifted forward), so what you see at t was computed in the past
- Future Kumo: Senkou A/B computed at t (unshifted) — this predicts future cloud shape but uses current data only
- Chikou span: compares close[t] vs historical data at t-26 — backward-looking
- Signals fire at close of bar t → entries execute at bar t+1

---

## 3. Rule-Based Strategies — Framework Design

### Q: How is the backtesting framework structured?

**A:** `src/rbs/base.py` implements a state machine with three states: `FLAT → PENDING → POSITION → FLAT`.

Key components:

- **TradeRecord dataclass** — captures entry/exit prices, dates, reasons, bars held, PnL (gross + net), and strategy-specific extras
- **IchimokuParams dataclass** — unified parameter object with Tenkan=9, Kijun=26, SenkouB=52, displacement=26
- **SizingMode enum** — FIXED or COMPOUNDING for portfolio simulation
- **Shared metrics computation** — win rate, profit factor, expectancy, skewness, kurtosis, streaks
- **Data loading** — filters out non-tradeable instruments, enforces minimum bar count

### Q: Why use buy-stop orders instead of market orders?

**A:** Buy-stop orders at a higher level (e.g., 9-period high) serve as a **confirmation filter**. The signal says "conditions are right," but the buy-stop says "prove it by making a new high." This:

1. Reduces false signals — if price can't break the level, the order expires (5-bar timeout)
2. Ensures the Tenkan-sen gets pulled upward (Ichimoku strategies)
3. Prevents buying into immediate weakness after a signal bar
4. Is realistic for NEPSE — limit orders are standard

The exception is RSI Mean Reversion, which uses market orders at next open — because mean reversion needs immediate action (the bounce is already happening).

### Q: How do you handle transaction costs?

**A:** 0.5% round-trip friction deducted from every trade's PnL: `net_pnl = gross_pnl - 0.5%`. This is conservative — actual NEPSE brokerage is ~0.36% + SEBON fee + DP charges. The portfolio simulator uses 1.5% for the ML-filtered version (higher to include slippage estimates). The RL agent uses TAU=0.0045 per unit of turnover.

---

## 4. Ichimoku Kumo Break Strategy

### Q: What is the Ichimoku Cloud and why use it?

**A:** Ichimoku Kinko Hyo ("one-glance equilibrium chart") is a complete trading system developed by Goichi Hosoda in 1968. It provides support/resistance (cloud), trend direction (Tenkan/Kijun), momentum confirmation (Chikou), and future structure (forward-projected cloud) — all in one indicator. It's particularly suited for trending markets and has well-defined entry/exit rules from Hosoda's original work and B.M. Sadekar's books.

### Q: Walk me through the 7-point entry checklist

**A:** All must be true simultaneously for a long entry:

1. **Price breaks above Kumo** — `close[t] > kumo_top[t]` AND `close[t-1] ≤ kumo_top[t-1]`. The breakout must be fresh.
2. **Future Kumo is bullish** — Senkou A > Senkou B (unshifted), meaning the cloud shape 26 bars ahead is green. Confirms structural bullishness.
3. **Chikou is free and clear** — `close[t] > max(highs in ±2 bars around t-26)` AND `close[t]` is above the historical Kumo at t-26. Chikou must have "room to run" without hitting past price action.
4. **Tenkan > Kijun** — fast momentum line above slow equilibrium line.
5. **Price above both lines** — `close > Tenkan AND close > Kijun`. Not just the cross, but price must be above both.
6. **Not over-extended** — `(close - Kijun) / ATR < 5.0`. Prevents chasing when price is too far from equilibrium.
7. **Senkou B flat filter** — if SB hasn't moved >0.1% over 15 bars (flat cloud = magnet zone), require candle body ≥ 0.5 × ATR to confirm genuine breakout, not noise.

### Q: Why check Chikou "free and clear" with a ±2 window?

**A:** The Chikou span plots current close 26 bars back. If it's tangled in past price action, the breakout has overhead resistance from historical supply. The ±2 window (checking bars at t-24 through t-28) accounts for the fact that support/resistance isn't a precise line — it's a zone. This is directly from Sadekar's methodology.

### Q: What's the Senkou B flatness filter and why does it matter?

**A:** Senkou B is a 52-period high-low midpoint. When it's flat (std < 0.1% over 15 bars), it means price has been range-bound for an extended period. A flat SB acts as a "magnet" — price tends to get pulled back into the cloud. To overcome this, we require a strong candle (body ≥ 0.5 × ATR) to confirm genuine breakout momentum, not just a wick poke.

### Q: Explain the 9-period vs 26-period high entry logic

**A:** The buy-stop goes at the 9-period high by default, which ensures the Tenkan-sen gets pulled upward on fill. But if the 26-period high is within 1.5 × ATR of the 9-period high, we use the 26-period high instead — this pulls the Kijun-sen up too, confirming both lines. It's a microstructure refinement that reduces false breakouts near consolidation tops.

### Q: How does the trailing stop work?

**A:** Initial stop: `Kijun[entry] - 1.5 × ATR[entry]`. Then every bar: `stop = max(current_stop, Kijun[t] - 1.5 × ATR[t])`. The key properties:

- **Ratchets up only** (asymmetric) — can never widen
- **Tracks the Kijun** — as Kijun rises in a trend, the stop follows
- **ATR buffer** — prevents whipsaw from normal volatility
- Exit triggers: gap stop (open ≤ stop), hard stop (low ≤ stop), or Kijun close (close < Kijun)

---

## 5. Ichimoku T/K Cross Strategy

### Q: How does T/K Cross differ from Kumo Break?

**A:** Two key differences:

1. **Signal trigger** — Kumo Break enters when price breaks above the cloud. T/K Cross enters when Tenkan crosses above Kijun (faster, more signals).
2. **No price-vs-Kumo requirement** — T/K Cross doesn't require price to be above the cloud at entry. The cross can happen below or inside the cloud. Sadekar gives this "less importance" than the Kumo Break but it catches trends earlier.

Everything else (Chikou check, future Kumo, stop mechanics, trailing logic) is identical.

### Q: What is cross strength and why track it?

**A:** Each T/K cross is classified:

- **Strong** — cross occurs with price above Kumo (bullish alignment)
- **Neutral** — cross occurs inside Kumo (ambiguous)
- **Weak** — cross occurs below Kumo (counter-trend)

This is recorded in `trade.extra["cross_strength"]` for post-analysis. The data shows strong crosses have significantly better win rates. It's also used as a feature for the ML filter.

### Q: Why does T/K Cross outperform Kumo Break in your data?

**A:** T/K Cross produced 991 trades (vs 837) with higher win rate (42.1% vs 39.2%) and better profit factor (2.34 vs 2.01). The likely reason: T/K Cross catches trends earlier — the Tenkan/Kijun cross happens before price breaks the cloud. By the time price breaks the cloud (Kumo Break signal), the easy part of the move is already done. The trade-off is that T/K Cross takes more "weak" signals, but the trailing stop mechanism manages the downside effectively.

---

## 6. EMA Crossover Strategy

### Q: Walk me through the EMA Crossover entry logic

**A:** Four conditions must all be true:

1. **Golden cross** — EMA(20) crosses above EMA(50). The fast line overtaking the slow line confirms short-term momentum shift.
2. **Price above both EMAs** — close > EMA(20) AND close > EMA(50). Prevents entering during a cross that happens below current price.
3. **ADX > 20** — Average Directional Index confirms the market is trending, not range-bound. ADX measures trend *strength* regardless of direction.
4. **Volume surge** — volume > 1.2 × 20-day volume MA. Ensures the move has institutional participation.

### Q: Why use ADX as a filter? How is it computed?

**A:** ADX measures trend strength (0–100) without directional bias. Computation:

1. Directional movement: `+DM = High[t] - High[t-1]` if positive and > `-DM`, else 0
2. Smoothed +DI and -DI over 14 periods
3. `DX = 100 × |+DI - -DI| / (+DI + -DI)`
4. `ADX = smoothed DX`

ADX > 20 means "trending." Below 20 is choppy/sideways where crossover signals produce whipsaws. This filter alone eliminates a large portion of losing trades in range-bound periods.

### Q: Why 1.2× volume threshold (vs 1.5× in Bollinger)?

**A:** EMA crossovers are trend-continuation signals — the trend is already established. A moderate volume confirmation (1.2×) is sufficient because we're confirming participation, not detecting a regime change. Bollinger breakouts need 1.5× because they're volatility expansion signals from a squeeze — that's a more dramatic event requiring stronger confirmation.

---

## 7. RSI Mean Reversion Strategy

### Q: How does mean reversion differ from trend-following in your framework?

**A:** Fundamental differences:

| Aspect | Trend-Following (6 strategies) | Mean Reversion (RSI) |
|--------|-------------------------------|---------------------|
| Entry timing | Buy-stop (confirm breakout) | Market order at next open (immediate) |
| Win rate target | ~40% (low WR, high RR) | Higher WR expected (buying oversold) |
| Exit strategy | Trailing stop (let winners run) | Fixed R-multiple target + time stop |
| Hold period | Unlimited (ride the trend) | Capped at 40 bars |
| Market regime | Works in trends | Works in mean-reverting ranges |

### Q: Explain the R-multiple targeting system

**A:** R = risk per trade = entry price - stop price. The profit target is set at 2R above entry: `target = entry + 2 × (entry - stop)`. This creates a fixed reward-to-risk ratio of 2:1. If you have a 40% win rate with 2R rewards, your expectancy is: `E = 0.4 × 2R - 0.6 × 1R = 0.2R` per trade — positive. This is Van Tharp's position-sizing methodology.

### Q: Why the SMA(200) filter?

**A:** We only buy oversold bounces when the long-term trend is up (close > SMA 200). Buying oversold in a bear market is catching a falling knife — the "oversold" condition can persist for weeks. The 200-day SMA filter ensures we're buying dips in an uptrend, not averaging down in a crash.

### Q: Why a 40-bar time stop?

**A:** Mean reversion trades have a built-in time decay — if the bounce hasn't happened in ~2 months, the thesis is dead. The stock likely isn't mean-reverting; it's in a new regime. Time stops prevent capital lockup in zombie positions and force reallocation to better opportunities.

---

## 8. Bollinger Breakout Strategy

### Q: What is a Bollinger squeeze and why is it predictive?

**A:** Bollinger Bands expand and contract with volatility. A "squeeze" = bandwidth in the bottom 20th percentile of its 120-bar history. Why it's predictive:

1. **Volatility is mean-reverting** — low vol periods are followed by high vol periods (empirically proven across all markets)
2. **Physics analogy** — compressed spring stores energy; compressed volatility stores directional energy
3. **Institutional explanation** — squeezes often occur during accumulation phases before breakouts

### Q: How do you compute bandwidth percentile?

**A:** `bandwidth = (upper - lower) / middle`. Then rank the current bandwidth against the last 120 bars: `percentile = rank_within_120_bars / 120`. If percentile ≤ 20, it's a squeeze. This is objective and normalized — a squeeze on NABIL (large cap) has the same statistical meaning as a squeeze on a micro-cap, even though absolute ATR values differ by 10x.

### Q: Why require ADX rising (not just ADX > threshold)?

**A:** ADX > threshold would tell you "the market is trending." ADX *rising* tells you "the trend is accelerating." For breakouts from a squeeze, we need acceleration — the transition from contraction to expansion. A high but falling ADX means the existing trend is exhausting, which is the opposite of what we want.

---

## 9. MACD Signal Cross Strategy

### Q: Why require MACD < 0 at the cross?

**A:** This is the "buying the turn" logic. When MACD is below zero, the 12-day EMA is below the 26-day EMA — short-term momentum is weak. A bullish cross *below zero* means momentum is shifting from bearish to bullish. Crosses *above zero* are continuation signals that often come too late. The below-zero filter catches reversals earlier.

### Q: Why require 2 bars of rising histogram?

**A:** `hist[t] > hist[t-1] > hist[t-2]` ensures MACD is accelerating, not just crossing. A single-bar cross can be noise — the histogram might flip positive for one bar and reverse. Two bars of rising histogram shows sustained momentum buildup. This is Gerald Appel's recommended confirmation from his original MACD work.

### Q: Why use SMA(50) for the trailing stop instead of an EMA?

**A:** SMA is smoother than EMA (less reactive to recent bars). For a trailing stop, you want stability — the stop should track the trend's "center of mass," not whipsaw with every volatile bar. EMA-based stops would tighten too aggressively during pullbacks within a trend, causing premature exits.

---

## 10. Donchian Breakout Strategy

### Q: What is the Turtle Trading System and how does your implementation differ?

**A:** The Turtle Trading System (Richard Dennis, 1983) used Donchian channel breakouts: buy at 20-day high, sell at 10-day low, with ATR-based stops. My implementation follows System 1 closely:

- 20-period entry channel, 10-period exit channel
- 2 × ATR initial stop (original Turtles used 2N where N=ATR)
- S1 filter: skip signal after a winning trade

Key differences from original Turtles:

- No pyramiding (adding to winners) — kept simple for NEPSE's liquidity
- 0.5% friction (Turtles traded futures with lower friction)
- 3-bar order timeout (Turtles used immediate fills)

### Q: Explain the S1 filter. Why skip signals after winners?

**A:** After a profitable trade, the same channel often produces another breakout signal immediately (the trend continues). The S1 filter skips this signal because:

1. **Diversification** — forces the system to wait for a fresh setup, reducing correlation between consecutive trades
2. **Mean reversion of streaks** — after a winner, the next signal is statistically more likely to be a false breakout (the easy money was already captured)
3. **Risk management** — prevents overexposure to one ticker/sector in momentum bursts

The filter resets after one skip: the next signal (whether it would have been a winner or loser) is taken.

### Q: Why asymmetric channels (20-entry, 10-exit)?

**A:** Entry channel (20) is wider → requires a stronger breakout to enter. Exit channel (10) is narrower → exits faster when the trend weakens. This creates an asymmetric payoff: slow to enter (fewer false signals), fast to exit (protect profits). The Turtles' original research showed 20/10 outperformed 20/20 or 10/10.

---

## 11. LightGBM Signal Quality Filter

### Q: What's the architecture of your ML layer?

**A:** It's a **signal quality filter**, not a direct predictor:

```
Rule-Based Signal → 38-Feature Snapshot → LightGBM → Quality Score → Accept/Skip
```

The model doesn't predict price direction — it predicts whether a *given strategy signal* will be profitable. This is a fundamentally easier problem because:

1. The signal already encodes directional conviction
2. The model only needs to learn *which market contexts* make the signal reliable
3. The base rate is the strategy's win rate (~40%), not 50/50

### Q: Why Huber loss instead of MSE?

**A:** Trade PnL distributions are heavy-tailed — a few trades make +100% while most make ±5%. MSE squares errors, so a single +100% outlier dominates the loss landscape. Huber loss is quadratic for small errors and linear for large errors: `L = 0.5x² if |x| < δ, else δ|x| - 0.5δ²`. This prevents outlier trades from distorting the model's predictions for typical trades. We also clip targets to [-30%, +30%].

### Q: Walk me through your walk-forward CV

**A:**

1. **Initial split** — train on all signals before 2024-07-01, test after
2. **Monthly refit** — in the OOS period, retrain every 3 calendar months using all data before the current window
3. **Purged gap** — 30-day gap between train and test sets at each fold boundary
4. **Why purge?** — trade signals can overlap in time (a signal on June 15 may predict PnL through July 15). Without the gap, the model could see the outcome of a training trade that overlaps with test trades.

### Q: What regularization do you use and why?

**A:** Multiple layers:

- `num_leaves=15` — shallow trees prevent memorizing individual trade patterns
- `min_child_samples=50` — each leaf needs ≥50 signals, forcing generalization over small clusters
- `subsample=0.7, colsample_bytree=0.6` — row and column bagging for ensemble diversity
- `reg_alpha=1.0` (L1), `reg_lambda=5.0` (L2) — explicit weight regularization
- `max_depth=4` — hard cap on tree depth

The key trade-off: we have relatively few training signals (~800 trades) compared to 38 features. Aggressive regularization prevents overfitting to specific trade configurations.

### Q: How much does the ML filter improve performance?

**A:** Taking only signals in the top 25th percentile by predicted PnL:

- Win rate: 36% → 41% (Kumo Break baseline → filtered)
- Profit factor: 1.37 → 1.73
- Fewer total trades (more selective), but higher average PnL per trade

The filter works because it learns context-dependent signal quality — e.g., a Kumo Break signal with high ADX, rising volume, and thick cloud is more reliable than one with flat ADX and thin cloud.

### Q: What's Information Coefficient (IC) and why do you use it?

**A:** IC = Spearman rank correlation between predicted quality scores and actual trade PnL. It measures *ranking accuracy*, not absolute prediction accuracy. For a signal filter, ranking is what matters — you want the model to correctly rank "good signal" above "bad signal," not predict exact PnL. IC of 0.05–0.10 is considered strong in quantitative finance.

---

## 12. Cross-Sectional Ranking Engine

### Q: How does the cross-sectional ranker differ from the signal filter?

**A:** The signal filter works *within* a single strategy: "should I take this Kumo Break signal?" The cross-sectional ranker works *across* all assets: "which 5 stocks should I hold for the next 5 days?"

Architecture:

- 10-dimensional feature set (trend, momentum, breakout, microstructure)
- Predicts 5-day forward log returns: `log(Close[t+5] / Close[t])`
- Non-overlapping 5-day holding windows
- Selects Top-K=5 assets per rebalance
- Benchmarked against equal-weight portfolio

### Q: Why 5-day forward returns as the target?

**A:** Five days is the sweet spot:

1. **Not too short** — 1-day returns are mostly noise; signal-to-noise ratio is too low for ML
2. **Not too long** — 20+ day returns introduce regime changes that invalidate features
3. **Practical** — weekly rebalancing is feasible for a retail NEPSE trader (not too many transactions)
4. **Non-overlapping** — eliminates autocorrelation in the target variable

### Q: What's the benchmark and how do you measure alpha?

**A:** Benchmark = equal-weight average return across all assets in the universe. Alpha = strategy return - benchmark return per rebalance. IC is computed at rebalance dates only (not daily) to measure actionable prediction skill. This is the standard quant fund evaluation — are your predictions better than "buy everything equally?"

---

## 13. Reinforcement Learning — PPO Portfolio Agent

### Q: Describe the state space, action space, and reward function

**A:**

**State (132 dimensions):**

- 11 features × 10 assets = 110 (CLV, stochastics, ATR, BB position, CMF, drawdown, EMA ribbon)
- 11 portfolio weight indicators (cash + 10 assets)
- 11 T+3 lock timers (countdown per slot)

**Actions (12 dimensions):**

- `[0:11]` — portfolio logits passed through `Softmax(temperature=2.0)` → target allocation weights
- `[11]` — macro veto signal: if < 0, liquidate everything to 100% cash

**Reward:**

- `R = ALPHA_SCALE × (portfolio_return - risk_parity_return) - friction`
- Pure *relative* reward — no absolute return component
- Friction = TAU × turnover (proportional to how much you rebalance)

### Q: What is T+3 action lock and why does it matter?

**A:** In NEPSE, when you buy a stock, you can't sell it for 3 trading days (T+3 settlement). The environment enforces this:

- On buy: set `lock_timer[asset] = 3`
- Each step: decrement all timers
- While `lock_timer > 0`: agent cannot reduce allocation to that asset

Without this, the agent learns to day-trade (buy/sell same day), which is physically impossible on NEPSE. The lock forces the agent to think in 3+ day swings and builds genuine conviction into positions.

### Q: Why use relative reward (vs risk parity) instead of absolute returns?

**A:** Absolute returns have a huge noise component — if the whole market goes up 3%, even a random portfolio returns 3%. The agent would learn to just "be invested" during bull markets and claim credit for beta. Relative reward against risk parity (inverse-volatility weighting) forces the agent to generate *alpha* — returns above what a naive diversification strategy produces. This is the standard approach in quant portfolio management.

### Q: Explain the macro veto mechanism

**A:** Action dimension 11 is a scalar. If < 0, the entire portfolio liquidates to cash regardless of individual asset signals. The agent learns to use this during bear markets or high-volatility regimes. It's a "risk-off" switch — much more effective than individually reducing each asset's weight because it's a single decisive action. The agent doesn't get penalized for going to cash; it just earns zero alpha while paying no friction.

### Q: How does walk-forward training work for the RL agent?

**A:**

1. **7 chronological folds** spanning 2019–2025
2. **Fold 1**: 5M timesteps (extensive initial learning on earliest data)
3. **Folds 2–7**: 1M timesteps each (rapid adaptation via transfer learning)
4. **Transfer**: neural network weights carry forward across folds — the agent doesn't start from scratch
5. **Temporal firewall**: train strictly before split date, evaluate after
6. **Random universe**: during training, each episode samples a random 10-asset subset to prevent memorization

### Q: What PPO hyperparameters did you use and why?

**A:**

- `learning_rate`: linear schedule 2e-4 → 1e-5 (anneal for stability)
- `n_steps=4096, batch_size=4096` — full-batch updates for stable GAE estimates
- `gamma=0.995` — long-horizon discount (portfolio decisions have multi-week consequences)
- `gae_lambda=0.95` — high lambda for low-bias advantage estimation
- `ent_coef=0.015` — moderate exploration encouragement
- `clip_range=0.2` — standard PPO clipping for trust region
- Network: `[512, 512, 256]` × 2 (separate policy and value heads)

---

## 14. RL — Hyperscale Single-Asset Agent

### Q: How does the single-asset agent differ from the portfolio agent?

**A:**

| Aspect | Portfolio v12 | Hyperscale v6 |
|--------|--------------|---------------|
| Universe | 10 assets, weight allocation | 1 random asset per episode |
| Action | 12D continuous (weights + veto) | 2D discrete (buy/sell/hold) |
| Reward | Relative to risk parity | Regime-conditioned alpha with opportunity cost |
| Envs | 16 SubprocVecEnv | 32 SubprocVecEnv |
| Network | [512,512,256] | [256,256,128] |
| Memory | Standard | Pre-compiled contiguous NumPy arrays (O(1) lookup) |

### Q: What is the regime-conditioned reward?

**A:** The drawdown penalty coefficient adapts to market regime:

```
κ(t) = κ_base × (1 - 0.5 × max(0, ribbon_align))
```

In confirmed bull trends (`ribbon_align > 0`), the drawdown penalty is halved. This lets the agent tolerate normal pullbacks during uptrends without panic-selling. In bearish regimes, full penalty applies. This is implicit regime detection — no explicit HMM or regime classifier needed.

### Q: What is opportunity cost and how do you model it?

**A:** When the agent sits in cash during a bull trend, it should be penalized for missing gains:

```
OC = OC_base × δ × ribbon_align × (1 + max(0, CMF))
```

Where δ = daily return of the asset, ribbon_align = EMA ribbon bullishness, CMF = Chaikin Money Flow. The opportunity cost only applies when:

1. The asset is going up (δ > 0)
2. The trend is bullish (ribbon_align > 0)
3. Volume confirms the move (CMF > 0)

This prevents the agent from learning to always sit in cash (a local optimum in noisy markets).

### Q: How do you achieve 1M timesteps/min throughput?

**A:** Pre-compiled NumPy arrays:

```python
ticker_arrays[tk][feature] = np.ascontiguousarray(data, dtype=np.float32)
```

All OHLCV + features are stored as contiguous float32 arrays at initialization. During `step()`, feature lookups are direct array indexing — no Pandas `.loc`, no string column names, no DataFrame overhead. With 32 parallel envs on 32 CPU cores, this achieves ~1M timesteps/min.

---

## 15. MDP Debugging & Reward Shaping

### Q: What was the pathological local optimum and how did you diagnose it?

**A:** The agent converged to: avg reward -0.75 to -0.95, action distribution 81% cash / 19% long. It learned that *taking any position guarantees penalties* and the optimal strategy is "never trade."

Diagnosis:

1. Plotted action distributions over training — saw cash-preference increasing monotonically
2. Decomposed reward into components — found the forced liquidation penalty was triple-stacking: log_return (loss) + TAU (friction) + 0.05 (penalty)
3. Checked feature distributions — found Z-scored features (range [-3,3]) dominated gradients over [0,1]-normalized features

### Q: Walk me through each fix

**A:**

**Fix 1 — Remove forced exit penalty:**
Before: trailing stop hits → reward = log_return - TAU - 0.05 (triple punishment)
After: trailing stop hits → reward = log_return - TAU (only actual loss + friction)
Rationale: trailing stops are *risk management*, not mistakes. Penalizing risk management teaches the agent to avoid taking risk entirely.

**Fix 2 — Feature standardization:**
Before: mixed scales (pct_k in [0,1], natr in [-3,3]). Z-scored features had 3× the gradient magnitude.
After: all features → (value - mean) / std, clipped to [-1, +1]. Uniform gradient flow.

**Fix 3 — PPO hyperparameters:**

- `ent_coef`: 0.05 → 0.005 (10× reduction). High entropy = random exploration = friction bleeds equity.
- `batch_size`: 256 → 512. Larger batches = smoother GAE estimates = more stable policy updates.
- `learning_rate`: 3e-4 → 1e-4. Less aggressive updates = fewer policy collapse events.

**Fix 4 — ATR multiplier:**

- 2.5 → 3.5. NEPSE has leptokurtic returns (fat tails, kurtosis >> 3). Tight stops get hit by normal volatility spikes, causing false exits. Widening the multiplier accommodates the tail risk.

### Q: What is entropy coefficient and why does it matter?

**A:** In PPO, the entropy bonus `H(π)` is added to the loss: `L = L_clip + ent_coef × H(π)`. Higher entropy = policy assigns more uniform probabilities = more exploration. Too high (0.05): agent randomizes too much, bleeding friction on unnecessary trades. Too low: agent exploits immediately and gets stuck in local optima. 0.005 was the sweet spot — enough exploration to discover profitable patterns without excessive random trading.

### Q: What is leptokurtosis and why does it affect stop-loss design?

**A:** Leptokurtic distributions have fatter tails and higher peaks than normal distributions (kurtosis > 3). NEPSE daily returns are leptokurtic — extreme moves (±5% days) happen more often than a Gaussian model predicts. With ATR_MULT=2.5, the trailing stop is set too tight — it gets triggered by "normal" extreme moves that aren't actually trend reversals. ATR_MULT=3.5 gives the stop room to accommodate these fat-tailed returns.

---

## 16. Backtesting Methodology & Bias Prevention

### Q: What biases can affect backtesting? How do you prevent each?

**A:**

| Bias | Definition | Prevention |
|------|-----------|------------|
| **Lookahead** | Using future data in current decisions | All indicators use rolling windows; signals at close t → entry at t+1; Ichimoku displacement is backward-looking |
| **Survivorship** | Only testing on stocks that survived to today | Use full NEPSE universe including delisted tickers (API provides historical data) |
| **Overfitting** | Optimizing parameters to fit historical data | Use standard indicator parameters (9/26/52 from Hosoda); OOS validation; walk-forward CV |
| **Selection** | Cherry-picking the best strategy after testing many | Report ALL 7 strategies head-to-head; framework forces identical friction and rules |
| **Transaction cost** | Ignoring real-world friction | 0.5% per trade baked into PnL; portfolio sim uses 1.5% |
| **Slippage** | Assuming fills at exact prices | Buy-stop orders with gap-up fills at open (realistic); market impact modeled via higher friction |

### Q: How do you handle order execution realistically?

**A:** Three mechanisms:

1. **Buy-stop orders** — entry at a higher level (not at signal price), confirming breakout
2. **Gap handling** — if open gaps above the buy-stop level, fill at open (not at the stop level). This is conservative.
3. **5-bar timeout** — if the buy-stop isn't hit in 5 bars, cancel the order. The opportunity has passed.
4. **Gap-down stops** — if open gaps below the trailing stop, exit at open (worst price), not at stop level

### Q: What's the difference between in-sample and out-of-sample in your framework?

**A:** Split date: 2024-07-01. All trades with entry before this date are in-sample (used for parameter selection and strategy development). All trades after are out-of-sample (never seen during development). The OOS results are the true performance estimate. For the ML layer, there's an additional 30-day purged gap to prevent information leakage.

---

## 17. Risk Management & Position Sizing

### Q: How do ATR-based trailing stops work?

**A:** ATR (Average True Range, 14 periods) measures "normal" volatility. The stop is set at `reference_level - ATR_MULT × ATR`. This means:

- In volatile stocks, stops are wider (more room to breathe)
- In calm stocks, stops are tighter (less noise to accommodate)
- The stop adapts to each stock's individual volatility profile

The "reference level" varies by strategy: Kijun-sen (Ichimoku), Slow EMA (EMA crossover), Middle Band (Bollinger), SMA(50) (MACD), 10-period low (Donchian).

### Q: What's the difference between gap stop, hard stop, and signal exit?

**A:**

1. **Gap stop** — open[t] ≤ stop_px. Price gaps below the stop overnight. Exit at open (worst case, no chance to exit at stop level). Happens in ~0.1% of trades but can cause large losses.
2. **Hard stop** — low[t] ≤ stop_px but open > stop. Intraday the price touches the stop level. Exit at stop_px (assumes a limit order was resting).
3. **Signal exit** — close[t] violates a condition (close < Kijun, close < EMA, MACD bearish cross). Exit at close. This is the most common exit type.

### Q: How does the portfolio simulator handle concurrent positions?

**A:** The simulator allows up to 8 concurrent trades:

- Each trade gets a fixed allocation (Rs 1,25,000 in ML version or portfolio_value / 8 in compounding mode)
- When a slot opens (trade exits), the next highest-quality signal fills it
- Mark-to-market: positions are valued daily using linear interpolation between entry and exit prices
- Equity curve tracks: base_capital + cash + sum(unrealized_pnl)

---

## 18. Testing & Software Engineering

### Q: How do you test trading strategies? You can't unit-test the market

**A:** 99 pytest cases using synthetic data:

1. **Synthetic OHLCV generator** — `make_ohlcv()` creates controllable price series with specified trend, volatility, and seed
2. **Indicator tests** — verify Tenkan, Kijun, Senkou, ATR computations against known inputs
3. **Signal tests** — inject specific price patterns that should/shouldn't trigger signals
4. **Portfolio tests** — verify equity curve computation, drawdown tracking, position sizing
5. **Integration tests** — run full strategy backtest on synthetic data, verify trade records
6. **Metrics tests** — verify win rate, profit factor, expectancy calculations with known trade sets

### Q: Why use synthetic data instead of real historical data for tests?

**A:** Three reasons:

1. **Determinism** — synthetic data with fixed seed produces identical results every run
2. **Controllability** — can create specific scenarios (trending, mean-reverting, gaps, data holes)
3. **Speed** — tests run in <60 seconds without downloading or reading large CSV files
4. **Independence** — tests don't break when new data is fetched or data format changes

### Q: What design patterns does your codebase use?

**A:**

- **Abstract Base Class** — `BacktestStrategy` ABC in `src/rbs/base.py` enforces consistent interface
- **Dataclass records** — `TradeRecord`, `IchimokuParams` for type-safe parameter passing
- **State machine** — `FLAT → PENDING → POSITION` lifecycle for each ticker
- **Strategy pattern** — each strategy overrides signal/exit logic while sharing execution/stop mechanics
- **Fixture injection** — pytest conftest provides reusable synthetic data generators

---

## 19. Market Microstructure — NEPSE Specifics

### Q: What makes NEPSE different from developed markets?

**A:**

1. **T+3 settlement** — can't sell for 3 days after buying; forces swing-trade minimum holding
2. **10% circuit breakers** — daily price movement capped at ±10%; prevents extreme gap moves but creates pinned-limit days
3. **Low liquidity** — many stocks trade <50 lots/day; large orders move prices
4. **Leptokurtic returns** — fatter tails than Gaussian; extreme days happen 3–5× more than developed markets
5. **No short-selling** — long-only constraint; can't profit from bear markets directly
6. **Limited instruments** — no derivatives, no ETFs, no options for hedging
7. **No institutional data feed** — must scrape nepsetrading.com (rate-limited, occasional gaps)

### Q: How do these constraints affect your strategy design?

**A:**

- **T+3**: RL agent has explicit lock timers; strategies use multi-day holding periods
- **Circuit breakers**: Gap-stop logic handles limit-down days where you can't exit at your stop price
- **Low liquidity**: No pyramiding (adding to positions); fixed position sizing
- **Fat tails**: ATR multiplier 3.5 (wider than the standard 2.0) for stop-loss
- **Long-only**: All strategies are buy-only; no short signals
- **No hedging**: Portfolio risk managed through position sizing and cash allocation only

---

## 20. Statistics & Evaluation Metrics

### Q: Define and interpret each metric you report

**A:**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Win Rate** | Winners / Total Trades | % of profitable trades. Sub-50% is fine if winners >> losers |
| **Profit Factor** | Gross Profits / Gross Losses | >1 = profitable. >2 = strong. Your 2.34 is excellent |
| **Expectancy** | (WR × AvgWin) - ((1-WR) × AvgLoss) | Average profit per trade. Positive = edge exists |
| **Max Drawdown** | Largest peak-to-trough decline in equity | Worst-case capital erosion. Your -18.2% is manageable |
| **Sharpe Ratio** | Mean(returns) / Std(returns) | Risk-adjusted return. >1 = good, >2 = excellent |
| **Calmar Ratio** | Annual Return / Max Drawdown | Return per unit of drawdown risk |
| **Spearman IC** | Rank correlation of predictions vs actuals | Measures ranking skill. 0.05–0.10 is strong in quant finance |
| **Profit per trade** | Total PnL / Total Trades | Average gain per trade including losers |

### Q: What's the difference between Sharpe and profit factor?

**A:** Sharpe measures *consistency* of returns (penalizes variance). Profit factor measures *magnitude* of wins vs losses (ignores timing). A strategy can have high PF but low Sharpe if gains are concentrated in a few large trades with many flat periods. Conversely, a low PF with high Sharpe means small but consistent gains. For systematic trading, both matter — PF confirms edge exists, Sharpe confirms it's tradeable.

### Q: How do you know if a profit factor of 2.34 is statistically significant?

**A:** Several checks:

1. **Sample size** — 991 trades is large enough for the law of large numbers to apply
2. **OOS consistency** — PF holds at 1.65 out-of-sample (not just in-sample artifact)
3. **Cross-strategy agreement** — both Ichimoku strategies show PF > 2.0
4. **Monte Carlo** — can shuffle trade order 10,000 times and verify PF > 1.0 in >95% of permutations (not implemented yet but straightforward to add)
5. **Regime robustness** — the strategy works across multiple year-regimes (bull 2020-2021, bear 2022, recovery 2023-2024)

### Q: What is walk-forward analysis and why is it the gold standard?

**A:** Walk-forward:

1. Train model on data up to date T
2. Test on data from T to T+Δ
3. Advance T by Δ, retrain, repeat

Why it's superior to simple train/test split:

- **No future contamination** — model always trained on past, tested on future
- **Adaptation** — model gets updated with recent data (market regimes change)
- **Multiple test periods** — instead of one lucky/unlucky OOS period, you get many
- **Realistic** — this is how you'd actually deploy the model (retrain monthly, trade next month)

---

## Rapid-Fire Questions (Expect These)

1. **"What would you do differently?"** — Add position sizing based on Kelly criterion, implement Monte Carlo simulation for confidence intervals, test on other emerging markets (DSE Bangladesh, CSE Sri Lanka).

2. **"Why not deep learning for price prediction?"** — For 441 tickers with ~3,000 bars each, we have ~1.3M data points. Not enough for LSTM/Transformer to outperform LightGBM. Gradient boosting is optimal for tabular data with <100 features (see Grinsztajn et al. 2022).

3. **"How would you deploy this?"** — Daily cron job: fetch new OHLCV → run signal detection → ML filter → rank by expected quality → execute top signals at market open via broker API. Risk monitor: track real-time equity curve, halt if drawdown exceeds -25%.

4. **"What's the biggest risk?"** — Regime change. These strategies work in trending/mean-reverting markets. A structural market change (new regulations, foreign investor influx changing microstructure) could invalidate the edge. Mitigation: walk-forward retraining, regime detection via ADX/volatility clustering, macro veto in RL agent.

5. **"Can you scale this to a larger market?"** — Yes, with modifications. Remove NEPSE-specific constraints (T+3 lock, circuit breaker handling), add short-selling capability, replace buy-stop with market orders for liquid markets, reduce ATR multiplier (developed markets are less leptokurtic). The framework architecture (ABC strategies, ML filter, RL agent) is market-agnostic.

6. **"What's your edge? Why would this work when most quant strategies fail?"** — Three edges: (a) market inefficiency — NEPSE has minimal institutional coverage, so systematic approaches capture alpha that's already arbitraged away in the S&P 500; (b) rigorous methodology — lookahead-free features, realistic friction, OOS validation; (c) diversification across strategy types — trend-following (6 strategies) + mean reversion (RSI) + ML filtering reduces strategy-specific risk.

7. **"Explain overfitting in the context of backtesting."** — Overfitting = the model learns patterns specific to historical data that won't repeat. Signs: high in-sample performance but poor OOS, extreme parameter sensitivity, strategy only works on the specific tickers tested. Prevention: standard parameters (not optimized), walk-forward CV, cross-strategy consistency checks, large trade sample sizes (991 trades).

8. **"What libraries did you use and why?"** — Pandas/NumPy for data (ubiquitous, well-tested); LightGBM for ML (fastest GBDT, handles categorical features, Huber loss built-in); Stable-Baselines3 for RL (well-tested PPO implementation, Gymnasium compatibility); pytest for testing (fixture injection, parametrized tests).
