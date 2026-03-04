# nepserl — Quantitative Trading Research Platform

**Time-series forecasting, rule-based strategies, ML signal filtering, and deep RL for the Nepal Stock Exchange.**

A research-grade framework for systematic trading on NEPSE — covering the full pipeline from data ingestion to portfolio-level backtesting. Implements 7 rule-based strategies, LightGBM signal quality models, and PPO-based reinforcement learning agents across 441 tickers and 14 years of daily OHLCV data (130k+ bars).

---

## Skills Demonstrated

| Domain | What This Project Shows |
|--------|------------------------|
| **Time-Series Forecasting** | Walk-forward cross-validation, 5-day forward return prediction, regime-aware feature engineering |
| **Feature Engineering** | 38 hand-crafted features across 6 groups (Ichimoku, trend, momentum, volatility, volume, candlestick) — all pure NumPy/Pandas, no TA-Lib |
| **Machine Learning** | LightGBM regressors with Huber loss, Spearman IC evaluation, quintile analysis, monthly refit schedules |
| **Deep Reinforcement Learning** | PPO with Gymnasium, multi-fold transfer learning, regime-conditioned rewards, T+3 action locks |
| **Backtesting & Risk** | Portfolio simulation with realistic friction (0.5%), ATR-based trailing stops, max drawdown tracking, Sharpe/Calmar ratios |
| **Software Engineering** | Modular strategy framework (ABC pattern), 99 pytest cases, parallel data pipelines, CLI interface |

---

## Project Stats

| Metric | Value |
|--------|-------|
| NEPSE Tickers | 441 (API universe) · 112 downloaded · 79 backtested |
| Data Span | 2012 – 2026 · 130,990 daily OHLCV bars · up to 3,231 bars/ticker |
| Rule-Based Strategies | 7 (Ichimoku ×2, EMA, RSI, Bollinger, MACD, Donchian) |
| ML Models | 2 (signal quality filter, cross-sectional ranker) |
| RL Agent Versions | 3 (hyperscale GPU, portfolio walk-forward, MDP-fixed) |
| Feature Dimensionality | 38 (LightGBM) · 12 (RL state) |
| Test Cases | 99 (pytest, all passing) |
| Lines of Code | 15,153 across 46 files (src: 1,664 · labs: 12,591 · tests: 898) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│  src/nepsetrading/  → Scraper, parallel OHLCV fetch     │
│  441 tickers · 14 years · 130k+ daily OHLCV bars        │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌───────────┐ ┌────────────┐
│  RULE-BASED  │ │  ML/LGBM  │ │  DEEP RL   │
│  7 strategies│ │  38 feats │ │  PPO agent │
│  ATR stops   │ │  walk-fwd │ │  16 envs   │
│  src/rbs/    │ │  labs/lgbm│ │  labs/rl/  │
└──────┬───────┘ └─────┬─────┘ └─────┬──────┘
       │               │             │
       └───────────────┼─────────────┘
                       ▼
          ┌─────────────────────────┐
          │   PORTFOLIO BACKTEST    │
          │  Equity curves · Sharpe │
          │  Drawdown · Win rate    │
          │  Yearly/monthly splits  │
          └─────────────────────────┘
```

---

## Strategy Lab — 7 Head-to-Head Systems

| # | Strategy | Entry Logic | Stop / Exit | Source |
|---|----------|-------------|-------------|--------|
| 1 | **Ichimoku Kumo Break** | Price closes above cloud | Kijun ± ATR(3.5) trail | [labs/rbs/ichimoku_kumo_break.py](labs/rbs/ichimoku_kumo_break.py) |
| 2 | **Ichimoku T/K Cross** | Tenkan crosses above Kijun | Close below Kijun | [labs/rbs/ichimoku_tk_cross.py](labs/rbs/ichimoku_tk_cross.py) |
| 3 | **EMA Crossover** | EMA(20) > EMA(50), ADX > 20 | Slow EMA ± ATR(1.5) trail | [labs/rbs/ema_crossover.py](labs/rbs/ema_crossover.py) |
| 4 | **RSI Mean Reversion** | RSI(14) bounces from oversold | Swing low ± ATR, 2.0R reward | [labs/rbs/rsi_mean_reversion.py](labs/rbs/rsi_mean_reversion.py) |
| 5 | **Bollinger Breakout** | Close above upper BB + squeeze | Middle band ± ATR trail | [labs/rbs/bollinger_breakout.py](labs/rbs/bollinger_breakout.py) |
| 6 | **MACD Signal Cross** | MACD crosses above signal | SMA(50) ± ATR trail | [labs/rbs/macd_cross.py](labs/rbs/macd_cross.py) |
| 7 | **Donchian Breakout** | Price above 20-period high | 10-period low ± ATR(2.0) | [labs/rbs/donchian_breakout.py](labs/rbs/donchian_breakout.py) |

All strategies share: 0.5% friction, long-only, buy-stop orders with 5-bar timeout, ATR-based trailing stops.

### Measured Backtest Results (79 tickers, 2012–2026)

| Strategy | Trades | Win Rate | Profit Factor | Expectancy | Max Drawdown |
|----------|--------|----------|---------------|------------|--------------|
| **Ichimoku Kumo Break** | 837 | 39.2% | 2.01 | +3.63% | -23.3% |
| **Ichimoku T/K Cross** | 991 | 42.1% | 2.34 | +4.51% | -18.2% |

Out-of-sample (≥ 2024-07-01): Kumo Break PF 1.57 / TK Cross PF 1.65 — positive expectancy holds on unseen data.

---

## ML Pipeline — LightGBM Signal Filtering

The ML layer filters raw strategy signals through a **38-feature LightGBM regressor** trained with walk-forward cross-validation:

**Feature Groups (38 total):**

| Group | Count | Examples |
|-------|-------|---------|
| Ichimoku | 11 | Cloud thickness, TK spread, Chikou vs. price, Senkou B flatness |
| Trend | 6 | SMA slope, price-to-SMA distance, ADX |
| Momentum | 6 | RSI, MACD histogram, Rate of Change |
| Volatility | 5 | ATR, Bollinger bandwidth, bandwidth squeeze percentile |
| Volume | 4 | OBV trend, volume surge %, CMF-20 |
| Candlestick | 4 | Body ratio, shadow ratio, CLV |
| Breakout | 2 | Donchian position, channel width |

**Training:** Huber loss, `min_child_samples=50`, `num_leaves=15`, monthly refit with median-split thresholds.  
**Evaluation:** Spearman IC, MSE, R², quintile P&L breakdown, regime-conditional analysis.

---

## Reinforcement Learning — PPO Agents

| Version | Key Innovation | Config |
|---------|---------------|--------|
| **Hyperscale** | 32 CPU + GPU, 12D orthogonal state space | Micro-structure + momentum + volatility + liquidity features |
| **Portfolio (v12)** | T+3 action lock, macro veto, relative momentum reward | 16 SubprocVecEnv, 5M timesteps fold-1, transfer learning |
| **MDP-Fixed** | Documented fixes for pathological optima | Reward topology + feature standardization + entropy tuning |

**MDP Debugging (documented in [docs/FIXES_SUMMARY.md](docs/FIXES_SUMMARY.md)):**

- Removed forced liquidation penalty — trailing stops are risk management, not punishment
- Standardized all features to [-1, +1] — fixed gradient dominance across feature scales
- Tuned entropy coefficient 0.05 → 0.005, batch size 256 → 512 for stable GAE estimates
- Adjusted ATR multiplier 2.5 → 3.5 for NEPSE's leptokurtic return distribution

---

## Technical Indicators — All Implemented from Scratch

No TA-Lib. Every indicator is pure NumPy/Pandas with **lookahead-safe** computation:

- **Ichimoku Cloud:** Tenkan, Kijun, Senkou A/B (displaced), Chikou, cloud thickness
- **Trend:** SMA(20/50/200), EMA(12/26), slope detection, ADX
- **Momentum:** RSI(14), MACD(12,26,9), Stochastic %K/%D, Rate of Change
- **Volatility:** ATR(14), Bollinger Bands(20,2.0), bandwidth squeeze detection
- **Volume:** OBV, volume surge %, Chaikin Money Flow
- **Breakout:** Donchian channels, Chandelier Exits
- **Candlestick:** Body/shadow ratios, Close-Location Value

---

## Quick Start

```bash
# Install (requires uv — https://docs.astral.sh/uv/)
uv sync

# Run tests
uv run pytest tests/ -v

# Run a rule-based strategy backtest
uv run python labs/rbs/ichimoku_kumo_break.py

# Run LightGBM signal filtering
uv run python labs/lgbm/ichimoku_lgbm.py

# Run RL training
uv run python labs/rl/nepserl_portfolio.py
```

---

## Project Structure

```
src/
  nepsetrading/          Data pipeline — scraper, parallel OHLCV fetch
  rbs/                   Strategy framework — base ABC, Ichimoku implementations
labs/
  rbs/                   7 rule-based strategy scripts (head-to-head backtests)
  lgbm/                  LightGBM signal filtering & portfolio simulation
  rl/                    PPO agents (hyperscale, portfolio, MDP-fixed)
  utils/                 Export, diagnostics, training config
tests/
  rbs/                   99 test cases — strategies, metrics, portfolio, integration
docs/
  FIXES_SUMMARY.md       MDP debugging documentation
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **RL** | Gymnasium, Stable-Baselines3, PyTorch (CUDA 12.4) |
| **ML** | LightGBM, scikit-learn, SciPy |
| **Data** | Pandas, NumPy, Requests (parallel fetching) |
| **Viz** | Matplotlib, Rich (CLI tables) |
| **Testing** | pytest (conftest fixtures, synthetic data generators) |
| **Tooling** | Python 3.12+, uv package manager |

---

## Key Design Decisions

- **No TA-Lib** — all indicators in pure NumPy/Pandas for portability and auditability
- **Lookahead-safe** — Ichimoku displacement, entry at bar-close → execution at bar+1, EOD-only trailing stops
- **NaN-safe observations** — handles tickers with sparse early history without crashing
- **Randomized episode sampling** — each RL `reset()` picks a random ticker + start date to prevent overfitting
- **Realistic friction** — 0.5% per trade, T+3 settlement locks, gap-down forced exits

---

## License

MIT
