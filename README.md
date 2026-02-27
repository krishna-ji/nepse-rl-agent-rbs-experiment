# NEPSE Universal RL Architecture: Stochastic Pullback Engine

Multi-asset Reinforcement Learning trading engine for the Nepal Stock Exchange (NEPSE).  
Fuses deterministic technical heuristics (Chandelier Exit, Stochastic Oscillator, Bollinger Bands) with PPO policy optimisation over a heterogeneous temporal dataset of 353+ tickers.

## Project Structure

```
src/
  data_loader.py   – Phase 1: Universal date-aligned MultiIndex loader
  features.py      – Phase 2: Vectorised feature engineering (Stochastic, ATR, BBW, …)
  environment.py   – Phase 3-5: Custom Gymnasium env with Chandelier-Exit TSL
  trainer.py       – Phase 6: PPO training harness (Stable-Baselines3)
  visualize.py     – Phase 7: Plotly diagnostic candlestick + overlay charts
runs/
  train.py         – Full training script
  evaluate.py      – Out-of-sample evaluation & HTML report generation
  quick_test.py    – Smoke test (≈ 20 s) to verify the pipeline
data/stocks/       – OHLCV CSVs (636 files, 555 with data)
```

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Smoke test – verifies data → features → env → PPO → eval → Plotly
uv run python main.py test

# Full training (≈ 500 K timesteps, ~15-30 min on CPU)
uv run python main.py train --timesteps 500000

# Evaluate a trained model on a specific ticker
uv run python main.py eval --ticker NABIL

# Evaluate on 10 random tickers
uv run python main.py eval --multi 10
```

## Architecture Overview

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `data_loader` | Loads 636 NEPSE CSVs → universal DatetimeIndex MultiIndex DataFrame, warm-up padding (200-day SMA lookback) |
| 2 | `features` | Stochastic %K/%D, NATR, BBW, SMA50/200, protected swing low, macro trend |
| 3-4 | `environment` | `gym.Env` with Discrete(2) actions, Chandelier-Exit ATR trailing stop, forced liquidation override |
| 5 | `environment` | Log-return reward, −0.015 transaction friction, −2.0 liquidation penalty |
| 6 | `trainer` | PPO (Stable-Baselines3), DummyVecEnv, checkpoint + eval callbacks |
| 7 | `visualize` | Interactive Plotly: candlestick, SMA200, swing low, TSL, buy/sell markers, stochastic subplot |

## Key Design Decisions

- **No ta-lib dependency** – all indicators implemented in pure NumPy/Pandas for Python 3.12+ compatibility
- **NaN-safe observations** – gracefully handles tickers with sparse early history
- **Randomised episode sampling** – each `reset()` picks a random ticker + random start date to prevent overfitting
