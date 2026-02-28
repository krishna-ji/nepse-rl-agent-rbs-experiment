# NEPSE Universal RL Architecture: Stochastic Pullback Engine

Multi-asset Reinforcement Learning trading engine for the Nepal Stock Exchange (NEPSE).  
Fuses deterministic technical heuristics (Chandelier Exit, Stochastic Oscillator, Bollinger Bands) with PPO policy optimisation over a heterogeneous temporal dataset of 353+ tickers.

## Project Structure

```
notebooks/
  simple_rl.ipynb  – Self-contained end-to-end notebook (data → features → env → PPO → eval → plots)
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

Everything lives in `notebooks/simple_rl.ipynb` — a single self-contained notebook covering:

| Phase | Description |
|-------|-------------|
| 1 | Load NEPSE CSVs → universal DatetimeIndex MultiIndex DataFrame, warm-up padding (200-day lookback) |
| 2 | Stochastic %K/%D, NATR, BBW, SMA50/200, protected swing low, macro trend |
| 3-4 | `gym.Env` with Discrete(2) actions, Chandelier-Exit ATR trailing stop, forced liquidation override |
| 5 | Log-return reward, −0.015 transaction friction, −2.0 liquidation penalty |
| 6 | PPO (Stable-Baselines3), DummyVecEnv, checkpoint + eval callbacks |
| 7 | Equity curves, buy/sell signals, metrics summary |

## Key Design Decisions

- **No ta-lib dependency** – all indicators implemented in pure NumPy/Pandas for Python 3.12+ compatibility
- **NaN-safe observations** – gracefully handles tickers with sparse early history
- **Randomised episode sampling** – each `reset()` picks a random ticker + random start date to prevent overfitting
