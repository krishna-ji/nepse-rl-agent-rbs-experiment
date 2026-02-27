"""
Phase 6 (cont.) – Evaluation & Diagnostic Visualisation
========================================================
• Deterministic out-of-sample roll-out
• Plotly interactive candlestick + overlays + momentum subplot
• Exported as a self-contained HTML artefact
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.environment import UniversalNepseEnv, run_deterministic_episode
from src.run_manager import get_logger


# ── Trajectory analysis helpers ───────────────────────────────────────────

def compute_metrics(traj: pd.DataFrame) -> dict:
    """Compute standard backtest statistics from a trajectory."""
    pv = traj["portfolio_value"].values
    total_return = pv[-1] / pv[0] - 1.0
    log_rets = np.diff(np.log(pv + 1e-10))
    sharpe = (np.mean(log_rets) / (np.std(log_rets) + 1e-10)) * np.sqrt(252)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / (running_max + 1e-10)
    max_dd = drawdowns.min()

    # Trade count
    actions = traj["action"].values
    positions = traj["position"].values
    buys = int(((positions[:-1] == 0) & (actions[:-1] == 1)).sum())
    sells_natural = int(((positions[:-1] == 1) & (actions[:-1] == 0)).sum())
    forced = int(traj.get("forced_liquidation", pd.Series([False])).sum())

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "num_buys": buys,
        "num_sells": sells_natural,
        "forced_liquidations": forced,
        "final_portfolio_value": pv[-1],
    }


# ── Plotly Visualisation ──────────────────────────────────────────────────

def plot_trajectory(
    traj: pd.DataFrame,
    feat_df: pd.DataFrame,
    ticker: str,
    output_path: str | pathlib.Path = "outputs/trajectory.html",
    show: bool = False,
) -> None:
    """Generate an interactive Plotly HTML with:
        - Candlestick OHLC
        - SMA200 & protected_swing_low overlays
        - Chandelier Exit TSL (dashed red)
        - Buy / Sell markers
        - %K / %D momentum subplot
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dates = traj["date"]
    close = traj["close"]

    # Pull OHLC from feature DataFrame
    ohlc = feat_df.loc[dates, ticker]
    o = ohlc["open"].values
    h = ohlc["high"].values
    l = ohlc["low"].values
    c = ohlc["close"].values

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[f"{ticker} – RL Trajectory", "Portfolio Value", "Stochastic %K / %D"],
    )

    # 1) Candlestick
    fig.add_trace(
        go.Candlestick(
            x=dates, open=o, high=h, low=l, close=c,
            name="OHLC",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1, col=1,
    )

    # SMA200 overlay
    sma200 = feat_df.loc[dates, (ticker, "sma200")].values
    fig.add_trace(
        go.Scatter(x=dates, y=sma200, mode="lines",
                   line=dict(color="blue", width=1, dash="dot"),
                   name="SMA 200"),
        row=1, col=1,
    )

    # Protected swing low
    psl = feat_df.loc[dates, (ticker, "protected_swing_low")].values
    fig.add_trace(
        go.Scatter(x=dates, y=psl, mode="lines",
                   line=dict(color="orange", width=1, dash="dash"),
                   name="Protected Swing Low"),
        row=1, col=1,
    )

    # TSL line (only while holding)
    tsl_vals = traj["tsl_level"].values
    fig.add_trace(
        go.Scatter(x=dates, y=tsl_vals, mode="lines",
                   line=dict(color="red", width=1.5, dash="dash"),
                   name="Chandelier TSL",
                   connectgaps=False),
        row=1, col=1,
    )

    # Buy markers (0→1)
    actions = traj["action"].values
    positions = traj["position"].values
    buy_mask = (positions == 0) & (actions == 1)
    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[buy_mask],
                y=l[buy_mask] * 0.98,
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="lime",
                            line=dict(width=1, color="darkgreen")),
                name="BUY",
            ),
            row=1, col=1,
        )

    # Sell markers (1→0)
    sell_mask = (positions == 1) & (actions == 0)
    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[sell_mask],
                y=h[sell_mask] * 1.02,
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="red",
                            line=dict(width=1, color="darkred")),
                name="SELL",
            ),
            row=1, col=1,
        )

    # Forced liquidation markers
    if "forced_liquidation" in traj.columns:
        fl_mask = traj["forced_liquidation"].fillna(False).astype(bool).values
        if fl_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[fl_mask],
                    y=h[fl_mask] * 1.04,
                    mode="markers",
                    marker=dict(symbol="x", size=14, color="magenta",
                                line=dict(width=2, color="darkmagenta")),
                    name="FORCED EXIT",
                ),
                row=1, col=1,
            )

    # 2) Portfolio value
    fig.add_trace(
        go.Scatter(x=dates, y=traj["portfolio_value"],
                   mode="lines", line=dict(color="royalblue", width=2),
                   name="Portfolio"),
        row=2, col=1,
    )

    # 3) Stochastic %K / %D
    pct_k = feat_df.loc[dates, (ticker, "pct_k")].values * 100
    pct_d = feat_df.loc[dates, (ticker, "pct_d")].values * 100
    fig.add_trace(
        go.Scatter(x=dates, y=pct_k, mode="lines",
                   line=dict(color="dodgerblue", width=1), name="%K"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=pct_d, mode="lines",
                   line=dict(color="orange", width=1), name="%D"),
        row=3, col=1,
    )
    # Oversold threshold
    fig.add_hline(y=20, line_dash="dot", line_color="gray",
                  annotation_text="Oversold 20", row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="gray",
                  annotation_text="Overbought 80", row=3, col=1)

    # Layout
    fig.update_layout(
        height=900,
        title_text=f"NEPSE RL Pullback Engine – {ticker}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.update_yaxes(title_text="Stochastic", row=3, col=1)

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    log = get_logger("rl_nepse.viz")
    log.info(f"Trajectory HTML -> {output_path}")
    if show:
        fig.show()


def evaluate_and_plot(
    model,
    feat_df: pd.DataFrame,
    valid_start_dates: dict,
    ticker: str | None = None,
    output_dir: str | pathlib.Path = "outputs/eval",
    episode_length: int = 252,
    plots_dir: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    """Full eval pipeline: deterministic episode → metrics → Plotly HTML + PNGs."""
    log = get_logger("rl_nepse.eval")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir is None:
        plots_dir = output_dir
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    env = UniversalNepseEnv(
        feat_df=feat_df,
        valid_start_dates=valid_start_dates,
        episode_length=episode_length,
        seed=123,
    )

    traj = run_deterministic_episode(env, model, ticker=ticker)
    actual_ticker = traj["ticker"].iloc[0]

    metrics = compute_metrics(traj)
    log.info(f"{'='*50}")
    log.info(f"  Evaluation - {actual_ticker}")
    log.info(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            log.info(f"  {k:>25s}: {v:+.4f}")
        else:
            log.info(f"  {k:>25s}: {v}")
    log.info(f"{'='*50}")

    # Plotly HTML
    html_path = output_dir / f"trajectory_{actual_ticker}.html"
    plot_trajectory(traj, feat_df, actual_ticker, output_path=html_path)

    # PNG plots
    from src.plots import (
        plot_portfolio_curve,
        plot_price_with_signals,
        plot_metrics_summary,
    )
    try:
        p = plot_portfolio_curve(traj, actual_ticker, plots_dir)
        log.info(f"PNG -> {p}")
    except Exception as e:
        log.warning(f"Portfolio PNG failed: {e}")

    try:
        p = plot_price_with_signals(traj, feat_df, actual_ticker, plots_dir)
        log.info(f"PNG -> {p}")
    except Exception as e:
        log.warning(f"Price signals PNG failed: {e}")

    try:
        p = plot_metrics_summary(metrics, actual_ticker, plots_dir)
        log.info(f"PNG -> {p}")
    except Exception as e:
        log.warning(f"Metrics PNG failed: {e}")

    # Post-inference telemetry: trade ledger + tear sheet
    from src.metrics import generate_tear_sheet, export_trade_ledger
    try:
        export_trade_ledger(traj, str(output_dir), actual_ticker)
    except Exception as e:
        log.warning(f"Trade ledger failed: {e}")
    try:
        generate_tear_sheet(traj, str(plots_dir), actual_ticker)
    except Exception as e:
        log.warning(f"Tear sheet failed: {e}")

    # Save trajectory CSV
    csv_path = output_dir / f"trajectory_{actual_ticker}.csv"
    traj.to_csv(csv_path, index=False)
    log.info(f"Trajectory CSV -> {csv_path}")

    return traj
