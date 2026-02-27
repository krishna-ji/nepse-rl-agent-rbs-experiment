"""
NEPSE RL Stochastic Pullback Engine – Streamlit Dashboard
==========================================================
TradingView-style candlestick charts with RL buy/sell signals
and Stochastic Oscillator indicator.

Run:
    streamlit run src/dashboard/app.py
    python main.py dash
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DATA_DIR = ROOT / "data" / "stocks"

# ─── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEPSE RL Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark theme CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-card .label {
        font-size: 12px;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .positive { color: #00e676; }
    .negative { color: #ff5252; }
    .neutral  { color: #42a5f5; }
    div[data-testid="stSidebar"] {
        background: #16161e;
    }
    .trade-row-buy  { color: #00e676; }
    .trade-row-sell { color: #ff5252; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading Helpers
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def scan_runs() -> list[dict]:
    """Scan outputs/ for runs that have evaluation trajectories."""
    if not OUTPUTS.exists():
        return []
    runs = []
    for d in sorted(OUTPUTS.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        if d.name in ("models", "eval", "models_test", "eval_test"):
            continue
        eval_dir = d / "eval"
        if not eval_dir.exists():
            continue
        traj_csvs = sorted(eval_dir.glob("trajectory_*.csv"))
        tickers = [f.stem.replace("trajectory_", "") for f in traj_csvs]
        if not tickers:
            continue
        runs.append({
            "name": d.name,
            "path": d,
            "tickers": tickers,
            "eval_dir": eval_dir,
            "plots_dir": d / "plots",
        })
    return runs


@st.cache_data(ttl=300)
def load_trajectory(eval_dir: str, ticker: str) -> pd.DataFrame:
    """Load a trajectory CSV for a given ticker from an eval directory."""
    csv_path = pathlib.Path(eval_dir) / f"trajectory_{ticker}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


@st.cache_data(ttl=300)
def load_ohlcv(ticker: str) -> pd.DataFrame:
    """Load raw OHLCV from data/stocks/{ticker}.csv."""
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("date", "timestamp"):
            col_map[c] = "date"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl in ("close", "ltp"):
            col_map[c] = "close"
        elif cl == "volume":
            col_map[c] = "volume"
    df = df.rename(columns=col_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Strip timezone info for clean merge with trajectory data
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=300)
def load_tear_sheet(plots_dir: str, ticker: str) -> dict | None:
    """Load tear sheet JSON for a ticker."""
    json_path = pathlib.Path(plots_dir) / f"{ticker}_tear_sheet.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_trade_ledger(eval_dir: str, ticker: str) -> pd.DataFrame:
    """Load trade ledger CSV."""
    csv_path = pathlib.Path(eval_dir) / f"{ticker}_trade_ledger.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, parse_dates=["date"])


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Compute Stochastic Oscillator %K and %D."""
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    raw_k = 100.0 * (close - lowest) / (highest - lowest + 1e-10)
    pct_k = raw_k.rolling(d_period, min_periods=d_period).mean()
    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean()
    return pct_k, pct_d


# ═══════════════════════════════════════════════════════════════════════════
#  Charting – TradingView-style Candlestick
# ═══════════════════════════════════════════════════════════════════════════

def build_chart(
    ohlcv: pd.DataFrame,
    traj: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Build a TradingView-style interactive chart with:
    - Candlestick OHLC
    - Volume bars
    - Buy / Sell / Forced Exit markers
    - TSL line
    - Stochastic %K / %D subplot
    - Portfolio equity subplot
    """
    # Merge OHLCV with trajectory on date
    merged = traj.merge(ohlcv[["date", "open", "high", "low", "volume"]],
                        on="date", how="left")

    dates = merged["date"]
    o = merged["open"].values
    h = merged["high"].values
    lo = merged["low"].values
    c = merged["close"].values
    vol = merged["volume"].fillna(0).infer_objects(copy=False).values

    # Compute Stochastic on the merged OHLC
    pct_k, pct_d = compute_stochastic(
        pd.Series(h), pd.Series(lo), pd.Series(c)
    )

    # Volume colors
    vol_colors = np.where(c >= o, "rgba(0,230,118,0.4)", "rgba(255,82,82,0.4)")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.45, 0.12, 0.18, 0.25],
        subplot_titles=[
            f"{ticker} – RL Trading Signals",
            "Volume",
            "Stochastic Oscillator (%K / %D)",
            "Portfolio Value",
        ],
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=dates, open=o, high=h, low=lo, close=c,
            name="OHLC",
            increasing_line_color="#00e676",
            increasing_fillcolor="#00e676",
            decreasing_line_color="#ff5252",
            decreasing_fillcolor="#ff5252",
            whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    # TSL line (only while holding)
    tsl = merged["tsl_level"].values.copy()
    tsl_masked = np.where(np.isnan(tsl), None, tsl)
    fig.add_trace(
        go.Scatter(
            x=dates, y=tsl_masked,
            mode="lines",
            line=dict(color="#ff9800", width=1.5, dash="dot"),
            name="Chandelier TSL",
            connectgaps=False,
        ),
        row=1, col=1,
    )

    # Buy markers
    actions = merged["action"].values
    positions = merged["position"].values
    buy_mask = (positions == 0) & (actions == 1)
    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[buy_mask],
                y=lo[buy_mask] * 0.97,
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=14,
                    color="#00e676",
                    line=dict(width=1.5, color="#004d25"),
                ),
                name="BUY",
                hovertemplate="BUY @ %{y:.2f}<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Sell markers
    sell_mask = (positions == 1) & (actions == 0)
    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=dates[sell_mask],
                y=h[sell_mask] * 1.03,
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=14,
                    color="#ff5252",
                    line=dict(width=1.5, color="#7f0000"),
                ),
                name="SELL",
                hovertemplate="SELL @ %{y:.2f}<br>%{x}<extra></extra>",
            ),
            row=1, col=1,
        )

    # Forced exit markers
    forced_col = merged.get("forced_liquidation", pd.Series(dtype=float))
    if forced_col is not None:
        fl_vals = forced_col.fillna(False).infer_objects(copy=False)
        # Handle string 'True'/'False' from CSV
        if fl_vals.dtype == object:
            fl_vals = fl_vals.map({"True": True, "False": False, True: True, False: False}).fillna(False)
        fl_mask = fl_vals.astype(bool).values
        if fl_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[fl_mask],
                    y=h[fl_mask] * 1.05,
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        size=14,
                        color="#e040fb",
                        line=dict(width=2, color="#7b1fa2"),
                    ),
                    name="FORCED EXIT",
                    hovertemplate="FORCED EXIT @ %{y:.2f}<br>%{x}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ── Row 2: Volume bars ────────────────────────────────────────────
    fig.add_trace(
        go.Bar(
            x=dates, y=vol,
            marker_color=vol_colors.tolist(),
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Row 3: Stochastic Oscillator ─────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dates, y=pct_k,
            mode="lines",
            line=dict(color="#42a5f5", width=1.5),
            name="%K",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=pct_d,
            mode="lines",
            line=dict(color="#ffa726", width=1.5),
            name="%D",
        ),
        row=3, col=1,
    )

    # Overbought / Oversold zones
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,82,82,0.08)",
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(0,230,118,0.08)",
                  line_width=0, row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#555",
                  line_width=0.8, row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#555",
                  line_width=0.8, row=3, col=1)

    # ── Row 4: Portfolio Value ────────────────────────────────────────
    pv = merged["portfolio_value"].values
    pv_color = "#00e676" if pv[-1] >= 1.0 else "#ff5252"
    fig.add_trace(
        go.Scatter(
            x=dates, y=pv,
            mode="lines",
            line=dict(color=pv_color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({38 if pv[-1]>=1 else 255},{230 if pv[-1]>=1 else 82},{118 if pv[-1]>=1 else 82},0.08)",
            name="Portfolio",
        ),
        row=4, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#555",
                  line_width=0.8, row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        height=1000,
        template="plotly_dark",
        paper_bgcolor="#0e0e16",
        plot_bgcolor="#0e0e16",
        font=dict(family="JetBrains Mono, Consolas, monospace", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=30, t=80, b=40),
        hovermode="x unified",
    )

    # Grid styling per subplot
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor="#1a1a2e",
            showgrid=True,
            zeroline=False,
            row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="#1a1a2e",
            showgrid=True,
            zeroline=False,
            row=i, col=1,
        )

    fig.update_yaxes(title_text="Price (NPR)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="Stoch", range=[-5, 105], row=3, col=1)
    fig.update_yaxes(title_text="PV", row=4, col=1)

    # Remove rangeslider for all x axes
    fig.update_xaxes(rangeslider_visible=False)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Metric Cards
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, color_class: str = "neutral") -> str:
    """Return HTML for a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {color_class}">{value}</div>
    </div>
    """


def render_metrics(ts: dict):
    """Render tear sheet metrics as styled cards."""
    ret = ts.get("total_return_agent", 0)
    ret_class = "positive" if ret > 0 else "negative" if ret < 0 else "neutral"

    sharpe = ts.get("annualized_sharpe", 0)
    sharpe_class = "positive" if sharpe > 0.5 else "negative" if sharpe < 0 else "neutral"

    wr = ts.get("win_rate", 0)
    wr_class = "positive" if wr >= 0.5 else "negative" if wr > 0 else "neutral"

    pf = ts.get("profit_factor", 0)
    pf_val = pf if isinstance(pf, (int, float)) else 0
    pf_class = "positive" if pf_val > 1 else "negative" if pf_val > 0 else "neutral"

    mdd = ts.get("max_drawdown_agent", 0)
    mdd_class = "negative" if mdd < -0.1 else "positive" if mdd > -0.05 else "neutral"

    cols = st.columns(6)
    with cols[0]:
        st.markdown(metric_card("Total Return", f"{ret:+.2%}", ret_class), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Sharpe Ratio", f"{sharpe:.2f}", sharpe_class), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card("Win Rate", f"{wr:.1%}", wr_class), unsafe_allow_html=True)
    with cols[3]:
        pf_str = f"{pf:.1f}" if isinstance(pf, (int, float)) else str(pf)
        st.markdown(metric_card("Profit Factor", pf_str, pf_class), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_card("Max Drawdown", f"{mdd:.2%}", mdd_class), unsafe_allow_html=True)
    with cols[5]:
        nt = ts.get("num_trades", 0)
        fl = ts.get("forced_liquidations", 0)
        st.markdown(metric_card("Trades / Forced", f"{nt} / {fl}", "neutral"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-ticker Comparison
# ═══════════════════════════════════════════════════════════════════════════

def render_comparison_table(run_info: dict):
    """Show a comparison table of all tickers in a run."""
    rows = []
    for ticker in run_info["tickers"]:
        ts = load_tear_sheet(str(run_info["plots_dir"]), ticker)
        if ts is None:
            continue
        rows.append({
            "Ticker": ticker,
            "Return": f"{ts.get('total_return_agent', 0):+.2%}",
            "Sharpe": f"{ts.get('annualized_sharpe', 0):.2f}",
            "Win Rate": f"{ts.get('win_rate', 0):.1%}",
            "Profit Factor": ts.get("profit_factor", 0),
            "Max DD": f"{ts.get('max_drawdown_agent', 0):.2%}",
            "Trades": ts.get("num_trades", 0),
            "Forced Exits": ts.get("forced_liquidations", 0),
            "Final PV": f"{ts.get('final_portfolio_value', 1):.4f}",
            "Exposure": f"{ts.get('exposure_pct', 0):.1%}",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Trade Ledger View
# ═══════════════════════════════════════════════════════════════════════════

def render_trade_ledger(eval_dir: str, ticker: str):
    """Display enriched trade ledger with color-coded transitions."""
    ledger = load_trade_ledger(eval_dir, ticker)
    if ledger.empty:
        st.info("No trade ledger found for this ticker.")
        return

    # Filter to only actionable rows (BUY, SELL, FORCED_EXIT)
    if "transition" in ledger.columns:
        action_rows = ledger[ledger["transition"].isin(
            ["BUY (0->1)", "SELL (1->0)", "FORCED_EXIT"]
        )].copy()
    else:
        action_rows = ledger[(ledger["action"] != -1)].copy()

    if action_rows.empty:
        st.info("No trades executed.")
        return

    display_cols = ["date", "close", "portfolio_value"]
    if "transition" in action_rows.columns:
        display_cols.insert(1, "transition")
    if "trade_id" in action_rows.columns:
        display_cols.append("trade_id")

    available = [c for c in display_cols if c in action_rows.columns]
    st.dataframe(
        action_rows[available].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main App
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 NEPSE RL Engine")
        st.markdown("---")

        runs = scan_runs()
        if not runs:
            st.error("No evaluation runs found in outputs/")
            st.stop()

        # Button to evaluate all stocks
        if "eval_all_status" not in st.session_state:
            st.session_state.eval_all_status = "idle"
        if st.button("Evaluate All Stocks", key="eval_all_btn"):
            st.session_state.eval_all_status = "running"

        if st.session_state.eval_all_status == "running":
            st.info("Running evaluation on all stocks. This may take several minutes...")
            # Run evaluation backend
            import subprocess
            import time
            # Find best model
            model_path = str(ROOT / "outputs" / "models" / "best" / "best_model")
            # Run evaluation script with all tickers
            eval_script = str(ROOT / "runs" / "run_evaluation.py")
            data_dir = str(DATA_DIR)
            # Use a timestamped output directory
            out_dir = str(ROOT / "outputs" / f"eval_all_{int(time.time())}")
            cmd = [
                "uv", "run", "python", eval_script,
                "--data_dir", data_dir,
                "--model_path", model_path,
                "--ticker", "",
                "--multi", "999",
                "--episode_length", "252",
                "--run_dir", out_dir,
            ]
            try:
                subprocess.run(cmd, check=True)
                st.session_state.eval_all_status = "done"
            except Exception as e:
                st.session_state.eval_all_status = "error"
                st.error(f"Evaluation failed: {e}")

        if st.session_state.eval_all_status == "done":
            st.success("Evaluation complete! Refresh to see new results.")
        elif st.session_state.eval_all_status == "error":
            st.error("Evaluation failed.")

        # Run selector
        run_names = [r["name"] for r in runs]
        selected_run_name = st.selectbox(
            "Select Run",
            run_names,
            index=0,
            help="Choose a completed run to view results",
        )
        run_info = next(r for r in runs if r["name"] == selected_run_name)

        st.markdown("---")

        # Ticker selector
        selected_ticker = st.selectbox(
            "Select Ticker",
            run_info["tickers"],
            index=0,
        )

        st.markdown("---")

        # View selector
        view = st.radio(
            "View",
            ["Chart", "Comparison", "Trade Ledger"],
            index=0,
        )

        st.markdown("---")

        # Run info
        st.markdown(f"**Tickers:** {len(run_info['tickers'])}")
        st.markdown(f"**Path:** `{run_info['path'].name}`")

    # ── Main Area ─────────────────────────────────────────────────────
    if view == "Chart":
        # Load data
        traj = load_trajectory(str(run_info["eval_dir"]), selected_ticker)
        if traj.empty:
            st.error(f"No trajectory data for {selected_ticker}")
            st.stop()

        ohlcv = load_ohlcv(selected_ticker)
        if ohlcv.empty:
            st.error(f"No OHLCV data for {selected_ticker}")
            st.stop()

        # Metrics row
        ts = load_tear_sheet(str(run_info["plots_dir"]), selected_ticker)
        if ts:
            render_metrics(ts)
            st.markdown("")

        # Build and render chart
        fig = build_chart(ohlcv, traj, selected_ticker)
        st.plotly_chart(fig, width="stretch", config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "displaylogo": False,
        })

        # Expandable: Raw tear sheet JSON
        if ts:
            with st.expander("Raw Tear Sheet JSON"):
                st.json(ts)

    elif view == "Comparison":
        st.markdown(f"### All Tickers – {selected_run_name}")
        render_comparison_table(run_info)

        # Overlay equity curves
        st.markdown("### Equity Curves Overlay")
        fig_eq = go.Figure()
        for ticker in run_info["tickers"]:
            t = load_trajectory(str(run_info["eval_dir"]), ticker)
            if t.empty:
                continue
            fig_eq.add_trace(go.Scatter(
                x=t["date"], y=t["portfolio_value"],
                mode="lines", name=ticker, line=dict(width=2),
            ))
        fig_eq.add_hline(y=1.0, line_dash="dash", line_color="#555")
        fig_eq.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="#0e0e16",
            plot_bgcolor="#0e0e16",
            yaxis_title="Portfolio Value",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_eq, width="stretch")

    elif view == "Trade Ledger":
        st.markdown(f"### Trade Ledger – {selected_ticker}")
        render_trade_ledger(str(run_info["eval_dir"]), selected_ticker)


if __name__ == "__main__":
    main()
else:
    main()
