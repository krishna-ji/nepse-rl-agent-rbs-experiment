#!/usr/bin/env python3
"""
Ichimoku RBS — Comprehensive Multi-Page PDF Report
====================================================
Generates a full analytics report from both Ichimoku strategies
(Kumo Break + T/K Cross) with the following pages:

  Page 1  : Year × Strategy return heatmap (green/red intensity)
  Page 2  : Ticker × Year PnL heatmap (top tickers, green/red grid)
  Page 3  : Year-wise aggregate bar charts (PnL, Win Rate, Trades)
  Page 4  : Monthly returns heatmap (Year × Month calendar)
  Page 5  : PnL % distribution — histogram + KDE + stats
  Page 6  : Trade duration distribution + bars-held analysis
  Page 7  : Win/Loss streaks + consecutive analysis
  Page 8  : Rolling metrics (60-trade rolling WR, PF, expectancy)
  Page 9  : Exit reason breakdown + Top/Bottom tickers
  Page 10 : Strategy comparison side-by-side summary table
"""

import warnings; warnings.filterwarnings("ignore")

import pathlib, sys, datetime, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "labs" / "rbs"))

import ichimoku_kumo_break as kumo_mod
import ichimoku_tk_cross   as tk_mod

FILTER_DATE = "2010-01-01"
FRICTION_PCT = 1.5

# Custom diverging colormap: deep red → white → deep green
CMAP_RG = LinearSegmentedColormap.from_list("RedGreen", [
    (0.0,  "#8B0000"),   # dark red
    (0.15, "#DC143C"),   # crimson
    (0.35, "#FF6B6B"),   # light red
    (0.5,  "#FFFFFF"),   # white (zero)
    (0.65, "#90EE90"),   # light green
    (0.85, "#228B22"),   # forest green
    (1.0,  "#006400"),   # dark green
])

CMAP_GREENS = LinearSegmentedColormap.from_list("Greens", [
    (0.0, "#FFFFFF"), (0.5, "#90EE90"), (1.0, "#006400"),
])

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "font.size":        9,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "figure.dpi":       150,
})

# ============================================================================
# DATA LOADING
# ============================================================================

def generate_all_trades():
    """Run both strategies, return combined DataFrame."""
    null_log = logging.getLogger("rbs_report_null")
    null_log.handlers = [logging.NullHandler()]
    frames = kumo_mod.load_ohlcv(null_log)
    print(f"[DATA] Loaded OHLCV for {len(frames)} tickers")

    print("[KUMO] Running Kumo Break backtest...")
    kumo_trades = []
    for tk, df in sorted(frames.items()):
        t = kumo_mod.backtest_ticker(tk, df)
        if t: kumo_trades.extend(t)
    print(f"[KUMO] {len(kumo_trades)} trades")

    print("[TK]   Running T/K Cross backtest...")
    tk_trades = []
    for tk, df in sorted(frames.items()):
        t = tk_mod.backtest_ticker(tk, df)
        if t: tk_trades.extend(t)
    print(f"[TK]   {len(tk_trades)} trades")

    df_k = pd.DataFrame(kumo_trades); df_k["strategy"] = "Kumo Break"
    df_t = pd.DataFrame(tk_trades);   df_t["strategy"] = "T/K Cross"
    combined = pd.concat([df_k, df_t], ignore_index=True)
    combined["entry_date"] = pd.to_datetime(combined["entry_date"])
    combined["exit_date"]  = pd.to_datetime(combined["exit_date"])
    combined = combined[combined["entry_date"] >= pd.Timestamp(FILTER_DATE)].copy()
    # Apply friction
    combined["adj_pnl_pct"] = combined["net_pnl_pct"] - FRICTION_PCT
    combined["year"]  = combined["entry_date"].dt.year
    combined["month"] = combined["entry_date"].dt.month
    combined["win"]   = (combined["adj_pnl_pct"] > 0).astype(int)
    return combined.sort_values("entry_date").reset_index(drop=True)


# ============================================================================
# HELPER
# ============================================================================

def _annotate_heatmap(ax, data, fmt="{:.1f}", fontsize=7):
    """Add text annotations to a heatmap."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if pd.notna(val):
                color = "white" if abs(val) > data.abs().max().max() * 0.6 else "black"
                ax.text(j + 0.5, i + 0.5, fmt.format(val),
                        ha="center", va="center", fontsize=fontsize, color=color)


def _compute_stats(df) -> dict:
    n = len(df)
    if n == 0:
        return {}
    wins = df[df["adj_pnl_pct"] > 0]
    losses = df[df["adj_pnl_pct"] <= 0]
    gp = wins["adj_pnl_pct"].sum() if len(wins) else 0
    gl = abs(losses["adj_pnl_pct"].sum()) if len(losses) else 0
    return {
        "trades": n,
        "win_rate": len(wins) / n * 100,
        "avg_pnl": df["adj_pnl_pct"].mean(),
        "median_pnl": df["adj_pnl_pct"].median(),
        "std_pnl": df["adj_pnl_pct"].std(),
        "avg_win": wins["adj_pnl_pct"].mean() if len(wins) else 0,
        "avg_loss": losses["adj_pnl_pct"].mean() if len(losses) else 0,
        "best": df["adj_pnl_pct"].max(),
        "worst": df["adj_pnl_pct"].min(),
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "expectancy": df["adj_pnl_pct"].mean(),
        "avg_bars": df["bars_held"].mean() if "bars_held" in df.columns else 0,
        "skew": df["adj_pnl_pct"].skew(),
        "kurtosis": df["adj_pnl_pct"].kurtosis(),
    }


# ============================================================================
# PAGE 1: Year × Strategy Return Heatmap
# ============================================================================

def page_year_strategy_heatmap(pdf, df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    fig.suptitle("Year × Strategy Performance Heatmap", fontsize=14, fontweight="bold", y=0.98)

    metrics = [
        ("Mean PnL %", "adj_pnl_pct", "mean"),
        ("Win Rate %", "win", "mean"),
        ("Trade Count", "adj_pnl_pct", "count"),
    ]

    for ax, (title, col, agg) in zip(axes, metrics):
        pivot = df.pivot_table(values=col, index="year", columns="strategy",
                               aggfunc=agg, fill_value=0)
        if agg == "mean" and col == "win":
            pivot *= 100

        if "count" in agg:
            im = ax.imshow(pivot.values, cmap=CMAP_GREENS, aspect="auto")
        else:
            vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.1)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(pivot.values, cmap=CMAP_RG, norm=norm, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8, rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Annotate
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                fmt = f"{val:.1f}" if "count" not in agg else f"{int(val)}"
                tcolor = "white" if abs(val) > pivot.abs().values.max() * 0.55 else "black"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=7, color=tcolor)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 2: Ticker × Year PnL Heatmap (top N tickers)
# ============================================================================

def page_ticker_year_heatmap(pdf, df, top_n=40):
    # Get tickers with most trades
    top_tickers = df["ticker"].value_counts().head(top_n).index.tolist()
    sub = df[df["ticker"].isin(top_tickers)]

    pivot = sub.pivot_table(values="adj_pnl_pct", index="ticker", columns="year",
                            aggfunc="mean", fill_value=np.nan)
    # Sort by total PnL
    pivot["_total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_total", ascending=False).drop(columns="_total")

    fig, ax = plt.subplots(figsize=(16, max(10, len(pivot) * 0.35)))
    fig.suptitle(f"Ticker × Year Avg PnL % (Top {len(pivot)} by Trade Count)",
                 fontsize=14, fontweight="bold", y=0.99)

    vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(pivot.values, cmap=CMAP_RG, norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Year"); ax.set_ylabel("Ticker")

    # Annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                c = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=c)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Avg PnL %")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 3: Year-wise Aggregate Bar Charts
# ============================================================================

def page_yearly_bars(pdf, df):
    yearly = df.groupby("year").agg(
        trades=("adj_pnl_pct", "count"),
        total_pnl=("adj_pnl_pct", "sum"),
        avg_pnl=("adj_pnl_pct", "mean"),
        win_rate=("win", "mean"),
        median_pnl=("adj_pnl_pct", "median"),
    )
    yearly["win_rate"] *= 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Year-wise Aggregate Performance", fontsize=14, fontweight="bold", y=0.99)

    # Total PnL
    ax = axes[0, 0]
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in yearly["total_pnl"]]
    ax.bar(yearly.index.astype(str), yearly["total_pnl"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Total PnL % (sum of all trades)", fontweight="bold")
    ax.set_ylabel("Cumulative PnL %")
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    # Win Rate
    ax = axes[0, 1]
    colors = ["#228B22" if v >= 35 else "#DC143C" if v < 25 else "#FFA500" for v in yearly["win_rate"]]
    ax.bar(yearly.index.astype(str), yearly["win_rate"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.axhline(33, color="gray", linestyle="--", linewidth=0.8, label="33% baseline")
    ax.set_title("Win Rate % by Year", fontweight="bold")
    ax.set_ylabel("Win Rate %")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)

    # Trade Count
    ax = axes[1, 0]
    kumo_yr = df[df["strategy"] == "Kumo Break"].groupby("year").size()
    tk_yr   = df[df["strategy"] == "T/K Cross"].groupby("year").size()
    all_yrs = sorted(df["year"].unique())
    kumo_vals = [kumo_yr.get(y, 0) for y in all_yrs]
    tk_vals   = [tk_yr.get(y, 0) for y in all_yrs]
    x = np.arange(len(all_yrs))
    ax.bar(x - 0.2, kumo_vals, 0.4, label="Kumo Break", color="#e8710a", alpha=0.8)
    ax.bar(x + 0.2, tk_vals, 0.4, label="T/K Cross", color="#1a73e8", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in all_yrs], rotation=45, fontsize=7)
    ax.set_title("Trade Count by Strategy & Year", fontweight="bold")
    ax.set_ylabel("# Trades")
    ax.legend(fontsize=8)

    # Avg PnL + Median
    ax = axes[1, 1]
    ax.bar(yearly.index.astype(str), yearly["avg_pnl"],
           color=["#228B22" if v >= 0 else "#DC143C" for v in yearly["avg_pnl"]],
           alpha=0.6, label="Mean PnL %", edgecolor="gray", linewidth=0.3)
    ax.plot(yearly.index.astype(str), yearly["median_pnl"], "ko-", markersize=4,
            linewidth=1.2, label="Median PnL %")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Mean & Median PnL % by Year", fontweight="bold")
    ax.set_ylabel("PnL %")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 4: Monthly Returns Heatmap (Year × Month Calendar)
# ============================================================================

def page_monthly_heatmap(pdf, df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Monthly Returns Heatmap (Year × Month)", fontsize=14, fontweight="bold", y=0.99)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for ax, (strat, sub) in zip(axes, df.groupby("strategy")):
        pivot = sub.pivot_table(values="adj_pnl_pct", index="year", columns="month",
                                aggfunc="mean", fill_value=np.nan)
        # Ensure all 12 months
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[sorted(pivot.columns)]

        vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 0.1)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(pivot.values, cmap=CMAP_RG, norm=norm, aspect="auto")

        ax.set_xticks(range(12))
        ax.set_xticklabels(month_names, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(f"{strat}", fontsize=11, fontweight="bold")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    c = "white" if abs(val) > vmax * 0.55 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=c)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 5: PnL Distribution — Histogram + KDE + Stats
# ============================================================================

def page_pnl_distribution(pdf, df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PnL % Distribution Analysis", fontsize=14, fontweight="bold", y=0.99)

    # Full distribution
    ax = axes[0, 0]
    pnl = df["adj_pnl_pct"].dropna()
    bins = np.linspace(pnl.min(), pnl.max(), 60)
    ax.hist(pnl, bins=bins, color="#4a90d9", alpha=0.7, edgecolor="white", linewidth=0.3, density=True)
    # KDE
    if len(pnl) > 5:
        kde = sp_stats.gaussian_kde(pnl)
        x_kde = np.linspace(pnl.min(), pnl.max(), 200)
        ax.plot(x_kde, kde(x_kde), color="#DC143C", linewidth=1.5, label="KDE")
    ax.axvline(pnl.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean: {pnl.mean():.2f}%")
    ax.axvline(pnl.median(), color="green", linestyle="--", linewidth=1, label=f"Median: {pnl.median():.2f}%")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("All Trades — PnL Distribution", fontweight="bold")
    ax.set_xlabel("PnL %"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # By strategy
    ax = axes[0, 1]
    for strat, color in [("Kumo Break", "#e8710a"), ("T/K Cross", "#1a73e8")]:
        sub = df[df["strategy"] == strat]["adj_pnl_pct"].dropna()
        ax.hist(sub, bins=40, color=color, alpha=0.5, edgecolor="white",
                linewidth=0.3, density=True, label=f"{strat} (μ={sub.mean():.2f}%)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("PnL Distribution by Strategy", fontweight="bold")
    ax.set_xlabel("PnL %"); ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Box plot by strategy
    ax = axes[1, 0]
    strategies = df["strategy"].unique()
    data_bp = [df[df["strategy"] == s]["adj_pnl_pct"].dropna().values for s in strategies]
    bp = ax.boxplot(data_bp, labels=strategies, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
    colors_bp = ["#e8710a", "#1a73e8"]
    for patch, c in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(c); patch.set_alpha(0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("PnL Box Plot by Strategy", fontweight="bold")
    ax.set_ylabel("PnL %")

    # Stats table
    ax = axes[1, 1]
    ax.axis("off")
    stats_all = _compute_stats(df)
    stats_kumo = _compute_stats(df[df["strategy"] == "Kumo Break"])
    stats_tk = _compute_stats(df[df["strategy"] == "T/K Cross"])

    headers = ["Metric", "All", "Kumo Break", "T/K Cross"]
    rows = [
        ["Trades", f"{stats_all['trades']}", f"{stats_kumo['trades']}", f"{stats_tk['trades']}"],
        ["Mean PnL %", f"{stats_all['avg_pnl']:.2f}", f"{stats_kumo['avg_pnl']:.2f}", f"{stats_tk['avg_pnl']:.2f}"],
        ["Median PnL %", f"{stats_all['median_pnl']:.2f}", f"{stats_kumo['median_pnl']:.2f}", f"{stats_tk['median_pnl']:.2f}"],
        ["Std Dev %", f"{stats_all['std_pnl']:.2f}", f"{stats_kumo['std_pnl']:.2f}", f"{stats_tk['std_pnl']:.2f}"],
        ["Win Rate %", f"{stats_all['win_rate']:.1f}", f"{stats_kumo['win_rate']:.1f}", f"{stats_tk['win_rate']:.1f}"],
        ["Avg Win %", f"{stats_all['avg_win']:.2f}", f"{stats_kumo['avg_win']:.2f}", f"{stats_tk['avg_win']:.2f}"],
        ["Avg Loss %", f"{stats_all['avg_loss']:.2f}", f"{stats_kumo['avg_loss']:.2f}", f"{stats_tk['avg_loss']:.2f}"],
        ["Best Trade %", f"{stats_all['best']:.2f}", f"{stats_kumo['best']:.2f}", f"{stats_tk['best']:.2f}"],
        ["Worst Trade %", f"{stats_all['worst']:.2f}", f"{stats_kumo['worst']:.2f}", f"{stats_tk['worst']:.2f}"],
        ["Profit Factor", f"{stats_all['profit_factor']:.2f}", f"{stats_kumo['profit_factor']:.2f}", f"{stats_tk['profit_factor']:.2f}"],
        ["Skewness", f"{stats_all['skew']:.2f}", f"{stats_kumo['skew']:.2f}", f"{stats_tk['skew']:.2f}"],
        ["Kurtosis", f"{stats_all['kurtosis']:.2f}", f"{stats_kumo['kurtosis']:.2f}", f"{stats_tk['kurtosis']:.2f}"],
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    # Color header
    for j, key in enumerate(headers):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Color data rows
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            table[i, j].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

    ax.set_title("Statistical Summary", fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 6: Trade Duration Analysis
# ============================================================================

def page_duration_analysis(pdf, df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Trade Duration & Holding Period Analysis", fontsize=14, fontweight="bold", y=0.99)

    # Duration histogram
    ax = axes[0, 0]
    bars = df["bars_held"].dropna()
    ax.hist(bars, bins=range(0, int(bars.max()) + 2), color="#4a90d9", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(bars.mean(), color="red", linestyle="--", label=f"Mean: {bars.mean():.1f}")
    ax.axvline(bars.median(), color="green", linestyle="--", label=f"Median: {bars.median():.0f}")
    ax.set_title("Bars Held Distribution", fontweight="bold")
    ax.set_xlabel("Bars Held"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Duration vs PnL scatter
    ax = axes[0, 1]
    wins = df[df["adj_pnl_pct"] > 0]
    loss = df[df["adj_pnl_pct"] <= 0]
    ax.scatter(loss["bars_held"], loss["adj_pnl_pct"], c="#DC143C", alpha=0.3, s=10, label="Losses")
    ax.scatter(wins["bars_held"], wins["adj_pnl_pct"], c="#228B22", alpha=0.3, s=10, label="Wins")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("PnL vs Bars Held", fontweight="bold")
    ax.set_xlabel("Bars Held"); ax.set_ylabel("PnL %")
    ax.legend(fontsize=8)

    # Duration bins — avg PnL
    ax = axes[1, 0]
    df_copy = df.copy()
    df_copy["dur_bin"] = pd.cut(df_copy["bars_held"], bins=[0, 5, 10, 15, 20, 30, 50, 100, 500],
                                 labels=["1-5", "6-10", "11-15", "16-20", "21-30", "31-50", "51-100", "100+"])
    dur_stats = df_copy.groupby("dur_bin", observed=True).agg(
        avg_pnl=("adj_pnl_pct", "mean"),
        count=("adj_pnl_pct", "count"),
        wr=("win", "mean"),
    )
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in dur_stats["avg_pnl"]]
    bars_plot = ax.bar(dur_stats.index.astype(str), dur_stats["avg_pnl"], color=colors,
                       alpha=0.7, edgecolor="gray", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    # Add count labels
    for bar, cnt in zip(bars_plot, dur_stats["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={cnt}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Avg PnL % by Holding Period", fontweight="bold")
    ax.set_xlabel("Bars Held"); ax.set_ylabel("Avg PnL %")

    # Win rate by duration bin
    ax = axes[1, 1]
    dur_stats["wr_pct"] = dur_stats["wr"] * 100
    colors = ["#228B22" if v >= 35 else "#DC143C" if v < 25 else "#FFA500"
              for v in dur_stats["wr_pct"]]
    ax.bar(dur_stats.index.astype(str), dur_stats["wr_pct"], color=colors,
           alpha=0.7, edgecolor="gray", linewidth=0.3)
    ax.axhline(33, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Win Rate % by Holding Period", fontweight="bold")
    ax.set_xlabel("Bars Held"); ax.set_ylabel("Win Rate %")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 7: Win/Loss Streaks
# ============================================================================

def page_streaks(pdf, df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Win/Loss Streak & Consecutive Trade Analysis", fontsize=14, fontweight="bold", y=0.99)

    for idx, (strat, sub) in enumerate(df.groupby("strategy")):
        sub = sub.sort_values("entry_date").reset_index(drop=True)
        wins_arr = sub["win"].values

        # Compute streaks
        streaks = []
        current_val = wins_arr[0]
        current_len = 1
        for i in range(1, len(wins_arr)):
            if wins_arr[i] == current_val:
                current_len += 1
            else:
                streaks.append(("Win" if current_val == 1 else "Loss", current_len))
                current_val = wins_arr[i]
                current_len = 1
        streaks.append(("Win" if current_val == 1 else "Loss", current_len))

        win_streaks  = [s[1] for s in streaks if s[0] == "Win"]
        loss_streaks = [s[1] for s in streaks if s[0] == "Loss"]

        # Streak histogram
        ax = axes[0, idx]
        max_streak = max(max(win_streaks, default=0), max(loss_streaks, default=0))
        bins = range(1, max_streak + 2)
        ax.hist(win_streaks, bins=bins, color="#228B22", alpha=0.6, label="Win Streaks", edgecolor="white")
        ax.hist(loss_streaks, bins=bins, color="#DC143C", alpha=0.6, label="Loss Streaks", edgecolor="white")
        ax.set_title(f"{strat} — Streak Distribution", fontweight="bold")
        ax.set_xlabel("Streak Length"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

        # Rolling PnL
        ax = axes[1, idx]
        rolling_pnl = sub["adj_pnl_pct"].cumsum()
        ax.plot(range(len(rolling_pnl)), rolling_pnl.values, color="#1a73e8", linewidth=0.8)
        ax.fill_between(range(len(rolling_pnl)), 0, rolling_pnl.values,
                        where=rolling_pnl.values >= 0, color="#34a853", alpha=0.15)
        ax.fill_between(range(len(rolling_pnl)), 0, rolling_pnl.values,
                        where=rolling_pnl.values < 0, color="#ea4335", alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{strat} — Cumulative PnL % (Trade Sequence)", fontweight="bold")
        ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL %")

        # Add streak stats
        max_w = max(win_streaks, default=0)
        max_l = max(loss_streaks, default=0)
        avg_w = np.mean(win_streaks) if win_streaks else 0
        avg_l = np.mean(loss_streaks) if loss_streaks else 0
        txt = f"Max Win Streak: {max_w}  |  Avg: {avg_w:.1f}\nMax Loss Streak: {max_l}  |  Avg: {avg_l:.1f}"
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 8: Rolling Metrics (60-trade window)
# ============================================================================

def page_rolling_metrics(pdf, df):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Rolling Metrics (60-Trade Window)", fontsize=14, fontweight="bold", y=0.99)

    window = 60
    for strat, color in [("Kumo Break", "#e8710a"), ("T/K Cross", "#1a73e8")]:
        sub = df[df["strategy"] == strat].sort_values("entry_date").reset_index(drop=True)
        if len(sub) < window:
            continue

        # Rolling win rate
        roll_wr = sub["win"].rolling(window).mean() * 100
        axes[0].plot(sub["entry_date"], roll_wr, color=color, linewidth=1, label=strat, alpha=0.8)

        # Rolling avg PnL (expectancy)
        roll_pnl = sub["adj_pnl_pct"].rolling(window).mean()
        axes[1].plot(sub["entry_date"], roll_pnl, color=color, linewidth=1, label=strat, alpha=0.8)

        # Rolling profit factor
        def _rolling_pf(series, w):
            pf_vals = []
            for i in range(len(series)):
                if i < w - 1:
                    pf_vals.append(np.nan)
                else:
                    chunk = series.iloc[i - w + 1:i + 1]
                    gp = chunk[chunk > 0].sum()
                    gl = abs(chunk[chunk <= 0].sum())
                    pf_vals.append(gp / gl if gl > 0 else 5.0)  # cap at 5
            return pd.Series(pf_vals, index=series.index)

        roll_pf = _rolling_pf(sub["adj_pnl_pct"], window)
        axes[2].plot(sub["entry_date"], roll_pf, color=color, linewidth=1, label=strat, alpha=0.8)

    axes[0].axhline(33, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Win Rate %"); axes[0].set_title("Rolling Win Rate", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Avg PnL %"); axes[1].set_title("Rolling Expectancy (Mean PnL %)", fontweight="bold")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Profit Factor"); axes[2].set_title("Rolling Profit Factor", fontweight="bold")
    axes[2].set_ylim(0, 5)
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Date")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 9: Exit Reason Breakdown + Top/Bottom Tickers
# ============================================================================

def page_exit_and_tickers(pdf, df):
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Exit Reason Breakdown & Ticker Performance", fontsize=14, fontweight="bold", y=0.99)

    # Exit reason pie charts
    for idx, (strat, sub) in enumerate(df.groupby("strategy")):
        ax = fig.add_subplot(gs[0, idx])
        reasons = sub["exit_reason"].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(reasons)))
        wedges, texts, autotexts = ax.pie(
            reasons.values, labels=reasons.index, autopct="%1.1f%%",
            colors=colors, textprops={"fontsize": 8}, pctdistance=0.8)
        ax.set_title(f"{strat} — Exit Reasons", fontweight="bold")

    # Top 15 tickers by total PnL
    ax = fig.add_subplot(gs[1, 0])
    ticker_pnl = df.groupby("ticker")["adj_pnl_pct"].agg(["sum", "count", "mean"])
    ticker_pnl = ticker_pnl[ticker_pnl["count"] >= 3]  # min 3 trades
    top15 = ticker_pnl.sort_values("sum", ascending=False).head(15)
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in top15["sum"]]
    bars = ax.barh(top15.index[::-1], top15["sum"][::-1], color=colors[::-1],
                   edgecolor="gray", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Top 15 Tickers (Total PnL %)", fontweight="bold")
    ax.set_xlabel("Total PnL %")
    # Add trade count
    for bar, cnt in zip(bars, top15["count"][::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" n={int(cnt)}", va="center", fontsize=7)

    # Bottom 15 tickers
    ax = fig.add_subplot(gs[1, 1])
    bot15 = ticker_pnl.sort_values("sum", ascending=True).head(15)
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in bot15["sum"]]
    bars = ax.barh(bot15.index[::-1], bot15["sum"][::-1], color=colors[::-1],
                   edgecolor="gray", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Bottom 15 Tickers (Total PnL %)", fontweight="bold")
    ax.set_xlabel("Total PnL %")
    for bar, cnt in zip(bars, bot15["count"][::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" n={int(cnt)}", va="center", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# PAGE 10: Yearly Returns Table + Risk Metrics
# ============================================================================

def page_risk_table(pdf, df):
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1]})
    fig.suptitle("Comprehensive Risk & Return Metrics", fontsize=14, fontweight="bold", y=0.99)

    # Yearly returns table
    ax = axes[0]
    ax.axis("off")
    yearly = df.groupby(["year", "strategy"]).agg(
        trades=("adj_pnl_pct", "count"),
        total_pnl=("adj_pnl_pct", "sum"),
        avg_pnl=("adj_pnl_pct", "mean"),
        wr=("win", "mean"),
        best=("adj_pnl_pct", "max"),
        worst=("adj_pnl_pct", "min"),
    ).reset_index()
    yearly["wr"] *= 100

    # Build table data
    years = sorted(df["year"].unique())
    headers = ["Year", "Kumo Trades", "Kumo PnL%", "Kumo WR%", "TK Trades", "TK PnL%", "TK WR%"]
    rows = []
    for y in years:
        row = [str(y)]
        for s in ["Kumo Break", "T/K Cross"]:
            mask = (yearly["year"] == y) & (yearly["strategy"] == s)
            sub = yearly[mask]
            if len(sub) > 0:
                r = sub.iloc[0]
                row.extend([f"{int(r['trades'])}", f"{r['total_pnl']:.1f}", f"{r['wr']:.0f}"])
            else:
                row.extend(["0", "—", "—"])
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)
    # Color cells by PnL
    for i, row in enumerate(rows):
        for j in [2, 5]:  # PnL columns
            try:
                val = float(row[j])
                bg = "#c8e6c9" if val > 0 else "#ffcdd2" if val < 0 else "white"
                table[i + 1, j].set_facecolor(bg)
            except (ValueError, IndexError):
                pass
        for j in range(len(headers)):
            if i % 2 == 0:
                if table[i + 1, j].get_facecolor() == (1.0, 1.0, 1.0, 1.0):
                    table[i + 1, j].set_facecolor("#f5f5f5")
    ax.set_title("Yearly Return Summary by Strategy", fontweight="bold", pad=15)

    # Risk metrics scatter: Win Rate vs Avg Win/Loss ratio
    ax = axes[1]
    for strat, color, marker in [("Kumo Break", "#e8710a", "o"), ("T/K Cross", "#1a73e8", "s")]:
        for y in years:
            sub = df[(df["strategy"] == strat) & (df["year"] == y)]
            if len(sub) < 3:
                continue
            wr = sub["win"].mean() * 100
            wins_sub = sub[sub["adj_pnl_pct"] > 0]["adj_pnl_pct"]
            loss_sub = sub[sub["adj_pnl_pct"] <= 0]["adj_pnl_pct"]
            avg_w = wins_sub.mean() if len(wins_sub) else 0
            avg_l = abs(loss_sub.mean()) if len(loss_sub) else 0.01
            ratio = avg_w / avg_l if avg_l > 0 else 5
            ax.scatter(wr, ratio, c=color, marker=marker, s=40, alpha=0.6, edgecolor="gray", linewidth=0.3)
            ax.annotate(str(y), (wr, ratio), fontsize=5, ha="center", va="bottom")

    # Breakeven lines
    wr_range = np.linspace(10, 80, 100)
    be_ratio = (100 - wr_range) / wr_range  # breakeven W/L ratio
    ax.plot(wr_range, be_ratio, "k--", linewidth=0.8, label="Breakeven")
    ax.scatter([], [], c="#e8710a", marker="o", label="Kumo Break")
    ax.scatter([], [], c="#1a73e8", marker="s", label="T/K Cross")
    ax.set_xlabel("Win Rate %"); ax.set_ylabel("Avg Win / Avg Loss Ratio")
    ax.set_title("Win Rate vs Reward/Risk Ratio (by Year) — Above Line = Profitable",
                 fontweight="bold")
    ax.set_xlim(10, 80); ax.set_ylim(0, 8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = PROJECT_ROOT / f"runs/{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = run_dir / "rbs_ichimoku_report.pdf"

    print("=" * 70)
    print("  ICHIMOKU RBS — COMPREHENSIVE PDF REPORT")
    print("=" * 70)

    df = generate_all_trades()
    print(f"\n[DATA] {len(df)} trades after 2010 filter")
    print(f"       Kumo Break: {(df['strategy']=='Kumo Break').sum()}")
    print(f"       T/K Cross:  {(df['strategy']=='T/K Cross').sum()}")

    # Save raw data
    df.to_csv(run_dir / "rbs_all_trades.csv", index=False)

    print(f"\n[PDF] Generating report...")
    with PdfPages(output_pdf) as pdf:
        print("  Page 1: Year × Strategy heatmap...")
        page_year_strategy_heatmap(pdf, df)

        print("  Page 2: Ticker × Year PnL heatmap...")
        page_ticker_year_heatmap(pdf, df)

        print("  Page 3: Year-wise bar charts...")
        page_yearly_bars(pdf, df)

        print("  Page 4: Monthly returns heatmap...")
        page_monthly_heatmap(pdf, df)

        print("  Page 5: PnL distribution + stats...")
        page_pnl_distribution(pdf, df)

        print("  Page 6: Trade duration analysis...")
        page_duration_analysis(pdf, df)

        print("  Page 7: Win/Loss streaks...")
        page_streaks(pdf, df)

        print("  Page 8: Rolling metrics...")
        page_rolling_metrics(pdf, df)

        print("  Page 9: Exit reasons + top/bottom tickers...")
        page_exit_and_tickers(pdf, df)

        print("  Page 10: Risk table + Win Rate vs R:R...")
        page_risk_table(pdf, df)

    print(f"\n[DONE] Report saved → {output_pdf.relative_to(PROJECT_ROOT)}")
    print(f"       Trades CSV  → {(run_dir / 'rbs_all_trades.csv').relative_to(PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
