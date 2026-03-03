"""
RBS Backtest Framework — Monolith Base
======================================
Everything in one file:
  data loading → backtest → portfolio sim → metrics → PDF report.

Usage::

    from src.rbs.kumo_break import KumoBreak

    s = KumoBreak()        # or KumoBreak(data_dir="path/to/csvs")
    s.run()                # backtest all tickers
    s.simulate()           # portfolio-level sim
    print(s.summary())     # metrics summary
    s.report("runs/out/")  # export PDF
"""

from __future__ import annotations

import abc
import datetime as dt
import json
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class IchimokuParams:
    """All tuneable params — sensible Sadekar defaults."""
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    displacement: int = 26
    min_rows: int = 250
    warmup: int = 80
    long_only: bool = True
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    transaction_cost: float = 0.005
    order_timeout: int = 5
    chikou_free_half: int = 2
    max_kijun_dist: float = 5.0
    hh_proximity: float = 1.5
    flat_sb_lookback: int = 15
    flat_sb_tol: float = 0.001
    flat_sb_strong: float = 0.5
    seed: int = 42


class SizingMode(Enum):
    FIXED = "fixed"
    COMPOUNDING = "compounding"


@dataclass
class TradeRecord:
    """Single trade output."""
    ticker: str
    direction: str
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    bars_held: int
    pnl_pct: float
    net_pnl_pct: float
    exit_reason: str
    strategy: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "ticker": self.ticker, "strategy": self.strategy,
            "direction": self.direction,
            "entry_date": self.entry_date, "exit_date": self.exit_date,
            "entry_price": self.entry_price, "exit_price": self.exit_price,
            "bars_held": self.bars_held,
            "pnl_pct": self.pnl_pct, "net_pnl_pct": self.net_pnl_pct,
            "exit_reason": self.exit_reason,
        }
        d.update(self.extra)
        return d


@dataclass
class StrategyMetrics:
    """Aggregate trade metrics."""
    n_trades: int = 0
    n_tickers: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    median_pnl: float = 0.0
    std_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    avg_bars_held: float = 0.0
    median_bars_held: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


# ============================================================================
# ICHIMOKU HELPERS
# ============================================================================

def compute_ichimoku(df: pd.DataFrame, p: IchimokuParams) -> Dict[str, np.ndarray]:
    """All Ichimoku components, lookahead-free."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tenkan = (h.rolling(p.tenkan_period).max() + l.rolling(p.tenkan_period).min()) / 2
    kijun  = (h.rolling(p.kijun_period).max()  + l.rolling(p.kijun_period).min())  / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (h.rolling(p.senkou_b_period).max() + l.rolling(p.senkou_b_period).min()) / 2
    senkou_a = senkou_a_raw.shift(p.displacement)
    senkou_b = senkou_b_raw.shift(p.displacement)
    kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    kumo_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p.atr_period).mean()
    return {
        "tenkan": tenkan.values, "kijun": kijun.values,
        "senkou_a": senkou_a.values, "senkou_b": senkou_b.values,
        "kumo_top": kumo_top.values, "kumo_bot": kumo_bot.values,
        "future_sa": senkou_a_raw.values, "future_sb": senkou_b_raw.values,
        "atr": atr.values,
    }


def chikou_is_free(c_arr, h_arr, l_arr, t, direction, sa_arr, sb_arr,
                   displacement, half_window):
    center = t - displacement
    lo = max(0, center - half_window)
    hi = min(len(c_arr), center + half_window + 1)
    if lo >= hi or center < 0:
        return False
    if direction == "long":
        barrier = np.nanmax(h_arr[lo:hi])
        if np.isnan(barrier) or c_arr[t] <= barrier:
            return False
        kumo_top_hist = max(sa_arr[center], sb_arr[center])
        if not np.isnan(kumo_top_hist) and c_arr[t] <= kumo_top_hist:
            return False
        return True
    else:
        barrier = np.nanmin(l_arr[lo:hi])
        if np.isnan(barrier) or c_arr[t] >= barrier:
            return False
        kumo_bot_hist = min(sa_arr[center], sb_arr[center])
        if not np.isnan(kumo_bot_hist) and c_arr[t] >= kumo_bot_hist:
            return False
        return True


def is_senkou_b_flat(sb_arr, t, lookback, tol):
    if t < lookback:
        return False
    recent = sb_arr[t - lookback : t + 1]
    if np.any(np.isnan(recent)):
        return False
    return (np.nanmax(recent) - np.nanmin(recent)) / max(np.nanmean(recent), 1e-10) < tol


def flat_sb_strong_candle(o_t, c_t, atr_t, direction, threshold):
    body = (c_t - o_t) if direction == "long" else (o_t - c_t)
    return body >= threshold * atr_t


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def _compute_streaks(pnl_arr: np.ndarray):
    win_streaks, loss_streaks = [], []
    if len(pnl_arr) == 0:
        return win_streaks, loss_streaks
    is_win = pnl_arr > 0
    current_val = is_win[0]
    current_len = 1
    for i in range(1, len(pnl_arr)):
        if is_win[i] == current_val:
            current_len += 1
        else:
            (win_streaks if current_val else loss_streaks).append(current_len)
            current_val = is_win[i]
            current_len = 1
    (win_streaks if current_val else loss_streaks).append(current_len)
    return win_streaks, loss_streaks


def compute_metrics(df: pd.DataFrame, pnl_col: str = "net_pnl_pct") -> StrategyMetrics:
    m = StrategyMetrics()
    if df.empty:
        return m
    pnl = df[pnl_col]
    m.n_trades = len(df)
    m.n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    wins = df[pnl > 0]
    losses = df[pnl <= 0]
    m.n_wins = len(wins)
    m.n_losses = len(losses)
    m.win_rate = m.n_wins / m.n_trades * 100 if m.n_trades else 0
    m.total_pnl = pnl.sum()
    m.avg_pnl = pnl.mean()
    m.median_pnl = pnl.median()
    m.std_pnl = pnl.std() if m.n_trades > 1 else 0
    m.expectancy = m.avg_pnl
    m.avg_win = wins[pnl_col].mean() if len(wins) else 0
    m.avg_loss = losses[pnl_col].mean() if len(losses) else 0
    m.best_trade = pnl.max()
    m.worst_trade = pnl.min()
    gross_win = wins[pnl_col].sum() if len(wins) else 0
    gross_loss = abs(losses[pnl_col].sum()) if len(losses) else 0
    m.profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else float("inf")
    if "bars_held" in df.columns:
        m.avg_bars_held = df["bars_held"].mean()
        m.median_bars_held = df["bars_held"].median()
    m.skewness = pnl.skew() if m.n_trades > 2 else 0
    m.kurtosis = pnl.kurtosis() if m.n_trades > 3 else 0
    ws, ls = _compute_streaks(pnl.values)
    m.max_win_streak = max(ws) if ws else 0
    m.max_loss_streak = max(ls) if ls else 0
    m.avg_win_streak = np.mean(ws) if ws else 0
    m.avg_loss_streak = np.mean(ls) if ls else 0
    return m


def yearly_metrics(df: pd.DataFrame, pnl_col: str = "net_pnl_pct") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    rows = []
    for year, grp in df.groupby("year"):
        g = grp[pnl_col]
        w = grp[g > 0]
        lo = grp[g <= 0]
        gw = w[pnl_col].sum() if len(w) else 0
        gl = abs(lo[pnl_col].sum()) if len(lo) else 0
        rows.append({
            "year": year, "trades": len(grp), "total_pnl": g.sum(),
            "avg_pnl": g.mean(), "median_pnl": g.median(),
            "win_rate": len(w) / len(grp) * 100 if len(grp) else 0,
            "profit_factor": gw / gl if gl > 0 else float("inf"),
        })
    return pd.DataFrame(rows).set_index("year")


def ticker_metrics(df: pd.DataFrame, pnl_col: str = "net_pnl_pct",
                   min_trades: int = 1) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for ticker, grp in df.groupby("ticker"):
        if len(grp) < min_trades:
            continue
        g = grp[pnl_col]
        w = grp[g > 0]
        rows.append({
            "ticker": ticker, "trades": len(grp), "total_pnl": g.sum(),
            "avg_pnl": g.mean(), "win_rate": len(w) / len(grp) * 100,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("total_pnl", ascending=False).set_index("ticker")
    return result


def monthly_returns(df: pd.DataFrame, pnl_col: str = "net_pnl_pct") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    df["month"] = pd.to_datetime(df["entry_date"]).dt.month
    return df.pivot_table(values=pnl_col, index="year", columns="month",
                          aggfunc="mean", fill_value=np.nan)


# ============================================================================
# DATA LOADING
# ============================================================================

def _build_exclude_set(project_root: pathlib.Path,
                       extra: set | None = None) -> set:
    excl = set(extra or set())
    stocks_json = project_root / "data/stocks.json"
    if stocks_json.exists():
        with open(stocks_json, encoding="utf-8") as f:
            for s in json.load(f):
                if s.get("sector") == "PROMOTSHARE":
                    excl.add(s["script"])
    for fname in ("mutual.json", "corpdeben.json"):
        p = project_root / "data" / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for s in json.load(f):
                    excl.add(s["script"])
    return excl


def load_ohlcv(
    data_dir: pathlib.Path,
    project_root: pathlib.Path,
    min_rows: int = 250,
    extra_exclude: set | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load all qualifying OHLCV CSVs. Returns {ticker: DataFrame}."""
    exclude = _build_exclude_set(project_root, extra_exclude)
    frames: Dict[str, pd.DataFrame] = {}

    for csv in sorted(data_dir.glob("*.csv")):
        ticker = csv.stem
        if ticker in exclude:
            continue
        try:
            df = pd.read_csv(csv, parse_dates=["Timestamp"])
            if df.empty or len(df) < min_rows:
                continue
            df = df.rename(columns={"Timestamp": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(df.columns):
                continue
            df = df[list(needed)].dropna()
            if len(df) < min_rows:
                continue
            frames[ticker] = df
        except Exception:
            continue
    return frames


# ============================================================================
# PORTFOLIO SIMULATOR
# ============================================================================

def simulate_portfolio(
    signals: pd.DataFrame,
    initial_capital: float = 1_000_000,
    max_slots: int = 8,
    friction_pct: float = 1.5,
    sizing: str = "compounding",
    seed: int = 42,
    filter_after: str | None = "2010-01-01",
) -> dict:
    """
    Day-by-day portfolio sim.

    Returns dict with keys: trades_df, equity, total_return_pct,
    max_drawdown_pct, n_trades, win_rate, skipped, initial_capital.
    """
    np.random.seed(seed)
    sig = signals.copy()
    if filter_after:
        sig = sig[sig["entry_date"] >= pd.Timestamp(filter_after)].copy()
    if sig.empty:
        eq = pd.Series([initial_capital], index=[pd.Timestamp.now()], name="equity")
        return {
            "trades_df": pd.DataFrame(), "equity": eq,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "n_trades": 0, "win_rate": 0.0, "skipped": 0,
            "initial_capital": initial_capital,
        }

    sig_by_day = sig.groupby("entry_date")
    all_dates = sorted(sig["entry_date"].unique())
    exit_dates = sig["exit_date"].dropna().unique()
    calendar = sorted(set(all_dates) | set(exit_dates))

    open_trades: list = []
    trades_log: list = []
    pv = float(initial_capital)
    skipped = 0
    eq_dates, eq_vals = [], []

    for day in calendar:
        # close
        still = []
        for t in open_trades:
            if t["exit_date"] <= day:
                adj = t["net_pnl_pct"] - friction_pct
                pnl_npr = t["trade_size"] * adj / 100.0
                pv += pnl_npr
                trades_log.append({**t, "adj_pnl_pct": adj, "pnl_npr": pnl_npr,
                                   "friction_pct": friction_pct})
            else:
                still.append(t)
        open_trades = still

        # open
        free = max_slots - len(open_trades)
        if free > 0 and day in sig_by_day.groups:
            ds = sig_by_day.get_group(day).sample(frac=1.0, random_state=seed)
            tsz = max(pv / max_slots, 0) if sizing == "compounding" else initial_capital / max_slots
            for _, row in ds.iterrows():
                if free <= 0:
                    skipped += 1
                    continue
                open_trades.append({
                    "ticker": row["ticker"], "strategy": row.get("strategy", ""),
                    "entry_date": row["entry_date"], "exit_date": row["exit_date"],
                    "direction": row["direction"],
                    "net_pnl_pct": row["net_pnl_pct"], "trade_size": tsz,
                })
                free -= 1
        elif day in sig_by_day.groups:
            skipped += len(sig_by_day.get_group(day))

        # mtm
        mtm = 0.0
        for t in open_trades:
            td = (t["exit_date"] - t["entry_date"]).days
            el = (day - t["entry_date"]).days
            frac = min(el / td, 1.0) if td > 0 else 1.0
            mtm += t["trade_size"] * (t["net_pnl_pct"] - friction_pct) / 100.0 * frac
        eq_dates.append(day)
        eq_vals.append(pv + mtm)

    equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates), name="equity")
    tdf = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    peak = equity.expanding().max()
    dd = ((equity - peak) / peak * 100).min()
    wr = 0.0
    if trades_log:
        wr = sum(1 for t in trades_log if t["adj_pnl_pct"] > 0) / len(trades_log) * 100

    return {
        "trades_df": tdf,
        "equity": equity,
        "total_return_pct": (equity.iloc[-1] / initial_capital - 1) * 100,
        "max_drawdown_pct": dd,
        "n_trades": len(trades_log),
        "win_rate": wr,
        "skipped": skipped,
        "initial_capital": initial_capital,
    }


# ============================================================================
# PDF REPORT
# ============================================================================

CMAP_RG = LinearSegmentedColormap.from_list("RedGreen", [
    (0.0,  "#8B0000"), (0.15, "#DC143C"), (0.35, "#FF6B6B"),
    (0.5,  "#FFFFFF"),
    (0.65, "#90EE90"), (0.85, "#228B22"), (1.0,  "#006400"),
])

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FAFAFA",
    "font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.dpi": 150,
})


def _page_yearly_bars(pdf, df, pnl_col):
    ym = yearly_metrics(df, pnl_col)
    if ym.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Year-wise Aggregate Performance", fontsize=14, fontweight="bold", y=0.99)
    ax = axes[0, 0]
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in ym["total_pnl"]]
    ax.bar(ym.index.astype(str), ym["total_pnl"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=0.5); ax.set_title("Total PnL %", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax = axes[0, 1]
    colors = ["#228B22" if v >= 35 else "#DC143C" if v < 25 else "#FFA500" for v in ym["win_rate"]]
    ax.bar(ym.index.astype(str), ym["win_rate"], color=colors, edgecolor="gray", linewidth=0.3)
    ax.axhline(33, color="gray", linestyle="--", linewidth=0.8); ax.set_title("Win Rate %", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax = axes[1, 0]
    ax.bar(ym.index.astype(str), ym["trades"], color="#4a90d9", edgecolor="gray", linewidth=0.3)
    ax.set_title("Trade Count", fontweight="bold"); ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax = axes[1, 1]
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in ym["avg_pnl"]]
    ax.bar(ym.index.astype(str), ym["avg_pnl"], color=colors, alpha=0.6, edgecolor="gray", linewidth=0.3, label="Mean")
    ax.plot(ym.index.astype(str), ym["median_pnl"], "ko-", markersize=4, linewidth=1.2, label="Median")
    ax.axhline(0, color="black", linewidth=0.5); ax.set_title("Mean & Median PnL %", fontweight="bold")
    ax.legend(fontsize=8); ax.tick_params(axis="x", rotation=45, labelsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _page_monthly_heatmap(pdf, df, pnl_col):
    mr = monthly_returns(df, pnl_col)
    if mr.empty:
        return
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    strategies = df["strategy"].unique() if "strategy" in df.columns else ["All"]
    fig, axes = plt.subplots(1, max(len(strategies), 1), figsize=(16, 8), squeeze=False)
    fig.suptitle("Monthly Returns Heatmap", fontsize=14, fontweight="bold", y=0.99)
    for idx, strat in enumerate(strategies):
        ax = axes[0, idx]
        sub = df[df["strategy"] == strat] if "strategy" in df.columns else df
        pivot = sub.pivot_table(values=pnl_col, index="year", columns="month",
                                aggfunc="mean", fill_value=np.nan)
        for mo in range(1, 13):
            if mo not in pivot.columns:
                pivot[mo] = np.nan
        pivot = pivot[sorted(pivot.columns)]
        vmax = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 0.1)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        ax.imshow(pivot.values, cmap=CMAP_RG, norm=norm, aspect="auto")
        ax.set_xticks(range(12)); ax.set_xticklabels(month_names, fontsize=8)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title(f"{strat}", fontsize=11, fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    c = "white" if abs(val) > vmax * 0.55 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=c)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _page_pnl_dist(pdf, df, pnl_col):
    from scipy import stats as sp_stats
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PnL Distribution Analysis", fontsize=14, fontweight="bold", y=0.99)
    pnl = df[pnl_col].dropna()
    ax = axes[0, 0]
    ax.hist(pnl, bins=60, color="#4a90d9", alpha=0.7, edgecolor="white", linewidth=0.3, density=True)
    if len(pnl) > 5:
        kde = sp_stats.gaussian_kde(pnl)
        x_kde = np.linspace(pnl.min(), pnl.max(), 200)
        ax.plot(x_kde, kde(x_kde), color="#DC143C", linewidth=1.5)
    ax.axvline(pnl.mean(), color="red", linestyle="--", label=f"Mean: {pnl.mean():.2f}%")
    ax.axvline(pnl.median(), color="green", linestyle="--", label=f"Median: {pnl.median():.2f}%")
    ax.set_title("PnL Distribution", fontweight="bold"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    if "strategy" in df.columns:
        for strat, color in zip(df["strategy"].unique(), ["#e8710a", "#1a73e8", "#34a853", "#ea4335"]):
            sub = df[df["strategy"] == strat][pnl_col].dropna()
            ax.hist(sub, bins=40, color=color, alpha=0.5, density=True, label=f"{strat} (μ={sub.mean():.2f}%)")
    ax.set_title("By Strategy", fontweight="bold"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    if "strategy" in df.columns:
        strats = df["strategy"].unique()
        data = [df[df["strategy"] == s][pnl_col].dropna().values for s in strats]
        bp = ax.boxplot(data, tick_labels=strats, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=5))
        for patch, c in zip(bp["boxes"], ["#e8710a", "#1a73e8", "#34a853", "#ea4335"]):
            patch.set_facecolor(c); patch.set_alpha(0.4)
    ax.axhline(0, color="black", linewidth=0.5); ax.set_title("Box Plot", fontweight="bold")
    ax = axes[1, 1]; ax.axis("off")
    m = compute_metrics(df, pnl_col)
    rows = [
        ["Trades", f"{m.n_trades}"], ["Win Rate %", f"{m.win_rate:.1f}"],
        ["Mean PnL %", f"{m.avg_pnl:.2f}"], ["Median PnL %", f"{m.median_pnl:.2f}"],
        ["Std Dev %", f"{m.std_pnl:.2f}"], ["Profit Factor", f"{m.profit_factor:.2f}"],
        ["Payoff Ratio", f"{m.payoff_ratio:.2f}"], ["Best Trade %", f"{m.best_trade:.2f}"],
        ["Worst Trade %", f"{m.worst_trade:.2f}"], ["Skewness", f"{m.skewness:.2f}"],
        ["Kurtosis", f"{m.kurtosis:.2f}"],
    ]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.5)
    for j in range(2):
        table[0, j].set_facecolor("#333333"); table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Summary Statistics", fontweight="bold", pad=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _page_duration(pdf, df, pnl_col):
    if "bars_held" not in df.columns:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Trade Duration Analysis", fontsize=14, fontweight="bold", y=0.99)
    bars = df["bars_held"].dropna()
    ax = axes[0, 0]
    ax.hist(bars, bins=range(0, int(bars.max()) + 2), color="#4a90d9", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(bars.mean(), color="red", linestyle="--", label=f"Mean: {bars.mean():.1f}")
    ax.axvline(bars.median(), color="green", linestyle="--", label=f"Median: {bars.median():.0f}")
    ax.set_title("Bars Held Distribution", fontweight="bold"); ax.legend(fontsize=8)
    ax = axes[0, 1]
    w = df[df[pnl_col] > 0]; lo = df[df[pnl_col] <= 0]
    ax.scatter(lo["bars_held"], lo[pnl_col], c="#DC143C", alpha=0.3, s=10, label="Losses")
    ax.scatter(w["bars_held"], w[pnl_col], c="#228B22", alpha=0.3, s=10, label="Wins")
    ax.axhline(0, color="black", linewidth=0.5); ax.set_title("PnL vs Duration", fontweight="bold")
    ax.legend(fontsize=8)
    dc = df.copy()
    dc["dur_bin"] = pd.cut(dc["bars_held"], bins=[0, 5, 10, 15, 20, 30, 50, 100, 500],
                           labels=["1-5", "6-10", "11-15", "16-20", "21-30", "31-50", "51-100", "100+"])
    dur = dc.groupby("dur_bin", observed=True).agg(avg_pnl=(pnl_col, "mean"), count=(pnl_col, "count"), wr=("win", "mean"))
    ax = axes[1, 0]
    colors = ["#228B22" if v >= 0 else "#DC143C" for v in dur["avg_pnl"]]
    bp = ax.bar(dur.index.astype(str), dur["avg_pnl"], color=colors, alpha=0.7, edgecolor="gray", linewidth=0.3)
    for bar, cnt in zip(bp, dur["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={cnt}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5); ax.set_title("Avg PnL by Duration", fontweight="bold")
    ax = axes[1, 1]
    dur["wr_pct"] = dur["wr"] * 100
    colors = ["#228B22" if v >= 35 else "#DC143C" if v < 25 else "#FFA500" for v in dur["wr_pct"]]
    ax.bar(dur.index.astype(str), dur["wr_pct"], color=colors, alpha=0.7, edgecolor="gray", linewidth=0.3)
    ax.axhline(33, color="gray", linestyle="--", linewidth=0.8); ax.set_title("Win Rate by Duration", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _page_top_bottom(pdf, df, pnl_col):
    tm = ticker_metrics(df, pnl_col, min_trades=3)
    if tm.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Top & Bottom Tickers (min 3 trades)", fontsize=14, fontweight="bold", y=0.99)
    for ax, subset, title in [(axes[0], tm.head(15), "Top 15"),
                               (axes[1], tm.tail(15).iloc[::-1], "Bottom 15")]:
        colors = ["#228B22" if v >= 0 else "#DC143C" for v in subset["total_pnl"]]
        bars = ax.barh(subset.index, subset["total_pnl"], color=colors, edgecolor="gray", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.5); ax.set_title(f"{title} (Total PnL %)", fontweight="bold")
        for bar, cnt in zip(bars, subset["trades"]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" n={int(cnt)}", va="center", fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def _page_equity(pdf, equity, initial_capital):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Portfolio Equity Curve", fontsize=14, fontweight="bold", y=0.99)
    eq = equity
    ax = axes[0]
    ax.plot(eq.index, eq.values, color="#1a73e8", linewidth=1)
    ax.fill_between(eq.index, initial_capital, eq.values, where=eq.values >= initial_capital, color="#34a853", alpha=0.15)
    ax.fill_between(eq.index, initial_capital, eq.values, where=eq.values < initial_capital, color="#ea4335", alpha=0.15)
    ax.axhline(initial_capital, color="gray", linestyle="--", linewidth=0.5)
    ret = (eq.iloc[-1] / initial_capital - 1) * 100
    ax.set_title(f"Return: {ret:+.2f}%", fontweight="bold")
    ax.set_ylabel("Portfolio Value (Rs)"); ax.grid(True, alpha=0.3)
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    ax = axes[1]
    ax.fill_between(eq.index, 0, dd.values, color="#ea4335", alpha=0.4)
    ax.set_ylabel("Drawdown %"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)


def generate_report(
    df: pd.DataFrame,
    output_path: pathlib.Path | str,
    equity: pd.Series | None = None,
    initial_capital: float = 1_000_000,
    pnl_col: str = "net_pnl_pct",
) -> pathlib.Path:
    """Generate multi-page PDF report. Returns output path."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["entry_date"].dt.year
    df["month"] = df["entry_date"].dt.month
    df["win"] = (df[pnl_col] > 0).astype(int)
    with PdfPages(output_path) as pdf:
        _page_yearly_bars(pdf, df, pnl_col)
        _page_monthly_heatmap(pdf, df, pnl_col)
        _page_pnl_dist(pdf, df, pnl_col)
        _page_duration(pdf, df, pnl_col)
        _page_top_bottom(pdf, df, pnl_col)
        if equity is not None:
            _page_equity(pdf, equity, initial_capital)
    return output_path


# ============================================================================
# THE BASE CLASS — everything wired together
# ============================================================================

class BacktestStrategy(abc.ABC):
    """
    Monolith base class.

    Subclass, implement ``name`` + ``backtest_ticker()``, then::

        s = MyStrategy()
        s.run()
        s.simulate()
        print(s.summary())
        s.report("runs/out/")
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path = "data/ohlcv/1D/stocks",
        project_root: str | pathlib.Path | None = None,
        params: IchimokuParams | None = None,
        min_rows: int = 250,
    ) -> None:
        self.params = params or IchimokuParams()
        self._project_root = pathlib.Path(
            project_root or pathlib.Path(__file__).resolve().parents[2]
        )
        self._data_dir = pathlib.Path(data_dir)
        if not self._data_dir.is_absolute():
            self._data_dir = self._project_root / self._data_dir
        self._min_rows = min_rows

        self._frames: Dict[str, pd.DataFrame] = {}
        self._trades: List[TradeRecord] = []
        self._has_run = False
        self._sim_result: dict | None = None

    # ── Abstract ─────────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def backtest_ticker(self, ticker: str, df: pd.DataFrame) -> List[TradeRecord]: ...

    # ── Data ─────────────────────────────────────────────────────────

    def load_data(self, *, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data. Cached after first call."""
        if self._frames and not force:
            return self._frames
        self._frames = load_ohlcv(
            self._data_dir, self._project_root, self._min_rows,
        )
        return self._frames

    @property
    def tickers(self) -> list[str]:
        return sorted(self._frames.keys())

    @property
    def n_loaded(self) -> int:
        return len(self._frames)

    # ── Run ──────────────────────────────────────────────────────────

    def run(self, *, tickers: list[str] | None = None) -> List[TradeRecord]:
        """Backtest all (or selected) tickers."""
        frames = self.load_data()
        target = tickers or sorted(frames.keys())
        self._trades = []
        for tk in target:
            df = frames.get(tk)
            if df is None:
                continue
            trades = self.backtest_ticker(tk, df)
            for t in trades:
                t.strategy = self.name
            self._trades.extend(trades)
        self._has_run = True
        return self._trades

    # ── Trades access ────────────────────────────────────────────────

    @property
    def trades(self) -> List[TradeRecord]:
        if not self._has_run:
            raise RuntimeError("Call .run() first")
        return self._trades

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self._has_run:
            raise RuntimeError("Call .run() first")
        rows = [t.to_dict() for t in self._trades]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        return df.sort_values("entry_date").reset_index(drop=True)

    # ── Metrics ──────────────────────────────────────────────────────

    @property
    def metrics(self) -> StrategyMetrics:
        return compute_metrics(self.trades_df)

    def yearly(self) -> pd.DataFrame:
        return yearly_metrics(self.trades_df)

    def by_ticker(self, min_trades: int = 3) -> pd.DataFrame:
        return ticker_metrics(self.trades_df, min_trades=min_trades)

    def monthly(self) -> pd.DataFrame:
        return monthly_returns(self.trades_df)

    # ── Portfolio sim ────────────────────────────────────────────────

    def simulate(
        self,
        capital: float = 1_000_000,
        slots: int = 8,
        friction: float = 1.5,
        sizing: str = "compounding",
        filter_after: str | None = "2010-01-01",
    ) -> dict:
        """Run day-by-day portfolio simulation on this strategy's trades."""
        self._sim_result = simulate_portfolio(
            self.trades_df,
            initial_capital=capital,
            max_slots=slots,
            friction_pct=friction,
            sizing=sizing,
            seed=self.params.seed,
            filter_after=filter_after,
        )
        return self._sim_result

    @property
    def portfolio(self) -> dict:
        if self._sim_result is None:
            raise RuntimeError("Call .simulate() first")
        return self._sim_result

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"Strategy        : {self.name}",
            f"Tickers Loaded  : {self.n_loaded}",
            f"Trades          : {m.n_trades}",
            f"Win Rate        : {m.win_rate:.1f}%",
            f"Total PnL       : {m.total_pnl:+.2f}%",
            f"Avg PnL         : {m.avg_pnl:+.2f}%",
            f"Profit Factor   : {m.profit_factor:.2f}",
            f"Payoff Ratio    : {m.payoff_ratio:.2f}",
            f"Best Trade      : {m.best_trade:+.2f}%",
            f"Worst Trade     : {m.worst_trade:+.2f}%",
            f"Max Win Streak  : {m.max_win_streak}",
            f"Max Loss Streak : {m.max_loss_streak}",
        ]
        if self._sim_result is not None:
            sr = self._sim_result
            lines += [
                f"\n--- Portfolio ---",
                f"Return          : {sr['total_return_pct']:+.2f}%",
                f"Max Drawdown    : {sr['max_drawdown_pct']:.2f}%",
                f"Executed        : {sr['n_trades']}",
                f"Skipped         : {sr['skipped']}",
                f"Port Win Rate   : {sr['win_rate']:.1f}%",
            ]
        return "\n".join(lines)

    # ── Report ───────────────────────────────────────────────────────

    def report(self, output_dir: str | pathlib.Path = "runs") -> pathlib.Path:
        """Generate multi-page PDF report."""
        out = pathlib.Path(output_dir)
        if not out.suffix:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = out / stamp
        out.mkdir(parents=True, exist_ok=True)
        pdf_path = out / f"{self.name.lower().replace(' ', '_').replace('/', '_')}_report.pdf"

        equity = self._sim_result["equity"] if self._sim_result else None
        cap = self._sim_result["initial_capital"] if self._sim_result else 1_000_000
        generate_report(self.trades_df, pdf_path, equity=equity, initial_capital=cap)

        # also dump CSVs
        self.trades_df.to_csv(out / "trades.csv", index=False)
        if self._sim_result is not None:
            eq = self._sim_result["equity"]
            eq.reset_index().to_csv(out / "equity_curve.csv", index=False)
            self.yearly().to_csv(out / "yearly_metrics.csv")

        return pdf_path

    def __repr__(self) -> str:
        n = len(self._trades) if self._has_run else "?"
        return f"<{self.__class__.__name__} '{self.name}' trades={n}>"
