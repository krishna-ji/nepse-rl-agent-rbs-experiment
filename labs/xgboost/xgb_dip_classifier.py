#!/usr/bin/env python
"""
XGBoost Dip-Buy Mean-Reversion Classifier
==========================================
Single-file pipeline:
  1. Extract "Dip Events" (EMA200 trend + Stoch<20 + Close < Lower BB)
  2. Label via Triple-Barrier Method (±6%/−3%, 10-bar horizon)
  3. Engineer stationary feature matrix
  4. Train XGBoost with monotone constraints & purged walk-forward CV
  5. Backtest ML-filtered vs baseline dip-buy
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import sys
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "ohlcv" / "1D" / "stocks"
RUN_DIR      = PROJECT_ROOT / "runs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
SPLIT_DATE       = "2024-07-01"
MIN_ROWS         = 300        # minimum bars to qualify a ticker
SEED             = 42
TRANSACTION_COST = 0.005      # 0.5 % per side

# triple-barrier params
TB_HORIZON       = 10         # forward look window (bars)
TB_TP_PCT        = 0.06       # +6 % take-profit
TB_SL_PCT        = 0.03       # −3 % stop-loss

# ML threshold
ML_PROB_THRESH   = 0.55       # minimum P(Win) to take trade (tuned below)

# ── logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger("xgb_dip")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); log.addHandler(_sh)
_fh = logging.FileHandler(RUN_DIR / "run.log"); _fh.setFormatter(_fmt); log.addHandler(_fh)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 : Data Loading & Universe
# ═══════════════════════════════════════════════════════════════════════════════

def _build_exclude_set() -> set[str]:
    excl: set[str] = set()
    stocks_path = PROJECT_ROOT / "data" / "stocks.json"
    if stocks_path.exists():
        for rec in json.loads(stocks_path.read_text()):
            if rec.get("sector") == "PROMOTSHARE":
                excl.add(rec["script"])
    for fname in ("mutual.json", "corpdeben.json"):
        fp = PROJECT_ROOT / "data" / fname
        if fp.exists():
            for rec in json.loads(fp.read_text()):
                excl.add(rec.get("script", rec.get("symbol", "")))
    return excl


def load_ohlcv(csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["Timestamp"])
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 : Technical Indicator Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _slow_stochastic(
    h: pd.Series, l: pd.Series, c: pd.Series,
    k_period: int = 14, k_smooth: int = 3, d_smooth: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Return (slow_%K, slow_%D)."""
    lowest  = l.rolling(k_period, min_periods=k_period).min()
    highest = h.rolling(k_period, min_periods=k_period).max()
    fast_k  = 100 * (c - lowest) / (highest - lowest + 1e-10)
    slow_k  = fast_k.rolling(k_smooth, min_periods=k_smooth).mean()   # smoothed %K
    slow_d  = slow_k.rolling(d_smooth, min_periods=d_smooth).mean()   # %D
    return slow_k, slow_d


def _bollinger(c: pd.Series, n: int = 20, k: float = 2.0):
    sma = _sma(c, n)
    std = c.rolling(n, min_periods=n).std()
    return sma, sma + k * std, sma - k * std  # mid, upper, lower


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 : Event Extraction & Triple-Barrier Labeling
# ═══════════════════════════════════════════════════════════════════════════════

def extract_dip_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where ALL three conditions hold:
      1. Close > EMA(200)  (bull regime)
      2. Slow Stoch %K < 20  (oversold)
      3. Close < Lower Bollinger Band (20, 2)  (stretched)
    Returns a boolean-indexed subset of df.
    """
    c = df["Close"]
    ema200 = _ema(c, 200)
    stoch_k, _ = _slow_stochastic(df["High"], df["Low"], c)
    _, _, bb_lower = _bollinger(c, 20, 2.0)

    cond_trend     = c > ema200
    cond_stoch     = stoch_k < 20
    cond_bollinger = c < bb_lower

    mask = cond_trend & cond_stoch & cond_bollinger
    return mask


def triple_barrier_label(
    df: pd.DataFrame, events_idx: pd.DatetimeIndex,
    tp_pct: float = TB_TP_PCT, sl_pct: float = TB_SL_PCT,
    horizon: int = TB_HORIZON,
) -> pd.Series:
    """
    For each event T0, look forward `horizon` bars:
      1 = hit +tp_pct first,  0 = hit −sl_pct first or time expires.
    Returns pd.Series aligned to events_idx.
    """
    labels = pd.Series(np.nan, index=events_idx, dtype=float)
    closes = df["Close"].values
    idx_all = df.index
    pos_map = {d: i for i, d in enumerate(idx_all)}  # date → iloc

    for t0 in events_idx:
        i0 = pos_map.get(t0)
        if i0 is None:
            continue
        entry = closes[i0]
        upper = entry * (1 + tp_pct)
        lower = entry * (1 - sl_pct)
        end = min(i0 + horizon, len(closes) - 1)
        label = 0  # default: time expiry / stop-loss
        for j in range(i0 + 1, end + 1):
            if closes[j] >= upper:
                label = 1
                break
            if closes[j] <= lower:
                label = 0
                break
        labels[t0] = label
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 : Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct stationary, bounded features for every row.
    (Only event rows will be used for modelling.)
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    stoch_k, stoch_d = _slow_stochastic(h, l, c)
    sma20 = _sma(c, 20)
    std20 = c.rolling(20, min_periods=20).std()
    ema200 = _ema(c, 200)
    atr5  = _atr(h, l, c, 5)
    atr14 = _atr(h, l, c, 14)
    atr20 = _atr(h, l, c, 20)
    bb_mid, bb_upper, bb_lower = _bollinger(c, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    bb_width_max_100 = bb_width.rolling(100, min_periods=50).max()
    vol_sma20 = _sma(v, 20)

    feats = pd.DataFrame(index=df.index)

    # 1. Oscillator Dynamics
    feats["stoch_k_level"]  = stoch_k
    feats["stoch_d_level"]  = stoch_d
    feats["stoch_k_slope"]  = stoch_k - stoch_k.shift(3)

    # stoch divergence: price lower-low but %K higher-low over last 10 bars
    price_ll = c == c.rolling(10, min_periods=5).min()
    stoch_hl = stoch_k > stoch_k.rolling(10, min_periods=5).min().shift(1)
    feats["stoch_divergence"] = (price_ll & stoch_hl).astype(int)

    # 2. Standardized Price Distance
    feats["dist_sma20_z"]    = (c - sma20) / (std20 + 1e-10)
    feats["dist_ema200_pct"] = (c - ema200) / (ema200 + 1e-10)

    # 3. Volatility Context
    feats["bb_width_norm"] = bb_width / (bb_width_max_100 + 1e-10)
    feats["atr_ratio"]     = atr5 / (atr20 + 1e-10)
    feats["atr14_pct"]     = atr14 / (c + 1e-10)   # relative volatility

    # 4. Volume Exhaustion
    feats["vol_surge_ratio"] = v / (vol_sma20 + 1e-10)

    # 5. Extra useful bounded features
    feats["close_vs_bb_lower"] = (c - bb_lower) / (atr14 + 1e-10)  # ATR-normalised
    feats["bar_range_ratio"]   = (h - l) / (atr14 + 1e-10)

    return feats


FEATURE_COLS = [
    "stoch_k_level", "stoch_d_level", "stoch_k_slope", "stoch_divergence",
    "dist_sma20_z", "dist_ema200_pct",
    "bb_width_norm", "atr_ratio", "atr14_pct",
    "vol_surge_ratio",
    "close_vs_bb_lower", "bar_range_ratio",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 : Dataset Assembly (All Tickers)
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_dataset() -> pd.DataFrame:
    """
    Iterate over all tickers, extract dip events, label, feature-ize,
    return a single DataFrame with columns: [features..., label, ticker, Date].
    """
    excl = _build_exclude_set()
    csvs = sorted(DATA_DIR.glob("*.csv"))
    rows_list: list[pd.DataFrame] = []
    tickers_scanned = 0
    tickers_with_events = 0

    for csv in csvs:
        ticker = csv.stem
        if ticker in excl:
            continue
        df = load_ohlcv(csv)
        if len(df) < MIN_ROWS:
            continue
        tickers_scanned += 1

        event_mask = extract_dip_events(df)
        events_idx = df.index[event_mask]
        if len(events_idx) == 0:
            continue

        labels = triple_barrier_label(df, events_idx)
        feats  = build_features(df)

        ev = feats.loc[events_idx, FEATURE_COLS].copy()
        ev["label"]  = labels
        ev["ticker"] = ticker
        ev = ev.dropna(subset=FEATURE_COLS + ["label"])
        if len(ev) > 0:
            rows_list.append(ev)
            tickers_with_events += 1

    if not rows_list:
        log.error("No dip events found across any ticker!")
        sys.exit(1)

    dataset = pd.concat(rows_list).sort_index()
    dataset.index.name = "Date"
    log.info(
        f"Dataset: {len(dataset)} events from {tickers_with_events} tickers "
        f"(scanned {tickers_scanned})"
    )
    log.info(f"Label distribution:\n{dataset['label'].value_counts().to_string()}")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 : Purged Walk-Forward Cross-Validation & Training
# ═══════════════════════════════════════════════════════════════════════════════

def purged_time_series_split(
    dates: pd.DatetimeIndex, n_splits: int = 5, purge_days: int = 15,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window time-series CV with purge gap.
    """
    unique_dates = np.sort(dates.unique())
    fold_size = len(unique_dates) // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end_dt = unique_dates[(i + 1) * fold_size]
        val_start_dt = train_end_dt + pd.Timedelta(days=purge_days)
        val_end_idx  = min((i + 2) * fold_size, len(unique_dates) - 1)
        val_end_dt   = unique_dates[val_end_idx]

        train_mask = dates <= train_end_dt
        val_mask   = (dates >= val_start_dt) & (dates <= val_end_dt)

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]
        if len(val_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


def train_xgb(dataset: pd.DataFrame):
    """
    Train XGBoost with walk-forward CV, then refit on full train set.
    Returns (model, cv_results_df).
    """
    X = dataset[FEATURE_COLS].values
    y = dataset["label"].values.astype(int)
    dates = pd.DatetimeIndex(dataset.index)

    # ── monotone constraints ─────────────────────────────────────────────
    # stoch_divergence: higher → more likely Class 1 (monotone increasing = +1)
    # vol_surge_ratio:  higher → more likely capitulation → Class 1 (+1)
    # dist_sma20_z:     more negative → deeper dip (no strict mono — leave 0)
    mono = [0] * len(FEATURE_COLS)
    if "stoch_divergence" in FEATURE_COLS:
        mono[FEATURE_COLS.index("stoch_divergence")] = 1
    # vol_surge_ratio can be ambiguous; leave 0 for now

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,      # L1
        reg_lambda=3.0,     # L2
        min_child_weight=10,
        gamma=1.0,
        monotone_constraints=tuple(mono),
        seed=SEED,
        tree_method="hist",
    )

    # ── walk-forward CV ──────────────────────────────────────────────────
    splits = purged_time_series_split(dates, n_splits=5, purge_days=15)
    cv_results = []

    for fold, (tr_i, va_i) in enumerate(splits):
        dtrain = xgb.DMatrix(X[tr_i], label=y[tr_i], feature_names=FEATURE_COLS)
        dval   = xgb.DMatrix(X[va_i], label=y[va_i], feature_names=FEATURE_COLS)
        bst = xgb.train(
            params, dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        preds = bst.predict(dval)
        auc = roc_auc_score(y[va_i], preds) if len(np.unique(y[va_i])) > 1 else np.nan
        acc = accuracy_score(y[va_i], (preds >= 0.5).astype(int))
        prec = precision_score(y[va_i], (preds >= 0.5).astype(int), zero_division=0)
        rec  = recall_score(y[va_i], (preds >= 0.5).astype(int), zero_division=0)
        n_tr, n_va = len(tr_i), len(va_i)
        win_rate_va = y[va_i].mean()
        cv_results.append(dict(
            fold=fold, n_train=n_tr, n_val=n_va,
            val_win_rate=win_rate_va,
            auc=auc, accuracy=acc, precision=prec, recall=rec,
            best_round=bst.best_iteration,
        ))
        log.info(
            f"  Fold {fold}: train={n_tr} val={n_va} AUC={auc:.3f} "
            f"Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f}"
        )

    cv_df = pd.DataFrame(cv_results)
    log.info(f"\n  CV Mean AUC : {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
    log.info(f"  CV Mean Prec: {cv_df['precision'].mean():.4f}")

    # ── chronological train / OOS split ──────────────────────────────────
    split_dt = pd.Timestamp(SPLIT_DATE)
    train_mask = dates < split_dt
    test_mask  = dates >= split_dt

    dtrain_full = xgb.DMatrix(
        X[train_mask], label=y[train_mask], feature_names=FEATURE_COLS,
    )
    best_rounds = int(cv_df["best_round"].median()) + 20  # small buffer
    final_model = xgb.train(params, dtrain_full, num_boost_round=best_rounds)

    # OOS evaluation
    if test_mask.sum() > 0:
        dtest = xgb.DMatrix(
            X[test_mask], label=y[test_mask], feature_names=FEATURE_COLS,
        )
        oos_preds = final_model.predict(dtest)
        y_test = y[test_mask]
        if len(np.unique(y_test)) > 1:
            oos_auc = roc_auc_score(y_test, oos_preds)
        else:
            oos_auc = np.nan
        oos_acc = accuracy_score(y_test, (oos_preds >= 0.5).astype(int))
        log.info(f"\n  OOS ({SPLIT_DATE}+): n={test_mask.sum()} AUC={oos_auc:.3f} Acc={oos_acc:.3f}")
        log.info(
            f"  OOS classification report:\n"
            + classification_report(
                y_test, (oos_preds >= 0.5).astype(int),
                target_names=["Trap", "Win"], zero_division=0,
            )
        )

    return final_model, cv_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 : Back-test  (Baseline vs ML-Filtered)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    bars_held: int
    pnl_pct: float
    net_pnl_pct: float
    exit_reason: str
    ml_prob: float = np.nan


def backtest_events(
    dataset: pd.DataFrame,
    model: xgb.Booster | None = None,
    prob_threshold: float = ML_PROB_THRESH,
    oos_only: bool = True,
) -> list[TradeResult]:
    """
    Walk through dip events and simulate trades.
    If model is None → baseline (take every dip event).
    If model is provided → only take trades with P(Win) >= prob_threshold.
    """
    if oos_only:
        dataset = dataset.loc[dataset.index >= SPLIT_DATE]

    if model is not None:
        dm = xgb.DMatrix(dataset[FEATURE_COLS].values, feature_names=FEATURE_COLS)
        probs = model.predict(dm)
        dataset = dataset.copy()
        dataset["ml_prob"] = probs
    else:
        dataset = dataset.copy()
        dataset["ml_prob"] = 1.0  # all pass

    # group by ticker and simulate
    trades: list[TradeResult] = []
    per_ticker_data: dict[str, pd.DataFrame] = {}

    # pre-load OHLCV for tickers in dataset
    for ticker in dataset["ticker"].unique():
        csv = DATA_DIR / f"{ticker}.csv"
        if csv.exists():
            per_ticker_data[ticker] = load_ohlcv(csv)

    for _, row in dataset.iterrows():
        if model is not None and row["ml_prob"] < prob_threshold:
            continue  # filtered out by ML

        ticker = row["ticker"]
        t0 = row.name  # Date index
        ohlcv = per_ticker_data.get(ticker)
        if ohlcv is None:
            continue

        if t0 not in ohlcv.index:
            continue
        i0 = ohlcv.index.get_loc(t0)
        entry_price = ohlcv["Close"].iloc[i0]

        upper = entry_price * (1 + TB_TP_PCT)
        lower = entry_price * (1 - TB_SL_PCT)
        end_i = min(i0 + TB_HORIZON, len(ohlcv) - 1)

        exit_reason = "time_expiry"
        exit_price = ohlcv["Close"].iloc[end_i]
        exit_i = end_i

        for j in range(i0 + 1, end_i + 1):
            p = ohlcv["Close"].iloc[j]
            if p >= upper:
                exit_price = upper
                exit_i = j
                exit_reason = "take_profit"
                break
            if p <= lower:
                exit_price = lower
                exit_i = j
                exit_reason = "stop_loss"
                break

        raw_pnl = (exit_price - entry_price) / entry_price
        net_pnl = raw_pnl - 2 * TRANSACTION_COST  # round-trip cost

        trades.append(TradeResult(
            ticker=ticker,
            entry_date=str(ohlcv.index[i0].date()),
            exit_date=str(ohlcv.index[exit_i].date()),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            bars_held=exit_i - i0,
            pnl_pct=round(raw_pnl * 100, 3),
            net_pnl_pct=round(net_pnl * 100, 3),
            exit_reason=exit_reason,
            ml_prob=round(row["ml_prob"], 4),
        ))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 : Reporting & Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _npr(val: float) -> str:
    """Readable Nepali Rupees."""
    if abs(val) >= 1e7:
        return f"NPR {val / 1e7:,.2f} Cr"
    if abs(val) >= 1e5:
        return f"NPR {val / 1e5:,.2f} L"
    return f"NPR {val:,.0f}"


def summarise_trades(trades: list[TradeResult], tag: str) -> dict:
    if not trades:
        log.info(f"  [{tag}] No trades.")
        return {}
    df = pd.DataFrame([t.__dict__ for t in trades])

    n = len(df)
    wins = (df["net_pnl_pct"] > 0).sum()
    win_rate = wins / n
    avg_pnl = df["net_pnl_pct"].mean()
    median_pnl = df["net_pnl_pct"].median()
    total_pnl = df["net_pnl_pct"].sum()
    avg_bars = df["bars_held"].mean()
    avg_win  = df.loc[df["net_pnl_pct"] > 0, "net_pnl_pct"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["net_pnl_pct"] <= 0, "net_pnl_pct"].mean() if (n - wins) > 0 else 0
    profit_factor = abs(avg_win * wins) / (abs(avg_loss * (n - wins)) + 1e-10)

    exit_counts = df["exit_reason"].value_counts().to_dict()

    metrics = dict(
        tag=tag, n_trades=n, wins=wins, win_rate=round(win_rate, 4),
        avg_pnl_pct=round(avg_pnl, 3), median_pnl_pct=round(median_pnl, 3),
        total_pnl_pct=round(total_pnl, 2),
        avg_bars_held=round(avg_bars, 1),
        avg_win_pct=round(avg_win, 3), avg_loss_pct=round(avg_loss, 3),
        profit_factor=round(profit_factor, 3),
        exits=exit_counts,
    )

    log.info(f"\n{'='*60}")
    log.info(f"  [{tag}] TRADE SUMMARY (OOS from {SPLIT_DATE})")
    log.info(f"{'='*60}")
    log.info(f"  Trades       : {n}")
    log.info(f"  Win Rate     : {win_rate:.1%}")
    log.info(f"  Avg PnL      : {avg_pnl:+.3f}%")
    log.info(f"  Median PnL   : {median_pnl:+.3f}%")
    log.info(f"  Total PnL    : {total_pnl:+.2f}%")
    log.info(f"  Avg Bars Held: {avg_bars:.1f}")
    log.info(f"  Avg Win      : {avg_win:+.3f}%  |  Avg Loss: {avg_loss:+.3f}%")
    log.info(f"  Profit Factor: {profit_factor:.3f}")
    log.info(f"  Exit Reasons : {exit_counts}")

    # save trades CSV
    df.to_csv(RUN_DIR / f"trades_{tag.lower().replace(' ', '_')}.csv", index=False)
    return metrics


def plot_feature_importance(model: xgb.Booster):
    imp = model.get_score(importance_type="gain")
    imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(imp_sorted.keys())[::-1], list(imp_sorted.values())[::-1])
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    log.info(f"  Feature importance plot saved.")


def plot_equity_curves(
    baseline_trades: list[TradeResult],
    ml_trades: list[TradeResult],
):
    """Simple cumulative PnL chart."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for trades, label, color in [
        (baseline_trades, "Baseline (all dips)", "grey"),
        (ml_trades, f"ML-Filtered (P>{ML_PROB_THRESH})", "steelblue"),
    ]:
        if not trades:
            continue
        df = pd.DataFrame([t.__dict__ for t in trades])
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df = df.sort_values("entry_date")
        df["cum_pnl"] = df["net_pnl_pct"].cumsum()
        ax.plot(df["entry_date"], df["cum_pnl"], label=label, color=color, lw=1.5)

    ax.set_ylabel("Cumulative Net PnL (%)")
    ax.set_title("Dip-Buy Backtest: Baseline vs ML-Filtered (OOS)")
    ax.legend()
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)
    log.info(f"  Equity curve plot saved.")


def plot_prob_distribution(dataset: pd.DataFrame, model: xgb.Booster):
    """Distribution of ML probabilities coloured by actual label."""
    oos = dataset.loc[dataset.index >= SPLIT_DATE].copy()
    if len(oos) == 0:
        return
    dm = xgb.DMatrix(oos[FEATURE_COLS].values, feature_names=FEATURE_COLS)
    oos["prob"] = model.predict(dm)

    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, color, lbl in [(1, "green", "Win (Class 1)"), (0, "red", "Trap (Class 0)")]:
        subset = oos[oos["label"] == lab]
        ax.hist(subset["prob"], bins=30, alpha=0.5, color=color, label=lbl, density=True)
    ax.axvline(ML_PROB_THRESH, color="black", ls="--", lw=1, label=f"Threshold={ML_PROB_THRESH}")
    ax.set_xlabel("P(Win)")
    ax.set_ylabel("Density")
    ax.set_title("OOS Probability Distribution by True Label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "prob_distribution.png", dpi=150)
    plt.close(fig)


def sweep_thresholds(dataset: pd.DataFrame, model: xgb.Booster) -> pd.DataFrame:
    """Evaluate trade metrics across different probability thresholds."""
    oos = dataset.loc[dataset.index >= SPLIT_DATE].copy()
    if len(oos) == 0:
        return pd.DataFrame()
    dm = xgb.DMatrix(oos[FEATURE_COLS].values, feature_names=FEATURE_COLS)
    oos["prob"] = model.predict(dm)

    results = []
    for thresh in np.arange(0.40, 0.80, 0.05):
        trades = backtest_events(dataset, model, prob_threshold=thresh, oos_only=True)
        if not trades:
            continue
        df = pd.DataFrame([t.__dict__ for t in trades])
        n = len(df)
        wr = (df["net_pnl_pct"] > 0).sum() / n
        avg = df["net_pnl_pct"].mean()
        tot = df["net_pnl_pct"].sum()
        results.append(dict(threshold=round(thresh, 2), n_trades=n, win_rate=round(wr, 3),
                            avg_pnl=round(avg, 3), total_pnl=round(tot, 2)))
    sweep_df = pd.DataFrame(results)
    log.info(f"\n  Threshold Sweep:\n{sweep_df.to_string(index=False)}")
    sweep_df.to_csv(RUN_DIR / "threshold_sweep.csv", index=False)
    return sweep_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 : Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("  XGBoost Dip-Buy Mean-Reversion Classifier")
    log.info(f"  Run dir: {RUN_DIR}")
    log.info("=" * 60)

    # ── Step 1-3: Assemble dataset ───────────────────────────────────────
    log.info("\n[1] Assembling dataset (events + labels + features) ...")
    dataset = assemble_dataset()
    dataset.to_csv(RUN_DIR / "dataset.csv")

    # ── Step 4-5: Train model ────────────────────────────────────────────
    log.info("\n[2] Training XGBoost with purged walk-forward CV ...")
    model, cv_df = train_xgb(dataset)
    cv_df.to_csv(RUN_DIR / "cv_results.csv", index=False)
    model.save_model(str(RUN_DIR / "model.json"))
    log.info(f"  Model saved to {RUN_DIR / 'model.json'}")

    # ── Feature importance ───────────────────────────────────────────────
    plot_feature_importance(model)

    # ── Step 6: Backtest ─────────────────────────────────────────────────
    log.info("\n[3] Back-testing (OOS) ...")
    baseline_trades = backtest_events(dataset, model=None, oos_only=True)
    ml_trades       = backtest_events(dataset, model=model, prob_threshold=ML_PROB_THRESH, oos_only=True)

    base_metrics = summarise_trades(baseline_trades, "Baseline")
    ml_metrics   = summarise_trades(ml_trades, "ML-Filtered")

    # ── Comparison ───────────────────────────────────────────────────────
    if base_metrics and ml_metrics:
        log.info(f"\n{'='*60}")
        log.info(f"  COMPARISON")
        log.info(f"{'='*60}")
        for key in ["n_trades", "win_rate", "avg_pnl_pct", "total_pnl_pct", "profit_factor"]:
            bv = base_metrics.get(key, "N/A")
            mv = ml_metrics.get(key, "N/A")
            log.info(f"  {key:20s}  Baseline={bv}  ML={mv}")

    # ── Plots ────────────────────────────────────────────────────────────
    log.info("\n[4] Generating plots ...")
    plot_equity_curves(baseline_trades, ml_trades)
    plot_prob_distribution(dataset, model)

    # ── Threshold sweep ──────────────────────────────────────────────────
    log.info("\n[5] Threshold sweep ...")
    sweep_thresholds(dataset, model)

    # ── Summary CSV ──────────────────────────────────────────────────────
    summary = pd.DataFrame([base_metrics, ml_metrics])
    summary.to_csv(RUN_DIR / "summary_metrics.csv", index=False)

    log.info(f"\n  All outputs saved to: {RUN_DIR}")
    log.info("  Done.")


if __name__ == "__main__":
    main()
