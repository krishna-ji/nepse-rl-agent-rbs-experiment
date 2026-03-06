#!/usr/bin/env python
"""
XGBoost Dip Detector - Bottom-Finding Signal Generator (v2.1)
==============================================================
Three critical fixes over v2.0:
  1. Right-edge extrema guard  -- NaN-invalidates last `order` bars per ticker
  2. Stratified calibration    -- random 15% holdout (no regime leak)
  3. Exogenous NEPSE features  -- index stoch_k + dist_sma200 for every bar

Diagnostics exported to RUN_DIR:
  - feature_importance.png        XGB gain-based bar chart
  - calibration_curve.png         reliability diagram (10 bins)
  - cv_fold_metrics.png           AUC / AUCPR per fold
  - probability_distribution.png  OOS probability histogram
  - label_rate_by_year.png        dip-rate drift check
  - stochk_vs_prob_scatter.png    stoch_k vs P(dip) coloured by label
  - feature_correlation.png       Pearson heatmap of the 7-feature matrix
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import argrelextrema
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
np.random.seed(42)

# ===========================================================================
# PATHS & CONSTANTS
# ===========================================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data" / "ohlcv" / "1D" / "stocks"
RUN_DIR      = PROJECT_ROOT / "runs" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS       = 300
SEED           = 42
EXTREMA_ORDER  = 5
MIN_OP_THRESH  = 0.10           # floor: discard absolute garbage below this
TOP_N_SIGNALS  = 5              # keep only the N best setups per day
NEPSE_TICKER   = "NEPSE"        # composite index CSV stem

# -- Trade engine constants -------------------------------------------------
CAPITAL          = 1_000_000
FRICTION         = 0.005       # round-trip friction
TP_ATR_MULT      = 2.0         # take-profit = Entry + 2.0 * ATR14
SL_ATR_MULT      = 1.5         # stop-loss   = Entry - 1.5 * ATR14
MAX_HOLD_BARS    = 10          # time stop at T+10 (EV peak)
MAX_WEIGHT       = 0.20        # max 20% of equity per trade

# -- logging ----------------------------------------------------------------
log = logging.getLogger("xgb_dip")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); log.addHandler(_sh)
_fh = logging.FileHandler(RUN_DIR / "run.log"); _fh.setFormatter(_fmt); log.addHandler(_fh)


# ===========================================================================
# SECTION 0 : Data Loading
# ===========================================================================

def _build_exclude_set() -> set[str]:
    excl: set[str] = set()
    stocks_path = PROJECT_ROOT / "data" / "stocks.json"
    if stocks_path.exists():
        for rec in json.loads(stocks_path.read_text()):
            if rec.get("sector") in ("PROMOTSHARE",):
                excl.add(rec["script"])
    for fname in ("mutual.json", "corpdeben.json"):
        fp = PROJECT_ROOT / "data" / fname
        if fp.exists():
            for rec in json.loads(fp.read_text()):
                excl.add(rec.get("script", rec.get("symbol", "")))
    return excl


def _build_sector_map() -> dict[str, str]:
    fp = PROJECT_ROOT / "data" / "stocks.json"
    if not fp.exists():
        return {}
    return {
        rec["script"]: rec.get("sector", "OTHERS")
        for rec in json.loads(fp.read_text())
    }


def load_ohlcv(csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["Timestamp"])
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def _load_nepse_index() -> pd.DataFrame:
    """Load NEPSE composite index OHLCV and compute index-level features."""
    csv = DATA_DIR / f"{NEPSE_TICKER}.csv"
    if not csv.exists():
        log.warning(f"  NEPSE index CSV not found at {csv}")
        return pd.DataFrame()
    df = load_ohlcv(csv)
    c, h, l = df["Close"], df["High"], df["Low"]
    stoch_k, _ = _slow_stochastic(h, l, c)
    sma200 = _sma(c, 200)

    idx = pd.DataFrame(index=df.index)
    idx["idx_stoch_k"]       = stoch_k
    idx["idx_dist_sma200"]   = (c - sma200) / (sma200 + 1e-10)
    idx["idx_ret5"]          = c.pct_change(5)      # 5-bar index return
    idx["idx_daily_ret"]     = c.pct_change(1)      # for rolling correlation
    return idx


# ===========================================================================
# SECTION 1 : Indicator Helpers
# ===========================================================================

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1,
    ).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def _slow_stochastic(h, l, c, k_per=14, k_sm=3, d_sm=3):
    lowest  = l.rolling(k_per, min_periods=k_per).min()
    highest = h.rolling(k_per, min_periods=k_per).max()
    fast_k  = 100 * (c - lowest) / (highest - lowest + 1e-10)
    slow_k  = fast_k.rolling(k_sm, min_periods=k_sm).mean()
    slow_d  = slow_k.rolling(d_sm, min_periods=d_sm).mean()
    return slow_k, slow_d


def _zscore(s: pd.Series, w: int) -> pd.Series:
    """Rolling Z-score: (x - rolling_mean) / rolling_std."""
    mu = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std(ddof=1)
    return (s - mu) / (sd + 1e-10)


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def _linreg_slope(s: pd.Series, w: int) -> pd.Series:
    """Rolling ordinary least-squares slope over *w* bars."""
    # Vectorised via covariance formula: slope = cov(x, y) / var(x)
    x = pd.Series(np.arange(len(s), dtype=float), index=s.index)
    xy_cov = s.rolling(w, min_periods=w).cov(x)
    x_var  = x.rolling(w, min_periods=w).var(ddof=1)
    return xy_cov / (x_var + 1e-10)


def _mfi(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, n: int = 5) -> pd.Series:
    """Money Flow Index over *n* bars."""
    tp = (h + l + c) / 3.0
    mf = tp * v
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(n, min_periods=n).sum()
    neg_mf = mf.where(delta <= 0, 0.0).rolling(n, min_periods=n).sum()
    ratio = pos_mf / (neg_mf + 1e-10)
    return 100.0 - 100.0 / (1.0 + ratio)


# ===========================================================================
# SECTION 2 : Automated Extrema Labeling (Oracle Target)
# ===========================================================================

def auto_label_extrema(df: pd.DataFrame, order: int = EXTREMA_ORDER) -> pd.Series:
    """Label local minima of Low in uptrend (Close > SMA_200).

    FIX #1: Right-edge guard -- last `order` bars are set to NaN because
    argrelextrema cannot resolve whether they are true minima without
    future data.  These rows MUST be dropped before training.
    """
    sma200 = _sma(df["Close"], 200)
    low_vals = df["Low"].values
    min_idx = argrelextrema(low_vals, np.less_equal, order=order)[0]

    # float target to allow NaN injection
    target = pd.Series(0.0, index=df.index)
    target.iloc[min_idx] = 1.0

    # Uptrend constraint
    uptrend = df["Close"] > sma200
    target = target.where(uptrend, 0.0)

    # CRITICAL: invalidate unresolved right-edge bars
    target.iloc[-order:] = np.nan

    return target


# ===========================================================================
# SECTION 3 : Feature Engineering (high-dimensional, zero forward leakage)
# ===========================================================================

FEATURE_COLS = [
    # -- Oscillators & trend --
    "stoch_k", "stoch_d", "stoch_k_delta",
    "rsi_14", "stoch_k_slope10",
    # -- Distance / volatility --
    "dist_sma200_pct", "dist_sma200_z50", "atr14_pct",
    # -- Volume dynamics --
    "vol_surge", "mfi_5",
    # -- Structural memory / divergence --
    "dist_prev_pivot", "stoch_divergence",
    # -- Exogenous NEPSE index --
    "idx_stoch_k", "idx_dist_sma200", "idx_ret5", "corr_idx_20",
]

def build_features(df: pd.DataFrame, idx_feats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Strictly lagged features -- computable at bar close with no future data.

    16-feature matrix:
      Oscillators  : stoch_k, stoch_d, stoch_k_delta, rsi_14, stoch_k_slope10
      Distance/vol : dist_sma200_pct, dist_sma200_z50, atr14_pct
      Volume       : vol_surge, mfi_5
      Struct memory: dist_prev_pivot, stoch_divergence
      Index macro  : idx_stoch_k, idx_dist_sma200, idx_ret5, corr_idx_20
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    stoch_k, stoch_d = _slow_stochastic(h, l, c)
    sma200 = _sma(c, 200)
    atr14 = _atr(h, l, c, 14)
    dist_sma200 = (c - sma200) / (sma200 + 1e-10)

    f = pd.DataFrame(index=df.index)

    # -- Oscillators & trend ------------------------------------------------
    f["stoch_k"]         = stoch_k
    f["stoch_d"]         = stoch_d
    f["stoch_k_delta"]   = stoch_k - stoch_k.shift(1)
    f["rsi_14"]          = _rsi(c, 14)
    f["stoch_k_slope10"] = _linreg_slope(stoch_k, 10)

    # -- Distance / volatility ----------------------------------------------
    f["dist_sma200_pct"] = dist_sma200
    f["dist_sma200_z50"] = _zscore(dist_sma200, 50)
    f["atr14_pct"]       = atr14 / (c + 1e-10)

    # -- Volume dynamics ----------------------------------------------------
    vol_ma20 = v.rolling(20, min_periods=20).mean()
    f["vol_surge"]       = v / (vol_ma20 + 1e-10)
    f["mfi_5"]           = _mfi(h, l, c, v, 5)

    # -- Structural memory / divergence -------------------------------------
    # Previous pivot low: 15-bar rolling min of Low, shifted by 1 to avoid look-ahead
    roll_min = l.rolling(15, min_periods=5).min().shift(1)
    # Capture the actual Low value when it matches the rolling min (pivot bar)
    is_pivot = (l.shift(1) == roll_min)
    prev_pivot_low = l.shift(1).where(is_pivot).ffill()
    # Distance from current close to last established trough
    dist_pp = (c / prev_pivot_low) - 1.0
    f["dist_prev_pivot"] = dist_pp.replace([np.inf, -np.inf], np.nan)
    # Stoch %K at the exact pivot bar, forward-filled
    stoch_at_pivot = stoch_k.shift(1).where(is_pivot).ffill()
    # Bullish divergence: price makes lower low but stoch is higher than at pivot
    lower_low = c < prev_pivot_low
    higher_stoch = stoch_k > stoch_at_pivot
    f["stoch_divergence"] = (lower_low & higher_stoch).where(prev_pivot_low.notna(), np.nan).astype(float)

    # -- Exogenous NEPSE index features (date-aligned) ----------------------
    if idx_feats is not None and len(idx_feats) > 0:
        f = f.join(idx_feats[["idx_stoch_k", "idx_dist_sma200", "idx_ret5", "idx_daily_ret"]], how="left")
        for col in ("idx_stoch_k", "idx_dist_sma200", "idx_ret5", "idx_daily_ret"):
            f[col] = f[col].ffill()
        # Rolling 20-bar correlation: ticker daily return vs index daily return
        ticker_ret = c.pct_change(1)
        f["corr_idx_20"] = ticker_ret.rolling(20, min_periods=20).corr(f["idx_daily_ret"])
        f.drop(columns=["idx_daily_ret"], inplace=True)
    else:
        f["idx_stoch_k"]     = np.nan
        f["idx_dist_sma200"] = np.nan
        f["idx_ret5"]        = np.nan
        f["corr_idx_20"]     = np.nan

    return f


# ===========================================================================
# SECTION 4 : Dataset Assembly
# ===========================================================================

def assemble_dataset() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Pool all tickers into a single training dataset.

    Returns
    -------
    dataset   : All bars (post warm-up, NaN target dropped) with features + target + ticker.
    ohlcv_map : ticker -> full OHLCV DataFrame (for scoring latest bars).
    idx_feats : NEPSE index features DataFrame (for scoring).
    """
    excl = _build_exclude_set()
    # Also exclude index CSVs from the training pool
    index_names = {
        "NEPSE", "SENSITIVE", "FLOAT", "SENFLOAT", "BANKING", "HYDROPOWER",
        "FINANCE", "HOTELS", "TRADING", "MANUFACTURE", "OTHERS", "MICROFINANCE",
        "LIFEINSU", "NONLIFEINSU", "INVESTMENT", "DEVBANK",
    }
    excl.update(index_names)

    # Load NEPSE index features first
    idx_feats = _load_nepse_index()
    if len(idx_feats) > 0:
        log.info(f"  NEPSE index: {len(idx_feats):,} bars loaded for exogenous features")
    else:
        log.warning("  NEPSE index features unavailable -- proceeding without them")

    csvs = sorted(DATA_DIR.glob("*.csv"))
    rows: list[pd.DataFrame] = []
    ohlcv_map: dict[str, pd.DataFrame] = {}
    n_scanned = n_used = 0

    for csv in csvs:
        ticker = csv.stem
        if ticker in excl:
            continue
        df = load_ohlcv(csv)
        if len(df) < MIN_ROWS:
            continue
        n_scanned += 1
        ohlcv_map[ticker] = df

        target = auto_label_extrema(df)
        feats = build_features(df, idx_feats)

        chunk = feats[FEATURE_COLS].copy()
        chunk["Close"] = df["Close"]            # needed for forward return analysis
        chunk["target_dip"] = target
        chunk["ticker"] = ticker

        # Drop rows where features are NaN (warm-up) OR target is NaN (right-edge guard)
        chunk = chunk.dropna(subset=FEATURE_COLS + ["target_dip"])

        if len(chunk) > 0:
            rows.append(chunk)
            n_used += 1

    if not rows:
        log.error("No data found!")
        sys.exit(1)

    dataset = pd.concat(rows).sort_index()
    dataset.index.name = "Date"
    dataset["target_dip"] = dataset["target_dip"].astype(int)

    n_pos = int(dataset["target_dip"].sum())
    n_neg = len(dataset) - n_pos
    n_dropped = sum(len(ohlcv_map[t]) for t in ohlcv_map) - len(dataset)  # approx
    log.info(f"  Dataset : {len(dataset):,} bars  |  {n_used} tickers (scanned {n_scanned})")
    log.info(f"  Labels  : dip=1 -> {n_pos:,}  dip=0 -> {n_neg:,}  ({n_pos/len(dataset):.2%} dip rate)")
    log.info(f"  Right-edge guard: ~{n_used * EXTREMA_ORDER} bars invalidated (NaN target)")

    return dataset, ohlcv_map, idx_feats


# ===========================================================================
# SECTION 5 : Walk-Forward CV + Production Model
# ===========================================================================

def train_model(
    dataset: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[xgb.Booster, IsotonicRegression, pd.DataFrame, dict]:
    """Walk-forward CV for quality metrics, then train prod model on ALL data.

    FIX #2: Calibrator holdout is a stratified 15% random sample (not the
    tail of the timeline) to avoid regime-leak into calibration.

    Returns (prod_model, prod_calibrator, cv_df, oos_bundle).
    oos_bundle contains concatenated OOS predictions for diagnostics.
    """
    dataset = dataset.sort_index()
    X = dataset[FEATURE_COLS].values.astype(np.float32)
    y = dataset["target_dip"].values.astype(int)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos = n_neg / (n_pos + 1) if n_pos > 0 else 1.0
    log.info(f"  scale_pos_weight: {scale_pos:.1f}")

    params = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=scale_pos,
        seed=SEED,
        tree_method="hist",
    )
    xgb_params = {k: v for k, v in params.items() if k != "n_estimators"}

    # -- Walk-forward CV (model quality check) -------------------------------
    cv_results: list[dict] = []
    oos_probs: list[np.ndarray] = []
    oos_labels: list[np.ndarray] = []
    oos_dates: list[np.ndarray] = []
    last_best_round = 200

    tscv = TimeSeriesSplit(n_splits=n_splits)
    dates_arr = pd.DatetimeIndex(dataset.index)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        hold_n = max(int(len(tr_idx) * 0.10), 20)
        sub_tr = tr_idx[:-hold_n]
        hold   = tr_idx[-hold_n:]

        dtrain = xgb.DMatrix(X[sub_tr], label=y[sub_tr], feature_names=FEATURE_COLS)
        dhold  = xgb.DMatrix(X[hold],   label=y[hold],   feature_names=FEATURE_COLS)
        dtest  = xgb.DMatrix(X[te_idx], label=y[te_idx],  feature_names=FEATURE_COLS)

        bst = xgb.train(
            xgb_params, dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dtrain, "train"), (dhold, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        last_best_round = bst.best_iteration

        hold_raw = bst.predict(dhold)
        cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        cal.fit(hold_raw, y[hold])

        test_cal = cal.predict(bst.predict(dtest))
        y_te = y[te_idx]

        oos_probs.append(test_cal)
        oos_labels.append(y_te)
        oos_dates.append(dates_arr[te_idx].values)

        has_both = len(np.unique(y_te)) > 1
        auc   = roc_auc_score(y_te, test_cal) if has_both else np.nan
        aucpr = average_precision_score(y_te, test_cal) if has_both else np.nan

        cv_results.append(dict(
            fold=fold,
            n_train=len(sub_tr), n_test=len(te_idx),
            test_range=f"{dates_arr[te_idx[0]]:%Y-%m-%d} -> {dates_arr[te_idx[-1]]:%Y-%m-%d}",
            auc=round(auc, 4) if not np.isnan(auc) else np.nan,
            aucpr=round(aucpr, 4) if not np.isnan(aucpr) else np.nan,
            best_round=bst.best_iteration,
        ))
        log.info(
            f"  Fold {fold}: AUC={auc:.3f}  AUCPR={aucpr:.4f}  "
            f"best_round={bst.best_iteration}"
        )

    cv_df = pd.DataFrame(cv_results)
    log.info(f"  CV Mean AUC  : {cv_df['auc'].mean():.4f} +/- {cv_df['auc'].std():.4f}")
    log.info(f"  CV Mean AUCPR: {cv_df['aucpr'].mean():.4f} +/- {cv_df['aucpr'].std():.4f}")

    # -- Production model (all data) -----------------------------------------
    log.info("  Training production model on ALL data ...")
    prod_rounds = max(last_best_round + 50, 200)

    # FIX #2: stratified random 15% holdout for calibration (not tail)
    rng = np.random.RandomState(SEED)
    hold_frac = 0.15
    hold_n = max(int(len(X) * hold_frac), 100)
    all_idx = np.arange(len(X))
    hold_mask = np.zeros(len(X), dtype=bool)
    hold_mask[rng.choice(len(X), size=hold_n, replace=False)] = True
    train_mask = ~hold_mask

    dtrain_all = xgb.DMatrix(X[train_mask], label=y[train_mask], feature_names=FEATURE_COLS)
    dhold_all  = xgb.DMatrix(X[hold_mask],  label=y[hold_mask],  feature_names=FEATURE_COLS)

    prod_model = xgb.train(
        xgb_params, dtrain_all,
        num_boost_round=prod_rounds,
        evals=[(dtrain_all, "train"), (dhold_all, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    # Calibrate on holdout
    hold_raw = prod_model.predict(dhold_all)
    prod_cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    prod_cal.fit(hold_raw, y[hold_mask])

    log.info(f"  Prod model: {prod_model.best_iteration} rounds  (cal holdout: {hold_n} random bars)")

    # bundle OOS predictions for diagnostics
    oos_bundle = dict(
        probs=np.concatenate(oos_probs),
        labels=np.concatenate(oos_labels),
        dates=np.concatenate(oos_dates),
    )

    return prod_model, prod_cal, cv_df, oos_bundle


# ===========================================================================
# SECTION 6 : Score Latest Bars -> Emit Signals
# ===========================================================================

def score_latest_signals(
    ohlcv_map: dict[str, pd.DataFrame],
    model: xgb.Booster,
    calibrator: IsotonicRegression,
    idx_feats: pd.DataFrame | None = None,
    min_thresh: float = MIN_OP_THRESH,
    top_n: int = TOP_N_SIGNALS,
    lookback: int = 3,
) -> pd.DataFrame:
    """Score the last *lookback* bars of every ticker.

    Cross-sectional ranking: keep only the top *top_n* signals per day
    above the *min_thresh* floor.
    """
    sector_map = _build_sector_map()
    signals: list[dict] = []

    for ticker, df in ohlcv_map.items():
        if len(df) < MIN_ROWS:
            continue
        feats = build_features(df, idx_feats)
        tail = feats.tail(lookback).dropna(subset=FEATURE_COLS)
        if tail.empty:
            continue

        X = tail[FEATURE_COLS].values.astype(np.float32)
        dm = xgb.DMatrix(X, feature_names=FEATURE_COLS)
        raw = model.predict(dm)
        cal = calibrator.predict(raw)

        for i, (date, row) in enumerate(tail.iterrows()):
            p = float(cal[i])
            if p >= min_thresh:
                signals.append(dict(
                    ticker=ticker,
                    sector=sector_map.get(ticker, ""),
                    date=date.strftime("%Y-%m-%d"),
                    close=round(float(df.loc[date, "Close"]), 2),
                    stoch_k=round(float(row["stoch_k"]), 1),
                    dist_sma200=f"{row['dist_sma200_pct']:.1%}",
                    idx_stoch_k=round(float(row["idx_stoch_k"]), 1) if pd.notna(row["idx_stoch_k"]) else None,
                    prob=round(p, 4),
                ))

    sig_df = pd.DataFrame(signals)
    if len(sig_df) > 0:
        # Cross-sectional ranking: top N per day
        sig_df = (
            sig_df
            .sort_values("prob", ascending=False)
            .groupby("date", sort=False)
            .head(top_n)
            .reset_index(drop=True)
        )
    return sig_df


# ===========================================================================
# SECTION 7 : Watchlist (top N by latest-bar probability)
# ===========================================================================

def _build_watchlist(
    ohlcv_map: dict[str, pd.DataFrame],
    model: xgb.Booster,
    calibrator: IsotonicRegression,
    idx_feats: pd.DataFrame | None = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """Score the *latest* bar of every ticker and return top-N by P(dip)."""
    sector_map = _build_sector_map()
    rows: list[dict] = []

    for ticker, df in ohlcv_map.items():
        if len(df) < MIN_ROWS:
            continue
        feats = build_features(df, idx_feats)
        last_row = feats.iloc[[-1]].dropna(subset=FEATURE_COLS)
        if last_row.empty:
            continue

        X = last_row[FEATURE_COLS].values.astype(np.float32)
        dm = xgb.DMatrix(X, feature_names=FEATURE_COLS)
        raw = model.predict(dm)
        cal = calibrator.predict(raw)
        p = float(cal[0])
        date = last_row.index[0]

        rows.append(dict(
            ticker=ticker,
            sector=sector_map.get(ticker, ""),
            date=date.strftime("%Y-%m-%d"),
            close=round(float(df.loc[date, "Close"]), 2),
            stoch_k=round(float(last_row["stoch_k"].iloc[0]), 1),
            dist_sma200=f"{last_row['dist_sma200_pct'].iloc[0]:.1%}",
            idx_stoch_k=round(float(last_row["idx_stoch_k"].iloc[0]), 1) if pd.notna(last_row["idx_stoch_k"].iloc[0]) else None,
            prob=round(p, 4),
        ))

    if not rows:
        return pd.DataFrame()
    wdf = pd.DataFrame(rows).sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
    return wdf


# ===========================================================================
# SECTION 8 : Diagnostic Graphs
# ===========================================================================

def _export_diagnostics(
    model: xgb.Booster,
    cv_df: pd.DataFrame,
    oos_bundle: dict,
    dataset: pd.DataFrame,
) -> None:
    """Export all diagnostic graphs to RUN_DIR."""
    log.info("  Exporting diagnostic graphs ...")
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Feature Importance (gain) ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    imp = model.get_score(importance_type="gain")
    # sort by gain descending
    imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
    ax.barh(list(imp_sorted.keys())[::-1], list(imp_sorted.values())[::-1], color="#4C72B0")
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (Gain)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # 2. Calibration Curve (reliability diagram) ----------------------------
    probs = oos_bundle["probs"]
    labels = oos_bundle["labels"]

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", label="Model", linewidth=2)
    except ValueError:
        ax.text(0.5, 0.5, "Insufficient data for calibration curve",
                ha="center", va="center", transform=ax.transAxes)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (OOS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)

    # 3. CV Fold Metrics bar chart ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    folds = cv_df["fold"].values
    for ax, metric, color in zip(axes, ["auc", "aucpr"], ["#4C72B0", "#DD8452"]):
        vals = cv_df[metric].values
        ax.bar(folds, vals, color=color, alpha=0.8)
        ax.axhline(np.nanmean(vals), color="red", ls="--", lw=1.5, label=f"Mean={np.nanmean(vals):.4f}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"CV {metric.upper()} per Fold")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "cv_fold_metrics.png", dpi=150)
    plt.close(fig)

    # 4. OOS Probability Distribution --------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, label="dip=0", color="#4C72B0", density=True)
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, label="dip=1", color="#C44E52", density=True)
    ax.axvline(MIN_OP_THRESH, color="black", ls="--", lw=1.5, label=f"Min thresh={MIN_OP_THRESH}")
    ax.set_xlabel("Calibrated P(dip)")
    ax.set_ylabel("Density")
    ax.set_title("OOS Probability Distribution by True Label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN_DIR / "probability_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Label Rate by Year (dip-rate drift check) --------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ds = dataset.copy()
    ds["year"] = pd.DatetimeIndex(ds.index).year
    yearly = ds.groupby("year")["target_dip"].agg(["mean", "sum", "count"])
    yearly.columns = ["dip_rate", "n_dips", "n_bars"]
    ax.bar(yearly.index, yearly["dip_rate"], color="#55A868", alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Dip Rate")
    ax.set_title("Label Rate (dip=1 fraction) by Year")
    for i, (yr, row) in enumerate(yearly.iterrows()):
        ax.text(yr, row["dip_rate"] + 0.001, f"{int(row['n_dips'])}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "label_rate_by_year.png", dpi=150)
    plt.close(fig)

    # 6. Stoch %K vs P(dip) scatter -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use OOS data: need to reconstruct stoch_k from dataset
    oos_dates = oos_bundle["dates"]
    # find matching rows in dataset
    oos_mask = dataset.index.isin(pd.DatetimeIndex(oos_dates))
    oos_ds = dataset.loc[oos_mask].copy()
    if len(oos_ds) >= len(probs):
        oos_ds = oos_ds.iloc[:len(probs)]
    if "stoch_k" in oos_ds.columns and len(oos_ds) == len(probs):
        scatter_colors = ["#C44E52" if lab == 1 else "#4C72B0" for lab in labels]
        ax.scatter(oos_ds["stoch_k"].values, probs, c=scatter_colors, alpha=0.15, s=3)
        ax.set_xlabel("Stochastic %K")
        ax.set_ylabel("Calibrated P(dip)")
        ax.set_title("Stoch %K vs P(dip) [red=dip, blue=no-dip]")
        ax.axhline(MIN_OP_THRESH, color="black", ls="--", lw=1, alpha=0.7)
    else:
        ax.text(0.5, 0.5, "OOS length mismatch -- skipped",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "stochk_vs_prob_scatter.png", dpi=150)
    plt.close(fig)

    # 7. Feature Correlation Heatmap ----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = dataset[FEATURE_COLS].corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURE_COLS, fontsize=8)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation (Pearson)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)

    log.info(f"  Saved 7 diagnostic graphs to {RUN_DIR}")


# ===========================================================================
# SECTION 9 : Fixed-Horizon Forward Return Analysis
# ===========================================================================

def evaluate_forward_returns(
    dataset: pd.DataFrame,
    probs: np.ndarray,
    oos_dates: np.ndarray,
    min_thresh: float = MIN_OP_THRESH,
    top_n: int = TOP_N_SIGNALS,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate raw predictive edge of OOS signals via forward returns.

    Cross-sectional ranking: for each OOS date, keep only the top *top_n*
    signals above *min_thresh*.  Then compute H-bar forward returns grouped
    by ticker to prevent cross-asset leakage.

    Returns a DataFrame indexed by horizon with columns:
        signals, win_rate_pct, mean_ret_pct, median_ret_pct
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]

    # Reconstruct the OOS slice of the dataset aligned with probs
    oos_idx = pd.DatetimeIndex(oos_dates)
    oos_mask = dataset.index.isin(oos_idx)
    oos_ds = dataset.loc[oos_mask].copy()

    if len(oos_ds) > len(probs):
        oos_ds = oos_ds.iloc[:len(probs)]
    if len(oos_ds) != len(probs):
        log.warning(f"  OOS length mismatch: dataset={len(oos_ds)}, probs={len(probs)} -- skipping forward returns")
        return pd.DataFrame()

    oos_ds["prob"] = probs

    # Compute forward returns per ticker (no cross-asset leakage)
    for h in horizons:
        oos_ds[f"fwd_ret_{h}"] = (
            oos_ds.groupby("ticker")["Close"].shift(-h) / oos_ds["Close"] - 1.0
        )

    # Cross-sectional ranking: floor filter then top-N per day
    oos_ds["_date"] = pd.DatetimeIndex(oos_ds.index).normalize()
    sig_ds = oos_ds.loc[oos_ds["prob"] >= min_thresh].copy()

    if len(sig_ds) == 0:
        log.info(f"  No OOS events at P >= {min_thresh}")
        return pd.DataFrame()

    sig_ds = (
        sig_ds
        .sort_values("prob", ascending=False)
        .groupby("_date", sort=False)
        .head(top_n)
    )
    n_days = sig_ds["_date"].nunique()
    log.info(f"  OOS daily top-{top_n} ranking: {len(sig_ds)} events across {n_days} days (P >= {min_thresh})")

    # Aggregate metrics per horizon
    rows: list[dict] = []
    for h in horizons:
        col = f"fwd_ret_{h}"
        valid = sig_ds[col].dropna()
        n = len(valid)
        if n == 0:
            rows.append(dict(horizon=f"T+{h}", signals=0, win_rate_pct=np.nan,
                             mean_ret_pct=np.nan, median_ret_pct=np.nan))
            continue
        rows.append(dict(
            horizon=f"T+{h}",
            signals=n,
            win_rate_pct=round(float((valid > 0).mean()) * 100, 2),
            mean_ret_pct=round(float(valid.mean()) * 100, 4),
            median_ret_pct=round(float(valid.median()) * 100, 4),
        ))

    fwd_df = pd.DataFrame(rows).set_index("horizon")
    return fwd_df


# ===========================================================================
# SECTION 10 : Trade Management Engine (Dynamic Exit & Inverse Vol Sizing)
# ===========================================================================

def _precompute_trade_data(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute ATR14 (absolute + pct) for anchored ATR bracket exits."""
    td = ohlcv.copy()
    atr14 = _atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], 14)
    td["atr14"]     = atr14
    td["atr14_pct"] = atr14 / (ohlcv["Close"] + 1e-10)
    return td


def _simulate_single_trade(
    td: pd.DataFrame,
    entry_date: pd.Timestamp,
    max_hold: int = MAX_HOLD_BARS,
) -> dict | None:
    """Anchored ATR Bracket trade.

    Entry at T0 Close.
    At T0, lock atr_entry = ATR14 of that bar.
    take_profit = Entry + TP_ATR_MULT * atr_entry   (static ceiling)
    stop_loss   = Entry - SL_ATR_MULT * atr_entry   (static floor)
    Time stop   = Close at T+max_hold (10 bars).

    Returns trade dict or None if the signal cannot be executed.
    """
    if entry_date not in td.index:
        return None
    loc = td.index.get_loc(entry_date)
    # Resolve possible slice (duplicate dates)
    if not isinstance(loc, (int, np.integer)):
        arr = np.arange(len(td))[loc]
        loc = int(arr[0])

    entry_bar    = td.iloc[loc]
    entry_price  = float(entry_bar["Close"])
    entry_atr    = float(entry_bar["atr14"])
    entry_atr_p  = float(entry_bar["atr14_pct"])

    if pd.isna(entry_atr) or entry_atr <= 0 or entry_price <= 0:
        return None

    # --- Static exit nodes anchored at T0 -----------------------------------
    take_profit = entry_price + TP_ATR_MULT * entry_atr
    stop_loss   = entry_price - SL_ATR_MULT * entry_atr

    for offset in range(1, max_hold + 1):
        bar_loc = loc + offset
        if bar_loc >= len(td):
            # Data exhausted before max hold
            last = len(td) - 1
            return dict(
                entry_date=entry_date,
                entry_price=round(entry_price, 2),
                exit_date=td.index[last],
                exit_price=round(float(td.iloc[last]["Close"]), 2),
                exit_reason="data_end",
                hold_bars=last - loc,
                atr_pct=entry_atr_p,
            )

        bar      = td.iloc[bar_loc]
        bar_low  = float(bar["Low"])
        bar_high = float(bar["High"])

        # --- 1. Stop-loss hit (check worst case first) ----------------------
        if bar_low <= stop_loss:
            return dict(
                entry_date=entry_date,
                entry_price=round(entry_price, 2),
                exit_date=td.index[bar_loc],
                exit_price=round(stop_loss, 2),
                exit_reason="stop_loss",
                hold_bars=offset,
                atr_pct=entry_atr_p,
            )

        # --- 2. Take-profit hit (limit fill at ceiling) ---------------------
        if bar_high >= take_profit:
            return dict(
                entry_date=entry_date,
                entry_price=round(entry_price, 2),
                exit_date=td.index[bar_loc],
                exit_price=round(take_profit, 2),
                exit_reason="take_profit",
                hold_bars=offset,
                atr_pct=entry_atr_p,
            )

        # --- 3. Time stop at max_hold (T+10) --------------------------------
        if offset == max_hold:
            return dict(
                entry_date=entry_date,
                entry_price=round(entry_price, 2),
                exit_date=td.index[bar_loc],
                exit_price=round(float(bar["Close"]), 2),
                exit_reason="time_stop",
                hold_bars=offset,
                atr_pct=entry_atr_p,
            )

    return None


def _compute_backtest_metrics(
    trade_log: pd.DataFrame,
    equity_df: pd.DataFrame,
    capital: float,
    friction: float,
) -> pd.DataFrame:
    """CAGR, Max DD, Sharpe, Profit Factor, Win Rate, Avg Win/Loss, Hold Time."""
    active = trade_log.dropna(subset=["dollar_pnl"])
    if len(active) == 0:
        return pd.DataFrame()

    pnl_pct  = active["pnl_pct"]
    dollar   = active["dollar_pnl"]
    wins_pct = pnl_pct[pnl_pct > 0]
    loss_pct = pnl_pct[pnl_pct <= 0]
    wins_d   = dollar[dollar > 0]
    loss_d   = dollar[dollar <= 0]

    win_rate  = len(wins_pct) / len(pnl_pct) * 100.0
    avg_win   = float(wins_pct.mean()) if len(wins_pct) else 0.0
    avg_loss  = float(loss_pct.mean()) if len(loss_pct) else 0.0
    pf        = abs(float(wins_d.sum()) / float(loss_d.sum())) if loss_d.sum() != 0 else np.inf
    avg_hold  = float(active["hold_bars"].mean())

    eq        = equity_df["equity"]
    final_eq  = float(eq.iloc[-1])
    n_days    = (eq.index[-1] - eq.index[0]).days
    years     = max(n_days / 365.25, 0.01)
    cagr      = (final_eq / capital) ** (1.0 / years) - 1.0 if final_eq > 0 else -1.0
    total_ret = final_eq / capital - 1.0

    peak   = eq.cummax()
    dd     = (eq - peak) / peak
    max_dd = float(dd.min())

    dr     = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0

    return pd.DataFrame([dict(
        CAGR=f"{cagr:.2%}",
        Max_Drawdown=f"{max_dd:.2%}",
        Sharpe_Ratio=round(sharpe, 3),
        Profit_Factor=round(pf, 3) if not np.isinf(pf) else "inf",
        Win_Rate=f"{win_rate:.1f}%",
        Avg_Win_Pct=f"{avg_win:.2f}%",
        Avg_Loss_Pct=f"{avg_loss:.2f}%",
        Avg_Hold_Bars=round(avg_hold, 1),
        Total_Trades=len(active),
        Total_Return=f"{total_ret:.2%}",
        Final_Equity=f"{final_eq:,.0f}",
    )])


def simulate_trades(
    dataset: pd.DataFrame,
    probs: np.ndarray,
    oos_dates: np.ndarray,
    ohlcv_map: dict[str, pd.DataFrame],
    capital: float = CAPITAL,
    friction: float = FRICTION,
    max_hold: int = MAX_HOLD_BARS,
    top_n: int = TOP_N_SIGNALS,
    max_weight: float = MAX_WEIGHT,
    min_thresh: float = MIN_OP_THRESH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full portfolio backtest: Chandelier Exit + Inverse Volatility Sizing.

    Pipeline
    --------
    1. Reconstruct OOS signal set (cross-sectional top-N ranking).
    2. Pre-compute ATR14 & roll_max_15 for every ticker.
    3. Simulate each trade's entry / exit mechanics.
    4. Chronological portfolio simulation with inverse-vol sizing.
    5. Compute summary metrics and export equity curve.

    Returns (trade_log, equity_curve_df, metrics_df).
    """
    # -- 1. Reconstruct OOS signals -------------------------------------------
    oos_idx = pd.DatetimeIndex(oos_dates)
    oos_mask = dataset.index.isin(oos_idx)
    oos_ds = dataset.loc[oos_mask].copy()
    if len(oos_ds) > len(probs):
        oos_ds = oos_ds.iloc[:len(probs)]
    if len(oos_ds) != len(probs):
        log.warning("  OOS length mismatch -- aborting trade simulation")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    oos_ds["prob"]  = probs
    oos_ds["_date"] = pd.DatetimeIndex(oos_ds.index).normalize()
    sig_ds = oos_ds.loc[oos_ds["prob"] >= min_thresh].copy()
    if len(sig_ds) == 0:
        log.info(f"  No OOS signals at P >= {min_thresh}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sig_ds = (
        sig_ds
        .sort_values("prob", ascending=False)
        .groupby("_date", sort=False)
        .head(top_n)
    )
    n_sig_days = sig_ds["_date"].nunique()
    log.info(f"  OOS signal pool: {len(sig_ds):,} signals across {n_sig_days:,} days")

    # -- 2. Pre-compute trade data for every ticker --------------------------
    log.info(f"  Pre-computing ATR14 & roll_max_15 for {len(ohlcv_map)} tickers ...")
    td_map: dict[str, pd.DataFrame] = {
        t: _precompute_trade_data(df) for t, df in ohlcv_map.items()
    }

    # -- 3. Simulate individual trade mechanics ------------------------------
    trades: list[dict] = []
    for row_date, row in sig_ds.iterrows():
        ticker = row["ticker"]
        if ticker not in td_map:
            continue
        t = _simulate_single_trade(td_map[ticker], row_date, max_hold)
        if t is not None:
            t["ticker"] = ticker
            t["prob"]   = float(row["prob"])
            trades.append(t)

    if not trades:
        log.warning("  No valid trades simulated")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trade_log = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    n_sma  = int((trade_log["exit_reason"] == "sma20_takeprofit").sum())
    n_cat  = int((trade_log["exit_reason"] == "catastrophic_stop").sum())
    n_time = int((trade_log["exit_reason"] == "time_stop").sum())
    n_dend = int((trade_log["exit_reason"] == "data_end").sum())
    log.info(
        f"  Simulated {len(trade_log):,} trades  "
        f"(sma20_tp={n_sma}, catastrophic={n_cat}, time_stop={n_time}, data_end={n_dend})"
    )

    # -- 4. Chronological portfolio simulation --------------------------------
    min_date = trade_log["entry_date"].min()
    max_date = trade_log["exit_date"].max()

    # Unified trading calendar
    all_dates_set: set = set()
    for ticker in td_map:
        idx = td_map[ticker].index
        mask = (idx >= min_date) & (idx <= max_date)
        all_dates_set.update(idx[mask].tolist())
    all_dates = sorted(all_dates_set)

    # Close-price lookup  dict[ticker][date] -> float
    trade_tickers = set(trade_log["ticker"].unique())
    close_lkup: dict[str, dict] = {}
    for ticker in trade_tickers:
        s = ohlcv_map[ticker]["Close"].reindex(pd.DatetimeIndex(all_dates), method="ffill")
        close_lkup[ticker] = s.to_dict()

    # Entry-date groups for quick lookup
    entry_groups: dict = {}
    for ed, grp in trade_log.groupby("entry_date"):
        entry_groups[ed] = grp

    # Portfolio state
    cash = float(capital)
    open_pos: dict[int, dict] = {}     # trade_log.index -> position dict
    equity_hist: list[dict] = []

    # Dollar tracking columns (filled during position exit)
    trade_log["allocation"] = np.nan
    trade_log["dollar_pnl"] = np.nan
    trade_log["pnl_pct"]    = np.nan

    for date in all_dates:
        # -- 4a. Close positions whose exit_date has arrived -----------------
        closed = []
        for tid, pos in open_pos.items():
            if date >= pos["exit_date"]:
                exit_val  = pos["shares"] * pos["exit_price"]
                fric_cost = friction * pos["shares"] * pos["entry_price"]
                realised  = exit_val - fric_cost
                cash += realised
                pnl_d = realised - pos["allocation"]
                pnl_p = (pos["exit_price"] / pos["entry_price"] - 1.0 - friction) * 100.0
                trade_log.at[tid, "allocation"] = pos["allocation"]
                trade_log.at[tid, "dollar_pnl"] = round(pnl_d, 2)
                trade_log.at[tid, "pnl_pct"]    = round(pnl_p, 4)
                closed.append(tid)
        for tid in closed:
            del open_pos[tid]

        # -- 4b. Total equity (for sizing) -----------------------------------
        total_equity = cash
        for pos in open_pos.values():
            px = close_lkup.get(pos["ticker"], {}).get(date, pos["entry_price"])
            total_equity += pos["shares"] * px

        # -- 4c. Open new positions (inverse-vol sizing) ---------------------
        if date in entry_groups:
            day_trades = entry_groups[date]
            if len(day_trades) > 0 and cash > 0:
                atr_pcts = day_trades["atr_pct"].values.astype(float)
                valid = atr_pcts > 0
                if valid.any():
                    inv_vol = np.where(valid, 1.0 / atr_pcts, 0.0)
                    s = inv_vol.sum()
                    weights = inv_vol / s if s > 0 else np.zeros_like(inv_vol)
                    weights = np.minimum(weights, max_weight)

                    for i, (idx, tr) in enumerate(day_trades.iterrows()):
                        w = weights[i]
                        if w <= 0 or cash <= 0:
                            continue
                        alloc = min(total_equity * w, cash)
                        if alloc <= 0:
                            continue
                        shares = alloc / tr["entry_price"]
                        cash -= alloc
                        open_pos[idx] = dict(
                            ticker=tr["ticker"],
                            shares=shares,
                            entry_price=tr["entry_price"],
                            exit_date=tr["exit_date"],
                            exit_price=tr["exit_price"],
                            allocation=alloc,
                        )

        # -- 4d. End-of-day mark-to-market -----------------------------------
        eod = cash
        for pos in open_pos.values():
            px = close_lkup.get(pos["ticker"], {}).get(date, pos["entry_price"])
            eod += pos["shares"] * px
        equity_hist.append(dict(date=date, equity=round(eod, 2)))

    equity_df = pd.DataFrame(equity_hist).set_index("date")

    # -- 5. Summary metrics --------------------------------------------------
    metrics = _compute_backtest_metrics(trade_log, equity_df, capital, friction)

    # -- 6. Debug: first 5 closed trades -------------------------------------
    first5 = trade_log.dropna(subset=["dollar_pnl"]).head(5)
    log.info("\n  +--- First 5 Closed Trades (Debug) -------------------------------------------+")
    for i, (_, t) in enumerate(first5.iterrows()):
        log.info(
            f"  | #{i+1}: {t['ticker']:<8} "
            f"Entry={pd.Timestamp(t['entry_date']).strftime('%Y-%m-%d')} @ {t['entry_price']:>10.2f}  "
            f"Exit={pd.Timestamp(t['exit_date']).strftime('%Y-%m-%d')} @ {t['exit_price']:>10.2f}  "
            f"{t['exit_reason']:<16} PnL={t['pnl_pct']:+.2f}%"
        )
    log.info("  +----------------------------------------------------------------------------+")

    # -- 7. Export ------------------------------------------------------------
    equity_df.to_csv(RUN_DIR / "equity_curve.csv")
    trade_log.to_csv(RUN_DIR / "trade_log.csv", index=False)
    metrics.to_csv(RUN_DIR / "backtest_metrics.csv", index=False)
    log.info(f"  Saved equity_curve.csv, trade_log.csv, backtest_metrics.csv")

    return trade_log, equity_df, metrics


# ===========================================================================
# SECTION 11 : Main
# ===========================================================================

def main():
    t0 = time.time()
    log.info("=" * 65)
    log.info("  XGBoost Dip Detector v2.1 - Bottom-Finding Signals")
    log.info(f"  Run: {RUN_DIR.name}")
    log.info("  Fixes: right-edge guard | stratified calibration | NEPSE index features")
    log.info("=" * 65)

    # -- 1. Data ---------------------------------------------------------------
    log.info("\n[1/6] Loading all tickers ...")
    dataset, ohlcv_map, idx_feats = assemble_dataset()

    # -- 2. Train --------------------------------------------------------------
    log.info("\n[2/6] Training (walk-forward CV + production model) ...")
    model, calibrator, cv_df, oos_bundle = train_model(dataset)
    cv_df.to_csv(RUN_DIR / "cv_results.csv", index=False)
    model.save_model(str(RUN_DIR / "model.json"))

    # -- 3. Diagnostics --------------------------------------------------------
    log.info("\n[3/6] Exporting diagnostics ...")
    _export_diagnostics(model, cv_df, oos_bundle, dataset)

    # -- 4. Forward return analysis (OOS, daily top-N ranking) ----------------
    log.info(f"\n[4/6] Forward return analysis (OOS, top-{TOP_N_SIGNALS}/day, P >= {MIN_OP_THRESH}) ...")
    fwd_df = evaluate_forward_returns(
        dataset, oos_bundle["probs"], oos_bundle["dates"],
        min_thresh=MIN_OP_THRESH,
        top_n=TOP_N_SIGNALS,
    )
    if len(fwd_df) > 0:
        log.info(f"\n{fwd_df.to_string()}")
        fwd_df.to_csv(RUN_DIR / "forward_returns_analysis.csv")
        log.info(f"  Saved forward_returns_analysis.csv")
    else:
        log.info("  Forward return analysis: no data")

    # -- 5. Trade Simulation (Anchored ATR Bracket + Inverse Vol Sizing) -----
    log.info(f"\n[5/6] Trade simulation (ATR Bracket {TP_ATR_MULT}R/{SL_ATR_MULT}R + InvVol, {MAX_HOLD_BARS}-bar max hold) ...")
    trade_log, equity_df, metrics = simulate_trades(
        dataset, oos_bundle["probs"], oos_bundle["dates"], ohlcv_map,
    )
    if len(metrics) > 0:
        log.info(f"\n  +--- Backtest Metrics -------------------------------------------------------+")
        for col in metrics.columns:
            log.info(f"  |  {col:<20} {metrics[col].iloc[0]}")
        log.info(f"  +--------------------------------------------------------------------------+")
    else:
        log.info("  Trade simulation: no trades executed")

    # -- 6. Score & emit signals (cross-sectional top-N) -----------------------
    log.info(f"\n[6/6] Scoring latest bars (top-{TOP_N_SIGNALS}/day, P >= {MIN_OP_THRESH}) ...")
    sig_df = score_latest_signals(ohlcv_map, model, calibrator, idx_feats)

    log.info(f"\n{'=' * 65}")
    if len(sig_df) == 0:
        log.info(f"  NO SIGNALS above P >= {MIN_OP_THRESH}")
    else:
        log.info(f"  TOP {TOP_N_SIGNALS} SIGNALS PER DAY  ({len(sig_df)} total, P >= {MIN_OP_THRESH})")
        log.info(f"{'=' * 65}")
        log.info(f"\n{sig_df.to_string(index=False)}")
        sig_df.to_csv(RUN_DIR / "signals.csv", index=False)
    log.info(f"{'=' * 65}")

    elapsed = time.time() - t0
    log.info(f"\n  All outputs -> {RUN_DIR}")
    log.info(f"  Total time: {elapsed:.1f}s")
    log.info("  Done.")


if __name__ == "__main__":
    main()
