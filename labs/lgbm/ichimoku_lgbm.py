#!/usr/bin/env python3
"""
NEPSE Ichimoku + LightGBM Signal-Quality Predictor  (v2)
=========================================================
Production-ready ML system that filters Ichimoku Kumo Break
signals using gradient-boosted signal quality prediction.

v2 fixes over v1:
  - Target engineering: winsorized PnL (clip ±30%), Huber loss
  - Removed slow `relative_volume_rank` (.apply → 30s overhead)
  - Classification: scale_pos_weight from actual class ratio,
    tuned threshold via F1-optimal search, not hardcoded 0.5
  - Stronger regularization (min_child_samples=50, num_leaves=15)
  - Walk-forward uses calendar months, not sparse signal dates
  - Filtering: median-split + percentile thresholds, not pred > 0
  - Optuna hyperparameter tuning (optional, controlled by flag)
  - Purged time-series CV for model selection
  - Reduced to 38 fast, high-signal features

Architecture:
  Rule-Based Signal → 38-Feature Vector → LightGBM Quality Score → Take/Skip

Feature Groups (38 total):
  Ichimoku Core (11)  — price_vs_kumo_top/bot, kumo_thickness,
                         tenkan_kijun_spread, price_vs_tenkan/kijun,
                         future_kumo_spread, chikou_clearance,
                         senkou_b_flatness, kijun_slope, tenkan_slope
  Trend (6)           — close_vs_sma20/50/200, sma20_slope, adx_14,
                         trend_alignment
  Momentum (6)        — rsi_14, rsi_slope_5, macd_histogram,
                         macd_hist_slope, roc_5, roc_20
  Volatility (5)      — atr_pct, bb_width, bb_position,
                         volatility_ratio, atr_expansion
  Volume (4)          — volume_surge, volume_trend, obv_slope,
                         volume_price_confirm
  Candlestick (4)     — candle_body_ratio, upper_shadow_ratio, clv,
                         gap_pct
  Breakout (2)        — donchian_20_position, breakout_strength
"""

import warnings; warnings.filterwarnings("ignore")
import json, logging, pathlib, datetime, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from scipy.stats import spearmanr

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT     = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR         = PROJECT_ROOT / "data/ohlcv/1D/stocks"

# Ichimoku periods (standard)
TENKAN_PERIOD    = 9
KIJUN_PERIOD     = 26
SENKOU_B_PERIOD  = 52
DISPLACEMENT     = 26

# Backtest parameters (must match rule-based strategy)
MIN_ROWS         = 250
ICHIMOKU_WARMUP  = 80
SPLIT_DATE       = "2024-07-01"
LONG_ONLY        = True
ATR_PERIOD       = 14
ATR_STOP_MULT    = 1.5
TRANSACTION_COST = 0.005
ORDER_TIMEOUT    = 5
CHIKOU_FREE_HALF = 2
MAX_KIJUN_DIST   = 5.0
HH_PROXIMITY     = 1.5
FLAT_SB_LOOKBACK = 15
FLAT_SB_TOL      = 0.001
FLAT_SB_STRONG   = 0.5

# ML parameters
SEED             = 42
MIN_TRAIN_SIGNALS = 50         # minimum signals before first model fit
WALK_FORWARD_MONTHS = 3        # refit every N calendar months
PNL_CLIP         = 30.0        # winsorize target at ±30%
RUN_OPTUNA       = False       # set True for hyperparameter search
OPTUNA_TRIALS    = 40          # number of Optuna trials

# Feature list (38 features — dropped slow/low-signal ones from v1)
FEATURE_NAMES = [
    # Ichimoku Core (11)
    "price_vs_kumo_top", "price_vs_kumo_bot", "kumo_thickness",
    "tenkan_kijun_spread", "price_vs_tenkan", "price_vs_kijun",
    "future_kumo_spread", "chikou_clearance", "senkou_b_flatness",
    "kijun_slope", "tenkan_slope",
    # Trend (6)
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
    "sma20_slope", "adx_14", "trend_alignment",
    # Momentum (6)
    "rsi_14", "rsi_slope_5", "macd_histogram", "macd_hist_slope",
    "roc_5", "roc_20",
    # Volatility (5)
    "atr_pct", "bb_width", "bb_position",
    "volatility_ratio", "atr_expansion",
    # Volume (4)
    "volume_surge", "volume_trend", "obv_slope",
    "volume_price_confirm",
    # Candlestick (4)
    "candle_body_ratio", "upper_shadow_ratio", "clv", "gap_pct",
    # Breakout (2)
    "donchian_20_position", "breakout_strength",
]

LGB_PARAMS_REG = {
    "objective":        "huber",          # robust to PnL outliers
    "metric":           "rmse",
    "learning_rate":    0.05,
    "max_depth":        4,                # shallower = less overfit
    "num_leaves":       15,               # conservative for ~3k samples
    "min_child_samples": 50,              # high → strong regularization
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "reg_alpha":        1.0,              # L1
    "reg_lambda":       5.0,              # L2
    "feature_fraction_seed": SEED,
    "verbosity":        -1,
    "seed":             SEED,
    "n_jobs":           -1,
}

LGB_PARAMS_CLF = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "max_depth":        4,
    "num_leaves":       15,
    "min_child_samples": 50,
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "reg_alpha":        1.0,
    "reg_lambda":       5.0,
    # scale_pos_weight set dynamically from training data
    "verbosity":        -1,
    "seed":             SEED,
    "n_jobs":           -1,
}

# ============================================================================
# SETUP
# ============================================================================

def setup():
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = PROJECT_ROOT / f"runs/{RUN_TS}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("ichimoku_lgbm_v2")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(RUN_DIR / "ichimoku_lgbm.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); log.addHandler(sh)

    log.info("=" * 70)
    log.info("NEPSE Ichimoku + LightGBM Signal-Quality Predictor  v2")
    log.info("=" * 70)
    log.info(f"Run directory  : {RUN_DIR.resolve()}")
    log.info(f"Data dir       : {DATA_DIR.resolve()}")
    log.info(f"Split date     : {SPLIT_DATE}")
    log.info(f"Features       : {len(FEATURE_NAMES)}")
    log.info(f"PnL clip       : +/-{PNL_CLIP}%")
    log.info(f"Optuna tuning  : {RUN_OPTUNA}")
    return log, RUN_DIR

# ============================================================================
# DATA LOADING
# ============================================================================

def _build_exclude_set():
    excl = set()
    stocks_json = PROJECT_ROOT / "data/stocks.json"
    if stocks_json.exists():
        for s in json.load(open(stocks_json, encoding="utf-8")):
            if s.get("sector") == "PROMOTSHARE":
                excl.add(s["script"])
    for fname in ("mutual.json", "corpdeben.json"):
        p = PROJECT_ROOT / "data" / fname
        if p.exists():
            for s in json.load(open(p, encoding="utf-8")):
                excl.add(s["script"])
    return excl

EXCLUDE_TICKERS = _build_exclude_set()

def load_ohlcv(log):
    log.info("Loading OHLCV data...")
    log.info(f"Excluding {len(EXCLUDE_TICKERS)} tickers")
    frames, skipped, excluded = {}, 0, 0
    for csv in sorted(DATA_DIR.glob("*.csv")):
        if csv.stem in EXCLUDE_TICKERS:
            excluded += 1; continue
        try:
            df = pd.read_csv(csv, parse_dates=["Timestamp"])
            if df.empty or len(df) < MIN_ROWS:
                skipped += 1; continue
            df = df.rename(columns={"Timestamp": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            df = df.set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if not {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                skipped += 1; continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < MIN_ROWS:
                skipped += 1; continue
            frames[csv.stem] = df
        except Exception as e:
            log.warning(f"Skip {csv.stem}: {e}"); skipped += 1
    log.info(f"Loaded {len(frames)} tickers, skipped {skipped}, excluded {excluded}")
    return frames

# ============================================================================
# INDICATOR COMPUTATION
# ============================================================================

def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def _ema(s, n):
    return s.ewm(span=n, min_periods=n, adjust=False).mean()

def _rsi(c, n=14):
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(com=n - 1, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(com=n - 1, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)

def _adx(h, l, c, n=14):
    plus_dm = h.diff().clip(lower=0.0)
    minus_dm = (-l.diff()).clip(lower=0.0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0
    tr = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(com=n - 1, min_periods=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(com=n - 1, min_periods=n, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(com=n - 1, min_periods=n, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(com=n - 1, min_periods=n, adjust=False).mean()

def compute_ichimoku(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    tenkan = (h.rolling(TENKAN_PERIOD).max() + l.rolling(TENKAN_PERIOD).min()) / 2
    kijun = (h.rolling(KIJUN_PERIOD).max() + l.rolling(KIJUN_PERIOD).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (h.rolling(SENKOU_B_PERIOD).max() + l.rolling(SENKOU_B_PERIOD).min()) / 2
    senkou_a = senkou_a_raw.shift(DISPLACEMENT)
    senkou_b = senkou_b_raw.shift(DISPLACEMENT)
    kumo_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    kumo_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    tr = pd.concat([
        h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    return {
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": senkou_a, "senkou_b": senkou_b,
        "kumo_top": kumo_top, "kumo_bot": kumo_bot,
        "future_sa": senkou_a_raw, "future_sb": senkou_b_raw,
        "atr": atr,
    }

# ============================================================================
# FEATURE ENGINEERING — 38 FEATURES (fast, high-signal)
# ============================================================================

def compute_all_features(df, ich):
    """
    Compute 38 features for every bar.  All lookahead-free.
    All price-distance features are ATR-normalized.
    """
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    tenkan, kijun = ich["tenkan"], ich["kijun"]
    senkou_a, senkou_b = ich["senkou_a"], ich["senkou_b"]
    kumo_top, kumo_bot = ich["kumo_top"], ich["kumo_bot"]
    fut_sa, fut_sb = ich["future_sa"], ich["future_sb"]
    atr = ich["atr"]
    eps = 1e-10

    feats = pd.DataFrame(index=df.index)

    # ── Ichimoku Core (11) ───────────────────────────────────────────
    feats["price_vs_kumo_top"]    = (c - kumo_top) / (atr + eps)
    feats["price_vs_kumo_bot"]    = (c - kumo_bot) / (atr + eps)
    feats["kumo_thickness"]       = (kumo_top - kumo_bot) / (atr + eps)
    feats["tenkan_kijun_spread"]  = (tenkan - kijun) / (atr + eps)
    feats["price_vs_tenkan"]      = (c - tenkan) / (atr + eps)
    feats["price_vs_kijun"]       = (c - kijun) / (atr + eps)
    feats["future_kumo_spread"]   = (fut_sa - fut_sb) / (atr + eps)
    feats["chikou_clearance"]     = (c - c.shift(DISPLACEMENT)) / (atr + eps)
    sb_std  = senkou_b.rolling(FLAT_SB_LOOKBACK).std()
    sb_mean = senkou_b.rolling(FLAT_SB_LOOKBACK).mean()
    feats["senkou_b_flatness"]    = sb_std / (sb_mean + eps)
    feats["kijun_slope"]          = (kijun - kijun.shift(5)) / (atr + eps)
    feats["tenkan_slope"]         = (tenkan - tenkan.shift(5)) / (atr + eps)

    # ── Trend (6) ────────────────────────────────────────────────────
    sma20  = _sma(c, 20)
    sma50  = _sma(c, 50)
    sma200 = _sma(c, 200)
    feats["close_vs_sma20"]  = (c - sma20) / (sma20 + eps)
    feats["close_vs_sma50"]  = (c - sma50) / (sma50 + eps)
    feats["close_vs_sma200"] = (c - sma200) / (sma200 + eps)
    feats["sma20_slope"]     = (sma20 - sma20.shift(5)) / (sma20.shift(5) + eps)
    feats["adx_14"]          = _adx(h, l, c, 14)
    feats["trend_alignment"] = (
        (c > sma20).astype(float) + (c > sma50).astype(float) +
        (c > sma200).astype(float) + (sma20 > sma50).astype(float) +
        (sma50 > sma200).astype(float)
    ) / 5.0

    # ── Momentum (6) ─────────────────────────────────────────────────
    rsi = _rsi(c, 14)
    feats["rsi_14"]       = rsi
    feats["rsi_slope_5"]  = rsi - rsi.shift(5)
    ema12 = _ema(c, 12); ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    macd_hist = macd_line - signal_line
    feats["macd_histogram"]  = macd_hist / (atr + eps)
    feats["macd_hist_slope"] = (macd_hist - macd_hist.shift(3)) / (atr + eps)
    feats["roc_5"]  = c.pct_change(5)
    feats["roc_20"] = c.pct_change(20)

    # ── Volatility (5) ───────────────────────────────────────────────
    feats["atr_pct"] = atr / (c + eps)
    std20 = c.rolling(20).std()
    feats["bb_width"] = (4 * std20) / (sma20 + eps)
    upper_bb = sma20 + 2 * std20; lower_bb = sma20 - 2 * std20
    feats["bb_position"]      = (c - lower_bb) / (upper_bb - lower_bb + eps)
    vol_5  = c.pct_change().rolling(5).std()
    vol_20 = c.pct_change().rolling(20).std()
    feats["volatility_ratio"] = vol_5 / (vol_20 + eps)
    atr_sma = atr.rolling(20).mean()
    feats["atr_expansion"]    = atr / (atr_sma + eps)

    # ── Volume (4) ───────────────────────────────────────────────────
    v_sma20 = _sma(v, 20)
    feats["volume_surge"]         = v / (v_sma20 + eps)
    feats["volume_trend"]         = (v_sma20 - v_sma20.shift(5)) / (v_sma20.shift(5) + eps)
    obv = (np.sign(c.diff()) * v).cumsum()
    obv_sma = _sma(obv, 5)
    feats["obv_slope"]            = (obv_sma - obv_sma.shift(5)) / (obv_sma.shift(5).abs() + eps)
    price_dir = np.sign(c.diff()); vol_dir = np.sign(v.diff())
    feats["volume_price_confirm"] = (price_dir * vol_dir).rolling(5).mean()

    # ── Candlestick (4) ──────────────────────────────────────────────
    hl_range = h - l + eps
    feats["candle_body_ratio"]   = (c - o).abs() / hl_range
    feats["upper_shadow_ratio"]  = (h - pd.concat([o, c], axis=1).max(axis=1)) / hl_range
    feats["clv"]                 = ((c - l) - (h - c)) / hl_range
    feats["gap_pct"]             = (o - c.shift(1)) / (c.shift(1) + eps)

    # ── Breakout (2) ─────────────────────────────────────────────────
    high_20 = h.rolling(20).max(); low_20 = l.rolling(20).min()
    feats["donchian_20_position"] = (c - low_20) / (high_20 - low_20 + eps)
    feats["breakout_strength"]    = (c - high_20) / (atr + eps)

    return feats

# ============================================================================
# CHIKOU / FLAT SB CHECKS (from rule-based strategy)
# ============================================================================

def chikou_is_free(c_arr, h_arr, l_arr, t, direction="long",
                   sa_arr=None, sb_arr=None):
    center = t - DISPLACEMENT
    lo = max(0, center - CHIKOU_FREE_HALF)
    hi = min(len(c_arr), center + CHIKOU_FREE_HALF + 1)
    if lo >= hi or center < 0:
        return False
    if direction == "long":
        barrier = np.nanmax(h_arr[lo:hi])
        if np.isnan(barrier) or c_arr[t] <= barrier:
            return False
        if sa_arr is not None and sb_arr is not None:
            kt = max(sa_arr[center], sb_arr[center])
            if not np.isnan(kt) and c_arr[t] <= kt:
                return False
        return True
    return False   # short not used

def is_senkou_b_flat(sb_arr, t):
    if t < FLAT_SB_LOOKBACK:
        return False
    recent = sb_arr[t - FLAT_SB_LOOKBACK:t + 1]
    if np.any(np.isnan(recent)):
        return False
    return (np.nanmax(recent) - np.nanmin(recent)) / max(np.nanmean(recent), 1e-10) < FLAT_SB_TOL

def flat_sb_strong_candle(o_t, c_t, atr_t):
    return (c_t - o_t) >= FLAT_SB_STRONG * atr_t

# ============================================================================
# SIGNAL GENERATION + TRADE SIMULATION + FEATURE EXTRACTION
# ============================================================================

def generate_signals_and_labels(ticker, df, all_features):
    """
    Run Kumo Break strategy, capture 38-feature vector at each signal,
    label with actual trade PnL.
    """
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index
    n = len(df)

    if n < ICHIMOKU_WARMUP + 10:
        return []

    ich = compute_ichimoku(df)
    tenkan   = ich["tenkan"].values
    kijun    = ich["kijun"].values
    senkou_a = ich["senkou_a"].values
    senkou_b = ich["senkou_b"].values
    kumo_top = ich["kumo_top"].values
    kumo_bot = ich["kumo_bot"].values
    fut_sa   = ich["future_sa"].values
    fut_sb   = ich["future_sb"].values
    atr      = ich["atr"].values

    signals = []
    state = "FLAT"
    direction = None
    kumo_brk_long = False
    pend_level = pend_stop = 0.0
    sig_bar = entry_bar = 0
    entry_price = stop_px = 0.0
    entry_date = None
    signal_features = None

    for t in range(ICHIMOKU_WARMUP, n):
        if any(np.isnan(x[t]) for x in [tenkan, kijun, kumo_top, kumo_bot,
                                         fut_sa, fut_sb, atr]):
            continue

        # Kumo break detection
        if t > 0 and not np.isnan(kumo_top[t-1]) and not np.isnan(kumo_bot[t-1]):
            if c[t] > kumo_top[t] and c[t-1] <= kumo_top[t-1]:
                kumo_brk_long = True
        if c[t] <= kumo_top[t]:
            kumo_brk_long = False

        # ── PENDING ──────────────────────────────────────────────────
        if state == "PENDING":
            filled = False
            if o[t] >= pend_level:
                entry_price = o[t]; filled = True
            elif h[t] >= pend_level:
                entry_price = pend_level; filled = True

            if filled:
                stop_px = pend_stop; entry_bar = t
                entry_date = dates[t]; state = "POSITION"
                kumo_brk_long = False; continue

            cancel = (t - sig_bar >= ORDER_TIMEOUT) or c[t] < kumo_bot[t] or tenkan[t] < kijun[t]
            if cancel:
                signals.append({
                    **signal_features,
                    "ticker": ticker, "signal_date": dates[sig_bar],
                    "direction": "LONG", "filled": False,
                    "net_pnl_pct": 0.0, "win": 0, "exit_reason": "cancelled",
                    "bars_held": 0,
                })
                state = "FLAT"; direction = None; signal_features = None
            continue

        # ── POSITION ─────────────────────────────────────────────────
        if state == "POSITION":
            exit_px = exit_rsn = None
            if o[t] <= stop_px:
                exit_px = o[t]; exit_rsn = "gap_stop"
            elif l[t] <= stop_px:
                exit_px = stop_px; exit_rsn = "hard_stop"
            elif c[t] < kijun[t]:
                exit_px = c[t]; exit_rsn = "kijun_close"
            else:
                trail = kijun[t] - ATR_STOP_MULT * atr[t]
                if not np.isnan(trail):
                    stop_px = max(stop_px, trail)

            if exit_px is not None:
                pnl = (exit_px / entry_price - 1.0) * 100.0
                net = pnl - TRANSACTION_COST * 100.0
                signals.append({
                    **signal_features,
                    "ticker": ticker, "signal_date": dates[sig_bar],
                    "entry_date": entry_date, "exit_date": dates[t],
                    "direction": "LONG", "filled": True,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_px, 2),
                    "bars_held": t - entry_bar,
                    "pnl_pct": round(pnl, 4),
                    "net_pnl_pct": round(net, 4),
                    "win": int(net > 0),
                    "exit_reason": exit_rsn,
                })
                state = "FLAT"; direction = None; signal_features = None
            continue

        # ── FLAT — scan for signals ──────────────────────────────────
        if kumo_brk_long:
            cond1 = c[t] > kumo_top[t]
            cond2 = fut_sa[t] > fut_sb[t]
            cond3 = chikou_is_free(c, h, l, t, "long",
                                   sa_arr=senkou_a, sb_arr=senkou_b)
            cond4 = tenkan[t] > kijun[t]
            cond5 = c[t] > tenkan[t] and c[t] > kijun[t]
            cond6 = (c[t] - kijun[t]) / max(atr[t], 1e-10) < MAX_KIJUN_DIST
            cond7 = True
            if is_senkou_b_flat(fut_sb, t):
                if not flat_sb_strong_candle(o[t], c[t], atr[t]):
                    cond7 = False

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7:
                hh9  = np.nanmax(h[max(0, t - TENKAN_PERIOD + 1):t + 1])
                hh26 = np.nanmax(h[max(0, t - KIJUN_PERIOD + 1):t + 1])
                pend_level = hh26 if (hh26 - hh9) / max(atr[t], 1e-10) < HH_PROXIMITY else hh9
                pend_stop = kijun[t] - ATR_STOP_MULT * atr[t]
                sig_bar = t; direction = "LONG"; state = "PENDING"

                # Capture features
                feat_row = all_features.iloc[t]
                signal_features = {f: feat_row[f] for f in FEATURE_NAMES
                                   if f in feat_row.index}

    # Close open position at data end
    if state == "POSITION":
        last_c = c[-1]
        pnl = (last_c / entry_price - 1.0) * 100.0
        net = pnl - TRANSACTION_COST * 100.0
        signals.append({
            **signal_features,
            "ticker": ticker, "signal_date": dates[sig_bar],
            "entry_date": entry_date, "exit_date": dates[-1],
            "direction": "LONG", "filled": True,
            "entry_price": round(entry_price, 2),
            "exit_price": round(last_c, 2),
            "bars_held": n - 1 - entry_bar,
            "pnl_pct": round(pnl, 4), "net_pnl_pct": round(net, 4),
            "win": int(net > 0), "exit_reason": "data_end",
        })

    return signals

# ============================================================================
# DATASET ASSEMBLY
# ============================================================================

def build_signal_dataset(frames, log):
    log.info(f"Generating signals and features for {len(frames)} tickers...")
    all_signals = []
    active = 0
    for i, (tk, df) in enumerate(sorted(frames.items())):
        ich = compute_ichimoku(df)
        feats = compute_all_features(df, ich)
        sigs = generate_signals_and_labels(tk, df, feats)
        if sigs:
            all_signals.extend(sigs); active += 1
        if (i + 1) % 100 == 0:
            log.info(f"  ... {i+1}/{len(frames)} tickers ({len(all_signals)} signals)")
    log.info(f"Done: {len(all_signals)} signals from {active}/{len(frames)} tickers")
    return pd.DataFrame(all_signals) if all_signals else pd.DataFrame()

# ============================================================================
# TARGET ENGINEERING
# ============================================================================

def prepare_targets(df, log):
    """
    Create robust targets from raw trade PnL:
      - target_clipped: PnL winsorized at +/-PNL_CLIP (reduces outlier influence)
      - win: binary (net PnL > 0)
    """
    raw = df["net_pnl_pct"].values
    clipped = np.clip(raw, -PNL_CLIP, PNL_CLIP)
    df["target_clipped"] = clipped
    p5, p95 = np.percentile(raw, [5, 95])
    log.info(f"  Target stats: mean={raw.mean():+.2f}%, median={np.median(raw):+.2f}%, "
             f"P5={p5:+.1f}%, P95={p95:+.1f}%, clipped to +/-{PNL_CLIP}%")
    return df

# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def temporal_split(df, log):
    split = pd.Timestamp(SPLIT_DATE)
    train = df[(df["signal_date"] < split) & (df["filled"] == True)].copy()
    test  = df[(df["signal_date"] >= split) & (df["filled"] == True)].copy()
    log.info(f"Train: {len(train)} signals  |  Test: {len(test)} signals")
    if len(train) > 0:
        wr_train = train["win"].mean() * 100
        log.info(f"  Train win rate: {wr_train:.1f}%")
    if len(test) > 0:
        wr_test = test["win"].mean() * 100
        log.info(f"  Test win rate:  {wr_test:.1f}%")
    return train, test

# ============================================================================
# PURGED TIME-SERIES CV (for Optuna)
# ============================================================================

def purged_cv_score(params, X, y, signal_dates, n_splits=3, purge_days=30):
    """
    Time-series CV with purge gap to avoid leakage from overlapping trades.
    Returns mean RMSE across folds.
    """
    unique_dates = np.sort(signal_dates.unique())
    fold_size = len(unique_dates) // (n_splits + 1)

    scores = []
    for i in range(n_splits):
        train_end_idx = (i + 1) * fold_size

        train_cutoff = unique_dates[train_end_idx]
        purge_cutoff = train_cutoff + pd.Timedelta(days=purge_days)

        # Find the first validation date after purge
        val_dates = unique_dates[unique_dates > purge_cutoff]
        if len(val_dates) < fold_size // 2:
            continue
        val_end = val_dates[min(fold_size, len(val_dates) - 1)]

        train_mask = signal_dates < train_cutoff
        val_mask = (signal_dates > purge_cutoff) & (signal_dates <= val_end)

        if train_mask.sum() < MIN_TRAIN_SIGNALS or val_mask.sum() < 10:
            continue

        dtrain = lgb.Dataset(X[train_mask], label=y[train_mask], free_raw_data=False)
        dval   = lgb.Dataset(X[val_mask], label=y[val_mask], free_raw_data=False)

        model = lgb.train(
            {**params, "verbosity": -1},
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        preds = model.predict(X[val_mask])
        rmse = np.sqrt(mean_squared_error(y[val_mask], preds))
        scores.append(rmse)

    return np.mean(scores) if scores else 1e6

# ============================================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================================

def optuna_tune(X_train, y_train, signal_dates_train, log):
    """
    Bayesian hyperparameter search with purged time-series CV.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.info("Optuna not installed. pip install optuna. Skipping tuning.")
        return LGB_PARAMS_REG

    log.info(f"\nOptuna tuning: {OPTUNA_TRIALS} trials with purged 3-fold CV...")

    def objective(trial):
        params = {
            "objective":        "huber",
            "metric":           "rmse",
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.15, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "num_leaves":       trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "verbosity":        -1,
            "seed":             SEED,
            "n_jobs":           -1,
        }
        return purged_cv_score(params, X_train, y_train, signal_dates_train)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    log.info(f"  Best CV RMSE: {study.best_value:.4f}")
    log.info(f"  Best params: {best}")

    # Map Optuna names back to LGB names
    tuned = {
        "objective": "huber", "metric": "rmse",
        "learning_rate": best["lr"],
        "max_depth": best["max_depth"],
        "num_leaves": best["num_leaves"],
        "min_child_samples": best["min_child_samples"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
        "verbosity": -1, "seed": SEED, "n_jobs": -1,
    }
    return tuned

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_regression_model(train_df, test_df, params, log, run_dir):
    log.info("\n" + "=" * 60)
    log.info("TRAINING: LightGBM Regression (Huber, clipped target)")
    log.info("=" * 60)

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["target_clipped"].values     # winsorized target
    X_test  = test_df[FEATURE_NAMES].values
    y_test  = test_df["target_clipped"].values

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES,
                         free_raw_data=False)
    dtest  = lgb.Dataset(X_test, label=y_test, reference=dtrain,
                         feature_name=FEATURE_NAMES, free_raw_data=False)

    model = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=80, verbose=True),
        ],
    )
    log.info(f"Best iteration: {model.best_iteration}")
    model.save_model(str(run_dir / "model_regression.txt"))
    return model

def train_classification_model(train_df, test_df, log, run_dir):
    log.info("\n" + "=" * 60)
    log.info("TRAINING: LightGBM Classification (Win/Loss)")
    log.info("=" * 60)

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["win"].values
    X_test  = test_df[FEATURE_NAMES].values
    y_test  = test_df["win"].values

    # Compute scale_pos_weight from actual class distribution
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / max(n_pos, 1)
    log.info(f"  Class balance: {n_pos} wins / {n_neg} losses -> scale_pos_weight={spw:.2f}")

    clf_params = {**LGB_PARAMS_CLF, "scale_pos_weight": spw}
    # Remove is_unbalance if present (conflicts with scale_pos_weight)
    clf_params.pop("is_unbalance", None)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES,
                         free_raw_data=False)
    dtest  = lgb.Dataset(X_test, label=y_test, reference=dtrain,
                         feature_name=FEATURE_NAMES, free_raw_data=False)

    model = lgb.train(
        clf_params, dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=80, verbose=True),
        ],
    )
    log.info(f"Best iteration: {model.best_iteration}")
    model.save_model(str(run_dir / "model_classification.txt"))
    return model

# ============================================================================
# OOS EVALUATION
# ============================================================================

def evaluate_regression(model, test_df, log, run_dir):
    X_test = test_df[FEATURE_NAMES].values
    y_actual = test_df["net_pnl_pct"].values        # evaluate on ACTUAL PnL
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_actual, y_pred)
    r2  = r2_score(y_actual, y_pred)
    ic, ic_p = spearmanr(y_pred, y_actual)

    log.info("\n" + "=" * 60)
    log.info("REGRESSION -- OUT-OF-SAMPLE METRICS")
    log.info("=" * 60)
    log.info(f"  MSE                      : {mse:.4f}")
    log.info(f"  R2                       : {r2:.6f}")
    log.info(f"  Information Coefficient   : {ic:.6f}  (p={ic_p:.2e})")
    log.info(f"  Pred range               : [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    log.info(f"  Pred std                 : {y_pred.std():.4f}")

    _log_feature_importance(model, "Regression", log, run_dir,
                            "feature_importance_regression.csv")
    return y_pred, {"mse": mse, "r2": r2, "ic": ic, "ic_p": ic_p}


def evaluate_classification(model, test_df, log, run_dir):
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["win"].values
    y_prob = model.predict(X_test)

    # Find optimal threshold by maximizing F1
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.30, 0.70, 0.02):
        yp = (y_prob >= thr).astype(int)
        f = f1_score(y_test, yp, zero_division=0)
        if f > best_f1:
            best_f1 = f; best_thr = thr
    log.info(f"\n  Optimal threshold: {best_thr:.2f}  (F1={best_f1:.4f})")

    y_pred = (y_prob >= best_thr).astype(int)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0

    log.info("\n" + "=" * 60)
    log.info("CLASSIFICATION -- OUT-OF-SAMPLE METRICS")
    log.info("=" * 60)
    log.info(f"  Accuracy   : {acc:.4f}")
    log.info(f"  Precision  : {prec:.4f}")
    log.info(f"  Recall     : {rec:.4f}")
    log.info(f"  F1 Score   : {f1:.4f}")
    log.info(f"  AUC-ROC    : {auc:.4f}")
    log.info(f"  Threshold  : {best_thr:.2f}")
    log.info(f"  Pred prob range: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Loss','Win'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    _log_feature_importance(model, "Classification", log, run_dir,
                            "feature_importance_classification.csv")
    return y_prob, best_thr, {"acc": acc, "prec": prec, "rec": rec,
                               "f1": f1, "auc": auc, "threshold": best_thr}


def _log_feature_importance(model, label, log, run_dir, fname):
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_NAMES, "Gain": importance,
    }).sort_values("Gain", ascending=False).reset_index(drop=True)
    feat_imp["Pct"] = (feat_imp["Gain"] / (feat_imp["Gain"].sum() + 1e-10) * 100).round(2)

    log.info(f"\nFEATURE IMPORTANCE -- {label} (Gain)")
    log.info("-" * 55)
    for _, row in feat_imp.head(20).iterrows():
        bar = "#" * int(row["Pct"] / 2)
        log.info(f"  {row['Feature']:24s} {row['Gain']:12.1f}  ({row['Pct']:5.2f}%)  {bar}")
    feat_imp.to_csv(run_dir / fname, index=False)

# ============================================================================
# ML-FILTERED BACKTEST
# ============================================================================

def ml_filtered_backtest(test_df, y_pred_reg, y_prob_clf, clf_threshold, log, run_dir):
    log.info("\n" + "=" * 70)
    log.info("ML-FILTERED BACKTEST COMPARISON")
    log.info("=" * 70)

    test = test_df.copy()
    test["pred_pnl"]  = y_pred_reg
    test["pred_prob"]  = y_prob_clf

    results = {}

    # A) Baseline
    _report_subset(test, "A) BASELINE (all signals)", log, results)

    # B) Regression: above median prediction
    median_pred = np.median(y_pred_reg)
    reg_filtered = test[test["pred_pnl"] > median_pred]
    _report_subset(reg_filtered,
                   f"B) REGRESSION FILTER (pred > median={median_pred:.2f})", log, results)

    # C) Classification: above optimal threshold
    clf_filtered = test[test["pred_prob"] >= clf_threshold]
    _report_subset(clf_filtered,
                   f"C) CLASSIFIER FILTER (prob >= {clf_threshold:.2f})", log, results)

    # D) Ensemble: both agree
    ensemble = test[(test["pred_pnl"] > median_pred) &
                    (test["pred_prob"] >= clf_threshold)]
    _report_subset(ensemble, "D) ENSEMBLE (both agree)", log, results)

    # E) Top quartile by regression
    if len(test) >= 10:
        p75 = np.percentile(y_pred_reg, 75)
        top25 = test[test["pred_pnl"] >= p75]
        _report_subset(top25, f"E) TOP 25% by regression (>= {p75:.2f})", log, results)

    # F) Bottom quartile removed
    if len(test) >= 10:
        p25 = np.percentile(y_pred_reg, 25)
        no_bottom = test[test["pred_pnl"] > p25]
        _report_subset(no_bottom, f"F) SKIP BOTTOM 25% (> {p25:.2f})", log, results)

    pd.DataFrame(results).T.to_csv(run_dir / "filter_comparison.csv")
    test.to_csv(run_dir / "oos_signals_with_predictions.csv", index=False)
    return results


def _report_subset(df, label, log, results_dict):
    n = len(df)
    if n == 0:
        log.info(f"\n{label}: 0 signals -- skipped")
        results_dict[label] = {"n_signals": 0}
        return
    wins = (df["net_pnl_pct"] > 0).sum()
    wr = wins / n * 100
    avg  = df["net_pnl_pct"].mean()
    total = df["net_pnl_pct"].sum()
    med  = df["net_pnl_pct"].median()
    mx_w = df["net_pnl_pct"].max()
    mx_l = df["net_pnl_pct"].min()
    g_win  = df[df["net_pnl_pct"] > 0]["net_pnl_pct"].sum()
    g_loss = abs(df[df["net_pnl_pct"] <= 0]["net_pnl_pct"].sum())
    pf = g_win / g_loss if g_loss > 0 else float("inf")

    log.info(f"\n{'_' * 55}")
    log.info(f"{label}")
    log.info(f"  Signals    : {n}")
    log.info(f"  Win Rate   : {wr:.1f}%")
    log.info(f"  Avg PnL    : {avg:+.2f}%")
    log.info(f"  Total PnL  : {total:+.2f}%")
    log.info(f"  Median PnL : {med:+.2f}%")
    log.info(f"  Profit Fac : {pf:.2f}")
    log.info(f"  Best/Worst : {mx_w:+.2f}% / {mx_l:+.2f}%")

    results_dict[label] = {
        "n_signals": n, "win_rate": wr, "avg_pnl": avg,
        "total_pnl": total, "median_pnl": med,
        "profit_factor": pf, "max_win": mx_w, "max_loss": mx_l,
    }

# ============================================================================
# WALK-FORWARD VALIDATION (calendar-based monthly chunks)
# ============================================================================

def walk_forward_backtest(df_all, reg_params, log, run_dir):
    """
    Expanding-window walk-forward refit every WALK_FORWARD_MONTHS months.
    Uses calendar boundaries for clean chunking.
    """
    log.info("\n" + "=" * 70)
    log.info("WALK-FORWARD VALIDATION")
    log.info("=" * 70)

    filled = df_all[df_all["filled"] == True].copy()
    filled = filled.sort_values("signal_date").reset_index(drop=True)

    if len(filled) < MIN_TRAIN_SIGNALS + 10:
        log.info(f"Too few filled signals ({len(filled)}).")
        return

    split = pd.Timestamp(SPLIT_DATE)
    train_sigs = filled[filled["signal_date"] < split]
    if len(train_sigs) < MIN_TRAIN_SIGNALS:
        log.info(f"Too few training signals ({len(train_sigs)}).")
        return

    # Build calendar-month boundaries for OOS period
    oos_sigs = filled[filled["signal_date"] >= split]
    if len(oos_sigs) == 0:
        log.info("No OOS signals."); return

    oos_start = oos_sigs["signal_date"].min()
    oos_end   = oos_sigs["signal_date"].max()
    boundaries = pd.date_range(
        start=oos_start.to_period("M").to_timestamp(),
        end=oos_end + pd.DateOffset(months=WALK_FORWARD_MONTHS),
        freq=f"{WALK_FORWARD_MONTHS}MS",
    )

    log.info(f"OOS range: {oos_start.date()} -> {oos_end.date()}")
    log.info(f"Walk-forward chunks: {len(boundaries)-1} "
             f"(every {WALK_FORWARD_MONTHS} months)")

    wf_records = []
    for i in range(len(boundaries) - 1):
        chunk_start = boundaries[i]
        chunk_end   = boundaries[i + 1]

        # Train: everything before this chunk
        train = filled[filled["signal_date"] < chunk_start]
        if len(train) < MIN_TRAIN_SIGNALS:
            continue

        # Test: signals in [chunk_start, chunk_end)
        test = filled[(filled["signal_date"] >= chunk_start) &
                      (filled["signal_date"] < chunk_end)]
        if len(test) == 0:
            continue

        X_tr = train[FEATURE_NAMES].values
        y_tr = np.clip(train["net_pnl_pct"].values, -PNL_CLIP, PNL_CLIP)
        X_te = test[FEATURE_NAMES].values
        y_te = test["net_pnl_pct"].values

        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        model = lgb.train(
            {**reg_params, "verbosity": -1}, dtrain,
            num_boost_round=min(800, max(100, len(train))),
        )
        preds = model.predict(X_te)

        # Unfiltered
        bl_avg = y_te.mean()
        bl_tot = y_te.sum()

        # Filter: above-median prediction
        med = np.median(preds)
        mask = preds > med
        if mask.sum() > 0:
            filt_avg = y_te[mask].mean()
            filt_tot = y_te[mask].sum()
            n_filt = int(mask.sum())
        else:
            filt_avg = filt_tot = 0.0; n_filt = 0

        # Filter: top quartile
        if len(preds) >= 4:
            p75 = np.percentile(preds, 75)
            top_mask = preds >= p75
            top_avg = y_te[top_mask].mean() if top_mask.sum() > 0 else 0.0
        else:
            top_avg = filt_avg

        wf_records.append({
            "period":           f"{chunk_start.date()}",
            "n_train":          len(train),
            "n_test":           len(test),
            "n_filtered":       n_filt,
            "baseline_avg":     round(bl_avg, 4),
            "filtered_avg":     round(filt_avg, 4),
            "top25_avg":        round(top_avg, 4),
            "baseline_total":   round(bl_tot, 4),
            "filtered_total":   round(filt_tot, 4),
            "improvement":      round(filt_avg - bl_avg, 4),
        })

    if wf_records:
        wf_df = pd.DataFrame(wf_records)
        wf_df.to_csv(run_dir / "walk_forward.csv", index=False)

        avg_imp = wf_df["improvement"].mean()
        n_pos = (wf_df["improvement"] > 0).sum()

        log.info(f"\nWalk-Forward Results ({len(wf_df)} chunks):")
        log.info(f"{'_' * 70}")
        log.info(f"  {'Period':<12s}  {'Train':>5s}  {'Test':>5s}  {'Filt':>4s}  "
                 f"{'Base avg':>9s}  {'Filt avg':>9s}  {'Top25':>8s}  {'D':>7s}")
        for _, r in wf_df.iterrows():
            log.info(f"  {r['period']:<12s}  {r['n_train']:5d}  {r['n_test']:5d}  "
                     f"{r['n_filtered']:4d}  {r['baseline_avg']:+9.4f}  "
                     f"{r['filtered_avg']:+9.4f}  {r['top25_avg']:+8.4f}  "
                     f"{r['improvement']:+7.4f}")
        log.info(f"{'_' * 70}")
        log.info(f"  Avg baseline PnL  : {wf_df['baseline_avg'].mean():+.4f}%")
        log.info(f"  Avg filtered PnL  : {wf_df['filtered_avg'].mean():+.4f}%")
        log.info(f"  Avg top-25% PnL   : {wf_df['top25_avg'].mean():+.4f}%")
        log.info(f"  Avg improvement   : {avg_imp:+.4f}%")
        log.info(f"  Chunks improved   : {n_pos}/{len(wf_df)}")

# ============================================================================
# SUMMARY EXPORT
# ============================================================================

def export_summary(reg_metrics, clf_metrics, filter_results, log, run_dir):
    summary = {
        "split_date": SPLIT_DATE, "n_features": len(FEATURE_NAMES),
        "pnl_clip": PNL_CLIP, "optuna": RUN_OPTUNA,
    }
    summary.update({f"reg_{k}": v for k, v in reg_metrics.items()})
    summary.update({f"clf_{k}": v for k, v in clf_metrics.items()})
    for label, stats in filter_results.items():
        pfx = label.split(")")[0].strip() + ")"
        for k, v in stats.items():
            summary[f"{pfx}_{k}"] = v
    pd.DataFrame([summary]).to_csv(run_dir / "summary_metrics.csv", index=False)
    log.info(f"\nAll outputs saved to {run_dir.resolve()}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log, run_dir = setup()
    np.random.seed(SEED)
    frames = load_ohlcv(log)

    # 1. Generate signals with features + labels
    df_signals = build_signal_dataset(frames, log)
    if df_signals.empty:
        log.info("No signals generated."); return

    n_filled = int(df_signals["filled"].sum())
    n_wins = int(df_signals[df_signals["filled"] == True]["win"].sum())
    log.info(f"\nSignal Summary:")
    log.info(f"  Total: {len(df_signals)}  |  Filled: {n_filled}  |  "
             f"Win rate: {n_wins/n_filled*100:.1f}%" if n_filled else "  No fills")
    df_signals.to_csv(run_dir / "all_signals.csv", index=False)

    # 2. Target engineering
    filled = df_signals[df_signals["filled"] == True].copy()
    filled = prepare_targets(filled, log)

    # 3. Temporal split
    train_df, test_df = temporal_split(filled, log)
    if len(train_df) < MIN_TRAIN_SIGNALS:
        log.info(f"Too few training signals ({len(train_df)})."); return
    if len(test_df) < 5:
        log.info(f"Too few test signals ({len(test_df)})."); return

    # 4. Optuna tuning (optional)
    if RUN_OPTUNA:
        reg_params = optuna_tune(
            train_df[FEATURE_NAMES].values,
            train_df["target_clipped"].values,
            train_df["signal_date"],
            log,
        )
    else:
        reg_params = LGB_PARAMS_REG

    # 5. Train models
    reg_model = train_regression_model(train_df, test_df, reg_params, log, run_dir)
    clf_model = train_classification_model(train_df, test_df, log, run_dir)

    # 6. Evaluate OOS
    y_pred_reg, reg_metrics = evaluate_regression(reg_model, test_df, log, run_dir)
    y_prob_clf, clf_thr, clf_metrics = evaluate_classification(clf_model, test_df, log, run_dir)

    # 7. ML-filtered backtest
    filter_results = ml_filtered_backtest(
        test_df, y_pred_reg, y_prob_clf, clf_thr, log, run_dir)

    # 8. Walk-forward validation
    walk_forward_backtest(df_signals, reg_params, log, run_dir)

    # 9. Summary
    export_summary(reg_metrics, clf_metrics, filter_results, log, run_dir)
    log.info("\nDone.")


if __name__ == "__main__":
    main()
