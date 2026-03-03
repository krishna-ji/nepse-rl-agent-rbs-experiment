#!/usr/bin/env python3
"""
NEPSE Cross-Sectional Ranking Engine — LightGBM
================================================
Supervised regression model predicting 5-day forward log returns
across all NEPSE assets, with Top-K execution backtest.

Architecture:
  - Same 11-dimensional feature tensor (CLV, NATR Z-Score, Ribbon, CMF…)
  - Target: 5-day forward log return  →  np.log(Close[t+5] / Close[t])
  - LightGBM regression with early stopping
  - OOS: MSE, R², Information Coefficient (Spearman rank corr)
  - Top-K=10 backtest vs. universe benchmark
"""

import warnings; warnings.filterwarnings("ignore")
import logging, pathlib, datetime, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_ROWS       = 250          # minimum rows to include a ticker
WARMUP         = 200          # feature warmup rows
SPLIT_DATE     = "2024-07-01" # temporal train/test boundary
FORWARD_DAYS   = 5            # prediction horizon (swing-trade)
TOP_K          = 10           # top-K assets per rebalance day
SEED           = 42
DATA_DIR       = pathlib.Path("data/ohlcv/1D/stocks")

OBS_FEATURES = [
    "clv", "lower_wick", "pct_k", "pct_d", "natr", "bb_pctb",
    "cmf", "d_low", "dd_state", "ribbon_align", "ribbon_disp",
]

LGB_PARAMS = {
    "objective":     "regression",
    "metric":        "rmse",
    "learning_rate": 0.05,
    "max_depth":     5,
    "num_leaves":    31,
    "min_child_samples": 50,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "verbosity":     -1,
    "seed":          SEED,
    "n_jobs":        -1,
}

# ============================================================================
# SETUP
# ============================================================================

def setup():
    RUN_TS  = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    RUN_DIR = pathlib.Path(f"runs/{RUN_TS}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("nepserl_lgbm")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(RUN_DIR / "nepserl_lgbm.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); log.addHandler(sh)

    log.info("=" * 70)
    log.info("NEPSE Cross-Sectional Ranking Engine — LightGBM")
    log.info("=" * 70)
    log.info(f"Run directory : {RUN_DIR.resolve()}")
    log.info(f"Data dir      : {DATA_DIR.resolve()}")
    log.info(f"Split date    : {SPLIT_DATE}")
    log.info(f"Forward days  : {FORWARD_DAYS}")
    log.info(f"Top-K         : {TOP_K}")
    return log, RUN_DIR

# ============================================================================
# DATA LOADING
# ============================================================================

def load_ohlcv(log):
    """Load OHLCV CSVs → dict of per-ticker DataFrames."""
    log.info("Loading OHLCV data...")
    frames, skipped = {}, 0
    for csv in sorted(DATA_DIR.glob("*.csv")):
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
            frames[csv.stem] = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            log.warning(f"Skip {csv.stem}: {e}"); skipped += 1
    log.info(f"Loaded {len(frames)} tickers, skipped {skipped}")
    return frames

# ============================================================================
# FEATURE ENGINEERING  (11-dimensional orthogonal state tensor)
# ============================================================================

def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

def _atr(h, l, c, n=14):
    return _true_range(h, l, c).ewm(span=n, min_periods=n, adjust=False).mean()

def _stochastic(h, l, c, kp=14, dp=3):
    lo = l.rolling(kp, min_periods=kp).min()
    hi = h.rolling(kp, min_periods=kp).max()
    raw_k = 100.0 * (c - lo) / (hi - lo + 1e-10)
    pk = raw_k.rolling(dp, min_periods=dp).mean()
    pd_ = pk.rolling(dp, min_periods=dp).mean()
    return pk, pd_

def _bb_pctb(c, n=20, ns=2.0):
    s_ = _sma(c, n)
    std = c.rolling(n, min_periods=n).std()
    upper = s_ + ns * std
    lower = s_ - ns * std
    return ((c - lower) / (upper - lower + 1e-10) * 2.0 - 1.0).clip(-3.0, 3.0)

def _psl(l, w=60):
    return l.rolling(w, min_periods=w).min()

def compute_features_for_ticker(o, h, l, c, v):
    """Compute 11 features for a single ticker. Returns a DataFrame."""
    atr14 = _atr(h, l, c, 14)
    psl_  = _psl(l, 60)
    sma10, sma20, sma50 = _sma(c, 10), _sma(c, 20), _sma(c, 50)
    sma100, sma200      = _sma(c, 100), _sma(c, 200)

    feats = pd.DataFrame(index=c.index)

    # 1. CLV ∈ [-1, 1]
    feats["clv"] = ((c - l) - (h - c)) / (h - l + 1e-10)

    # 2. Lower wick (ATR-normalized) ∈ [0, 3]
    feats["lower_wick"] = ((c - l) / (atr14 + 1e-10)).clip(0.0, 3.0)

    # 3. Stochastic %K (zero-centered) ∈ [-1, 1]
    pk, pd_ = _stochastic(h, l, c)
    feats["pct_k"] = (pk / 50.0) - 1.0

    # 4. Stochastic %D (zero-centered) ∈ [-1, 1]
    feats["pct_d"] = (pd_ / 50.0) - 1.0

    # 5. NATR → Z-scored over 100d rolling window ∈ [-3, 3]
    natr_raw = atr14 / (c + 1e-10)
    rm = natr_raw.rolling(100, min_periods=100).mean()
    rs = natr_raw.rolling(100, min_periods=100).std()
    feats["natr"] = ((natr_raw - rm) / (rs + 1e-8)).clip(-3.0, 3.0)

    # 6. Bollinger %B ∈ [-3, 3]
    feats["bb_pctb"] = _bb_pctb(c, 20, 2.0)

    # 7. CMF-20 ∈ [-1, 1]
    mf_mult = ((c - l) - (h - c)) / (h - l + 1e-10)
    mf_vol  = mf_mult * v
    cmf_20  = (mf_vol.rolling(20, min_periods=20).sum()
               / (v.rolling(20, min_periods=20).sum() + 1e-10))
    feats["cmf"] = cmf_20.clip(-1.0, 1.0)

    # 8. D_low — distance to protected swing low (ATR-normalized) ∈ [-3, 3]
    feats["d_low"] = ((c - psl_) / (atr14 + 1e-10)).clip(-3.0, 3.0)

    # 9. DD_state — drawdown from 20d high (ATR-normalized) ∈ [0, 5]
    rolling_high_20 = h.rolling(20, min_periods=20).max()
    feats["dd_state"] = ((rolling_high_20 - c) / (atr14 + 1e-10)).clip(0.0, 5.0)

    # 10. Ribbon Alignment ∈ [-1, 1]
    bull_count = (
        (sma10 > sma20).astype(float) + (sma20 > sma50).astype(float)
        + (sma50 > sma100).astype(float) + (sma100 > sma200).astype(float)
    )
    feats["ribbon_align"] = bull_count / 4.0 * 2.0 - 1.0

    # 11. Ribbon Dispersion ∈ [-3, 3]
    feats["ribbon_disp"] = ((sma10 - sma200) / (sma200 + 1e-10) * 10.0).clip(-3.0, 3.0)

    return feats

# ============================================================================
# DATASET ASSEMBLY  (MultiIndex: Date × Ticker)
# ============================================================================

def build_dataset(frames, log):
    """
    Build a flat (Date, Ticker)-indexed DataFrame with 11 features
    plus the 5-day forward log return target. Drop NaN rows.
    """
    log.info("Building cross-sectional dataset...")
    all_pieces = []
    for tk, raw in frames.items():
        o, h, l, c, v = raw["Open"], raw["High"], raw["Low"], raw["Close"], raw["Volume"]
        feats = compute_features_for_ticker(o, h, l, c, v)
        # Target: 5-day forward log return
        feats["target_return"] = np.log(c.shift(-FORWARD_DAYS) / c)
        feats["Ticker"] = tk
        all_pieces.append(feats)

    df = pd.concat(all_pieces, axis=0)
    df.index.name = "Date"
    df = df.reset_index().set_index(["Date", "Ticker"]).sort_index()

    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    log.info(f"Dataset: {n_after:,} rows  ({n_before - n_after:,} NaN rows dropped)")
    log.info(f"Tickers: {df.index.get_level_values('Ticker').nunique()}")
    log.info(f"Date range: {df.index.get_level_values('Date').min().date()} → "
             f"{df.index.get_level_values('Date').max().date()}")
    return df

# ============================================================================
# TEMPORAL SPLIT
# ============================================================================

def temporal_split(df, log):
    split = pd.Timestamp(SPLIT_DATE)
    dates = df.index.get_level_values("Date")
    train_mask = dates < split
    test_mask  = dates >= split

    X_train = df.loc[train_mask, OBS_FEATURES]
    y_train = df.loc[train_mask, "target_return"]
    X_test  = df.loc[test_mask,  OBS_FEATURES]
    y_test  = df.loc[test_mask,  "target_return"]

    log.info(f"Train: {len(X_train):,} rows  "
             f"({dates[train_mask].min().date()} → {dates[train_mask].max().date()})")
    log.info(f"Test:  {len(X_test):,} rows  "
             f"({dates[test_mask].min().date()} → {dates[test_mask].max().date()})")
    return X_train, y_train, X_test, y_test

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, X_test, y_test, log):
    log.info("Training LightGBM regression model...")
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dtest  = lgb.Dataset(X_test,  label=y_test,  reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=50, verbose=True),
    ]

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dtest],
        valid_names=["train", "test"],
        callbacks=callbacks,
    )
    log.info(f"Best iteration: {model.best_iteration}")
    return model

# ============================================================================
# OOS EVALUATION
# ============================================================================

def evaluate_oos(model, X_test, y_test, log, run_dir):
    """Compute MSE, R², and Information Coefficient (Spearman rank corr)."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    ic, ic_p = spearmanr(y_pred, y_test)

    log.info("=" * 50)
    log.info("OUT-OF-SAMPLE METRICS")
    log.info("=" * 50)
    log.info(f"  MSE                     : {mse:.8f}")
    log.info(f"  R²                      : {r2:.6f}")
    log.info(f"  Information Coefficient  : {ic:.6f}  (p={ic_p:.2e})")
    log.info("=" * 50)

    # ── Feature Importance (gain-based) ──
    importance = model.feature_importance(importance_type="gain")
    feat_imp = pd.DataFrame({
        "Feature": OBS_FEATURES,
        "Gain": importance,
    }).sort_values("Gain", ascending=False).reset_index(drop=True)
    feat_imp["Pct"] = (feat_imp["Gain"] / feat_imp["Gain"].sum() * 100).round(2)

    log.info("")
    log.info("FEATURE IMPORTANCE (Gain)")
    log.info("-" * 40)
    for _, row in feat_imp.iterrows():
        bar = "█" * int(row["Pct"] / 2)
        log.info(f"  {row['Feature']:18s} {row['Gain']:12.1f}  ({row['Pct']:5.2f}%)  {bar}")

    feat_imp.to_csv(run_dir / "feature_importance.csv", index=False)
    return y_pred, {"mse": mse, "r2": r2, "ic": ic, "ic_p": ic_p}

# ============================================================================
# TOP-K CROSS-SECTIONAL BACKTEST
# ============================================================================

def topk_backtest(X_test, y_test, y_pred, log, run_dir):
    """
    For each OOS date:
      - Pick Top-K tickers by predicted return
      - Strategy return  = mean actual 5d return of Top-K
      - Benchmark return = mean actual 5d return of all assets
    Accumulate across dates.
    """
    df_bt = X_test.copy()
    df_bt["actual"]    = y_test.values
    df_bt["predicted"] = y_pred

    dates = df_bt.index.get_level_values("Date").unique().sort_values()
    log.info(f"\nTop-K Backtest: {len(dates)} unique OOS dates, K={TOP_K}")

    records = []
    cum_strategy  = 0.0
    cum_benchmark = 0.0

    for dt in dates:
        day = df_bt.loc[dt]
        if len(day) < TOP_K:
            continue  # skip days with fewer assets than K

        # Pick top-K by predicted return
        topk = day.nlargest(TOP_K, "predicted")
        strat_ret  = topk["actual"].mean()
        bench_ret  = day["actual"].mean()
        cum_strategy  += strat_ret
        cum_benchmark += bench_ret

        records.append({
            "Date":            dt,
            "n_assets":        len(day),
            "strategy_return": strat_ret,
            "benchmark_return": bench_ret,
            "cum_strategy":    cum_strategy,
            "cum_benchmark":   cum_benchmark,
            "cum_alpha":       cum_strategy - cum_benchmark,
        })

    results = pd.DataFrame(records)
    results.to_csv(run_dir / "topk_backtest.csv", index=False)

    log.info("")
    log.info("=" * 60)
    log.info("TOP-K BACKTEST RESULTS")
    log.info("=" * 60)
    log.info(f"  OOS days evaluated       : {len(results)}")
    log.info(f"  Cumulative Strategy      : {cum_strategy:+.6f}  "
             f"({np.expm1(cum_strategy)*100:+.2f}%)")
    log.info(f"  Cumulative Benchmark     : {cum_benchmark:+.6f}  "
             f"({np.expm1(cum_benchmark)*100:+.2f}%)")
    alpha = cum_strategy - cum_benchmark
    log.info(f"  Cumulative Alpha         : {alpha:+.6f}  "
             f"({np.expm1(alpha)*100:+.2f}%)")
    log.info("=" * 60)

    # ── Daily IC (per-date rank correlation) ──
    daily_ics = []
    for dt in dates:
        day = df_bt.loc[dt]
        if len(day) >= 10:
            corr, _ = spearmanr(day["predicted"], day["actual"])
            if not np.isnan(corr):
                daily_ics.append(corr)
    if daily_ics:
        mean_ic = np.mean(daily_ics)
        ic_ir   = mean_ic / (np.std(daily_ics) + 1e-10)
        log.info(f"  Mean Daily IC            : {mean_ic:.6f}")
        log.info(f"  ICIR (IC / std(IC))      : {ic_ir:.4f}")
        log.info(f"  IC Hit Rate (>0)         : {np.mean(np.array(daily_ics) > 0)*100:.1f}%")
    log.info("=" * 60)

    return results

# ============================================================================
# SUMMARY EXPORT
# ============================================================================

def export_summary(metrics, results, run_dir, log):
    """Save summary_metrics.csv for easy comparison across runs."""
    if not results.empty:
        summary = {
            "mse":              metrics["mse"],
            "r2":               metrics["r2"],
            "ic":               metrics["ic"],
            "ic_p":             metrics["ic_p"],
            "cum_strategy":     results["cum_strategy"].iloc[-1],
            "cum_benchmark":    results["cum_benchmark"].iloc[-1],
            "cum_alpha":        results["cum_alpha"].iloc[-1],
            "oos_days":         len(results),
            "split_date":       SPLIT_DATE,
            "forward_days":     FORWARD_DAYS,
            "top_k":            TOP_K,
        }
    else:
        summary = metrics
    pd.DataFrame([summary]).to_csv(run_dir / "summary_metrics.csv", index=False)
    log.info(f"\nAll outputs saved to {run_dir.resolve()}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log, run_dir = setup()

    # 1. Load OHLCV
    frames = load_ohlcv(log)

    # 2. Feature engineering + target creation → flat MultiIndex DataFrame
    df = build_dataset(frames, log)

    # 3. Temporal split
    X_train, y_train, X_test, y_test = temporal_split(df, log)

    # 4. Train LightGBM
    model = train_model(X_train, y_train, X_test, y_test, log)
    model.save_model(str(run_dir / "model.txt"))
    log.info(f"Model saved → {run_dir / 'model.txt'}")

    # 5. OOS evaluation
    y_pred, metrics = evaluate_oos(model, X_test, y_test, log, run_dir)

    # 6. Top-K backtest
    results = topk_backtest(X_test, y_test, y_pred, log, run_dir)

    # 7. Summary
    export_summary(metrics, results, run_dir, log)
    log.info("Done.")

if __name__ == "__main__":
    main()
