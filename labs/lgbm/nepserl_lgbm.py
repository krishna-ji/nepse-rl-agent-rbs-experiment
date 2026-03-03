#!/usr/bin/env python3
"""
NEPSE Cross-Sectional Ranking Engine — LightGBM
================================================
Supervised regression model predicting 5-day forward log returns
across all NEPSE assets, with Top-K execution backtest.

Architecture:
  - 10-dimensional swing-trading feature tensor:
      Trend & Retracement: close_vs_sma20, close_vs_sma50,
                           retracement_depth, uptrend_pullback
      Momentum & Oversold: rsi_14, rsi_slope_3
      Breakout & Volume:   bb_width_ratio, volume_surge
      Micro-Structure:     clv, ribbon_disp
  - Target: 5-day forward log return  →  np.log(Close[t+5] / Close[t])
  - LightGBM regression with early stopping
  - OOS: MSE, R², Information Coefficient (Spearman rank corr)
  - Top-K=5 non-overlapping 5-day holding backtest vs. universe benchmark
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

PROJECT_ROOT   = pathlib.Path(__file__).resolve().parents[2]
MIN_ROWS       = 250          # minimum rows to include a ticker
WARMUP         = 200          # feature warmup rows
SPLIT_DATE     = "2024-07-01" # temporal train/test boundary
FORWARD_DAYS   = 5            # prediction horizon (swing-trade)
TOP_K          = 5            # top-K assets per rebalance (high conviction)
HOLDING_DAYS   = 5            # hold for 5 trading days before rebalancing
TRANSACTION_FEE= 0.005        # ~0.5% round-trip cost (brokerage + SEBON fees)
SEED           = 42
DATA_DIR       = PROJECT_ROOT / "data/ohlcv/1D/stocks"

OBS_FEATURES = [
    # ── Trend & Retracement ──
    "close_vs_sma20", "close_vs_sma50", "retracement_depth", "uptrend_pullback",
    # ── Momentum & Oversold ──
    "rsi_14", "rsi_slope_3",
    # ── Breakout & Volume ──
    "bb_width_ratio", "volume_surge",
    # ── Micro-Structure ──
    "clv", "ribbon_disp",
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
    RUN_DIR = PROJECT_ROOT / f"runs/{RUN_TS}"
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
# FEATURE ENGINEERING  (10-dimensional swing-trading tensor)
# ============================================================================

def _sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def _rsi(c, n=14):
    """Wilder RSI with EWM smoothing."""
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.ewm(com=n - 1, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(com=n - 1, min_periods=n, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)

def compute_features_for_ticker(o, h, l, c, v):
    """Compute 10 swing-trading features for a single ticker."""
    sma20  = _sma(c, 20)
    sma50  = _sma(c, 50)
    sma200 = _sma(c, 200)
    sma10  = _sma(c, 10)
    sma_v20 = _sma(v, 20)

    feats = pd.DataFrame(index=c.index)

    # ── Trend & Retracement ──────────────────────────────────────

    # 1. close_vs_sma20: distance from short-term trend
    feats["close_vs_sma20"] = (c - sma20) / (sma20 + 1e-10)

    # 2. close_vs_sma50: distance from medium trend
    feats["close_vs_sma50"] = (c - sma50) / (sma50 + 1e-10)

    # 3. retracement_depth: % drawdown from 20d high
    high_20 = h.rolling(20, min_periods=20).max()
    feats["retracement_depth"] = (high_20 - c) / (high_20 + 1e-10)

    # 4. uptrend_pullback: binary — macro uptrend AND short-term pullback
    feats["uptrend_pullback"] = (
        (sma50 > sma200) & (c < sma20)
    ).astype(np.float32)

    # ── Momentum & Oversold ──────────────────────────────────────

    # 5. rsi_14: standard 14-day RSI
    rsi = _rsi(c, 14)
    feats["rsi_14"] = rsi

    # 6. rsi_slope_3: RSI today minus RSI 3 days ago
    feats["rsi_slope_3"] = rsi - rsi.shift(3)

    # ── Breakout & Volume ────────────────────────────────────────

    # 7. bb_width_ratio: current BB width / SMA(BB width, 20)
    #    BB width = (SMA20 + 2*std) - (SMA20 - 2*std) = 4*std
    std20 = c.rolling(20, min_periods=20).std()
    bb_width = 4.0 * std20
    bb_width_ma = bb_width.rolling(20, min_periods=20).mean()
    feats["bb_width_ratio"] = bb_width / (bb_width_ma + 1e-10)

    # 8. volume_surge: today's volume / SMA(volume, 20)
    feats["volume_surge"] = v / (sma_v20 + 1e-10)

    # ── Micro-Structure ──────────────────────────────────────────

    # 9. clv: Close Location Value ∈ [-1, 1]
    feats["clv"] = ((c - l) - (h - c)) / (h - l + 1e-10)

    # 10. ribbon_disp: (SMA10 - SMA200) / SMA200 × 10
    feats["ribbon_disp"] = (sma10 - sma200) / (sma200 + 1e-10) * 10.0

    return feats

# ============================================================================
# DATASET ASSEMBLY  (MultiIndex: Date × Ticker)
# ============================================================================

def build_dataset(frames, log):
    """
    Build a flat (Date, Ticker)-indexed DataFrame with 11 features
    plus the 5-day forward log return target. Drop NaN rows.
    """
    log.info(f"Building cross-sectional dataset ({len(OBS_FEATURES)} features)...")
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
# TOP-K CROSS-SECTIONAL BACKTEST  (5-day holding period)
# ============================================================================

def topk_backtest(X_test, y_test, y_pred, log, run_dir):
    """
    Non-overlapping 5-day holding backtest:
      - Jump pointer by exactly HOLDING_DAYS (i += 5) each step.
      - On each rebalance date: rank tickers, pick Top-K=5.
      - Strategy return  = mean actual target_return of Top-K.
      - Benchmark return = mean actual target_return of all assets.
      - Days between rebalances are fully skipped (capital is locked).
    """
    df_bt = X_test.copy()
    df_bt["actual"]    = y_test.values
    df_bt["predicted"] = y_pred

    dates = df_bt.index.get_level_values("Date").unique().sort_values()
    n_dates = len(dates)
    log.info(f"\nTop-K Backtest: {n_dates} unique OOS dates, K={TOP_K}, "
             f"holding={HOLDING_DAYS}d, stride=+{HOLDING_DAYS}")

    records = []
    cum_strategy  = 0.0
    cum_benchmark = 0.0

    i = 0
    while i < n_dates:
        dt = dates[i]
        day = df_bt.loc[dt]

        if len(day) < TOP_K:
            i += HOLDING_DAYS          # skip forward even if sparse
            continue

        # ── Rebalance: rank and pick Top-K ──
        topk = day.nlargest(TOP_K, "predicted")
        strat_ret  = topk["actual"].mean() - TRANSACTION_FEE
        bench_ret  = day["actual"].mean()
        cum_strategy  += strat_ret
        cum_benchmark += bench_ret

        picks = topk.index.get_level_values("Ticker").tolist() \
                if "Ticker" in topk.index.names else []

        records.append({
            "Date":             dt,
            "n_assets":         len(day),
            "top_k_tickers":    ",".join(picks[:TOP_K]),
            "strategy_return":  strat_ret,
            "benchmark_return": bench_ret,
            "cum_strategy":     cum_strategy,
            "cum_benchmark":    cum_benchmark,
            "cum_alpha":        cum_strategy - cum_benchmark,
        })

        i += HOLDING_DAYS              # jump forward by exactly 5 days

    results = pd.DataFrame(records)
    results.to_csv(run_dir / "topk_backtest.csv", index=False)

    n_rebalances = len(results)
    log.info("")
    log.info("=" * 60)
    log.info("TOP-K BACKTEST RESULTS  (5-day non-overlapping holding)")
    log.info("=" * 60)
    log.info(f"  Rebalance events         : {n_rebalances}")
    log.info(f"  Cumulative Strategy      : {cum_strategy:+.6f}  "
             f"({np.expm1(cum_strategy)*100:+.2f}%)")
    log.info(f"  Cumulative Benchmark     : {cum_benchmark:+.6f}  "
             f"({np.expm1(cum_benchmark)*100:+.2f}%)")
    alpha = cum_strategy - cum_benchmark
    log.info(f"  Cumulative Alpha         : {alpha:+.6f}  "
             f"({np.expm1(alpha)*100:+.2f}%)")
    log.info("=" * 60)

    # ── Per-rebalance IC (rank correlation on rebalance days only) ──
    rebal_ics = []
    for _, row in results.iterrows():
        dt = row["Date"]
        day = df_bt.loc[dt]
        if len(day) >= 10:
            corr, _ = spearmanr(day["predicted"], day["actual"])
            if not np.isnan(corr):
                rebal_ics.append(corr)
    if rebal_ics:
        mean_ic = np.mean(rebal_ics)
        ic_ir   = mean_ic / (np.std(rebal_ics) + 1e-10)
        log.info(f"  Mean Rebalance IC        : {mean_ic:.6f}")
        log.info(f"  ICIR (IC / std(IC))      : {ic_ir:.4f}")
        log.info(f"  IC Hit Rate (>0)         : {np.mean(np.array(rebal_ics) > 0)*100:.1f}%")
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
