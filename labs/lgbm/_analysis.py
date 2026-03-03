#!/usr/bin/env python3
"""Quick deep analysis of the Ichimoku + LightGBM run data."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

filled = pd.read_csv("runs/20260303_233338/all_signals.csv")
filled = filled[filled["filled"]==True].copy()
filled["signal_date"] = pd.to_datetime(filled["signal_date"])
filled["year"] = filled["signal_date"].dt.year

oos = pd.read_csv("runs/20260303_233338/oos_signals_with_predictions.csv")

print("="*60)
print("DEEP ANALYSIS: Ichimoku Kumo Break + LightGBM")
print("="*60)

# 1. Overall
n = len(filled)
wins = (filled["net_pnl_pct"]>0).sum()
avg_win = filled[filled["net_pnl_pct"]>0]["net_pnl_pct"].mean()
avg_loss = filled[filled["net_pnl_pct"]<=0]["net_pnl_pct"].mean()
total_pnl = filled["net_pnl_pct"].sum()
wl_ratio = abs(avg_win/avg_loss)
print(f"\n1. OVERALL: {n} trades, {wins}/{n-wins} W/L")
print(f"   WR={wins/n*100:.1f}%, Avg Win={avg_win:+.1f}%, Avg Loss={avg_loss:+.1f}%")
print(f"   W/L ratio={wl_ratio:.2f}, Total PnL={total_pnl:+.0f}%")

# 2. Top/Bottom tickers
tk = filled.groupby("ticker").agg(
    n=("net_pnl_pct","count"),
    total=("net_pnl_pct","sum"),
    avg=("net_pnl_pct","mean"),
    wr=("win","mean"),
)
tk["wr"] *= 100
print("\n2. TOP 10 TICKERS (total PnL):")
for t, r in tk.sort_values("total", ascending=False).head(10).iterrows():
    print(f"   {t:10s} {int(r['n']):3d}t  total={r['total']:+8.1f}%  avg={r['avg']:+.1f}%  WR={r['wr']:.0f}%")
print("\n   BOTTOM 10:")
for t, r in tk.sort_values("total", ascending=True).head(10).iterrows():
    print(f"   {t:10s} {int(r['n']):3d}t  total={r['total']:+8.1f}%  avg={r['avg']:+.1f}%  WR={r['wr']:.0f}%")

# 3. ML analysis
print("\n3. ML MODEL - OOS ANALYSIS:")
pred = oos["pred_pnl"].values
actual = oos["net_pnl_pct"].values
ic, p = spearmanr(pred, actual)
print(f"   IC={ic:.4f} (p={p:.4f})")

# Quintile analysis
oos2 = oos.copy()
oos2["q"] = pd.qcut(oos2["pred_pnl"], 5, labels=["Q1(worst)","Q2","Q3","Q4","Q5(best)"])
print("\n   Quintile Analysis (by predicted quality):")
for q in ["Q1(worst)","Q2","Q3","Q4","Q5(best)"]:
    qd = oos2[oos2["q"]==q]
    gw = qd[qd["net_pnl_pct"]>0]["net_pnl_pct"].sum()
    gl = abs(qd[qd["net_pnl_pct"]<=0]["net_pnl_pct"].sum())
    pf = gw/gl if gl>0 else 0
    print(f"   {q:10s}: {len(qd):3d}t  avg={qd['net_pnl_pct'].mean():+6.2f}%  "
          f"WR={qd['win'].mean()*100:.1f}%  PF={pf:.2f}")

# 4. Regime
print("\n4. REGIME PERFORMANCE:")
for y1, y2, label in [(2012,2016,"Bull 2012-16"),(2017,2018,"Sideways 17-18"),
                       (2019,2021,"Bull 2019-21"),(2022,2023,"Correction 22-23"),
                       (2024,2026,"Recent 24-26")]:
    d = filled[filled["year"].between(y1, y2)]
    if len(d) > 0:
        gw = d[d["net_pnl_pct"]>0]["net_pnl_pct"].sum()
        gl = abs(d[d["net_pnl_pct"]<=0]["net_pnl_pct"].sum())
        pf = gw/gl if gl>0 else 0
        print(f"   {label:20s}: {len(d):4d}t  avg={d['net_pnl_pct'].mean():+.2f}%  "
              f"WR={d['win'].mean()*100:.1f}%  PF={pf:.2f}  total={d['net_pnl_pct'].sum():+.0f}%")

# 5. Exit reason
print("\n5. EXIT REASON BREAKDOWN:")
for rsn, grp in filled.groupby("exit_reason"):
    gw = grp[grp["net_pnl_pct"]>0]["net_pnl_pct"].sum()
    gl = abs(grp[grp["net_pnl_pct"]<=0]["net_pnl_pct"].sum())
    pf = gw/gl if gl>0 else 0
    print(f"   {rsn:15s}: {len(grp):4d}t  avg={grp['net_pnl_pct'].mean():+6.2f}%  "
          f"WR={grp['win'].mean()*100:.1f}%  PF={pf:.2f}  bars={grp['bars_held'].mean():.0f}")

# 6. Seasonality
print("\n6. MONTHLY SEASONALITY:")
filled["month"] = filled["signal_date"].dt.month
for m, grp in filled.groupby("month"):
    print(f"   Month {m:2d}: {len(grp):4d}t  avg={grp['net_pnl_pct'].mean():+6.2f}%  "
          f"WR={grp['win'].mean()*100:.1f}%")
