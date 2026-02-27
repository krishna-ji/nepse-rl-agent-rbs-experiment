"""
Macro-Portfolio Aggregation & Gradient Telemetry Extraction
===========================================================
Parses all trade ledgers from the latest production run, computes
system-wide profitability, extracts PPO training losses, and
outputs the action distribution tensor.
"""

import csv
import json
import math
from pathlib import Path

run = Path("outputs/production_20260226_233129")
eval_dir = run / "eval"
plots_dir = run / "plots"

# ═══════════════════════════════════════════════════════════════
# 1. AGGREGATE SYSTEM TEAR SHEET
# ═══════════════════════════════════════════════════════════════

ledgers = sorted(eval_dir.glob("*_trade_ledger.csv"))
all_trades = []
total_action_0 = 0
total_action_1 = 0

for lf in ledgers:
    with open(lf, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        a = int(r["action"]) if r["action"] else 0
        if a == 0:
            total_action_0 += 1
        else:
            total_action_1 += 1
    trades: dict = {}
    for r in rows:
        tid = int(r["trade_id"]) if r["trade_id"] else 0
        if tid == 0:
            continue
        if tid not in trades:
            trades[tid] = {"ticker": r["ticker"], "entries": [], "exits": [], "forced": False}
        t = r["transition"]
        fl = r["forced_liquidation"] in ("True", "true", "1")
        if "BUY" in t or "0->1" in t:
            trades[tid]["entries"].append(r)
        elif "SELL" in t or "1->0" in t or "FORCED" in t:
            trades[tid]["exits"].append(r)
            if fl:
                trades[tid]["forced"] = True
    all_trades.extend(trades.values())

total_buys = len(all_trades)
forced_liquidations = sum(1 for t in all_trades if t["forced"])

wins = 0
losses = 0
win_returns: list[float] = []
loss_returns: list[float] = []

for trade in all_trades:
    if not trade["entries"] or not trade["exits"]:
        continue
    entry_price = float(trade["entries"][0]["close"]) if trade["entries"][0]["close"] else None
    exit_price = float(trade["exits"][-1]["close"]) if trade["exits"][-1]["close"] else None
    if entry_price and exit_price and entry_price > 0 and exit_price > 0:
        log_ret = math.log(exit_price / entry_price)
        if log_ret > 0:
            wins += 1
            win_returns.append(log_ret)
        else:
            losses += 1
            loss_returns.append(abs(log_ret))

total_closed = wins + losses
win_rate = wins / total_closed if total_closed > 0 else 0
all_log_rets = win_returns + [-x for x in loss_returns]
avg_return = sum(all_log_rets) / len(all_log_rets) if all_log_rets else 0
avg_win = sum(win_returns) / len(win_returns) if win_returns else 0
avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0
loss_rate = 1.0 - win_rate
expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

tear = {
    "Total_System_Trades": total_buys,
    "Total_Closed_Trades": total_closed,
    "Total_Forced_Liquidations": forced_liquidations,
    "Macro_Win_Rate": round(win_rate, 4),
    "Average_Return_Per_Trade_Pct": round(avg_return * 100, 4),
    "Average_Win_Log": round(avg_win, 6),
    "Average_Loss_Log": round(avg_loss, 6),
    "System_Expectancy": round(expectancy, 6),
    "Action_0_Count": total_action_0,
    "Action_1_Count": total_action_1,
    "Action_Ratio_0_to_1": round(total_action_0 / total_action_1, 4) if total_action_1 > 0 else None,
    "Execution_Sparsity_Pct": round(total_action_1 / (total_action_0 + total_action_1) * 100, 2),
}

print("=" * 70)
print("  SYSTEM TEAR SHEET (Macro-Portfolio Aggregation)")
print("=" * 70)
print(json.dumps(tear, indent=2))

out_path = eval_dir / "system_tear_sheet.json"
with open(out_path, "w") as f:
    json.dump(tear, f, indent=2)
print(f"\nSaved -> {out_path}")

# ═══════════════════════════════════════════════════════════════
# 2. PPO TRAINING LOSSES (Last 20 rows)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PPO TRAINING TELEMETRY (Last 20 non-zero rows)")
print("=" * 70)

losses_csv = plots_dir / "training_losses.csv"
with open(losses_csv) as f:
    reader = list(csv.DictReader(f))

valid_rows = [r for r in reader if float(r.get("policy_loss", 0)) != 0.0]
last20 = valid_rows[-20:]

header = f"{'timestep':>8} | {'policy_loss':>12} | {'value_loss':>11} | {'entropy_loss':>12} | {'clip_frac':>11} | {'approx_kl':>11} | {'lr':>6}"
print(header)
print("-" * len(header))
for r in last20:
    ts = int(r["timestep"])
    pl = float(r["policy_loss"])
    vl = float(r["value_loss"])
    el = float(r["entropy_loss"])
    cf = float(r["clip_fraction"])
    ak = float(r["approx_kl"])
    lr = float(r["lr"])
    print(f"{ts:>8} | {pl:>+12.8f} | {vl:>11.8f} | {el:>+12.8f} | {cf:>11.8f} | {ak:>11.8f} | {lr:.4f}")

fin = last20[-1]
print("\n--- FINAL CONVERGENCE STATE ---")
print(f"  policy_loss:  {float(fin['policy_loss']):+.10f}")
print(f"  value_loss:   {float(fin['value_loss']):.10f}")
print(f"  entropy_loss: {float(fin['entropy_loss']):+.10f}")
print(f"  approx_kl:    {float(fin['approx_kl']):.10f}")
print(f"  clip_frac:    {float(fin['clip_fraction']):.10f}")

ep_csv = plots_dir / "episode_rewards.csv"
with open(ep_csv) as f:
    ep_rows = list(csv.DictReader(f))
last_eps = ep_rows[-20:]
mean_final = sum(float(r["episode_reward"]) for r in last_eps) / len(last_eps)
print(f"  ep_rew_mean(last20): {mean_final:+.4f}")

# ═══════════════════════════════════════════════════════════════
# 3. ACTION DISTRIBUTION TENSOR
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ACTION DISTRIBUTION TENSOR")
print("=" * 70)
print(f"  Action=0 (HOLD/SELL):     {total_action_0:,}")
print(f"  Action=1 (BUY/HOLD_LONG): {total_action_1:,}")
if total_action_1 > 0:
    print(f"  Ratio (0:1):              {total_action_0 / total_action_1:.2f}:1")
    print(f"  Execution Sparsity:       {total_action_1 / (total_action_0 + total_action_1) * 100:.2f}% of timesteps in-market")
else:
    print("  Ratio: N/A (no action=1)")
