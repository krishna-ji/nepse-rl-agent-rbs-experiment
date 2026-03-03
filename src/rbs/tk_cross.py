"""
T/K Cross Strategy — Sadekar Ch.5
==================================
Tenkan crosses above/below Kijun.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.rbs.base import (
    BacktestStrategy,
    TradeRecord,
    chikou_is_free,
    compute_ichimoku,
    flat_sb_strong_candle,
    is_senkou_b_flat,
)


def classify_cross_strength(close_t: float, kumo_top_t: float,
                            kumo_bot_t: float, direction: str) -> str:
    """Classify T/K cross strength by location relative to Kumo."""
    if close_t > kumo_top_t:
        return "strong" if direction == "LONG" else "weak"
    elif close_t < kumo_bot_t:
        return "weak" if direction == "LONG" else "strong"
    return "neutral"


class TKCross(BacktestStrategy):
    """Tenkan/Kijun Cross strategy."""

    @property
    def name(self) -> str:
        return "T/K Cross"

    def backtest_ticker(self, ticker: str, df: pd.DataFrame) -> List[TradeRecord]:
        p = self.params
        o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
        dates = df.index
        n = len(df)
        if n < p.warmup + 10:
            return []

        ich = compute_ichimoku(df, p)
        tenkan, kijun = ich["tenkan"], ich["kijun"]
        senkou_a, senkou_b = ich["senkou_a"], ich["senkou_b"]
        kumo_top, kumo_bot = ich["kumo_top"], ich["kumo_bot"]
        fut_sa, fut_sb = ich["future_sa"], ich["future_sb"]
        atr = ich["atr"]

        trades: List[TradeRecord] = []
        state = "FLAT"
        direction = None
        tk_cross_long = tk_cross_short = False
        pend_level = pend_stop = 0.0
        sig_bar = 0
        cross_str = "neutral"
        entry_price = stop_px = 0.0
        entry_bar = 0
        entry_date = None
        trade_cross_str = "neutral"

        for t in range(p.warmup, n):
            if any(np.isnan(x[t]) for x in [tenkan, kijun, kumo_top, kumo_bot,
                                              fut_sa, fut_sb, atr]):
                continue

            # T/K cross detection
            if t > 0 and not np.isnan(tenkan[t - 1]) and not np.isnan(kijun[t - 1]):
                if tenkan[t] > kijun[t] and tenkan[t - 1] <= kijun[t - 1]:
                    tk_cross_long = True
                if not p.long_only:
                    if tenkan[t] < kijun[t] and tenkan[t - 1] >= kijun[t - 1]:
                        tk_cross_short = True
            if tenkan[t] <= kijun[t]:
                tk_cross_long = False
            if tenkan[t] >= kijun[t]:
                tk_cross_short = False

            # PENDING
            if state == "PENDING":
                filled = False
                if direction == "LONG":
                    if o[t] >= pend_level:
                        entry_price = o[t]; filled = True
                    elif h[t] >= pend_level:
                        entry_price = pend_level; filled = True
                elif direction == "SHORT":
                    if o[t] <= pend_level:
                        entry_price = o[t]; filled = True
                    elif l[t] <= pend_level:
                        entry_price = pend_level; filled = True
                if filled:
                    stop_px = pend_stop; entry_bar = t; entry_date = dates[t]
                    trade_cross_str = cross_str
                    state = "POSITION"; tk_cross_long = tk_cross_short = False
                    continue
                cancel = (t - sig_bar >= p.order_timeout)
                if direction == "LONG":
                    cancel = cancel or tenkan[t] <= kijun[t]
                elif direction == "SHORT":
                    cancel = cancel or tenkan[t] >= kijun[t]
                if cancel:
                    state = "FLAT"; direction = None
                continue

            # POSITION
            if state == "POSITION":
                exit_px = exit_rsn = None
                if direction == "LONG":
                    if o[t] <= stop_px:
                        exit_px = o[t]; exit_rsn = "gap_stop"
                    elif l[t] <= stop_px:
                        exit_px = stop_px; exit_rsn = "hard_stop"
                    elif c[t] < kijun[t]:
                        exit_px = c[t]; exit_rsn = "kijun_close"
                    else:
                        trail = kijun[t] - p.atr_stop_mult * atr[t]
                        if not np.isnan(trail):
                            stop_px = max(stop_px, trail)
                elif direction == "SHORT":
                    if o[t] >= stop_px:
                        exit_px = o[t]; exit_rsn = "gap_stop"
                    elif h[t] >= stop_px:
                        exit_px = stop_px; exit_rsn = "hard_stop"
                    elif c[t] > kijun[t]:
                        exit_px = c[t]; exit_rsn = "kijun_close"
                    else:
                        trail = kijun[t] + p.atr_stop_mult * atr[t]
                        if not np.isnan(trail):
                            stop_px = min(stop_px, trail)
                if exit_px is not None:
                    pnl = ((exit_px / entry_price - 1) * 100
                           if direction == "LONG"
                           else (1 - exit_px / entry_price) * 100)
                    net = pnl - p.transaction_cost * 100
                    trades.append(TradeRecord(
                        ticker=ticker, direction=direction,
                        entry_date=entry_date, exit_date=dates[t],
                        entry_price=round(entry_price, 2),
                        exit_price=round(exit_px, 2),
                        bars_held=t - entry_bar,
                        pnl_pct=round(pnl, 4), net_pnl_pct=round(net, 4),
                        exit_reason=exit_rsn,
                        extra={"cross_strength": trade_cross_str},
                    ))
                    state = "FLAT"; direction = None
                continue

            # FLAT — scan for entries
            if tk_cross_long:
                cond1 = fut_sa[t] > fut_sb[t]
                cond2 = chikou_is_free(c, h, l, t, "long", senkou_a, senkou_b,
                                       p.displacement, p.chikou_free_half)
                cond3 = c[t] > tenkan[t] and c[t] > kijun[t]
                cond4 = (c[t] - kijun[t]) / max(atr[t], 1e-10) < p.max_kijun_dist
                cond5 = True
                if is_senkou_b_flat(fut_sb, t, p.flat_sb_lookback, p.flat_sb_tol):
                    if not flat_sb_strong_candle(o[t], c[t], atr[t], "long", p.flat_sb_strong):
                        cond5 = False
                if cond1 and cond2 and cond3 and cond4 and cond5:
                    hh9  = np.nanmax(h[max(0, t - p.tenkan_period + 1):t + 1])
                    hh26 = np.nanmax(h[max(0, t - p.kijun_period  + 1):t + 1])
                    if (hh26 - hh9) / max(atr[t], 1e-10) < p.hh_proximity:
                        pend_level = hh26
                    else:
                        pend_level = hh9
                    pend_stop = kijun[t] - p.atr_stop_mult * atr[t]
                    sig_bar = t; direction = "LONG"
                    cross_str = classify_cross_strength(c[t], kumo_top[t], kumo_bot[t], "LONG")
                    state = "PENDING"

            elif not p.long_only and tk_cross_short:
                cond1 = fut_sa[t] < fut_sb[t]
                cond2 = chikou_is_free(c, h, l, t, "short", senkou_a, senkou_b,
                                       p.displacement, p.chikou_free_half)
                cond3 = c[t] < tenkan[t] and c[t] < kijun[t]
                cond4 = (kijun[t] - c[t]) / max(atr[t], 1e-10) < p.max_kijun_dist
                cond5 = True
                if is_senkou_b_flat(fut_sb, t, p.flat_sb_lookback, p.flat_sb_tol):
                    if not flat_sb_strong_candle(o[t], c[t], atr[t], "short", p.flat_sb_strong):
                        cond5 = False
                if cond1 and cond2 and cond3 and cond4 and cond5:
                    ll9  = np.nanmin(l[max(0, t - p.tenkan_period + 1):t + 1])
                    ll26 = np.nanmin(l[max(0, t - p.kijun_period  + 1):t + 1])
                    if (ll9 - ll26) / max(atr[t], 1e-10) < p.hh_proximity:
                        pend_level = ll26
                    else:
                        pend_level = ll9
                    pend_stop = kijun[t] + p.atr_stop_mult * atr[t]
                    sig_bar = t; direction = "SHORT"
                    cross_str = classify_cross_strength(c[t], kumo_top[t], kumo_bot[t], "SHORT")
                    state = "PENDING"

        # EOD close
        if state == "POSITION":
            last_c = c[-1]
            pnl = ((last_c / entry_price - 1) * 100
                   if direction == "LONG"
                   else (1 - last_c / entry_price) * 100)
            net = pnl - p.transaction_cost * 100
            trades.append(TradeRecord(
                ticker=ticker, direction=direction,
                entry_date=entry_date, exit_date=dates[-1],
                entry_price=round(entry_price, 2),
                exit_price=round(last_c, 2),
                bars_held=n - 1 - entry_bar,
                pnl_pct=round(pnl, 4), net_pnl_pct=round(net, 4),
                exit_reason="data_end",
                extra={"cross_strength": trade_cross_str},
            ))
        return trades
