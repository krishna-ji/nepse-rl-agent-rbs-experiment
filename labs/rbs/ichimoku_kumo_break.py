#!/usr/bin/env python3
"""
NEPSE Ichimoku Kumo Break Strategy — CLI Runner
================================================
Thin wrapper around ``src.rbs`` monolith framework.

Usage::

    python labs/rbs/ichimoku_kumo_break.py
"""

import sys, pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rbs.kumo_break import KumoBreak


def main():
    print("=" * 70)
    print("  NEPSE Ichimoku Kumo Break Strategy — Rule-Based Backtest")
    print("=" * 70)

    s = KumoBreak()
    s.run()
    s.simulate()
    print(s.summary())
    pdf = s.report()
    print(f"\nReport saved → {pdf.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
