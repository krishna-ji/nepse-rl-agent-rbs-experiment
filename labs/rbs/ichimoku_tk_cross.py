#!/usr/bin/env python3
"""
NEPSE Ichimoku T/K Cross Strategy — CLI Runner
===============================================
Thin wrapper around ``src.rbs`` monolith framework.

Usage::

    python labs/rbs/ichimoku_tk_cross.py
"""

import sys, pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rbs.tk_cross import TKCross


def main():
    print("=" * 70)
    print("  NEPSE Ichimoku T/K Cross Strategy — Rule-Based Backtest")
    print("=" * 70)

    s = TKCross()
    s.run()
    s.simulate()
    print(s.summary())
    pdf = s.report()
    print(f"\nReport saved → {pdf.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
