"""
Run Manager – timestamped output directories & logging setup
=============================================================
Every run gets its own folder: outputs/run_YYYYMMDD_HHMMSS/
Inside it:  logs/, plots/, models/, eval/
"""

from __future__ import annotations

import logging
import pathlib
import sys
from datetime import datetime
from typing import Optional


def create_run_dir(
    base_dir: str | pathlib.Path = "outputs",
    tag: str = "run",
) -> pathlib.Path:
    """Create a timestamped run directory and return its path.

    Layout created:
        outputs/run_20260225_143022/
            logs/
            plots/
            models/
            eval/
    """
    base = pathlib.Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for sub in ("logs", "plots", "models", "eval"):
        (run_dir / sub).mkdir(exist_ok=True)

    return run_dir


def setup_logging(
    run_dir: str | pathlib.Path,
    name: str = "rl_nepse",
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that writes to both console and a log file.

    Returns
    -------
    logger : logging.Logger
        Pre-configured logger.  Also writes to <run_dir>/logs/run.log
    """
    run_dir = pathlib.Path(run_dir)
    log_file = run_dir / "logs" / "run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler – DEBUG level (captures everything)
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler – INFO level (cleaner output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file:      {log_file}")

    return logger


def get_logger(name: str = "rl_nepse") -> logging.Logger:
    """Get the existing logger (or a default one if setup_logging wasn't called).
    
    Child loggers (e.g. rl_nepse.train) propagate to the parent, so only
    add a fallback handler if no ancestor has one.
    """
    logger = logging.getLogger(name)

    # Walk up the hierarchy to check if any ancestor has handlers
    check = logger
    has_handlers = False
    while check:
        if check.handlers:
            has_handlers = True
            break
        if not check.propagate:
            break
        check = check.parent

    if not has_handlers:
        # Minimal fallback (only if setup_logging was never called)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger
