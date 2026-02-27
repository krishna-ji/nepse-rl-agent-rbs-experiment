"""
NEPSE Universal RL Stochastic Pullback Engine
=============================================
Entry-point.  Delegates to runs/ scripts.

Usage:
    python main.py train       # full training
    python main.py eval        # out-of-sample evaluation
    python main.py test        # quick smoke test
    python main.py all         # smoke -> train -> eval (full pipeline)
    python main.py dash        # Streamlit dashboard
    python main.py menu        # interactive CLI menu

Or run any script directly (Code Runner / Play button):
    python runs/run_training.py
    python runs/run_evaluation.py
    python runs/run_smoketest.py
    python runs/run_all.py

Each run creates a timestamped folder under outputs/ with:
    logs/    – full debug log
    plots/   – PNG training/evaluation charts
    models/  – PPO checkpoints
    eval/    – trajectory HTMLs & CSVs
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2:
        # No args -> launch interactive CLI menu
        from runs.run_cli import cli_menu
        cli_menu()
        return

    cmd = sys.argv[1].lower()
    # Remove 'cmd' from argv so sub-script sees clean argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    DATA = str(ROOT / "data" / "stocks")

    if cmd == "train":
        from runs.run_training import run_training
        run_training(
            data_dir=DATA,
            timesteps=500_000,
            episode_length=252,
            n_envs=4,
            seed=42,
            device="auto",
        )
    elif cmd in ("evaluate", "eval"):
        from runs.run_evaluation import run_evaluation
        run_evaluation(
            data_dir=DATA,
            model_path=str(ROOT / "outputs" / "models" / "best" / "best_model"),
            ticker=None,
            multi=3,
            episode_length=252,
        )
    elif cmd == "test":
        from runs.run_smoketest import run_smoketest
        run_smoketest(
            data_dir=DATA,
            timesteps=200_000,
            episode_length=100,
            n_envs=4,
        )
    elif cmd == "all":
        # run_all.py has its own __main__ config; execute it
        import runpy
        runpy.run_path(str(ROOT / "runs" / "run_all.py"), run_name="__main__")
    elif cmd in ("dash", "dashboard"):
        from runs.run_dashboard import run_dashboard
        run_dashboard()
    elif cmd in ("menu", "cli"):
        from runs.run_cli import cli_menu
        cli_menu()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
