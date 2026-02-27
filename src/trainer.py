"""
Phase 6: PPO Training Harness
==============================
Wraps UniversalNepseEnv in Stable-Baselines3 PPO with sensible
hyper-parameters tuned for a swing-trading discrete-action setup.
"""

from __future__ import annotations

import csv
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.environment import UniversalNepseEnv
from src.run_manager import get_logger


# ── Metrics-collecting callback ───────────────────────────────────────────

class MetricsCollectorCallback(BaseCallback):
    """Collects per-update training metrics for later plotting.

    Stores:
        episode_timesteps, episode_rewards     – per completed episode
        update_timesteps, policy_losses,
        value_losses, entropy_losses, lrs      – per PPO update
    """

    def __init__(self, log_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._step = 0
        self.logger_inst = get_logger("rl_nepse.train")

        # Episode-level
        self.episode_timesteps: list[int] = []
        self.episode_rewards: list[float] = []

        # Update-level (grabbed from SB3 logger after each rollout)
        self.update_timesteps: list[int] = []
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropy_losses: list[float] = []
        self.learning_rates: list[float] = []
        self.clip_fractions: list[float] = []
        self.approx_kls: list[float] = []

    def _on_step(self) -> bool:
        self._step += 1
        # Capture completed-episode rewards
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_timesteps.append(self.num_timesteps)
                self.episode_rewards.append(info["episode"]["r"])

        if self._step % self.log_freq == 0:
            recent = self.episode_rewards[-20:]
            if recent:
                mean_r = np.mean(recent)
                self.logger_inst.info(
                    f"step {self.num_timesteps:>9,d} | "
                    f"mean_ep_reward(last20) {mean_r:+.4f}"
                )
        return True

    def _on_rollout_end(self) -> None:
        """Called after each PPO rollout – grab losses from SB3 logger."""
        try:
            log = self.model.logger.name_to_value
            self.update_timesteps.append(self.num_timesteps)
            self.policy_losses.append(log.get("train/policy_gradient_loss", 0.0))
            self.value_losses.append(log.get("train/value_loss", 0.0))
            self.entropy_losses.append(log.get("train/entropy_loss", 0.0))
            self.clip_fractions.append(log.get("train/clip_fraction", 0.0))
            self.approx_kls.append(log.get("train/approx_kl", 0.0))
            self.learning_rates.append(log.get("train/learning_rate", 0.0))
        except Exception:
            pass

    def save_csv(self, output_dir: str | pathlib.Path) -> None:
        """Dump collected metrics to CSVs."""
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Episode rewards
        if self.episode_rewards:
            with open(out / "episode_rewards.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestep", "episode_reward"])
                for ts, r in zip(self.episode_timesteps, self.episode_rewards):
                    w.writerow([ts, r])

        # Training losses
        if self.update_timesteps:
            with open(out / "training_losses.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestep", "policy_loss", "value_loss",
                            "entropy_loss", "clip_fraction", "approx_kl", "lr"])
                for i, ts in enumerate(self.update_timesteps):
                    w.writerow([
                        ts,
                        self.policy_losses[i],
                        self.value_losses[i],
                        self.entropy_losses[i],
                        self.clip_fractions[i],
                        self.approx_kls[i],
                        self.learning_rates[i],
                    ])


# ── Factory to build env closures ────────────────────────────────────────

def _make_env(
    feat_df: pd.DataFrame,
    valid_start_dates: Dict[str, pd.Timestamp],
    episode_length: int,
    seed: int,
):
    """Return a callable that creates a fresh env (needed by VecEnv)."""

    def _init():
        env = UniversalNepseEnv(
            feat_df=feat_df,
            valid_start_dates=valid_start_dates,
            episode_length=episode_length,
            seed=seed,
        )
        return Monitor(env)

    return _init


# ── Public API ────────────────────────────────────────────────────────────

def train_ppo(
    feat_df: pd.DataFrame,
    valid_start_dates: Dict[str, pd.Timestamp],
    total_timesteps: int = 500_000,
    episode_length: int = 252,
    n_envs: int = 4,
    save_dir: str | pathlib.Path = "outputs/models",
    seed: int = 42,
    device: str = "auto",
    plots_dir: str | pathlib.Path | None = None,
) -> PPO:
    """Train a PPO agent on the UniversalNepseEnv.

    Parameters
    ----------
    feat_df : pre-computed feature DataFrame (from features.compute_features).
    valid_start_dates : warm-up-adjusted start dates per ticker.
    total_timesteps : total env interactions to train.
    episode_length : trading days per episode.
    n_envs : number of parallel environments.
    save_dir : where to write model checkpoints.
    seed : random seed.
    device : "cpu", "cuda", or "auto".
    plots_dir : where to save PNG training plots (None = save_dir).

    Returns
    -------
    Trained PPO model.
    """
    log = get_logger("rl_nepse.train")
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir is None:
        plots_dir = save_dir.parent / "plots" if save_dir.name == "models" else save_dir
    plots_dir = pathlib.Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Build vectorised environments
    env_fns = [
        _make_env(feat_df, valid_start_dates, episode_length, seed + i)
        for i in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns)      # DummyVecEnv is safer on Windows

    # Eval env (single, deterministic-ish)
    eval_env = DummyVecEnv(
        [_make_env(feat_df, valid_start_dates, episode_length, seed + 999)]
    )

    # PPO hyper-parameters — conservative LR, moderate clip
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,          # maintain exploration pressure
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ),
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 10, 10_000),
        save_path=str(save_dir),
        name_prefix="ppo_nepse",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=max(total_timesteps // 20, 5_000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    metrics_cb = MetricsCollectorCallback(log_freq=5_000)

    log.info(f"PPO training | {total_timesteps:,} timesteps | {n_envs} envs | "
             f"device={model.device}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        progress_bar=True,
    )

    final_path = save_dir / "ppo_nepse_final"
    model.save(str(final_path))
    log.info(f"Model saved -> {final_path}")

    # ── Save metrics CSVs ──
    metrics_cb.save_csv(plots_dir)
    log.info(f"Training metrics CSVs -> {plots_dir}")

    # ── Generate PNG training plots ──
    _generate_training_plots(metrics_cb, save_dir, plots_dir, log)

    vec_env.close()
    eval_env.close()
    return model


def _generate_training_plots(
    cb: MetricsCollectorCallback,
    save_dir: pathlib.Path,
    plots_dir: pathlib.Path,
    log,
) -> None:
    """Create training diagnostic PNGs from collected metrics."""
    from src.plots import (
        plot_training_reward,
        plot_training_loss,
        plot_eval_reward,
        plot_learning_rate,
        plot_reward_distribution,
    )

    # 1. Episode reward curve
    if cb.episode_rewards:
        p = plot_training_reward(cb.episode_timesteps, cb.episode_rewards, plots_dir)
        log.info(f"Plot saved: {p}")

        p = plot_reward_distribution(cb.episode_rewards, plots_dir)
        log.info(f"Plot saved: {p}")

    # 2. Loss curves
    if cb.update_timesteps:
        p = plot_training_loss(
            cb.update_timesteps, cb.policy_losses,
            cb.value_losses, cb.entropy_losses, plots_dir,
        )
        log.info(f"Plot saved: {p}")

    # 3. Learning rate
    if cb.learning_rates:
        p = plot_learning_rate(cb.update_timesteps, cb.learning_rates, plots_dir)
        log.info(f"Plot saved: {p}")

    # 4. Eval reward (from SB3 EvalCallback's npz file)
    eval_npz = save_dir / "eval_logs" / "evaluations.npz"
    if eval_npz.exists():
        try:
            data = np.load(str(eval_npz))
            ts = data["timesteps"].tolist()
            mean_r = data["results"].mean(axis=1).tolist()
            std_r = data["results"].std(axis=1).tolist()
            p = plot_eval_reward(ts, mean_r, std_r, plots_dir)
            log.info(f"Plot saved: {p}")
        except Exception as e:
            log.warning(f"Could not plot eval reward: {e}")


def load_model(path: str | pathlib.Path) -> PPO:
    """Load a saved PPO model."""
    return PPO.load(str(path))
