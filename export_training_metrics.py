#!/usr/bin/env python3
"""
Export NEPSE RL Training Metrics to CSV
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import pathlib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class EnhancedRewardTracker(BaseCallback):
    """Enhanced callback to collect and export training metrics"""
    def __init__(self, run_dir):
        super().__init__(verbose=0)
        self.run_dir = pathlib.Path(run_dir)
        
        # Episode metrics
        self.ep_timesteps = []
        self.ep_rewards = []
        self.ep_lengths = []
        self._ep_count = 0
        
        # Training loss metrics
        self.update_ts = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.learning_rates = []
        self.explained_variances = []
        
        # Initialize logger
        self.log = logging.getLogger("nepserl_metrics")

    def _on_step(self):
        """Called after each environment step"""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_timesteps.append(self.num_timesteps)
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self._ep_count += 1
                
                if self._ep_count % 50 == 0:
                    avg_reward = np.mean(self.ep_rewards[-50:])
                    avg_length = np.mean(self.ep_lengths[-50:])
                    self.log.info(f"Episode {self._ep_count:4d} | Steps: {self.num_timesteps:7d} | "
                                f"Avg Reward (50): {avg_reward:+.4f} | Avg Length: {avg_length:.1f}")
                    
                    # Export intermediate results every 100 episodes
                    if self._ep_count % 100 == 0:
                        self.export_episode_metrics()
        return True

    def _on_rollout_end(self):
        """Called after each rollout (policy update)"""
        try:
            vals = self.model.logger.name_to_value
            self.update_ts.append(self.num_timesteps)
            self.policy_losses.append(vals.get("train/policy_gradient_loss", 0.0))
            self.value_losses.append(vals.get("train/value_loss", 0.0))
            self.entropy_losses.append(vals.get("train/entropy_loss", 0.0))
            self.learning_rates.append(vals.get("train/learning_rate", 0.0))
            self.explained_variances.append(vals.get("train/explained_variance", 0.0))
            
            # Export loss metrics every few updates
            if len(self.update_ts) % 10 == 0:
                self.export_loss_metrics()
                
        except Exception as e:
            self.log.warning(f"Could not extract training metrics: {e}")

    def export_episode_metrics(self):
        """Export episode rewards and metrics to CSV"""
        if not self.ep_rewards:
            return
            
        episode_df = pd.DataFrame({
            'timestep': self.ep_timesteps,
            'episode_reward': self.ep_rewards,
            'episode_length': self.ep_lengths,
            'episode_number': range(1, len(self.ep_rewards) + 1)
        })
        
        # Add moving averages
        for window in [10, 50, 100]:
            if len(self.ep_rewards) >= window:
                episode_df[f'reward_ma_{window}'] = episode_df['episode_reward'].rolling(window, min_periods=1).mean()
                episode_df[f'length_ma_{window}'] = episode_df['episode_length'].rolling(window, min_periods=1).mean()
        
        csv_path = self.run_dir / "episode_metrics.csv"
        episode_df.to_csv(csv_path, index=False)
        self.log.info(f"Exported episode metrics to {csv_path}")
        
    def export_loss_metrics(self):
        """Export training loss metrics to CSV"""
        if not self.update_ts:
            return
            
        loss_df = pd.DataFrame({
            'timestep': self.update_ts,
            'policy_loss': self.policy_losses,
            'value_loss': self.value_losses,
            'entropy_loss': self.entropy_losses,
            'learning_rate': self.learning_rates,
            'explained_variance': self.explained_variances,
            'update_number': range(1, len(self.update_ts) + 1)
        })
        
        csv_path = self.run_dir / "training_losses.csv"
        loss_df.to_csv(csv_path, index=False)
        self.log.info(f"Exported loss metrics to {csv_path}")

    def export_final_metrics(self):
        """Export all metrics at the end of training"""
        self.export_episode_metrics()
        self.export_loss_metrics()
        
        # Create summary statistics
        if self.ep_rewards:
            summary_stats = {
                'total_episodes': len(self.ep_rewards),
                'total_timesteps': max(self.ep_timesteps) if self.ep_timesteps else 0,
                'final_reward_mean': np.mean(self.ep_rewards[-50:]) if len(self.ep_rewards) >= 50 else np.mean(self.ep_rewards),
                'final_reward_std': np.std(self.ep_rewards[-50:]) if len(self.ep_rewards) >= 50 else np.std(self.ep_rewards),
                'best_reward': max(self.ep_rewards),
                'worst_reward': min(self.ep_rewards),
                'reward_improvement': np.mean(self.ep_rewards[-50:]) - np.mean(self.ep_rewards[:50]) if len(self.ep_rewards) >= 100 else 0,
            }
            
            if self.policy_losses:
                summary_stats.update({
                    'final_policy_loss': self.policy_losses[-1],
                    'final_value_loss': self.value_losses[-1],
                    'final_entropy_loss': self.entropy_losses[-1],
                    'total_updates': len(self.policy_losses)
                })
            
            summary_df = pd.DataFrame([summary_stats])
            summary_path = self.run_dir / "training_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            self.log.info(f"Exported training summary to {summary_path}")


def extract_metrics_from_existing_run(run_dir):
    """
    Extract metrics from an existing run directory if saved data exists
    This is a fallback for runs where metrics weren't saved during training
    """
    run_path = pathlib.Path(run_dir)
    log_file = run_path / "nepserl.log"
    
    if not log_file.exists():
        print(f"No log file found at {log_file}")
        return None, None
    
    print(f"Extracting metrics from {log_file}")
    
    # Parse episode rewards from log
    episode_data = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "Training — ep" in line and "avg_reward" in line:
                # Extract: ep 50, ts 13104, avg_reward(50)=-1.2211
                try:
                    parts = line.split("—")[1].strip()
                    ep_part = parts.split(",")[0].strip().split()[1]
                    ts_part = parts.split(",")[1].strip().split()[1] 
                    reward_part = parts.split("avg_reward(50)=")[1]
                    
                    episode_data.append({
                        'episode_number': int(ep_part),
                        'timestep': int(ts_part),
                        'avg_reward_50': float(reward_part)
                    })
                except (IndexError, ValueError) as e:
                    continue
    
    if episode_data:
        episode_df = pd.DataFrame(episode_data)
        csv_path = run_path / "extracted_episode_metrics.csv"
        episode_df.to_csv(csv_path, index=False)
        print(f"Exported {len(episode_data)} episode metrics to {csv_path}")
        return episode_df, None
    else:
        print("No training metrics found in log file")
        return None, None


if __name__ == "__main__":
    # Try to extract from existing run first
    run_dir = "C:/Users/krishna/Desktop/nepserl/runs/20260227_172411"
    
    print("Attempting to extract metrics from existing run...")
    episode_metrics, loss_metrics = extract_metrics_from_existing_run(run_dir)
    
    if episode_metrics is not None:
        print("✓ Successfully extracted episode metrics!")
        print(f"  Episodes: {len(episode_metrics)}")
        print(f"  Timestep range: {episode_metrics['timestep'].min()} - {episode_metrics['timestep'].max()}")
        print(f"  Final avg reward: {episode_metrics['avg_reward_50'].iloc[-1]:.4f}")
    else:
        print("✗ Could not extract metrics from existing run")
        print("  Run training again with the enhanced tracker to get CSV exports")