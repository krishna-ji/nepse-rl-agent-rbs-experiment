#!/usr/bin/env python3
"""
Enhanced NEPSE RL Training Configuration
Addresses convergence issues and provides improved hyperparameters
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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch


class AdvancedMetricsCallback(BaseCallback):
    """
    Enhanced callback to track detailed training metrics and export to CSV
    """
    def __init__(self, run_dir, eval_freq=10000, save_freq=50000):
        super().__init__(verbose=1)
        self.run_dir = pathlib.Path(run_dir)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []
        self.episode_count = 0
        
        # Training metrics tracking  
        self.training_timesteps = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.learning_rates = []
        self.explained_variances = []
        self.clip_fractions = []
        
        # Performance tracking
        self.best_mean_reward = -np.inf
        self.reward_threshold_reached = False
        
        # Set up logging
        self.metrics_log = logging.getLogger("metrics")
        
    def _on_step(self) -> bool:
        # Track individual episodes
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.episode_timesteps.append(self.num_timesteps)
                self.episode_count += 1
                
                # Log progress every 25 episodes
                if self.episode_count % 25 == 0:
                    recent_rewards = self.episode_rewards[-25:]
                    mean_reward = np.mean(recent_rewards)
                    std_reward = np.std(recent_rewards)
                    
                    self.metrics_log.info(f"Episode {self.episode_count:4d} | "
                                        f"Steps: {self.num_timesteps:7d} | "
                                        f"Reward: {mean_reward:+.4f}±{std_reward:.4f}")
                    
                    # Check for improvement
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.metrics_log.info(f"🎯 New best mean reward: {mean_reward:+.4f}")
                    
                    # Save intermediate results
                    if self.episode_count % 100 == 0:
                        self._save_episode_metrics()
        
        # Save model checkpoints
        if self.num_timesteps % self.save_freq == 0:
            model_path = self.run_dir / f"model_checkpoint_{self.num_timesteps}.zip"
            self.model.save(model_path)
            self.metrics_log.info(f"💾 Saved model checkpoint: {model_path}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called after each policy update"""
        try:
            # Extract training metrics from logger
            logger_vals = self.model.logger.name_to_value
            
            self.training_timesteps.append(self.num_timesteps)
            self.policy_losses.append(logger_vals.get("train/policy_gradient_loss", np.nan))
            self.value_losses.append(logger_vals.get("train/value_loss", np.nan))  
            self.entropy_losses.append(logger_vals.get("train/entropy_loss", np.nan))
            self.learning_rates.append(logger_vals.get("train/learning_rate", np.nan))
            self.explained_variances.append(logger_vals.get("train/explained_variance", np.nan))
            self.clip_fractions.append(logger_vals.get("train/clip_fraction", np.nan))
            
            # Save loss metrics periodically
            if len(self.training_timesteps) % 20 == 0:  # Every 20 updates
                self._save_loss_metrics()
                
        except Exception as e:
            self.metrics_log.warning(f"Could not extract training metrics: {e}")
    
    def _save_episode_metrics(self):
        """Save episode-level metrics to CSV"""
        if not self.episode_rewards:
            return
            
        episode_df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'timestep': self.episode_timesteps,
            'reward': self.episode_rewards,
            'length': self.episode_lengths
        })
        
        # Add rolling statistics
        for window in [10, 50, 100]:
            if len(self.episode_rewards) >= window:
                episode_df[f'reward_ma_{window}'] = episode_df['reward'].rolling(window, min_periods=1).mean()
                episode_df[f'reward_std_{window}'] = episode_df['reward'].rolling(window, min_periods=1).std()
        
        csv_path = self.run_dir / "training_episode_metrics.csv"
        episode_df.to_csv(csv_path, index=False)
        
    def _save_loss_metrics(self):
        """Save training loss metrics to CSV"""
        if not self.training_timesteps:
            return
            
        loss_df = pd.DataFrame({
            'timestep': self.training_timesteps,
            'policy_loss': self.policy_losses,
            'value_loss': self.value_losses,
            'entropy_loss': self.entropy_losses,
            'learning_rate': self.learning_rates,
            'explained_variance': self.explained_variances,
            'clip_fraction': self.clip_fractions,
            'update': range(1, len(self.training_timesteps) + 1)
        })
        
        csv_path = self.run_dir / "training_loss_metrics.csv"
        loss_df.to_csv(csv_path, index=False)
    
    def _on_training_end(self) -> None:
        """Save final metrics"""
        self._save_episode_metrics()
        self._save_loss_metrics()
        
        # Create final summary
        if self.episode_rewards:
            final_stats = {
                'total_episodes': len(self.episode_rewards),
                'total_timesteps': self.num_timesteps,
                'best_mean_reward': self.best_mean_reward,
                'final_mean_reward': np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards),
                'final_std_reward': np.std(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.std(self.episode_rewards),
                'total_improvement': np.mean(self.episode_rewards[-50:]) - np.mean(self.episode_rewards[:50]) if len(self.episode_rewards) >= 100 else 0
            }
            
            summary_df = pd.DataFrame([final_stats])
            summary_path = self.run_dir / "final_training_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            
            self.metrics_log.info(f"💯 Training complete! Final mean reward: {final_stats['final_mean_reward']:+.4f}")


def create_improved_training_config():
    """
    Returns improved PPO hyperparameters based on analysis of convergence issues
    """
    
    # Configuration 1: Conservative (for stable training)
    conservative_config = {
        'learning_rate': 1e-4,          # Reduced from 3e-4
        'n_steps': 4096,                # Increased from 2048
        'batch_size': 256,              # Keep same
        'n_epochs': 15,                 # Increased from 10
        'gamma': 0.995,                 # Increased from 0.99 (longer horizon)
        'gae_lambda': 0.98,             # Increased from 0.95
        'clip_range': 0.15,             # Reduced from 0.2 (more conservative)
        'ent_coef': 0.01,               # Reduced from 0.05 (less exploration)
        'vf_coef': 0.25,                # Reduced from 0.5
        'max_grad_norm': 0.3,           # Reduced from 0.5
        'policy_kwargs': dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Larger networks
            activation_fn=torch.nn.Tanh  # Different activation
        ),
        'description': 'Conservative - Stable learning with reduced exploration'
    }
    
    # Configuration 2: Aggressive (for faster learning)
    aggressive_config = {
        'learning_rate': 5e-4,          # Higher learning rate
        'n_steps': 2048,                # Standard
        'batch_size': 512,              # Larger batches
        'n_epochs': 8,                  # Fewer epochs per update
        'gamma': 0.99,                  # Standard
        'gae_lambda': 0.95,             # Standard
        'clip_range': 0.25,             # Larger clip range
        'ent_coef': 0.03,               # Moderate exploration
        'vf_coef': 0.5,                 # Standard
        'max_grad_norm': 1.0,           # Less gradient clipping
        'policy_kwargs': dict(
            net_arch=dict(pi=[512, 512], vf=[512, 512])  # Even larger networks
        ),
        'description': 'Aggressive - Faster learning with higher capacity'
    }
    
    # Configuration 3: Adaptive (recommended)
    adaptive_config = {
        'learning_rate': schedule_learning_rate,  # Custom schedule
        'n_steps': 3072,                # Compromise
        'batch_size': 384,              # Compromise
        'n_epochs': 12,                 # Higher than default
        'gamma': 0.992,                 # Slight increase
        'gae_lambda': 0.96,             # Slight increase
        'clip_range': schedule_clip_range,  # Scheduled clipping
        'ent_coef': schedule_entropy_coef,  # Scheduled entropy
        'vf_coef': 0.4,
        'max_grad_norm': 0.4,
        'policy_kwargs': dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # 3-layer networks
            activation_fn=torch.nn.ReLU
        ),
        'description': 'Adaptive - Scheduled hyperparameters for optimal convergence'
    }
    
    return conservative_config, aggressive_config, adaptive_config


def schedule_learning_rate(progress_remaining):
    """
    Learning rate schedule: Start high, decay over time
    """
    if progress_remaining > 0.8:
        return 3e-4
    elif progress_remaining > 0.5:
        return 2e-4
    elif progress_remaining > 0.2:
        return 1e-4
    else:
        return 5e-5


def schedule_clip_range(progress_remaining):
    """
    Clip range schedule: Start higher, reduce over time
    """
    return 0.25 * progress_remaining + 0.1


def schedule_entropy_coef(progress_remaining):
    """
    Entropy coefficient schedule: High exploration early, low later
    """
    return 0.05 * progress_remaining + 0.005


def train_with_enhanced_config(feat_df, valid_start_dates, run_dir, config_name='conservative'):
    """
    Train PPO with enhanced configuration and monitoring
    """
    # Get configurations
    conservative, aggressive, adaptive = create_improved_training_config()
    
    configs = {
        'conservative': conservative,
        'aggressive': aggressive, 
        'adaptive': adaptive
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Use: {list(configs.keys())}")
    
    config = configs[config_name]
    
    print(f"Training with {config_name} configuration:")
    print(f"Description: {config['description']}")
    
    # Training parameters
    TOTAL_TIMESTEPS = 1_000_000  # Increased from 500k
    N_ENVS = 6  # Increased from 4
    SEED = 42
    
    # Create environments
    def make_env(seed):
        def _init():
            return Monitor(NepseEnv(feat_df, valid_start_dates, episode_length=252, seed=seed))
        return _init
    
    # Use SubprocVecEnv for better parallelization
    vec_env = SubprocVecEnv([make_env(SEED + i) for i in range(N_ENVS)])
    eval_env = DummyVecEnv([make_env(SEED + 999)])
    
    # Create model with enhanced config
    model = PPO(
        "MlpPolicy",
        vec_env,
        **{k: v for k, v in config.items() if k not in ['description']},
        seed=SEED,
        device="auto",
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard_logs")
    )
    
    # Create enhanced callback
    metrics_callback = AdvancedMetricsCallback(run_dir)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=25000,  # Evaluate every 25k steps
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    print(f"\n🚀 Starting enhanced training...")
    print(f"   Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   Environments: {N_ENVS}")
    print(f"   Configuration: {config_name}")
    
    # Train the model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[metrics_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = run_dir / "final_model.zip"
    model.save(final_model_path)
    
    # Close environments
    vec_env.close()
    eval_env.close()
    
    print(f"✅ Training complete! Model saved to {final_model_path}")
    print(f"📊 Check {run_dir} for detailed CSV metrics and logs")
    
    return model, metrics_callback


if __name__ == "__main__":
    # This is a template/example - you would need to load your actual data
    print("Enhanced NEPSE RL Training Configuration")
    print("=" * 50)
    print("\nThis script provides improved training configurations")
    print("to address the convergence issues identified in your current model.")
    
    print("\n🔍 Issues identified:")
    print("   • Final rewards still quite negative (-0.7426)")  
    print("   • Training appears to plateau after ~300k timesteps")
    print("   • High variance in recent episodes")
    
    print("\n💡 Recommended improvements:")
    print("   • Increase training to 1M+ timesteps")
    print("   • Use scheduled hyperparameters (learning rate, entropy)")
    print("   • Larger network architectures")
    print("   • More frequent evaluation and model saving")
    print("   • Better monitoring and CSV export throughout training")
    
    print("\n📝 Usage:")
    print("   1. Load your feat_df and valid_start_dates")  
    print("   2. Create a new run directory") 
    print("   3. Call: train_with_enhanced_config(feat_df, valid_start_dates, run_dir)")
    
    print("\n🎯 Expected improvements:")
    print("   • Better convergence with scheduled hyperparameters")
    print("   • Detailed CSV exports throughout training")
    print("   • Model checkpoints and evaluation metrics")
    print("   • Reduced variance and improved final performance")