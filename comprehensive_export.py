#!/usr/bin/env python3
"""
Comprehensive NEPSE RL Training Data Extractor and CSV Exporter
Creates separate CSV files for training losses and episode rewards
"""

import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import pandas as pd
import re
from datetime import datetime


def extract_detailed_metrics_from_log(log_file_path):
    """
    Extract detailed training metrics from the log file
    Returns episode rewards and any available loss information
    """
    log_path = pathlib.Path(log_file_path)
    
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return None, None
    
    print(f"Parsing detailed metrics from {log_path}")
    
    # Data containers
    episode_rewards = []
    training_losses = []
    individual_episodes = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} log lines...")
    
    # Extract episode-level data
    episode_pattern = r"Training — ep (\d+), ts (\d+), avg_reward\(50\)=([-+]?\d*\.?\d+)"
    
    for line_num, line in enumerate(lines):
        # Extract training episode summaries (every 50 episodes)
        match = re.search(episode_pattern, line)
        if match:
            ep_num = int(match.group(1))
            timestep = int(match.group(2))
            avg_reward = float(match.group(3))
            
            episode_rewards.append({
                'episode_number': ep_num,
                'timestep': timestep,
                'avg_reward_50': avg_reward,
                'log_line': line_num + 1
            })
    
    # Convert to DataFrames
    episode_df = pd.DataFrame(episode_rewards) if episode_rewards else None
    loss_df = pd.DataFrame(training_losses) if training_losses else None
    
    return episode_df, loss_df


def create_synthetic_episode_data(episode_summary_df):
    """
    Create individual episode data by interpolating between summary points
    This gives us a more detailed view of the training progress
    """
    if episode_summary_df is None or len(episode_summary_df) == 0:
        return None
    
    detailed_episodes = []
    
    # For each interval between summary points, create individual episode estimates
    for i in range(len(episode_summary_df)):
        current = episode_summary_df.iloc[i]
        
        if i == 0:
            # First batch - assume episodes 1 to current_ep
            start_ep = 1
            start_ts = 0
            start_reward = current['avg_reward_50']  # Estimate
        else:
            prev = episode_summary_df.iloc[i-1] 
            start_ep = prev['episode_number'] + 1
            start_ts = prev['timestep']
            start_reward = prev['avg_reward_50']
        
        end_ep = current['episode_number']
        end_ts = current['timestep']
        end_reward = current['avg_reward_50']
        
        # Interpolate episodes between summary points
        ep_count = int(end_ep - start_ep + 1)
        if ep_count > 0:
            for j in range(ep_count):
                ep_num = start_ep + j
                # Linear interpolation of timestep and reward
                if ep_count == 1:
                    ts = end_ts
                    reward = end_reward
                else:
                    progress = j / (ep_count - 1)
                    ts = int(start_ts + progress * (end_ts - start_ts))
                    # Add some realistic noise to rewards
                    reward = start_reward + progress * (end_reward - start_reward)
                    reward += np.random.normal(0, 0.1)  # Add realistic noise
                
                detailed_episodes.append({
                    'episode': ep_num,
                    'timestep': ts,
                    'episode_reward': reward,
                    'is_interpolated': ep_count > 1
                })
    
    return pd.DataFrame(detailed_episodes)


def create_synthetic_loss_data(episode_df):
    """
    Create synthetic loss data based on training progress
    This is an approximation since actual loss data wasn't logged
    """
    if episode_df is None:
        return None
    
    # Create loss curves that reflect typical PPO training behavior
    timesteps = episode_df['timestep'].values
    
    # Synthetic loss curves (roughly based on typical PPO behavior)
    policy_losses = []
    value_losses = []
    entropy_losses = []
    
    for i, ts in enumerate(timesteps):
        # Policy loss typically starts higher and decreases with oscillations
        base_policy = 0.01 * np.exp(-ts / 200000)  # Exponential decay
        policy_noise = 0.005 * (np.random.random() - 0.5)  # Random oscillations
        policy_loss = -(base_policy + policy_noise)  # Negative because it's a loss
        
        # Value loss typically oscillates around a lower value
        base_value = 0.006 + 0.002 * np.sin(ts / 50000)  # Oscillating base
        value_noise = 0.001 * (np.random.random() - 0.5)
        value_loss = base_value + value_noise
        
        # Entropy decreases over time (exploration -> exploitation)
        base_entropy = -0.7 + 0.2 * np.exp(-ts / 300000)  # Approaches -0.5
        entropy_noise = 0.05 * (np.random.random() - 0.5)
        entropy = base_entropy + entropy_noise
        
        policy_losses.append(policy_loss)
        value_losses.append(value_loss) 
        entropy_losses.append(entropy)
    
    loss_df = pd.DataFrame({
        'timestep': timesteps,
        'policy_loss': policy_losses,
        'value_loss': value_losses,
        'entropy_loss': entropy_losses,
        'update_number': range(1, len(timesteps) + 1),
        'note': 'synthetic_data_based_on_typical_ppo_behavior'
    })
    
    return loss_df


def export_training_csvs(run_dir):
    """
    Main function to extract and export training data as CSV files
    """
    run_path = pathlib.Path(run_dir)
    log_file = run_path / "nepserl.log"
    
    print(f"Exporting training data from: {run_path}")
    print(f"Log file: {log_file}")
    
    # Extract what we can from logs
    episode_summary, loss_data = extract_detailed_metrics_from_log(log_file)
    
    if episode_summary is not None:
        print(f"✓ Found {len(episode_summary)} episode summary points")
        
        # Export episode summary
        summary_path = run_path / "episode_rewards_summary.csv"
        episode_summary.to_csv(summary_path, index=False)
        print(f"✓ Exported episode summary to: {summary_path}")
        
        # Create detailed episode data
        detailed_episodes = create_synthetic_episode_data(episode_summary)
        if detailed_episodes is not None:
            detail_path = run_path / "episode_rewards_detailed.csv"
            detailed_episodes.to_csv(detail_path, index=False)
            print(f"✓ Exported detailed episodes to: {detail_path}")
            print(f"  Contains {len(detailed_episodes)} individual episodes")
        
        # Create synthetic loss data (since actual losses weren't logged)
        synthetic_losses = create_synthetic_loss_data(episode_summary)
        if synthetic_losses is not None:
            loss_path = run_path / "training_losses_synthetic.csv"
            synthetic_losses.to_csv(loss_path, index=False)
            print(f"✓ Exported synthetic loss data to: {loss_path}")
            print(f"  Note: Loss data is synthetic/estimated since actual losses weren't logged")
    
    else:
        print("✗ No episode data found in log file")
    
    # Create metadata file
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'run_directory': str(run_path),
        'log_file': str(log_file),
        'episode_summaries_found': len(episode_summary) if episode_summary is not None else 0,
        'data_source': 'nepserl_log_file',
        'note': 'Detailed episode rewards are interpolated. Loss data is synthetic approximation.'
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_path = run_path / "export_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Exported metadata to: {metadata_path}")
    
    return episode_summary, detailed_episodes, synthetic_losses


def analyze_convergence(episode_df):
    """
    Analyze the training convergence and provide recommendations
    """
    if episode_df is None:
        return
    
    print("\n" + "="*50)
    print("CONVERGENCE ANALYSIS")
    print("="*50)
    
    rewards = episode_df['avg_reward_50'].values
    timesteps = episode_df['timestep'].values
    
    # Basic statistics
    initial_reward = np.mean(rewards[:3]) if len(rewards) >= 3 else rewards[0]
    final_reward = np.mean(rewards[-3:]) if len(rewards) >= 3 else rewards[-1]
    best_reward = np.max(rewards)
    worst_reward = np.min(rewards)
    
    print(f"Training Progress:")
    print(f"  Initial reward (first 3 avg):  {initial_reward:+.4f}")
    print(f"  Final reward (last 3 avg):    {final_reward:+.4f}")
    print(f"  Best reward achieved:          {best_reward:+.4f}")
    print(f"  Worst reward:                  {worst_reward:+.4f}")
    print(f"  Total improvement:             {final_reward - initial_reward:+.4f}")
    print(f"  Total episodes processed:      {episode_df['episode_number'].iloc[-1]}")
    print(f"  Total timesteps:               {timesteps[-1]:,}")
    
    # Convergence assessment
    recent_variance = np.var(rewards[-10:]) if len(rewards) >= 10 else np.var(rewards)
    trend_slope = np.polyfit(range(len(rewards)), rewards, 1)[0]  # Linear trend
    
    print(f"\nConvergence Metrics:")
    print(f"  Recent variance (last 10):     {recent_variance:.6f}")
    print(f"  Overall trend slope:           {trend_slope:+.6f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if abs(trend_slope) < 0.0001:
        print("  ⚠️  CONVERGENCE ISSUE: Training appears to have plateaued")
        print("      - Reward improvement has stagnated")
        print("      - Consider adjusting hyperparameters")
    
    if recent_variance > 0.01:
        print("  ⚠️  HIGH VARIANCE: Rewards are still oscillating significantly")
        print("     - Training may not have converged")
        print("     - Consider longer training or different settings")
    
    if final_reward < -0.5:
        print("  ⚠️  POOR PERFORMANCE: Final rewards are still quite negative")
        print("     - Model may not be learning effectively") 
        print("     - Consider reward function adjustment")
    
    print(f"\nSUGGESTED IMPROVEMENTS:")
    print(f"  1. Increase total timesteps to 1M+ for better convergence")
    print(f"  2. Try different learning rates (current appears to be 3e-4)")
    print(f"  3. Adjust PPO hyperparameters:")
    print(f"     - Reduce learning rate to 1e-4 or 1e-5")
    print(f"     - Increase n_steps (current: 2048) to 4096")
    print(f"     - Try entropy coefficient of 0.01 instead of 0.05")
    print(f"  4. Consider reward engineering or environment modifications")
    print(f"  5. Try different policy network architectures")


if __name__ == "__main__":
    # Set random seed for reproducible synthetic data
    np.random.seed(42)
    
    # Export CSV files for the current run
    run_directory = "C:/Users/krishna/Desktop/nepserl/runs/20260227_172411"
    
    print("NEPSE RL Training Data CSV Exporter")
    print("="*50)
    
    episode_summary, detailed_episodes, synthetic_losses = export_training_csvs(run_directory)
    
    if episode_summary is not None:
        analyze_convergence(episode_summary)
    
    print("\n✓ Export complete! Check the run directory for CSV files:")
    print(f"  📂 {run_directory}")
    print("\nFiles created:")
    print("  📄 episode_rewards_summary.csv    - Episode reward summaries (every 50 episodes)")
    print("  📄 episode_rewards_detailed.csv   - Individual episode rewards (interpolated)")
    print("  📄 training_losses_synthetic.csv  - Synthetic loss curves (approximated)")
    print("  📄 export_metadata.csv           - Export information and metadata")