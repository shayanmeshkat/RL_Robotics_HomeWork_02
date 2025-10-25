import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from dqn_discrete import Grid_World_v0
import save_results
import seaborn as sns
import random
import pickle
import argparse

def run_param_study(params_to_study=None):
    """
    Run hyperparameter study for DQN algorithm.
    
    Args:
        params_to_study: List of parameter names to study. 
                        Options: ['learning_rate', 'gamma', 'hidden_layer_size', 
                                 'replay_memory_size', 'mini_batch', 'epsilon_end']
                        If None, studies all parameters.
    """
    
    os.makedirs('./Results/param_study', exist_ok=True)
    os.makedirs('./Results/param_study/data', exist_ok=True)
    
    # Define hyperparameter ranges to test
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'gamma': [0.8, 0.9, 0.95, 0.99],
        'hidden_layer_size': [12, 24, 48, 96],
        'replay_memory_size': [1000, 2000, 5000],
        'mini_batch': [32, 64, 128],
        'epsilon_end': [0.01, 0.1, 0.3, 0.8]  # Final epsilon value (exploration floor)
    }
    
    # If no params specified, study all
    if params_to_study is None:
        params_to_study = list(param_grid.keys())
    
    # Validate parameter names
    invalid_params = set(params_to_study) - set(param_grid.keys())
    if invalid_params:
        raise ValueError(f"Invalid parameter names: {invalid_params}. "
                        f"Valid options are: {list(param_grid.keys())}")
    
    # Fixed parameters
    num_episodes = 1000
    steps_per_episode = 80
    num_runs = 3
    is_slippery = False
    
    # Store results
    results = []
    
    print(f"\nStudying parameters: {params_to_study}")
    print(f"Total parameter studies: {len(params_to_study)}")
    
    # Test learning rate
    if 'learning_rate' in params_to_study:
        print("\n" + "="*70)
        print("HYPERPARAMETER STUDY: LEARNING RATE")
        print("="*70)
        
        for lr in param_grid['learning_rate']:
            print(f"\nTesting learning_rate = {lr}")
            
            config_rewards = []
            config_test_rewards = []
            
            for run in range(num_runs):
                torch.manual_seed(run + 42)
                np.random.seed(run + 42)
                random.seed(run + 42)
                
                if os.path.exists("grid_world_v0_dqn.pt"):
                    os.remove("grid_world_v0_dqn.pt")
                
                grid_world = Grid_World_v0()
                grid_world.learning_rate = lr
                
                train_rewards, test_rewards = grid_world.train(
                    episodes=num_episodes,
                    steps_per_episode=steps_per_episode,
                    is_slippery=is_slippery
                )
                
                config_rewards.append(train_rewards)
                config_test_rewards.append(test_rewards)
            
            results.append({
                'param': 'learning_rate',
                'value': lr,
                'train_rewards': np.array(config_rewards),
                'test_rewards': config_test_rewards
            })
    
    # Test gamma
    if 'gamma' in params_to_study:
        print("\n" + "="*70)
        print("HYPERPARAMETER STUDY: GAMMA (DISCOUNT FACTOR)")
        print("="*70)
        
        for gamma in param_grid['gamma']:
            print(f"\nTesting gamma = {gamma}")
            
            config_rewards = []
            config_test_rewards = []
            
            for run in range(num_runs):
                torch.manual_seed(run + 42)
                np.random.seed(run + 42)
                random.seed(run + 42)
                
                if os.path.exists("grid_world_v0_dqn.pt"):
                    os.remove("grid_world_v0_dqn.pt")
                
                grid_world = Grid_World_v0()
                grid_world.gamma = gamma
                
                train_rewards, test_rewards = grid_world.train(
                    episodes=num_episodes,
                    steps_per_episode=steps_per_episode,
                    is_slippery=is_slippery
                )
                
                config_rewards.append(train_rewards)
                config_test_rewards.append(test_rewards)
            
            results.append({
                'param': 'gamma',
                'value': gamma,
                'train_rewards': np.array(config_rewards),
                'test_rewards': config_test_rewards
            })
    
    # Test hidden layer size
    if 'hidden_layer_size' in params_to_study:
        print("\n" + "="*70)
        print("HYPERPARAMETER STUDY: HIDDEN LAYER SIZE")
        print("="*70)
        
        for hidden_size in param_grid['hidden_layer_size']:
            print(f"\nTesting hidden_layer_size = {hidden_size}")
            
            config_rewards = []
            config_test_rewards = []
            
            for run in range(num_runs):
                torch.manual_seed(run + 42)
                np.random.seed(run + 42)
                random.seed(run + 42)
                
                if os.path.exists("grid_world_v0_dqn.pt"):
                    os.remove("grid_world_v0_dqn.pt")
                
                grid_world = Grid_World_v0()
                grid_world.hidden_layer_size = hidden_size
                
                train_rewards, test_rewards = grid_world.train(
                    episodes=num_episodes,
                    steps_per_episode=steps_per_episode,
                    is_slippery=is_slippery
                )
                
                config_rewards.append(train_rewards)
                config_test_rewards.append(test_rewards)
            
            results.append({
                'param': 'hidden_layer_size',
                'value': hidden_size,
                'train_rewards': np.array(config_rewards),
                'test_rewards': config_test_rewards
            })
    
    # Test replay memory size
    if 'replay_memory_size' in params_to_study:
        print("\n" + "="*70)
        print("HYPERPARAMETER STUDY: REPLAY MEMORY SIZE")
        print("="*70)
        
        for mem_size in param_grid['replay_memory_size']:
            print(f"\nTesting replay_memory_size = {mem_size}")
            
            config_rewards = []
            config_test_rewards = []
            
            for run in range(num_runs):
                torch.manual_seed(run + 42)
                np.random.seed(run + 42)
                random.seed(run + 42)
                
                if os.path.exists("grid_world_v0_dqn.pt"):
                    os.remove("grid_world_v0_dqn.pt")
                
                grid_world = Grid_World_v0()
                grid_world.replay_memory_size = mem_size
                
                train_rewards, test_rewards = grid_world.train(
                    episodes=num_episodes,
                    steps_per_episode=steps_per_episode,
                    is_slippery=is_slippery
                )
                
                config_rewards.append(train_rewards)
                config_test_rewards.append(test_rewards)
            
            results.append({
                'param': 'replay_memory_size',
                'value': mem_size,
                'train_rewards': np.array(config_rewards),
                'test_rewards': config_test_rewards
            })
    
    # Test mini batch size
    if 'mini_batch' in params_to_study:
        print("\n" + "="*70)
        print("HYPERPARAMETER STUDY: MINI BATCH SIZE")
        print("="*70)
        
        for batch_size in param_grid['mini_batch']:
            print(f"\nTesting mini_batch = {batch_size}")
            
            config_rewards = []
            config_test_rewards = []
            
            for run in range(num_runs):
                torch.manual_seed(run + 42)
                np.random.seed(run + 42)
                random.seed(run + 42)
                
                if os.path.exists("grid_world_v0_dqn.pt"):
                    os.remove("grid_world_v0_dqn.pt")
                
                grid_world = Grid_World_v0()
                grid_world.mini_batch = batch_size
                
                train_rewards, test_rewards = grid_world.train(
                    episodes=num_episodes,
                    steps_per_episode=steps_per_episode,
                    is_slippery=is_slippery
                )
                
                config_rewards.append(train_rewards)
                config_test_rewards.append(test_rewards)
            
            results.append({
                'param': 'mini_batch',
                'value': batch_size,
                'train_rewards': np.array(config_rewards),
                'test_rewards': config_test_rewards
            })
    
    # Test epsilon end value
    if 'epsilon_end' in params_to_study:
        print("\n" + "="*70)
        print("HYPERPARAMETER STUDY: EPSILON END (EXPLORATION FLOOR)")
        print("="*70)
        
        for eps_end in param_grid['epsilon_end']:
            print(f"\nTesting epsilon_end = {eps_end}")
            
            config_rewards = []
            config_test_rewards = []
            
            for run in range(num_runs):
                torch.manual_seed(run + 42)
                np.random.seed(run + 42)
                random.seed(run + 42)
                
                if os.path.exists("grid_world_v0_dqn.pt"):
                    os.remove("grid_world_v0_dqn.pt")
                
                grid_world = Grid_World_v0()
                grid_world.epsilon_end = eps_end
                
                train_rewards, test_rewards = grid_world.train(
                    episodes=num_episodes,
                    steps_per_episode=steps_per_episode,
                    is_slippery=is_slippery
                )
                
                config_rewards.append(train_rewards)
                config_test_rewards.append(test_rewards)
            
            results.append({
                'param': 'epsilon_end',
                'value': eps_end,
                'train_rewards': np.array(config_rewards),
                'test_rewards': config_test_rewards
            })
    
    return results, param_grid


def save_training_data(results, param_grid):
    """
    Save all training data to files for later plotting.
    """
    
    # Save complete results as pickle
    with open('./Results/param_study/data/complete_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'param_grid': param_grid}, f)
    print("\nSaved complete results to: ./Results/param_study/data/complete_results.pkl")
    
    # Save individual parameter data as numpy arrays
    for result in results:
        param_name = result['param']
        value = result['value']
        train_rewards = result['train_rewards']
        test_rewards = result['test_rewards']
        
        # Create filename
        filename = f"{param_name}_{value}"
        
        # Save training rewards
        np.save(f'./Results/param_study/data/{filename}_train.npy', train_rewards)
        
        # Save test rewards
        test_rewards_array = np.array(test_rewards, dtype=object)
        np.save(f'./Results/param_study/data/{filename}_test.npy', test_rewards_array, allow_pickle=True)
    
    print("Saved individual parameter data to: ./Results/param_study/data/")
    
    # Save metadata
    metadata = {
        'param_grid': param_grid,
        'param_names': list(set([r['param'] for r in results])),
        'description': 'DQN Hyperparameter Study Results'
    }
    with open('./Results/param_study/data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("Saved metadata to: ./Results/param_study/data/metadata.pkl")


def plot_training_curves_only(results):
    """
    Create training curve plots only (no bar charts) for each hyperparameter.
    """
    
    # Group results by parameter type
    param_groups = {}
    for result in results:
        param_name = result['param']
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(result)
    
    # Create a separate figure for each parameter showing only training curves
    for param_name, param_results in param_groups.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Training Curves: {param_name.upper()}', fontsize=16, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_results)))
        
        for idx, result in enumerate(param_results):
            value = result['value']
            train_rewards = result['train_rewards']
            
            # Calculate percentiles across runs (axis=0 is across different runs)
            percentile_25 = np.percentile(train_rewards, 25, axis=0)
            percentile_50 = np.percentile(train_rewards, 50, axis=0)  # Median
            percentile_75 = np.percentile(train_rewards, 75, axis=0)
            
            # Rolling average for smoothing
            window = 20
            rolling_median = pd.Series(percentile_50).rolling(window=window, min_periods=1).mean()
            rolling_p25 = pd.Series(percentile_25).rolling(window=window, min_periods=1).mean()
            rolling_p75 = pd.Series(percentile_75).rolling(window=window, min_periods=1).mean()
            
            episodes = np.arange(len(rolling_median))
            
            # Plot median with shaded percentile range
            ax.plot(episodes, rolling_median, label=f'{param_name}={value}', 
                   linewidth=2.5, color=colors[idx])
            ax.fill_between(episodes, 
                           rolling_p25, 
                           rolling_p75, 
                           alpha=0.2, color=colors[idx])
        
        ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
        ax.set_ylabel('Training Reward (Median)', fontsize=14, fontweight='bold')
        ax.set_title(f'Training Performance Comparison', fontsize=14)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add a text box with statistics
        final_rewards = []
        for result in param_results:
            final_perf = np.median(result['train_rewards'][:, -50:])
            final_rewards.append(final_perf)
        
        best_idx = np.argmax(final_rewards)
        best_value = param_results[best_idx]['value']
        best_reward = final_rewards[best_idx]
        
        textstr = f'Best {param_name}: {best_value}\nFinal Median Reward: {best_reward:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            #    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'./Results/param_study/{param_name}_training_curves.pdf', 
                   dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved training curves: ./Results/param_study/{param_name}_training_curves.pdf")
        plt.close()


def plot_results(results):
    """
    Create comprehensive visualizations of hyperparameter study results.
    """
    
    # Group results by parameter type
    param_groups = {}
    for result in results:
        param_name = result['param']
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(result)
    
    # Create a figure for each parameter
    for param_name, param_results in param_groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Hyperparameter Study: {param_name.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Training curves with percentiles
        ax1 = axes[0, 0]
        for result in param_results:
            value = result['value']
            train_rewards = result['train_rewards']
            
            # Calculate percentiles across runs
            percentile_25 = np.percentile(train_rewards, 25, axis=0)
            percentile_50 = np.percentile(train_rewards, 50, axis=0)  # Median
            percentile_75 = np.percentile(train_rewards, 75, axis=0)
            
            # Rolling average
            window = 20
            rolling_median = pd.Series(percentile_50).rolling(window=window, min_periods=1).mean()
            rolling_p25 = pd.Series(percentile_25).rolling(window=window, min_periods=1).mean()
            rolling_p75 = pd.Series(percentile_75).rolling(window=window, min_periods=1).mean()
            
            episodes = np.arange(len(rolling_median))
            ax1.plot(episodes, rolling_median, label=f'{param_name}={value}', linewidth=2)
            ax1.fill_between(episodes, 
                            rolling_p25, 
                            rolling_p75, 
                            alpha=0.2)
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Training Reward (Median)', fontsize=12)
        ax1.set_title('Training Performance', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Test performance
        ax2 = axes[0, 1]
        test_means = []
        test_stds = []
        labels = []
        
        for result in param_results:
            value = result['value']
            test_rewards = result['test_rewards']
            
            # Flatten test rewards
            all_test_rewards = []
            for run_tests in test_rewards:
                if isinstance(run_tests, list):
                    all_test_rewards.extend(run_tests)
                else:
                    all_test_rewards.append(run_tests)
            
            test_means.append(np.mean(all_test_rewards))
            test_stds.append(np.std(all_test_rewards))
            labels.append(f'{value}')
        
        x_pos = np.arange(len(labels))
        ax2.bar(x_pos, test_means, yerr=test_stds, capsize=5, alpha=0.7)
        ax2.set_xlabel(f'{param_name}', fontsize=12)
        ax2.set_ylabel('Average Test Reward', fontsize=12)
        ax2.set_title('Test Performance', fontsize=14)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Final performance comparison
        ax3 = axes[1, 0]
        final_train_medians = []
        final_train_p25 = []
        final_train_p75 = []
        
        for result in param_results:
            value = result['value']
            train_rewards = result['train_rewards']
            
            # Percentiles of last 50 episodes across runs
            last_50 = train_rewards[:, -50:]
            final_train_medians.append(np.median(last_50))
            final_train_p25.append(np.percentile(last_50, 25))
            final_train_p75.append(np.percentile(last_50, 75))
        
        x_pos = np.arange(len(labels))
        ax3.bar(x_pos, final_train_medians, 
               yerr=[np.array(final_train_medians) - np.array(final_train_p25),
                     np.array(final_train_p75) - np.array(final_train_medians)],
               capsize=5, alpha=0.7, color='green')
        ax3.set_xlabel(f'{param_name}', fontsize=12)
        ax3.set_ylabel('Final Training Reward (Median, Last 50 Episodes)', fontsize=12)
        ax3.set_title('Final Training Performance', fontsize=14)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Learning stability (coefficient of variation)
        ax4 = axes[1, 1]
        cv_values = []
        
        for result in param_results:
            train_rewards = result['train_rewards']
            # Calculate coefficient of variation for last 100 episodes
            last_episodes = train_rewards[:, -100:]
            cv = np.std(last_episodes) / (np.abs(np.mean(last_episodes)) + 1e-8)
            cv_values.append(cv)
        
        x_pos = np.arange(len(labels))
        ax4.bar(x_pos, cv_values, alpha=0.7, color='orange')
        ax4.set_xlabel(f'{param_name}', fontsize=12)
        ax4.set_ylabel('Coefficient of Variation', fontsize=12)
        ax4.set_title('Learning Stability (Lower is Better)', fontsize=14)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'./Results/param_study/{param_name}_study.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot: ./Results/param_study/{param_name}_study.png")
        plt.close()


def save_summary_statistics(results):
    """
    Save summary statistics to CSV file.
    """
    
    summary_data = []
    
    for result in results:
        param_name = result['param']
        value = result['value']
        train_rewards = result['train_rewards']
        test_rewards = result['test_rewards']
        
        # Calculate metrics
        final_train = np.mean(train_rewards[:, -50:])
        final_train_std = np.std(train_rewards[:, -50:])
        
        # Flatten test rewards
        all_test_rewards = []
        for run_tests in test_rewards:
            if isinstance(run_tests, list):
                all_test_rewards.extend(run_tests)
            else:
                all_test_rewards.append(run_tests)
        
        avg_test = np.mean(all_test_rewards)
        std_test = np.std(all_test_rewards)
        
        summary_data.append({
            'Parameter': param_name,
            'Value': value,
            'Final_Train_Mean': final_train,
            'Final_Train_Std': final_train_std,
            'Test_Mean': avg_test,
            'Test_Std': std_test,
            'CV': np.std(train_rewards[:, -100:]) / (np.abs(np.mean(train_rewards[:, -100:])) + 1e-8)
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv('./Results/param_study/summary_statistics.csv', index=False)
    print("\nSummary Statistics:")
    print(df.to_string())
    print("\nSaved summary to: ./Results/param_study/summary_statistics.csv")


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run DQN Hyperparameter Study')
    parser.add_argument('--params', nargs='+', 
                       choices=['learning_rate', 'gamma', 'hidden_layer_size', 
                               'replay_memory_size', 'mini_batch', 'epsilon_end', 'all'],
                       default=['all'],
                       help='Hyperparameters to study (default: all)')
    
    args = parser.parse_args()
    
    # Determine which parameters to study
    if 'all' in args.params:
        params_to_study = None  # Study all parameters
    else:
        params_to_study = args.params
    
    print("="*70)
    print("DQN HYPERPARAMETER STUDY")
    print("="*70)
    
    if params_to_study:
        print(f"Selected parameters: {', '.join(params_to_study)}")
    else:
        print("Selected parameters: ALL")
    
    # Run the parameter study
    results, param_grid = run_param_study(params_to_study)
    
    # Save training data
    print("\nSaving training data...")
    save_training_data(results, param_grid)
    
    # Plot training curves only (no bar charts)
    print("\nGenerating training curve plots...")
    plot_training_curves_only(results)
    
    # Plot complete results with bar charts
    print("\nGenerating complete analysis plots...")
    plot_results(results)
    
    # Save summary statistics
    save_summary_statistics(results)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER STUDY COMPLETE")
    print("="*70)
