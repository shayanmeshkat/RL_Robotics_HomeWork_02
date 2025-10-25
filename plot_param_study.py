import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import seaborn as sns


def load_training_data():
    """
    Load saved training data from param_study.
    """
    
    data_dir = './Results/param_study/data'
    
    if not os.path.exists(f'{data_dir}/complete_results.pkl'):
        raise FileNotFoundError("No saved training data found. Run param_study.py first.")
    
    # Load complete results
    with open(f'{data_dir}/complete_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    results = data['results']
    param_grid = data['param_grid']
    
    print("Loaded training data successfully!")
    print(f"Parameters studied: {list(param_grid.keys())}")
    
    return results, param_grid


def plot_training_curves_only(results, save_dir='./Results/param_study'):
    """
    Create training curve plots only (no bar charts) for each hyperparameter.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
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
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{param_name}_training_curves.pdf', 
                   dpi=300, format='pdf', bbox_inches='tight')
        plt.savefig(f'{save_dir}/{param_name}_training_curves.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Saved training curves: {save_dir}/{param_name}_training_curves.pdf/.png")
        plt.close()


def plot_complete_analysis(results, save_dir='./Results/param_study'):
    """
    Create comprehensive visualizations with 4 subplots for each hyperparameter.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
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
        plt.savefig(f'{save_dir}/{param_name}_complete_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved complete analysis: {save_dir}/{param_name}_complete_analysis.png")
        plt.close()


def plot_comparison_heatmap(results, save_dir='./Results/param_study'):
    """
    Create heatmap comparing all hyperparameters.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for heatmap
    param_names = []
    param_values = []
    final_rewards = []
    
    for result in results:
        param_name = result['param']
        value = result['value']
        train_rewards = result['train_rewards']
        
        param_names.append(f"{param_name}\n{value}")
        final_perf = np.median(train_rewards[:, -50:])
        final_rewards.append(final_perf)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group by parameter type
    param_groups = {}
    for i, result in enumerate(results):
        param_name = result['param']
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append((result['value'], final_rewards[i]))
    
    # Create grouped bar chart
    x_offset = 0
    x_ticks = []
    x_labels = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(param_groups)))
    
    for idx, (param_name, values) in enumerate(param_groups.items()):
        values_sorted = sorted(values, key=lambda x: x[0])
        param_vals = [v[0] for v in values_sorted]
        rewards = [v[1] for v in values_sorted]
        
        x_positions = np.arange(len(param_vals)) + x_offset
        ax.bar(x_positions, rewards, label=param_name, alpha=0.8, color=colors[idx])
        
        x_ticks.extend(x_positions)
        x_labels.extend([str(v) for v in param_vals])
        x_offset += len(param_vals) + 1
    
    ax.set_xlabel('Hyperparameter Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Training Reward (Median, Last 50 Episodes)', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {save_dir}/hyperparameter_comparison.png")
    plt.close()


def print_summary_statistics(results):
    """
    Print summary statistics for all hyperparameters.
    """
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for result in results:
        param_name = result['param']
        value = result['value']
        train_rewards = result['train_rewards']
        
        final_median = np.median(train_rewards[:, -50:])
        final_p25 = np.percentile(train_rewards[:, -50:], 25)
        final_p75 = np.percentile(train_rewards[:, -50:], 75)
        
        print(f"\n{param_name} = {value}")
        print(f"  Final Reward (Median): {final_median:.2f}")
        print(f"  Final Reward (25th percentile): {final_p25:.2f}")
        print(f"  Final Reward (75th percentile): {final_p75:.2f}")
        print(f"  IQR: {final_p75 - final_p25:.2f}")


def plot_all_training_curves_combined(results, save_dir='./Results/param_study'):
    """
    Create a single figure with all 5 hyperparameter training curves in subplots.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Group results by parameter type
    param_groups = {}
    for result in results:
        param_name = result['param']
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(result)
    
    # Create figure with 5 subplots (1 row, 5 columns or 2 rows)
    n_params = len(param_groups)
    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    axes = axes.flatten()
    
    fig.suptitle('DQN Hyperparameter Study: Training Curves Comparison', 
                 fontsize=36, fontweight='bold', y=0.995)
    
    # Plot each parameter in a subplot
    for idx, (param_name, param_results) in enumerate(param_groups.items()):
        ax = axes[idx]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_results)))
        
        for p_idx, result in enumerate(param_results):
            value = result['value']
            train_rewards = result['train_rewards']
            
            # Calculate percentiles across runs
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
            ax.plot(episodes, rolling_median, label=f'{value}', 
                   linewidth=2.5, color=colors[p_idx])
            ax.fill_between(episodes, 
                           rolling_p25, 
                           rolling_p75, 
                           alpha=0.2, color=colors[p_idx])
        
        # Format subplot
        ax.set_xlabel('Episode', fontsize=32, fontweight='bold')
        ax.set_ylabel('Training Reward', fontsize=32, fontweight='bold')
        ax.tick_params(axis='both', labelsize=32)
        ax.set_title(f'{param_name.replace("_", " ").title()}', fontsize=32, fontweight='bold')
        ax.legend(loc='best', fontsize=32, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide the extra subplot if we have fewer than 6 parameters
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_training_curves_combined.pdf', 
               dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/all_training_curves_combined.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved combined training curves: {save_dir}/all_training_curves_combined.pdf/.png")
    plt.close()
    exit()


def plot_all_training_curves_single_row(results, save_dir='./Results/param_study'):
    """
    Create a single figure with all 5 hyperparameter training curves in a single row.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Group results by parameter type
    param_groups = {}
    for result in results:
        param_name = result['param']
        if param_name not in param_groups:
            param_groups[param_name] = []
        param_groups[param_name].append(result)
    
    # Create figure with subplots in a single row
    n_params = len(param_groups)
    fig, axes = plt.subplots(1, n_params, figsize=(24, 5))
    
    fig.suptitle('DQN Hyperparameter Study: Training Curves Comparison', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Plot each parameter in a subplot
    for idx, (param_name, param_results) in enumerate(param_groups.items()):
        ax = axes[idx]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(param_results)))
        
        for p_idx, result in enumerate(param_results):
            value = result['value']
            train_rewards = result['train_rewards']
            
            # Calculate percentiles across runs
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
            ax.plot(episodes, rolling_median, label=f'{value}', 
                   linewidth=2.0, color=colors[p_idx])
            ax.fill_between(episodes, 
                           rolling_p25, 
                           rolling_p75, 
                           alpha=0.2, color=colors[p_idx])
        
        # Format subplot
        ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Training Reward (Median)', fontsize=11, fontweight='bold')
        ax.set_title(f'{param_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_training_curves_row.pdf', 
               dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/all_training_curves_row.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved combined training curves (row): {save_dir}/all_training_curves_row.pdf/.png")
    plt.close()


def plot_epsilon_study_standalone(results, save_dir='./Results/param_study'):
    """
    Create a standalone plot for epsilon end hyperparameter study.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter only epsilon_end results
    epsilon_results = [r for r in results if r['param'] == 'epsilon_end']
    
    if not epsilon_results:
        print("No epsilon_end results found in data.")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_results)))
    
    for idx, result in enumerate(epsilon_results):
        value = result['value']
        train_rewards = result['train_rewards']
        
        # Calculate percentiles across runs
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
        ax.plot(episodes, rolling_median, 
               label=f'ε-end = {value}', 
               linewidth=3.0, 
               color=colors[idx])
        ax.fill_between(episodes, 
                       rolling_p25, 
                       rolling_p75, 
                       alpha=0.25, 
                       color=colors[idx])
    
    # Format plot
    ax.set_xlabel('Episode', fontsize=32, fontweight='bold')
    ax.set_ylabel('Training Reward', fontsize=32, fontweight='bold')
    ax.set_title('Effect of Epsilon-Greedy Exploration on Learning', 
                fontsize=32, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=20, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.tick_params(axis='both', labelsize=20)

    # Add statistics text box
    final_rewards = []
    for result in epsilon_results:
        final_perf = np.median(result['train_rewards'][:, -50:])
        final_rewards.append(final_perf)
    
    best_idx = np.argmax(final_rewards)
    best_value = epsilon_results[best_idx]['value']
    best_reward = final_rewards[best_idx]
    
    # textstr = f'Best Min Epsilon: {best_value}\nFinal Median Reward: {best_reward:.2f}'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=13,
    #        verticalalignment='top', bbox=props, fontweight='bold')
    
    # # Add annotation about exploration-exploitation tradeoff
    # annotation_text = ('Lower ε-end → Less exploration\n'
    #                   'Higher ε-end → More exploration maintained')
    # ax.text(0.98, 0.02, annotation_text, transform=ax.transAxes, 
    #        fontsize=11, verticalalignment='bottom', horizontalalignment='right',
    #        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
    #        style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epsilon_end_study_standalone.pdf', 
               dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/epsilon_end_study_standalone.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved epsilon study standalone: {save_dir}/epsilon_end_study_standalone.pdf/.png")
    plt.close()


def plot_epsilon_with_decay_visualization(results, save_dir='./Results/param_study'):
    """
    Create a figure showing both training curves and epsilon decay curves with different end values.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter only epsilon_end results
    epsilon_results = [r for r in results if r['param'] == 'epsilon_end']
    
    if not epsilon_results:
        print("No epsilon_end results found in data.")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle('Epsilon-Greedy Exploration Strategy: Minimum Epsilon Analysis', 
                 fontsize=20, fontweight='bold', y=1.00)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_results)))
    
    # Plot 1: Training curves
    for idx, result in enumerate(epsilon_results):
        value = result['value']
        train_rewards = result['train_rewards']
        
        # Calculate percentiles across runs
        percentile_25 = np.percentile(train_rewards, 25, axis=0)
        percentile_50 = np.percentile(train_rewards, 50, axis=0)
        percentile_75 = np.percentile(train_rewards, 75, axis=0)
        
        # Rolling average for smoothing
        window = 20
        rolling_median = pd.Series(percentile_50).rolling(window=window, min_periods=1).mean()
        rolling_p25 = pd.Series(percentile_25).rolling(window=window, min_periods=1).mean()
        rolling_p75 = pd.Series(percentile_75).rolling(window=window, min_periods=1).mean()
        
        episodes = np.arange(len(rolling_median))
        
        ax1.plot(episodes, rolling_median, 
                label=f'ε-end = {value}', 
                linewidth=2.5, 
                color=colors[idx])
        ax1.fill_between(episodes, 
                        rolling_p25, 
                        rolling_p75, 
                        alpha=0.2, 
                        color=colors[idx])
    
    ax1.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Training Reward (Median)', fontsize=14, fontweight='bold')
    ax1.set_title('Training Performance', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=12)
    
    # Plot 2: Epsilon decay curves with different end values
    num_episodes = 1000
    episodes = np.arange(num_episodes)
    decay_rate = 0.005  # Default decay rate
    
    for idx, result in enumerate(epsilon_results):
        eps_end = result['value']
        
        # Calculate epsilon values over episodes
        eps_start = 1.0
        epsilon_values = [eps_end + (eps_start - eps_end) * np.exp(-decay_rate * ep) 
                         for ep in episodes]
        
        ax2.plot(episodes, epsilon_values, 
                label=f'ε-end = {eps_end}', 
                linewidth=2.5, 
                color=colors[idx])
    
    ax2.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Epsilon (ε) Value', fontsize=14, fontweight='bold')
    ax2.set_title('Exploration Rate Decay to Different Floors', fontsize=16, fontweight='bold')
    ax2.legend(loc='best', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylim([0, 1.05])
    
    # Add horizontal lines for reference
    ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1.5, label='50% exploration')
    ax2.axhline(y=0.1, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='10% exploration')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/epsilon_end_with_curves.pdf', 
               dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/epsilon_end_with_curves.png', 
               dpi=300, bbox_inches='tight')
    print(f"Saved epsilon study with decay curves: {save_dir}/epsilon_end_with_curves.pdf/.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("PLOTTING HYPERPARAMETER STUDY RESULTS")
    print("="*70)
    
    # Load saved training data
    print("\nLoading training data...")
    results, param_grid = load_training_data()
    
    # Plot epsilon decay study standalone
    print("\nGenerating epsilon decay standalone plot...")
    plot_epsilon_study_standalone(results)
    plot_epsilon_with_decay_visualization(results)
    
    # Plot all training curves combined in subplots
    print("\nGenerating combined training curves plot...")
    plot_all_training_curves_combined(results)
    plot_all_training_curves_single_row(results)
    
    # Plot training curves only
    print("\nGenerating training curve plots...")
    plot_training_curves_only(results)
    
    # Plot complete analysis
    print("\nGenerating complete analysis plots...")
    plot_complete_analysis(results)
    
    # Plot comparison heatmap
    print("\nGenerating comparison plot...")
    plot_comparison_heatmap(results)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    print("\n" + "="*70)
    print("PLOTTING COMPLETE")
    print("="*70)
    print("\nAll plots saved to: ./Results/param_study/")
