#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

from matplotlib import rcParams

# Set up consistent matplotlib styling
def setup_matplotlib_styling():
    """Configure matplotlib with consistent styling without LaTeX"""
    # Disable LaTeX rendering
    rcParams['text.usetex'] = False
    
    # Keep the rest of the styling
    # Font configuration
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['figure.titlesize'] = 18
    
    # Figure settings
    rcParams['figure.figsize'] = (10, 6)
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.format'] = 'pdf'
   
    plt.style.use('seaborn-v0_8-whitegrid')

def analyze_results(csv_file, output_dir="analysis"):
    """
    Analyze the plugin timing data and generate visualizations
    
    Args:
        csv_file: Path to the plugin timing summary CSV file
        output_dir: Directory to store analysis results
    """
    
    # Set up matplotlib styling
    setup_matplotlib_styling()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: Results file {csv_file} not found")
        return False
    
    # Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Ensure the required columns exist
    required_columns = ['proposal', 'avg_time_per_iter_ms']
    stddev_columns = ['stddev_time_ms', 'stddev_plugin_ms']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Required columns not found in {csv_file}: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    # Check if we have standard deviation columns
    has_stddev = all(col in df.columns for col in stddev_columns)
    if not has_stddev:
        print("Warning: Standard deviation columns not found, visualizations will not include error bars")
    
    # Set proposal as the index for better plotting
    df.set_index('proposal', inplace=True)
    
    # Calculate statistics
    stats = pd.DataFrame()
    for column in df.columns:
        if column != 'proposal':  # Skip the proposal column
            stats[column] = [
                df[column].mean(),
                df[column].std(),
                df[column].min(),
                df[column].max()
            ]
    
    stats.index = ["Mean", "Std Dev", "Min", "Max"]
    stats.to_csv(f"{output_dir}/timing_statistics.csv")
    
    # Generate visualizations
    
    # 1. Bar chart of average time per iteration with standard deviation
    plt.figure(figsize=(10, 6))
    ax = df['avg_time_per_iter_ms'].plot(kind="bar", color='skyblue', yerr=df['stddev_time_ms'] if has_stddev else None)
    plt.title("Average Time per Iteration by Proposal")
    plt.ylabel("Time (milliseconds)")
    plt.xlabel("Proposal")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add values on top of bars
    for i, v in enumerate(df['avg_time_per_iter_ms']):
        text = f"{v:.2f}"
        if has_stddev:
            text += f"\n±{df['stddev_time_ms'].iloc[i]:.2f}"
        ax.text(i, v + (max(df['avg_time_per_iter_ms']) * 0.02), text, 
                horizontalalignment='center', fontsize=9)
    
    plt.savefig(f"{output_dir}/avg_time_per_iter.pdf")
    
    # 2. If there are multiple metrics to compare, plot them side by side
    if all(col in df.columns for col in ['avg_time_per_iter_ms', 'avg_time_plugin_ms']):
        plt.figure(figsize=(12, 7))
        
        # For grouped bar charts with error bars, use regular bar plot
        x = np.arange(len(df.index))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot first set of bars
        rects1 = ax.bar(x - width/2, df['avg_time_per_iter_ms'], width, 
                        label='Avg Time per Iteration', color='cornflowerblue',
                        yerr=df['stddev_time_ms'] if has_stddev else None)
        
        # Plot second set of bars
        rects2 = ax.bar(x + width/2, df['avg_time_plugin_ms'], width, 
                        label='Avg Time per Algorithm', color='lightcoral',
                        yerr=df['stddev_plugin_ms'] if has_stddev else None)
        
        # Add labels, title and legend
        ax.set_xlabel('Proposal')
        ax.set_ylabel('Time (milliseconds)')
        ax.set_title('Comparison of Average Timing Metrics with Standard Deviation')
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.legend()
        
        # Add value annotations
        def autolabel(rects, stddevs=None):
            for i, rect in enumerate(rects):
                height = rect.get_height()
                text = f"{height:.2f}"
                if stddevs is not None:
                    text += f"\n±{stddevs.iloc[i]:.2f}"
                ax.annotate(text,
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1, df['stddev_time_ms'] if has_stddev else None)
        autolabel(rects2, df['stddev_plugin_ms'] if has_stddev else None)
        
        fig.tight_layout()
        plt.savefig(f"{output_dir}/timing_metrics_comparison.pdf")
    
    # 3. If total metrics are available, plot those too
    if all(col in df.columns for col in ['total_time_ms', 'total_time_plugin_ms']):
        plt.figure(figsize=(12, 7))
        
        # For grouped bar charts with error bars, use regular bar plot
        x = np.arange(len(df.index))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot first set of bars
        rects1 = ax.bar(x - width/2, df['total_time_ms'], width, 
                        label='Total Sum of Times', color='darkblue')
        
        # Plot second set of bars
        rects2 = ax.bar(x + width/2, df['total_time_plugin_ms'], width, 
                        label='Total Plugin Time (First to Last)', color='darkred')
        
        # Add labels, title and legend
        ax.set_xlabel('Proposal')
        ax.set_ylabel('Time (milliseconds)')
        ax.set_title('Total Time Metrics by Proposal')
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.legend()
        
        # Add value annotations
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(f"{output_dir}/total_timing_metrics.pdf")
    
    # 4. Statistical table visualization
    if has_stddev:
        plt.figure(figsize=(12, 6))
        # Extract key metrics for table
        table_data = pd.DataFrame({
            'Avg Time (ms)': df['avg_time_per_iter_ms'],
            'Std Dev (ms)': df['stddev_time_ms'],
            'CoV (%)': (df['stddev_time_ms'] / df['avg_time_per_iter_ms'] * 100)
        })
        
        # Round values for display
        table_data = table_data.round(2)
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.8 + 1))
        ax.axis('off')
        ax.axis('tight')
        
        table = ax.table(
            cellText=table_data.values,
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title("Statistical Summary of Plugin Timing Measurements")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_statistics_table.pdf")
    
    # Generate report
    with open(f"{output_dir}/timing_analysis_report.txt", "w") as f:
        f.write("StrongSwan QKD Plugin Timing Analysis\n")
        f.write("====================================\n\n")
        
        f.write("Statistical Summary:\n")
        f.write("-------------------\n")
        f.write(stats.to_string())
        f.write("\n\n")
        
        # Compare proposals for average time per iteration
        best_proposal = df['avg_time_per_iter_ms'].idxmin()
        worst_proposal = df['avg_time_per_iter_ms'].idxmax()
        
        f.write(f"Fastest proposal (lowest avg time): {best_proposal}\n")
        f.write(f"Average time per iteration: {df.loc[best_proposal, 'avg_time_per_iter_ms']:.3f} ms\n")
        if has_stddev:
            f.write(f"Standard deviation: {df.loc[best_proposal, 'stddev_time_ms']:.3f} ms\n")
            f.write(f"Coefficient of variation: {(df.loc[best_proposal, 'stddev_time_ms'] / df.loc[best_proposal, 'avg_time_per_iter_ms'] * 100):.2f}%\n")
        f.write("\n")
        
        f.write(f"Slowest proposal (highest avg time): {worst_proposal}\n")
        f.write(f"Average time per iteration: {df.loc[worst_proposal, 'avg_time_per_iter_ms']:.3f} ms\n")
        if has_stddev:
            f.write(f"Standard deviation: {df.loc[worst_proposal, 'stddev_time_ms']:.3f} ms\n")
            f.write(f"Coefficient of variation: {(df.loc[worst_proposal, 'stddev_time_ms'] / df.loc[worst_proposal, 'avg_time_per_iter_ms'] * 100):.2f}%\n")
        f.write("\n")
        
        # Performance comparison
        if worst_proposal != best_proposal:
            perf_ratio = df.loc[worst_proposal, 'avg_time_per_iter_ms'] / df.loc[best_proposal, 'avg_time_per_iter_ms']
            f.write(f"Performance ratio (slowest/fastest): {perf_ratio:.2f}x\n\n")
        
        f.write("Generated visualizations:\n")
        f.write(f"1. {output_dir}/avg_time_per_iter.pdf - Average time per iteration across proposals\n")
        f.write(f"2. {output_dir}/timing_statistics.csv - Detailed timing statistics\n")
        
        if all(col in df.columns for col in ['avg_time_per_iter_ms', 'avg_time_plugin_ms']):
            f.write(f"3. {output_dir}/timing_metrics_comparison.pdf - Comparison of average timing metrics\n")
        
        if all(col in df.columns for col in ['total_time_ms', 'total_time_plugin_ms']):
            f.write(f"4. {output_dir}/total_timing_metrics.pdf - Total timing metrics by proposal\n")
        
        if has_stddev:
            f.write(f"5. {output_dir}/timing_statistics_table.pdf - Statistical summary table\n")
    
    print(f"Analysis completed. Results saved to {output_dir}/")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <plugin_timing_file> [<output_dir>]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis"
    
    analyze_results(csv_file, output_dir)