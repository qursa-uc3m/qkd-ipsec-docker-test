#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from matplotlib import rcParams

# Configure matplotlib styling
def setup_matplotlib_styling():
    """Configure matplotlib with consistent styling without LaTeX"""
    rcParams['text.usetex'] = False
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['figure.titlesize'] = 18
    rcParams['figure.figsize'] = (10, 6)
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.format'] = 'pdf'
    plt.style.use('seaborn-v0_8-whitegrid')

def load_and_validate_data(csv_file, required_columns):
    """
    Load data from CSV and validate required columns
    
    Args:
        csv_file: Path to the CSV file
        required_columns: List of required column names
        
    Returns:
        DataFrame or None if validation fails
    """
    if not os.path.exists(csv_file):
        print(f"Error: Results file {csv_file} not found")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # Validate required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Required columns not found in {csv_file}: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Set proposal as index for plotting
    df.set_index('proposal', inplace=True)
    return df

def create_bar_chart(df, y_column, title, ylabel, output_path, 
                     yerr_column=None, log_scale=False, annotations=True):
    """
    Create a bar chart from DataFrame
    
    Args:
        df: DataFrame with data
        y_column: Column name for y-axis values
        title: Chart title
        ylabel: Y-axis label
        output_path: Path to save the chart
        yerr_column: Column name for error bars (optional)
        log_scale: Whether to use logarithmic scale for y-axis
        annotations: Whether to add value annotations to bars
    """
    plt.figure(figsize=(12, 7))
    ax = df[y_column].plot(kind="bar", color='skyblue', yerr=df[yerr_column] if yerr_column else None)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Proposal")
    plt.xticks(rotation=45, ha='right')

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    
    # Add value annotations if requested
    if annotations:
        for i, v in enumerate(df[y_column]):
            text = f"{v:.2f}"
            if yerr_column:
                text += f"\n±{df[yerr_column].iloc[i]:.2f}"
            
            # Adjust position based on scale type
            if log_scale:
                y_pos = v * 1.1  # Multiplicative position for log scale
            else:
                y_pos = v + (max(df[y_column]) * 0.02)  # Additive position for linear scale
                
            ax.text(i, y_pos, text, horizontalalignment='center', fontsize=9)
    
    plt.savefig(output_path)
    plt.close()

def create_grouped_bar_chart(df, y1_column, y2_column, y1_label, y2_label, 
                            title, ylabel, output_path, 
                            yerr1_column=None, yerr2_column=None, 
                            log_scale=False, color1='cornflowerblue', color2='lightcoral'):
    """Create a grouped bar chart comparing two metrics"""
    x = np.arange(len(df.index))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot first set of bars
    rects1 = ax.bar(x - width/2, df[y1_column], width, 
                   label=y1_label, color=color1,
                   yerr=df[yerr1_column] if yerr1_column else None)
    
    # Plot second set of bars
    rects2 = ax.bar(x + width/2, df[y2_column], width, 
                   label=y2_label, color=color2,
                   yerr=df[yerr2_column] if yerr2_column else None)
    
    if log_scale:
        ax.set_yscale('log')
    
    # Add labels, title and legend
    ax.set_xlabel('Proposal')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
                
            if log_scale:
                y_pos = height * 1.1
            else:
                y_pos = height + (ax.get_ylim()[1] * 0.01)
            
            ax.annotate(text,
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1, df[yerr1_column] if yerr1_column else None)
    autolabel(rects2, df[yerr2_column] if yerr2_column else None)
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_statistics_table(df, output_path):
    """Create a table visualization of key statistics"""
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
    plt.savefig(output_path)
    plt.close()

def generate_statistics(df):
    """Generate statistical summary from DataFrame"""
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
    return stats

def generate_report(df, stats, output_path, log_scale=False, has_stddev=False):
    """Generate a text report of the analysis results"""
    with open(output_path, "w") as f:
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
        # List of visualizations will be added by the main function

def analyze_results(csv_file, output_dir="analysis", log_scale=False):
    """Main analysis function"""
    # Set up matplotlib styling
    setup_matplotlib_styling()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define required columns
    required_columns = ['proposal', 'avg_time_per_iter_ms']
    
    # Load and validate data
    df = load_and_validate_data(csv_file, required_columns)
    if df is None:
        return False
    
    # Check for standard deviation columns
    stddev_columns = ['stddev_time_ms', 'stddev_plugin_ms']
    has_stddev = all(col in df.columns for col in stddev_columns)
    
    # Generate statistics
    stats = generate_statistics(df)
    stats.to_csv(f"{output_dir}/timing_statistics.csv")
    
    # Track generated visualizations for report
    visualizations = []
    
    # 1. Create average time per iteration bar chart
    avg_time_chart_path = f"{output_dir}/avg_time_per_iter.pdf"
    create_bar_chart(
        df, 
        'avg_time_per_iter_ms', 
        "Average Time per Iteration by Proposal", 
        "Time (milliseconds)", 
        avg_time_chart_path,
        'stddev_time_ms' if has_stddev else None,
        log_scale
    )
    visualizations.append(f"1. {avg_time_chart_path} - Average time per iteration across proposals")
    
    # 2. Create timing metrics comparison if available
    if all(col in df.columns for col in ['avg_time_per_iter_ms', 'avg_time_plugin_ms']):
        comparison_chart_path = f"{output_dir}/timing_metrics_comparison.pdf"
        create_grouped_bar_chart(
            df,
            'avg_time_per_iter_ms', 'avg_time_plugin_ms',
            'Avg Time per Iteration', 'Avg Time per Algorithm',
            'Comparison of Average Timing Metrics with Standard Deviation',
            'Time (milliseconds)',
            comparison_chart_path,
            'stddev_time_ms' if has_stddev else None,
            'stddev_plugin_ms' if has_stddev else None,
            log_scale
        )
        visualizations.append(f"2. {comparison_chart_path} - Comparison of average timing metrics")
    
    # 3. Create total timing metrics comparison if available
    if all(col in df.columns for col in ['total_time_ms', 'total_time_plugin_ms']):
        total_chart_path = f"{output_dir}/total_timing_metrics.pdf"
        create_grouped_bar_chart(
            df,
            'total_time_ms', 'total_time_plugin_ms',
            'Total Sum of Times', 'Total Plugin Time (First to Last)',
            'Total Time Metrics by Proposal',
            'Time (milliseconds)',
            total_chart_path,
            None, None,  # No error bars for totals
            log_scale,
            'darkblue', 'darkred'
        )
        visualizations.append(f"3. {total_chart_path} - Total timing metrics by proposal")
    
    # 4. Create statistics table if standard deviation available
    if has_stddev:
        table_path = f"{output_dir}/timing_statistics_table.pdf"
        create_statistics_table(df, table_path)
        visualizations.append(f"4. {table_path} - Statistical summary table")
    
    # 5. Generate CSV output
    csv_path = f"{output_dir}/timing_statistics.csv"
    visualizations.append(f"5. {csv_path} - Detailed timing statistics")
    
    # Generate report
    report_path = f"{output_dir}/timing_analysis_report.txt"
    generate_report(df, stats, report_path, log_scale, has_stddev)
    
    # Add visualizations to report
    with open(report_path, "a") as f:
        for viz in visualizations:
            f.write(f"{viz}\n")
        
        if log_scale:
            f.write("\nNote: All plots use logarithmic scale for y-axis\n")
    
    print(f"Analysis completed. Results saved to {output_dir}/")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <plugin_timing_file> [<output_dir>] [--log-scale]")
        sys.exit(1)
    
    # Extract the log_scale flag from arguments
    log_scale = "--log-scale" in sys.argv
    # Remove it from arguments if present
    if log_scale:
        sys.argv.remove("--log-scale")
    
    csv_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis"
    
    analyze_results(csv_file, output_dir, log_scale)