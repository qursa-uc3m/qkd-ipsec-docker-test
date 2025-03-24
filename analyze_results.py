#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from matplotlib import rcParams

# Set up consistent matplotlib styling
def setup_matplotlib_styling():
    """Configure matplotlib with consistent styling and LaTeX support for all plots"""
    # Enable LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'
    
    # Font configuration
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
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

def analyze_results(csv_file):
    """
    Analyze the test results and generate visualizations
    """
    
    # Set up matplotlib styling
    setup_matplotlib_styling()
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: Results file {csv_file} not found")
        return False
    
    # Load the data
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False
    
    # Create output directory
    os.makedirs("analysis", exist_ok=True)
    
    # Calculate statistics
    stats = pd.DataFrame()
    for column in df.columns:
        stats[column] = [
            df[column].mean(),
            df[column].std(),
            df[column].min(),
            df[column].max()
        ]
    
    stats.index = ["Mean", "Std Dev", "Min", "Max"]
    stats.to_csv("analysis/statistics.csv")
    
    # Generate visualizations
    
    # 1. Bar chart of average latencies
    plt.figure(figsize=(10, 6))
    stats.loc["Mean"].plot(kind="bar")
    plt.title("Average Handshake Latency by Proposal")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Proposal")
    plt.tight_layout()
    plt.savefig("analysis/avg_latency.pdf")
    
    # 2. Box plot for distribution
    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.title("Latency Distribution by Proposal")
    plt.ylabel("Latency (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("analysis/latency_distribution.pdf")
    
    # Generate report
    with open("analysis/analysis_report.txt", "w") as f:
        f.write("StrongSwan QKD Plugin Performance Analysis\n")
        f.write("=========================================\n\n")
        
        f.write("Statistical Summary:\n")
        f.write("-------------------\n")
        f.write(stats.to_string())
        f.write("\n\n")
        
        # Compare proposals
        best_proposal = stats.loc["Mean"].idxmin()
        f.write(f"Best performing proposal (lowest average latency): {best_proposal}\n")
        f.write(f"Average latency: {stats.loc['Mean', best_proposal]:.6f} seconds\n\n")
        
        # Variability
        most_consistent = stats.loc["Std Dev"].idxmin()
        f.write(f"Most consistent proposal (lowest std deviation): {most_consistent}\n")
        f.write(f"Standard deviation: {stats.loc['Std Dev', most_consistent]:.6f} seconds\n\n")
        
        f.write("Generated visualizations:\n")
        f.write("1. analysis/avg_latency.* - Average latencies across proposals\n")
        f.write("2. analysis/latency_distribution.* - Distribution of latencies\n")
        f.write("3. analysis/statistics.csv - Detailed statistics\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "results/results.csv"
    
    analyze_results(csv_file)