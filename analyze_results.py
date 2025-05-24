#!/usr/bin/env python3
import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors


# Configure matplotlib styling - matching your reference style
def setup_matplotlib_styling():
    """Configure matplotlib with consistent styling for all visualization types"""

    # Match the font settings from your reference
    rcParams["font.family"] = "Ubuntu"  # Or use 'Liberation Sans' as alternative
    rcParams["font.size"] = 11
    rcParams["axes.titlesize"] = 16
    rcParams["axes.labelsize"] = 12
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.labelweight"] = "bold"
    rcParams["xtick.labelsize"] = 11
    rcParams["ytick.labelsize"] = 11
    rcParams["legend.fontsize"] = 11
    rcParams["figure.titlesize"] = 16
    rcParams["figure.figsize"] = (11, 6)
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.format"] = "pdf"

    # Custom color palette
    color_palette = list(
        mcolors.LinearSegmentedColormap.from_list("", ["#9fcf69", "#33acdc"])(
            np.linspace(0, 1, 9)
        )
    )

    return color_palette


def load_raw_timing_data(csv_file):
    """
    Load raw timing data from CSV file

    Args:
        csv_file: Path to the CSV file with raw timing data

    Returns:
        DataFrame with timing data
    """
    if not os.path.exists(csv_file):
        print(f"Error: Results file {csv_file} not found")
        return None

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded data with columns: {df.columns.tolist()}")

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Convert timestamps to total microseconds
        df["create_total_us"] = (
            df["create_timestamp"] * 1_000_000 + df["create_microseconds"]
        )
        df["destroy_total_us"] = (
            df["destroy_timestamp"] * 1_000_000 + df["destroy_microseconds"]
        )

        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def calculate_timing_metrics(df):
    """
    Calculate total plugin time and in-plugin time for each proposal/iteration

    Returns:
        DataFrame with timing metrics per proposal
    """
    results = []

    # Group by proposal and iteration
    for (proposal, iteration), group in df.groupby(
        ["proposal", "iteration"], sort=False
    ):
        # Sort by operation_sequence to ensure correct order
        group = group.sort_values("operation_sequence")

        # Total plugin time: last destroy - first create
        total_plugin_time_us = (
            group["destroy_total_us"].iloc[-1] - group["create_total_us"].iloc[0]
        )
        total_plugin_time_ms = total_plugin_time_us / 1000.0

        # In-plugin time: sum of (destroy - create) for each operation
        in_plugin_times = group["destroy_total_us"] - group["create_total_us"]
        total_in_plugin_time_us = in_plugin_times.sum()
        total_in_plugin_time_ms = total_in_plugin_time_us / 1000.0

        results.append(
            {
                "proposal": proposal,
                "iteration": iteration,
                "total_plugin_time_ms": total_plugin_time_ms,
                "total_in_plugin_time_ms": total_in_plugin_time_ms,
                "num_operations": len(group),
            }
        )

    return pd.DataFrame(results)


def aggregate_timing_statistics(timing_df):
    """
    Calculate average and standard deviation for each proposal

    Returns:
        DataFrame with statistics per proposal
    """
    stats = (
        timing_df.groupby("proposal")
        .agg(
            {
                "total_plugin_time_ms": ["mean", "std", "count"],
                "total_in_plugin_time_ms": ["mean", "std"],
                "num_operations": "mean",
            }
        )
        .round(3)
    )

    # Flatten column names
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]
    stats.columns = [
        "avg_total_plugin_time_ms",
        "std_total_plugin_time_ms",
        "num_iterations",
        "avg_total_in_plugin_time_ms",
        "std_total_in_plugin_time_ms",
        "avg_num_operations",
    ]

    # Reset index to make proposal a column
    stats = stats.reset_index()

    return stats


def plot_boxplot_with_scatter(
    timing_df, title, ylabel, output_path, color_palette=None, log_scale=False
):
    """
    Create a boxplot with scatter points overlay, matching the reference style

    Args:
        timing_df: DataFrame with timing data
        title: Plot title
        ylabel: Y-axis label
        output_path: Output file path
        color_palette: Color palette to use
        log_scale: If True, use logarithmic y-axis scale
    """
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    fig, ax = plt.subplots(figsize=(11, 6))

    # Set logarithmic scale if requested
    if log_scale:
        ax.set_yscale("log")
        # Update ylabel to indicate log scale
        ylabel = f"{ylabel} (log scale)"

    # Set grid behind the data
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)

    # Prepare data for boxplot
    proposals = timing_df["proposal"].unique()

    # Create list of arrays for boxplot
    data_arrays = []
    positions = []

    for i, proposal in enumerate(proposals):
        data = timing_df[timing_df["proposal"] == proposal][
            "total_plugin_time_ms"
        ].values
        data_arrays.append(data)
        positions.append(i)

    # Create boxplot
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", alpha=0.3, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.5, color="darkblue"),
    )

    # Add scatter points on top
    for i, (proposal, data) in enumerate(zip(proposals, data_arrays)):
        # Add jitter to x-coordinates for better visibility
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(
            x,
            data,
            alpha=0.7,
            s=40,
            color=color_palette[i % len(color_palette)],
            edgecolor="black",
            linewidth=0.5,
        )

    # Customize plot
    ax.set_xticks(positions)
    ax.set_xticklabels(proposals, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Set labels and title
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, labelpad=20)
    ax.set_title(title, fontweight="bold", fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_network_conditions_comparison(
    all_data,
    proposals,
    network_conditions,
    title,
    ylabel,
    output_path,
    color_palette=None,
    log_scale=False,
):
    """
    Create a boxplot comparing different network conditions for each proposal
    with transparent boxes and colored scatter points
    """
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Set grid behind the data
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)

    if log_scale:
        ax.set_yscale("log")
        ylabel = f"{ylabel} (log scale)"

    # Setup positions
    n_conditions = len(network_conditions)
    n_proposals = len(proposals)
    width = 0.8 / n_conditions  # Width of each box

    # Colors for different network conditions
    colors = [
        color_palette[i]
        for i in np.linspace(0, len(color_palette) - 1, n_conditions, dtype=int)
    ]

    for i, proposal in enumerate(proposals):
        base_pos = i * 1.0  # Base position for this proposal

        for j, condition in enumerate(network_conditions):
            # Position for this condition within the proposal group
            pos = base_pos + (j - n_conditions / 2 + 0.5) * width

            # Get data for this proposal and condition
            data_key = f"{proposal}||{condition}"
            if data_key in all_data:
                data = all_data[data_key]

                # Create transparent boxplot
                bp = ax.boxplot(
                    [data],
                    positions=[pos],
                    widths=width * 0.8,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor="white", alpha=0.3, linewidth=1.2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    medianprops=dict(linewidth=1.5, color="darkblue"),
                )

                # Add colored scatter points with jitter
                x_scatter = np.random.normal(pos, width * 0.15, size=len(data))
                ax.scatter(
                    x_scatter,
                    data,
                    alpha=0.7,
                    s=40,
                    color=colors[j],
                    edgecolor="black",
                    linewidth=0.5,
                    label=condition if i == 0 else "",
                )

    # Set x-axis
    ax.set_xticks([i * 1.0 for i in range(n_proposals)])
    ax.set_xticklabels(proposals, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Set labels and title
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, labelpad=20)
    ax.set_title(title, fontweight="bold", fontsize=16, pad=20)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_timing_bar_chart(
    stats_df,
    metric_column,
    std_column,
    title,
    ylabel,
    output_path,
    color_palette=None,
    log_scale=False,
):
    """
    Create a bar chart with error bars for timing metrics
    """
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    fig, ax = plt.subplots(figsize=(11, 6))

    # Set logarithmic scale FIRST if requested
    if log_scale:
        ax.set_yscale("log")
        # Modify ylabel to indicate log scale (avoid duplicates)
        if "(log scale)" not in ylabel.lower():
            ylabel = f"{ylabel} (log scale)"

    # Set grid behind the data
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)

    # Create bar positions
    x = np.arange(len(stats_df))

    # For log scale, we need to handle error bars differently
    if log_scale:
        # In log scale, error bars should be multiplicative, not additive
        # Convert to log space for proper error bar calculation
        means = stats_df[metric_column].values
        stds = stats_df[std_column].values

        # Calculate asymmetric error bars for log scale
        lower_errors = means - np.maximum(
            means - stds, means * 0.1
        )  # Prevent going to 0 or negative
        upper_errors = (means + stds) - means

        error_bars = [lower_errors, upper_errors]
    else:
        error_bars = stats_df[std_column]

    # Create bars with error bars
    bars = ax.bar(
        x,
        stats_df[metric_column],
        yerr=stats_df[std_column],
        capsize=5,
        color=color_palette[-1],
        edgecolor="black",
        linewidth=1.2,
    )

    # Customize plot
    ax.set_title(title, fontweight="bold", fontsize=16, pad=20)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, labelpad=20)
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["proposal"], rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Add value annotations on bars
    for i, (bar, mean, std) in enumerate(
        zip(bars, stats_df[metric_column], stats_df[std_column])
    ):
        height = bar.get_height()

        # Position text appropriately for log/linear scale
        if log_scale:
            # For log scale, position text as a ratio above the bar
            text_y = height * 1.1
        else:
            # For linear scale, position text with fixed offset
            text_y = height + std + max(stats_df[metric_column]) * 0.01

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + max(stats_df[metric_column]) * 0.01,
            f"{mean:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_comparison_chart(stats_df, output_path, color_palette=None):
    """
    Create a bar chart showing the difference between total plugin time and in-plugin time
    """
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    # Calculate the difference (overhead)
    stats_df["overhead_ms"] = (
        stats_df["avg_total_plugin_time_ms"] - stats_df["avg_total_in_plugin_time_ms"]
    )
    stats_df["overhead_percent"] = (
        stats_df["overhead_ms"] / stats_df["avg_total_in_plugin_time_ms"]
    ) * 100

    fig, ax = plt.subplots(figsize=(11, 6))

    # Set grid behind the data
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)

    # Create bar positions
    x = np.arange(len(stats_df))

    # Create bars
    bars = ax.bar(
        x,
        stats_df["overhead_ms"],
        color=color_palette[4],  # Use a middle color from palette
        edgecolor="black",
        linewidth=1.2,
    )

    # Customize plot
    ax.set_title(
        "Plugin Overhead: Difference Between Total Time and Active Plugin Time",
        fontweight="bold",
        fontsize=16,
        pad=20,
    )
    ax.set_ylabel(
        "Overhead (milliseconds)", fontweight="bold", fontsize=12, labelpad=20
    )
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["proposal"], rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Add value annotations on bars with percentage
    for i, (bar, overhead_ms, overhead_pct) in enumerate(
        zip(bars, stats_df["overhead_ms"], stats_df["overhead_percent"])
    ):
        height = bar.get_height()
        # Position text above or below bar depending on sign
        va = "bottom" if height >= 0 else "top"
        y_offset = (
            0.01 * max(abs(stats_df["overhead_ms"]))
            if height >= 0
            else -0.01 * max(abs(stats_df["overhead_ms"]))
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + y_offset,
            f"{overhead_ms:.2f} ms\n({overhead_pct:.1f}%)",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_detailed_report(stats_df, timing_df, output_path):
    """Generate a detailed text report of the analysis"""
    with open(output_path, "w") as f:
        f.write("StrongSwan QKD Plugin Timing Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("Analysis Summary\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total proposals analyzed: {len(stats_df)}\n")
        f.write(f"Total measurements: {len(timing_df)}\n\n")

        # Best/worst performers for total plugin time
        f.write("Total Plugin Time Analysis (First Create to Last Destroy)\n")
        f.write("-" * 50 + "\n")
        best_idx = stats_df["avg_total_plugin_time_ms"].idxmin()
        worst_idx = stats_df["avg_total_plugin_time_ms"].idxmax()

        f.write(f"Fastest: {stats_df.loc[best_idx, 'proposal']}\n")
        f.write(
            f"  Average: {stats_df.loc[best_idx, 'avg_total_plugin_time_ms']:.3f} ms\n"
        )
        f.write(
            f"  Std Dev: {stats_df.loc[best_idx, 'std_total_plugin_time_ms']:.3f} ms\n\n"
        )

        f.write(f"Slowest: {stats_df.loc[worst_idx, 'proposal']}\n")
        f.write(
            f"  Average: {stats_df.loc[worst_idx, 'avg_total_plugin_time_ms']:.3f} ms\n"
        )
        f.write(
            f"  Std Dev: {stats_df.loc[worst_idx, 'std_total_plugin_time_ms']:.3f} ms\n\n"
        )

        # Performance ratios
        f.write("Performance Comparisons\n")
        f.write("-" * 30 + "\n")
        for _, row in stats_df.iterrows():
            overhead = (
                (row["avg_total_plugin_time_ms"] - row["avg_total_in_plugin_time_ms"])
                / row["avg_total_in_plugin_time_ms"]
                * 100
            )
            f.write(f"{row['proposal']}:\n")
            f.write(f"  Plugin overhead: {overhead:.1f}%\n")
            f.write(
                f"  Average operations per iteration: {row['avg_num_operations']:.1f}\n\n"
            )


def load_network_condition_data(result_dirs):
    """
    Load timing data from multiple network condition directories

    Args:
        result_dirs: List of directory paths containing test results

    Returns:
        Dictionary with combined data and network condition labels
    """
    all_data = {}
    network_conditions = []

    for result_dir in result_dirs:
        # Extract network condition from directory name
        dir_name = os.path.basename(result_dir)

        # Parse network condition from directory name
        if dir_name.startswith("no_network_conditions"):
            condition_label = "No Network Conditions"
        else:
            # Extract lat, jit, loss values from directory name
            parts = dir_name.split("_")
            lat = jit = loss = "0"

            for part in parts:
                if part.startswith("lat"):
                    lat = part[3:]
                elif part.startswith("jit"):
                    jit = part[3:]
                elif part.startswith("loss"):
                    loss = part[4:]

            condition_label = f"Lat:{lat}ms Jit:{jit}ms Loss:{loss}%"

        network_conditions.append(condition_label)

        # Look for plugin_timing_raw.csv in the directory
        raw_csv_path = os.path.join(result_dir, "plugin_timing_raw.csv")
        if os.path.exists(raw_csv_path):
            print(f"Loading data from {raw_csv_path}")
            df = load_raw_timing_data(raw_csv_path)
            if df is not None:
                timing_df = calculate_timing_metrics(df)

                # Store data for each proposal under this condition
                for proposal in timing_df["proposal"].unique():
                    data_key = f"{proposal}||{condition_label}"  # Use || as separator
                    proposal_data = timing_df[timing_df["proposal"] == proposal][
                        "total_plugin_time_ms"
                    ].values
                    all_data[data_key] = proposal_data
        else:
            print(f"Warning: No plugin_timing_raw.csv found in {result_dir}")
            print(f"  Looking for alternative files...")

            # Try to find the timing data in subdirectories or with different names
            found = False
            for root, dirs, files in os.walk(result_dir):
                if "plugin_timing_raw.csv" in files:
                    alt_path = os.path.join(root, "plugin_timing_raw.csv")
                    print(f"  Found at: {alt_path}")
                    df = load_raw_timing_data(alt_path)
                    if df is not None:
                        timing_df = calculate_timing_metrics(df)
                        for proposal in timing_df["proposal"].unique():
                            data_key = f"{proposal}||{condition_label}"
                            proposal_data = timing_df[
                                timing_df["proposal"] == proposal
                            ]["total_plugin_time_ms"].values
                            all_data[data_key] = proposal_data
                        found = True
                        break

            if not found:
                print(
                    f"  Could not find plugin_timing_raw.csv in {result_dir} or subdirectories"
                )

    return all_data, network_conditions


def analyze_network_conditions(
    result_dirs, output_dir="analysis_network_comparison", log_scale=False
):
    """
    Analyze and compare plugin timing across different network conditions

    Args:
        result_dirs: List of directories containing test results for different network conditions
        output_dir: Output directory for analysis results
        log_scale: If True, use logarithmic scale for comparison plot
    """
    # Set up matplotlib styling
    color_palette = setup_matplotlib_styling()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data from all directories
    print("Loading data from multiple network conditions...")
    all_data, network_conditions = load_network_condition_data(result_dirs)

    if not all_data:
        print("Error: No data loaded from any directory")
        return False

    # Extract unique proposals - use || separator
    proposals = list(set([key.split("||")[0] for key in all_data.keys()]))
    proposals.sort()  # Ensure consistent ordering

    print(f"Found proposals: {proposals}")
    print(f"Found network conditions: {network_conditions}")

    # Create comparison plot
    print("Creating network conditions comparison plot...")
    comparison_plot_path = f"{output_dir}/network_conditions_comparison.pdf"
    plot_network_conditions_comparison(
        all_data,
        proposals,
        network_conditions,
        f"Plugin Timing Comparison Across Network Conditions",
        "Time (milliseconds)",
        comparison_plot_path,
        color_palette,
        log_scale=log_scale,
    )
    print(f"Created comparison plot: {comparison_plot_path}")

    # Generate summary statistics
    stats_data = []
    for condition in network_conditions:
        for proposal in proposals:
            data_key = f"{proposal}||{condition}"
            if data_key in all_data:
                data = all_data[data_key]
                stats_data.append(
                    {
                        "network_condition": condition,
                        "proposal": proposal,
                        "mean_ms": np.mean(data),
                        "std_ms": np.std(data),
                        "min_ms": np.min(data),
                        "max_ms": np.max(data),
                        "count": len(data),
                    }
                )

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = f"{output_dir}/network_comparison_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics to {stats_csv_path}")

    # Generate report
    report_path = f"{output_dir}/network_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("Network Conditions Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analyzed {len(network_conditions)} network conditions\n")
        f.write(f"Proposals: {', '.join(proposals)}\n\n")

        for condition in network_conditions:
            f.write(f"\n{condition}:\n")
            f.write("-" * 30 + "\n")
            for proposal in proposals:
                data_key = f"{proposal}||{condition}"
                if data_key in all_data:
                    data = all_data[data_key]
                    f.write(
                        f"  {proposal}: {np.mean(data):.2f} ± {np.std(data):.2f} ms (n={len(data)})\n"
                    )

    print(f"Generated report: {report_path}")
    print(f"\nAnalysis completed! Results saved to {output_dir}/")

    return True


def analyze_plugin_timing(raw_csv_file, output_dir="analysis", log_scale=False):
    """Main analysis function for plugin timing data

    Args:
        raw_csv_file: Path to raw CSV file
        output_dir: Output directory for results
        log_scale: If True, use logarithmic scale for boxplots
    """
    # Set up matplotlib styling
    color_palette = setup_matplotlib_styling()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load raw timing data
    print("Loading raw timing data...")
    df = load_raw_timing_data(raw_csv_file)
    if df is None:
        return False

    # Calculate timing metrics
    print("Calculating timing metrics...")
    timing_df = calculate_timing_metrics(df)

    # Aggregate statistics by proposal
    print("Aggregating statistics...")
    stats_df = aggregate_timing_statistics(timing_df)

    # Save detailed timing data
    timing_csv_path = f"{output_dir}/detailed_timing_metrics.csv"
    timing_df.to_csv(timing_csv_path, index=False)
    print(f"Saved detailed timing metrics to {timing_csv_path}")

    # Save aggregated statistics
    stats_csv_path = f"{output_dir}/plugin_timing_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved aggregated statistics to {stats_csv_path}")

    # Create visualizations
    print("Creating visualizations...")

    # 1. Total plugin time boxplot with scatter
    total_plugin_boxplot = f"{output_dir}/total_plugin_time_boxplot.pdf"
    plot_boxplot_with_scatter(
        timing_df,
        f'Total Plugin Time by Proposal (N={int(stats_df["num_iterations"].iloc[0])})',
        "Time (milliseconds)",
        total_plugin_boxplot,
        color_palette,
        log_scale=log_scale,
    )
    print(f"Created total plugin time boxplot: {total_plugin_boxplot}")

    # 2. Average total plugin time bar chart
    total_plugin_chart = f"{output_dir}/avg_total_plugin_time.pdf"
    create_timing_bar_chart(
        stats_df,
        "avg_total_plugin_time_ms",
        "std_total_plugin_time_ms",
        "Average Total Plugin Time by Proposal\n(First Create to Last Destroy)",
        "Time (milliseconds)",
        total_plugin_chart,
        color_palette,
    )
    print(f"Created average total plugin time chart: {total_plugin_chart}")

    # 3. Overhead comparison chart
    comparison_chart = f"{output_dir}/plugin_overhead_comparison.pdf"
    create_comparison_chart(stats_df, comparison_chart, color_palette)
    print(f"Created overhead comparison chart: {comparison_chart}")

    # Generate report
    report_path = f"{output_dir}/plugin_timing_report.txt"
    generate_detailed_report(stats_df, timing_df, report_path)
    print(f"Generated analysis report: {report_path}")

    print(f"\nAnalysis completed successfully! All results saved to {output_dir}/")

    # Print summary to console
    print("\nQuick Summary:")
    print("-" * 50)
    print("Average Total Plugin Time (ms):")
    for _, row in stats_df.iterrows():
        print(
            f"  {row['proposal']}: {row['avg_total_plugin_time_ms']:.2f} ± {row['std_total_plugin_time_ms']:.2f}"
        )

    return True


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  Single file mode: python analyze_plugin_timing.py <raw_timing_csv> [<output_dir>] [--log-scale]"
        )
        print(
            "  Network comparison mode: python analyze_plugin_timing.py --network-compare <dir1> <dir2> ... [--output <output_dir>] [--log-scale]"
        )
        print("\nExamples:")
        print("  python analyze_plugin_timing.py timing_data.csv analysis_output")
        print(
            "  python analyze_plugin_timing.py timing_data.csv analysis_output --log-scale"
        )
        print(
            "  python analyze_plugin_timing.py --network-compare lat0_* lat100_* --output network_analysis --log-scale"
        )
        sys.exit(1)

    # Check for log-scale flag
    log_scale = "--log-scale" in sys.argv
    if log_scale:
        sys.argv.remove("--log-scale")  # Remove it from argv for easier parsing

    if sys.argv[1] == "--network-compare":
        # Network comparison mode
        result_dirs = []
        output_dir = "analysis_network_comparison"

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--output":
                if i + 1 < len(sys.argv):
                    output_dir = sys.argv[i + 1]
                    i += 2
                else:
                    print("Error: --output requires a directory name")
                    sys.exit(1)
            else:
                result_dirs.append(sys.argv[i])
                i += 1

        if len(result_dirs) < 2:
            print("Error: Network comparison mode requires at least 2 directories")
            sys.exit(1)

        # Run network conditions analysis
        success = analyze_network_conditions(
            result_dirs, output_dir, log_scale=log_scale
        )
    else:
        # Single file mode
        csv_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis"

        # Run standard analysis
        success = analyze_plugin_timing(csv_file, output_dir)

    sys.exit(0 if success else 1)
