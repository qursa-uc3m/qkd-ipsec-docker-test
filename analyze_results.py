#!/usr/bin/env python3
import sys
import os
import argparse
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
        Dictionary with combined data, network condition labels, and ordered proposals list
    """
    all_data = {}
    network_conditions = []
    proposal_order = None  # Will store the first encountered proposal order

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

                # Preserve proposal order from first file encountered
                if proposal_order is None:
                    # Get unique proposals in the order they first appear
                    proposal_order = timing_df["proposal"].drop_duplicates().tolist()

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

                        # Preserve proposal order from first file encountered
                        if proposal_order is None:
                            proposal_order = (
                                timing_df["proposal"].drop_duplicates().tolist()
                            )

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

    return all_data, network_conditions, proposal_order


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
    all_data, network_conditions, proposal_order = load_network_condition_data(
        result_dirs
    )

    if not all_data:
        print("Error: No data loaded from any directory")
        return False

    if proposal_order is None:
        # Fallback to extracting proposals if order wasn't preserved
        proposals = list(set([key.split("||")[0] for key in all_data.keys()]))
        proposals.sort()  # Keep alphabetical as fallback
        print("Warning: Could not preserve original proposal order, using alphabetical")
    else:
        proposals = proposal_order
        print(f"Using original proposal order: {proposals}")

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


import argparse
import sys
import os


def plot_ike_bytes_by_exchange(df_bytes, output_path, color_palette=None):
    """
    Create a grouped bar chart showing bytes transmitted for different IKE exchange types per proposal

    Args:
        df_bytes: DataFrame with byte transmission data per proposal
        output_path: Output file path for the plot
        color_palette: Color palette to use for different exchange types
    """
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Set grid behind the data
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)

    # Define exchange types to plot
    exchange_types = [
        ("ike_sa_init_bytes", "IKE_SA_INIT"),
        ("ike_intermediate_bytes", "IKE_INTERMEDIATE"),
        ("ike_auth_bytes", "IKE_AUTH"),
        ("informational_bytes", "INFORMATIONAL"),
        ("total_pcap_bytes", "TOTAL PCAP"),
    ]

    # Prepare data
    proposals = df_bytes["proposal"].tolist()
    n_proposals = len(proposals)
    n_exchanges = len(exchange_types)

    # Set up bar positions
    bar_width = 0.18
    x = np.arange(n_proposals)

    # Colors for different exchange types
    colors = [
        color_palette[1],
        color_palette[3],
        color_palette[5],
        color_palette[7],
        color_palette[8],
    ]

    # Create bars for each exchange type
    for i, (column, label) in enumerate(exchange_types):
        # Get byte values for this exchange type
        byte_values = df_bytes[column].values

        # Calculate bar positions
        bar_positions = x + (i - n_exchanges / 2 + 0.5) * bar_width

        # Create bars
        bars = ax.bar(
            bar_positions,
            byte_values,
            bar_width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=1.0,
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, value in zip(bars, byte_values):
            if value > 0:  # Only show labels for non-zero values
                height = bar.get_height()
                # Format bytes with commas for readability
                label_text = f"{int(value):,}" if value >= 1000 else f"{int(value)}"

                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(df_bytes["total_pcap_bytes"]) * 0.01,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=(
                        90 if value > max(df_bytes["total_pcap_bytes"]) * 0.1 else 0
                    ),
                )

    # Customize plot
    ax.set_title(
        "Bytes Transmitted by IKE Exchange Type and Proposal",
        fontweight="bold",
        fontsize=16,
        pad=20,
    )
    ax.set_ylabel("Bytes Transmitted", fontweight="bold", fontsize=12, labelpad=20)
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(proposals, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Add legend
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Format y-axis to show values with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created bytes transmission plot: {output_path}")


def analyze_bytes_data(bytes_csv_file, output_dir="analysis"):
    """
    Analyze bytes data and create visualization

    Args:
        bytes_csv_file: Path to CSV file with bytes data
        output_dir: Output directory for results
    """
    # Set up matplotlib styling
    color_palette = setup_matplotlib_styling()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load bytes data
    print(f"Loading bytes data from {bytes_csv_file}...")
    try:
        df_bytes = pd.read_csv(bytes_csv_file)
        print(f"Loaded {len(df_bytes)} proposals")

        # Create bytes plot
        bytes_plot_path = f"{output_dir}/ike_bytes_by_exchange.pdf"
        plot_ike_bytes_by_exchange(df_bytes, bytes_plot_path, color_palette)

        # Print summary to console
        print("\nIKE Bytes Summary:")
        print("-" * 50)
        for _, row in df_bytes.iterrows():
            proposal = row["proposal"]
            total_bytes = row["total_ike_bytes"]
            sa_init_bytes = row["ike_sa_init_bytes"]
            intermediate_bytes = row["ike_intermediate_bytes"]
            auth_bytes = row["ike_auth_bytes"]

            print(f"{proposal}:")
            print(f"  Total: {total_bytes:,} bytes")
            print(
                f"  IKE_SA_INIT: {sa_init_bytes:,} bytes ({sa_init_bytes/total_bytes*100:.1f}%)"
            )
            if intermediate_bytes > 0:
                print(
                    f"  IKE_INTERMEDIATE: {intermediate_bytes:,} bytes ({intermediate_bytes/total_bytes*100:.1f}%)"
                )
            print(
                f"  IKE_AUTH: {auth_bytes:,} bytes ({auth_bytes/total_bytes*100:.1f}%)"
            )
            print()

        return True

    except Exception as e:
        print(f"Error analyzing bytes data: {e}")
        return False


def compare_bytes_data(baseline_csv, comparison_csv, output_dir="analysis"):
    """
    Compare two CSV files and analyze the increase in bytes transmission

    Args:
        baseline_csv: Path to baseline CSV file (e.g., no network conditions)
        comparison_csv: Path to comparison CSV file (e.g., with packet loss)
        output_dir: Output directory for results
    """
    # Set up matplotlib styling
    color_palette = setup_matplotlib_styling()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load both CSV files
    print(f"Loading baseline data from {baseline_csv}...")
    try:
        df_baseline = pd.read_csv(baseline_csv)
        df_baseline = df_baseline.set_index("proposal")
    except Exception as e:
        print(f"Error loading baseline file: {e}")
        return False

    print(f"Loading comparison data from {comparison_csv}...")
    try:
        df_comparison = pd.read_csv(comparison_csv)
        df_comparison = df_comparison.set_index("proposal")
    except Exception as e:
        print(f"Error loading comparison file: {e}")
        return False

    # Find common proposals
    common_proposals = df_baseline.index.intersection(df_comparison.index)
    if len(common_proposals) == 0:
        print("Error: No common proposals found between the two files")
        return False

    print(
        f"Found {len(common_proposals)} common proposals: {common_proposals.tolist()}"
    )

    # Calculate differences
    comparison_data = []
    for proposal in common_proposals:
        baseline_ike = df_baseline.loc[proposal, "total_ike_bytes"]
        comparison_ike = df_comparison.loc[proposal, "total_ike_bytes"]
        baseline_pcap = df_baseline.loc[proposal, "total_pcap_bytes"]
        comparison_pcap = df_comparison.loc[proposal, "total_pcap_bytes"]

        # Calculate absolute increases (positive numbers)
        ike_increase = abs(comparison_ike - baseline_ike)
        pcap_increase = abs(comparison_pcap - baseline_pcap)

        # Calculate percentage increases
        ike_percent_increase = (
            (ike_increase / baseline_ike * 100) if baseline_ike > 0 else 0
        )
        pcap_percent_increase = (
            (pcap_increase / baseline_pcap * 100) if baseline_pcap > 0 else 0
        )

        comparison_data.append(
            {
                "proposal": proposal,
                "baseline_ike_bytes": baseline_ike,
                "comparison_ike_bytes": comparison_ike,
                "ike_bytes_increase": ike_increase,
                "ike_percent_increase": ike_percent_increase,
                "baseline_pcap_bytes": baseline_pcap,
                "comparison_pcap_bytes": comparison_pcap,
                "pcap_bytes_increase": pcap_increase,
                "pcap_percent_increase": pcap_percent_increase,
            }
        )

    # Create DataFrame with comparison results
    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison results to CSV
    comparison_csv_path = f"{output_dir}/bytes_comparison_results.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved comparison results to {comparison_csv_path}")

    # Create visualization
    plot_path = f"{output_dir}/bytes_increase_comparison.pdf"
    plot_bytes_increase_comparison(comparison_df, plot_path, color_palette)

    return True


def plot_bytes_increase_comparison(comparison_df, output_path, color_palette=None):
    """
    Create a grouped bar chart showing the increase in IKE and PCAP bytes per proposal

    Args:
        comparison_df: DataFrame with comparison results
        output_path: Output file path for the plot
        color_palette: Color palette to use
    """
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Set grid behind the data
    ax.grid(True, linestyle="--", which="both", color="grey", alpha=0.4)
    ax.set_axisbelow(True)

    # Prepare data
    proposals = comparison_df["proposal"].tolist()
    ike_increases = comparison_df["ike_bytes_increase"].values
    pcap_increases = comparison_df["pcap_bytes_increase"].values

    # Set up bar positions
    x = np.arange(len(proposals))
    bar_width = 0.35

    # Create bars
    bars1 = ax.bar(
        x - bar_width / 2,
        ike_increases,
        bar_width,
        label="IKE Bytes Increase",
        color=color_palette[3],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.8,
    )

    bars2 = ax.bar(
        x + bar_width / 2,
        pcap_increases,
        bar_width,
        label="PCAP Bytes Increase",
        color=color_palette[6],
        edgecolor="black",
        linewidth=1.0,
        alpha=0.8,
    )

    # Add value labels on bars
    def add_value_labels(bars, values, percentages):
        for bar, value, pct in zip(bars, values, percentages):
            if value > 0:
                height = bar.get_height()
                label_text = f"{int(value):,}\n({pct:.1f}%)"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(max(ike_increases), max(pcap_increases)) * 0.01,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    add_value_labels(bars1, ike_increases, comparison_df["ike_percent_increase"])
    add_value_labels(bars2, pcap_increases, comparison_df["pcap_percent_increase"])

    # Customize plot
    ax.set_title(
        "Bytes Transmission Increase by Proposal\n(Comparison vs Baseline)",
        fontweight="bold",
        fontsize=16,
        pad=20,
    )
    ax.set_ylabel("Bytes Increase", fontweight="bold", fontsize=12, labelpad=20)
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(proposals, rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Add legend
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    # Format y-axis to show values with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Created bytes increase comparison plot: {output_path}")


def load_and_compute_combined_timing(plugin_raw_csv, pcap_csv):
    """Load raw plugin data and PCAP data, compute statistics on the fly"""
    try:
        # Load and process raw plugin timing data (same as existing function)
        df_plugin_raw = load_raw_timing_data(plugin_raw_csv)
        if df_plugin_raw is None:
            return None

        df_pcap = pd.read_csv(pcap_csv)

        # Calculate timing metrics from raw plugin data (same logic as plugin timing analysis)
        plugin_timing_df = calculate_timing_metrics(df_plugin_raw)

        # Aggregate plugin statistics by proposal (same as existing)
        plugin_stats = aggregate_timing_statistics(plugin_timing_df)

        # Merge with PCAP data
        df_combined = pd.merge(plugin_stats, df_pcap, on="proposal", how="inner")
        if len(df_combined) == 0:
            raise ValueError("No common proposals found")

        # Convert and calculate timing metrics
        df_combined["network_time_ms"] = df_combined["ike_latency_avg"] * 1000
        df_combined["network_std_ms"] = df_combined["ike_latency_std"] * 1000
        df_combined["plugin_time_ms"] = df_combined[
            "avg_total_plugin_time_ms"
        ]  # From aggregated stats
        df_combined["plugin_std_ms"] = df_combined[
            "std_total_plugin_time_ms"
        ]  # From aggregated stats

        # Processing overhead = plugin time - network time
        df_combined["overhead_ms"] = (
            df_combined["plugin_time_ms"] - df_combined["network_time_ms"]
        )
        df_combined["overhead_std_ms"] = np.sqrt(
            df_combined["plugin_std_ms"] ** 2 + df_combined["network_std_ms"] ** 2
        )

        # Store raw individual measurements for boxplots
        df_combined["raw_plugin_data"] = df_combined["proposal"].apply(
            lambda prop: plugin_timing_df[plugin_timing_df["proposal"] == prop][
                "total_plugin_time_ms"
            ].values
        )

        return df_combined, plugin_timing_df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def generate_raw_boxplot_data(df_combined, plugin_timing_df):
    """Generate individual data points from raw measurements for boxplots"""
    data = []

    for _, row in df_combined.iterrows():
        proposal = row["proposal"]

        # Get raw plugin measurements for this proposal
        plugin_measurements = plugin_timing_df[
            plugin_timing_df["proposal"] == proposal
        ]["total_plugin_time_ms"].values

        # Generate network measurements based on statistics (same count as plugin measurements)
        n_measurements = len(plugin_measurements)
        network_mean = row["network_time_ms"]
        network_std = row["network_std_ms"]
        network_measurements = np.maximum(
            0, np.random.normal(network_mean, network_std, n_measurements)
        )

        # Calculate overhead for each measurement pair
        overhead_measurements = plugin_measurements - network_measurements

        # Add to data list
        for i in range(n_measurements):
            data.extend(
                [
                    {
                        "proposal": proposal,
                        "type": "Network",
                        "time_ms": network_measurements[i],
                    },
                    {
                        "proposal": proposal,
                        "type": "Overhead",
                        "time_ms": overhead_measurements[i],
                    },
                ]
            )

    return pd.DataFrame(data)


def plot_raw_timing_comparison(
    df_combined, plugin_timing_df, output_path, color_palette=None
):
    """Create side-by-side boxplots using raw individual measurements"""
    if color_palette is None:
        color_palette = setup_matplotlib_styling()

    # Generate raw data for boxplots
    df_plot = generate_raw_boxplot_data(df_combined, plugin_timing_df)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    proposals = df_combined["proposal"].tolist()
    colors = [color_palette[2], color_palette[6]]  # Blue, Orange

    for i, proposal in enumerate(proposals):
        prop_data = df_plot[df_plot["proposal"] == proposal]

        for j, timing_type in enumerate(["Network", "Overhead"]):
            data = prop_data[prop_data["type"] == timing_type]["time_ms"].values
            pos = i + (j - 0.5) * 0.35

            # Boxplot
            ax.boxplot(
                [data],
                positions=[pos],
                widths=0.25,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor="white", alpha=0.3, linewidth=1.2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                medianprops=dict(linewidth=1.5, color="darkblue"),
            )

            # Scatter points
            x_jitter = np.random.normal(pos, 0.03, len(data))
            ax.scatter(
                x_jitter,
                data,
                alpha=0.7,
                s=25,
                color=colors[j],
                edgecolor="black",
                linewidth=0.3,
                label=f"{timing_type} Time" if i == 0 else "",
            )

    # Formatting
    ax.set_title(
        "Network Time vs Processing Overhead\n(Plugin Time - Network Time)",
        fontweight="bold",
        fontsize=16,
        pad=20,
    )
    ax.set_ylabel("Time (milliseconds)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Proposal", fontweight="bold", fontsize=12)
    ax.set_xticks(range(len(proposals)))
    ax.set_xticklabels(proposals, rotation=45, ha="right")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_combined_timing(plugin_raw_csv, pcap_csv, output_dir="analysis"):
    """Main analysis function using raw plugin data"""
    color_palette = setup_matplotlib_styling()
    os.makedirs(output_dir, exist_ok=True)

    # Load and process raw data
    print("Loading raw plugin timing data and PCAP measurements...")
    result = load_and_compute_combined_timing(plugin_raw_csv, pcap_csv)
    if result is None:
        return False

    df_combined, plugin_timing_df = result

    # Save combined statistics
    df_combined_clean = df_combined.drop(
        columns=["raw_plugin_data"]
    )  # Remove the array column for CSV
    df_combined_clean.to_csv(f"{output_dir}/combined_timing_stats.csv", index=False)

    # Create plot using raw measurements
    plot_raw_timing_comparison(
        df_combined,
        plugin_timing_df,
        f"{output_dir}/network_vs_processing.pdf",
        color_palette,
    )

    # Console summary
    print("\nTiming Analysis Summary:")
    print("-" * 70)
    print(
        f"{'Proposal':<20} {'Network (ms)':<15} {'Overhead (ms)':<15} {'Ratio %':<10}"
    )
    print("-" * 70)

    for _, row in df_combined.iterrows():
        ratio = (
            (row["overhead_ms"] / row["network_time_ms"] * 100)
            if row["network_time_ms"] > 0
            else 0
        )
        print(
            f"{row['proposal']:<20} {row['network_time_ms']:>7.1f}±{row['network_std_ms']:>4.1f} "
            f"{row['overhead_ms']:>7.1f}±{row['overhead_std_ms']:>4.1f} {ratio:>7.1f}"
        )

    return True


def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="Analyze StrongSwan QKD Plugin Performance Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plugin timing analysis only
  python analyze_plugin_timing.py --plugin-timing plugin_timing_raw.csv --output analysis_output
  
  # PCAP bytes analysis only
  python analyze_plugin_timing.py --pcap-bytes pcap_measurements.csv --output analysis_output
  
  # Combined analysis
  python analyze_plugin_timing.py --plugin-timing plugin_raw.csv --pcap-bytes pcap_data.csv --output analysis_output
  
  # Network conditions comparison
  python analyze_plugin_timing.py --network-compare dir1 dir2 dir3 --output network_analysis
  
  # Bytes comparison between two files
  python analyze_plugin_timing.py --compare-files baseline.csv comparison.csv --output comparison_analysis
  
  # With log-scale plots
  python analyze_plugin_timing.py --plugin-timing data.csv --log-scale
        """,
    )

    # Analysis options
    parser.add_argument(
        "--plugin-timing",
        metavar="CSV_FILE",
        help="Path to plugin timing raw CSV file (plugin_timing_raw.csv)",
    )

    parser.add_argument(
        "--pcap-bytes",
        metavar="CSV_FILE",
        help="Path to PCAP bytes measurements CSV file (pcap_measurements.csv)",
    )

    parser.add_argument(
        "--network-compare",
        nargs="+",
        metavar="DIR",
        help="Compare plugin timing across multiple network condition directories",
    )

    parser.add_argument(
        "--compare-files",
        nargs=2,
        metavar=("BASELINE", "COMPARISON"),
        help="Compare bytes between two CSV files (baseline comparison)",
    )

    # Output and formatting options
    parser.add_argument(
        "--output",
        "-o",
        default="analysis",
        metavar="DIR",
        help="Output directory for analysis results (default: analysis)",
    )

    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for timing plots",
    )

    return parser.parse_args()


def main():
    """Main function with proper argument parsing"""
    args = parse_arguments()

    # Validate that at least one analysis is requested
    if not (
        args.plugin_timing
        or args.pcap_bytes
        or args.network_compare
        or args.compare_files
    ):
        print("Error: Please specify at least one analysis type:")
        print("  --plugin-timing <file>       for plugin timing analysis")
        print("  --pcap-bytes <file>          for PCAP bytes analysis")
        print("  --network-compare <dirs>     for network comparison")
        print("  --compare-files <baseline> <comparison> for file comparison")
        print("\nUse --help for detailed usage information")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    success = True

    # File comparison mode
    if args.compare_files:
        print("Running file comparison analysis...")
        baseline_file, comparison_file = args.compare_files
        comparison_success = compare_bytes_data(
            baseline_file, comparison_file, args.output
        )
        success = success and comparison_success

    # Network comparison mode
    elif args.network_compare:
        print("Running network conditions comparison analysis...")
        success = analyze_network_conditions(
            args.network_compare, args.output, log_scale=args.log_scale
        )

    else:
        # Individual analyses
        if args.plugin_timing:
            print("Running plugin timing analysis...")
            timing_success = analyze_plugin_timing(
                args.plugin_timing, args.output, log_scale=args.log_scale
            )
            success = success and timing_success

        if args.pcap_bytes:
            print("Running PCAP bytes analysis...")
            bytes_success = analyze_bytes_data(args.pcap_bytes, args.output)
            success = success and bytes_success

        # Automatic combined timing analysis (if both files are provided)
        if args.plugin_timing and args.pcap_bytes:
            print("Running combined timing analysis (Network vs Processing)...")
            combined_success = analyze_combined_timing(
                args.plugin_timing, args.pcap_bytes, args.output
            )
            success = success and combined_success

    if success:
        print(f"\n✓ Analysis completed successfully! Results saved to: {args.output}/")
    else:
        print("\n✗ Analysis completed with errors. Check the output above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
