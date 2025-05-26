#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# alice/data_proc/plugin_proc.py

import os
import pandas as pd


def collect_raw_plugin_timing_data(plugin_timing_log, prop, iteration, log_message):
    """
    Collect raw plugin timing data from temporary file for each iteration.
    Preserves all individual creation/destruction events without calculating differences.

    Args:
        plugin_timing_log: Path to the plugin timing log file
        prop: The cryptographic proposal being tested
        iteration: The iteration number
        log_message: Function for logging messages

    Returns:
        DataFrame with raw timing data, or empty DataFrame if no data
    """
    log_message(
        f"Collecting raw plugin timing data for proposal: {prop}, iteration: {iteration}"
    )

    # Check if the timing log exists
    if not os.path.exists(plugin_timing_log):
        log_message(f"Timing log file {plugin_timing_log} not found")
        return pd.DataFrame()

    try:
        # Read the timing log file
        timing_df = pd.read_csv(plugin_timing_log)

        # If the file is empty or has no data, return empty DataFrame
        if timing_df.empty:
            log_message("No plugin timing data found")
            return pd.DataFrame()

        # Remove any rows with NaN or empty method values
        timing_df = timing_df.dropna(subset=["method"])

        if timing_df.empty:
            log_message("No valid plugin timing data after filtering")
            return pd.DataFrame()

        # Convert timestamp columns to numeric types for consistency
        numeric_cols = [
            "create_timestamp",
            "create_microseconds",
            "destroy_timestamp",
            "destroy_microseconds",
        ]
        for col in numeric_cols:
            if col in timing_df.columns:
                timing_df[col] = pd.to_numeric(timing_df[col], errors="coerce")

        # Add metadata columns to identify this specific test run
        timing_df["proposal"] = prop
        timing_df["iteration"] = iteration

        # Add sequence number for operations within this iteration
        timing_df["operation_sequence"] = range(1, len(timing_df) + 1)

        # Keep all original columns and metadata - no time difference calculations
        log_message(
            f"Collected {len(timing_df)} raw timing measurements for {prop} iteration {iteration}"
        )

        # Log the methods captured for debugging
        methods_found = timing_df["method"].unique().tolist()
        log_message(f"Methods captured: {methods_found}")

        return timing_df

    except Exception as e:
        log_message(f"Error collecting raw plugin timing data: {e}")
        return pd.DataFrame()


def process_plugin_timing_data(plugin_timing_log, prop, log_message):
    """
    Process plugin timing data from temporary file for each proposal
    (Kept for backward compatibility with aggregated stats)

    Args:
        plugin_timing_log: Path to the plugin timing log file
        prop: The cryptographic proposal being tested
        log_message: Function for logging messages

    Returns:
        A dictionary with timing statistics
    """
    log_message(f"Processing plugin timing data for proposal: {prop}")

    # Check if the timing log exists
    if not os.path.exists(plugin_timing_log):
        log_message(f"Timing log file {plugin_timing_log} not found")
        return {
            "proposal": prop,
            "algorithms_count": 0,
            "total_time_ms": 0,
            "total_time_plugin_ms": 0,
            "avg_time_ms": 0,
            "min_time_ms": 0,
            "max_time_ms": 0,
        }

    try:
        # Read the timing log file
        timing_df = pd.read_csv(plugin_timing_log)

        # If the file is empty or has no data, return default values
        if timing_df.empty:
            log_message("No plugin timing data found")
            return {
                "proposal": prop,
                "algorithms_count": 0,
                "total_time_ms": 0,
                "total_time_plugin_ms": 0,
                "avg_time_ms": 0,
                "min_time_ms": 0,
                "max_time_ms": 0,
            }

        # Convert timestamp columns to numeric types
        numeric_cols = [
            "create_timestamp",
            "create_microseconds",
            "destroy_timestamp",
            "destroy_microseconds",
        ]
        for col in numeric_cols:
            if col in timing_df.columns:
                timing_df[col] = pd.to_numeric(timing_df[col], errors="coerce")

        # Calculate time differences in milliseconds
        timing_df["time_diff_ms"] = (
            timing_df["destroy_timestamp"] - timing_df["create_timestamp"]
        ) * 1000 + (
            timing_df["destroy_microseconds"] - timing_df["create_microseconds"]
        ) / 1000

        # Remove any rows with NaN or empty method values
        timing_df = timing_df.dropna(subset=["method"])
        # Count the number of algorithms (rows)
        algo_count = len(timing_df)

        # Calculate statistics
        total_time_ms = timing_df["time_diff_ms"].sum()

        # For total plugin time calculation
        if algo_count > 0:
            # Convert both timestamps to milliseconds first
            create_timestamps_ms = (
                timing_df["create_timestamp"] * 1000
                + timing_df["create_microseconds"] / 1000
            )
            destroy_timestamps_ms = (
                timing_df["destroy_timestamp"] * 1000
                + timing_df["destroy_microseconds"] / 1000
            )

            # Find the earliest and latest points
            first_create_ms = create_timestamps_ms.min()
            last_destroy_ms = destroy_timestamps_ms.max()

            # Calculate the total elapsed time
            total_time_plugin_ms = last_destroy_ms - first_create_ms
        else:
            total_time_plugin_ms = 0

        # Return the statistics
        return {
            "proposal": prop,
            "algorithms_count": algo_count,
            "total_time_ms": total_time_ms,
            "total_time_plugin_ms": total_time_plugin_ms,
        }

    except Exception as e:
        log_message(f"Error processing plugin timing data: {e}")
        return {
            "proposal": prop,
            "algorithms_count": 0,
            "total_time_ms": 0,
            "total_time_plugin_ms": 0,
            "avg_time_ms": 0,
            "min_time_ms": 0,
            "max_time_ms": 0,
        }


def aggregate_plugin_timing(proposal_iterations, prop, num_iterations, log_message):
    """
    Aggregate plugin timing data across multiple iterations for a proposal

    Args:
        proposal_iterations: List of dictionaries with iteration-level timing stats
        prop: The proposal name
        num_iterations: Number of iterations run
        log_message: Logging function

    Returns:
        Dictionary with aggregated timing statistics for the proposal
    """
    log_message(f"Aggregating plugin timing data for proposal: {prop}")

    # Check if algorithm count is consistent across iterations
    algorithm_counts = [
        iter_data["algorithms_count"] for iter_data in proposal_iterations
    ]

    if not algorithm_counts:
        log_message(f"Warning: No algorithm counts data available for {prop}")
        total_algorithms = 0
        is_consistent = True
    else:
        # Check if all counts are the same
        is_consistent = all(count == algorithm_counts[0] for count in algorithm_counts)
        total_algorithms = (
            algorithm_counts[0]
            if is_consistent
            else max(set(algorithm_counts), key=algorithm_counts.count)
        )
    if not is_consistent:
        log_message(
            f"Warning: Inconsistent algorithm counts across iterations for {prop}: {algorithm_counts}"
        )

    # Collect all the timing values
    all_times_ms = [iter_data["total_time_ms"] for iter_data in proposal_iterations]
    all_plugin_times_ms = [
        iter_data["total_time_plugin_ms"] for iter_data in proposal_iterations
    ]

    # Total time inside plugins
    total_time_ms = sum(all_times_ms)
    avg_time_in_plugin_per_iter_ms = (
        total_time_ms / num_iterations if num_iterations > 0 else 0
    )
    # Calculate standard deviation for total_time_ms
    if len(all_times_ms) > 1:
        stddev_time_ms = pd.Series(all_times_ms).std()
    else:
        stddev_time_ms = 0

    # Total time between first creation and last destruction
    total_time_plugin_ms = sum(all_plugin_times_ms)
    avg_elapsed_time_per_iter_ms = (
        total_time_plugin_ms / num_iterations if num_iterations > 0 else 0
    )
    # Calculate standard deviation for total_time_plugin_ms
    if len(all_plugin_times_ms) > 1:
        stddev_plugin_ms = pd.Series(all_plugin_times_ms).std()
    else:
        stddev_plugin_ms = 0

    # Create and return summary dictionary
    return {
        "proposal": prop,
        "iterations": num_iterations,
        "total_algorithms": total_algorithms,
        "total_time_ms": total_time_ms,
        "total_time_plugin_ms": total_time_plugin_ms,
        "avg_time_in_plugin_per_iter_ms": avg_time_in_plugin_per_iter_ms,
        "avg_elapsed_time_per_iter_ms": avg_elapsed_time_per_iter_ms,
        "stddev_time_ms": stddev_time_ms,
        "stddev_plugin_ms": stddev_plugin_ms,
    }


def generate_plugin_report(output_dir, proposals, df_plugin_timing, log_message):
    """Generate the plugin timing report"""
    with open(f"{output_dir}/plugin_report.txt", "w") as report_file:
        report_file.write("StrongSwan QKD Plugin Performance Test Results\n")
        report_file.write("===============================================\n\n")

        for prop in proposals:

            report_file.write(f"Proposal: {prop}\n")
            report_file.write("=" * (len(prop) + 10) + "\n\n")
            report_file.write("\n")

            # Plugin timing data (main focus)
            plugin_data = (
                df_plugin_timing[df_plugin_timing["proposal"] == prop]
                if not df_plugin_timing.empty
                else pd.DataFrame()
            )

            if not plugin_data.empty:
                row = plugin_data.iloc[0]

                # Check which columns exist before trying to access them
                available_cols = row.index.tolist()

                report_file.write("Plugin Performance Analysis:\n")

                if "total_algorithms" in available_cols:
                    report_file.write(
                        f"  Plugin Operations Count: {row['total_algorithms']}\n"
                    )

                if "total_time_ms" in available_cols:
                    report_file.write(
                        f"  Plugin Sum of Times: {row['total_time_ms']:.2f} ms\n"
                    )
                    if "stddev_time_ms" in available_cols:
                        report_file.write(
                            f"  Plugin Sum Std Dev: {row['stddev_time_ms']:.2f} ms\n"
                        )

                # Add the new metric - total plugin time (first creation to last destruction)
                if "total_time_plugin_ms" in available_cols:
                    report_file.write(
                        f"  Plugin Total Time (first to last): {row['total_time_plugin_ms']:.2f} ms\n"
                    )
                    if "stddev_plugin_ms" in available_cols:
                        report_file.write(
                            f"  Plugin Total Time Std Dev: {row['stddev_plugin_ms']:.2f} ms\n"
                        )

                # Average times with standard deviations
                if "avg_time_in_plugin_per_iter_ms" in available_cols:
                    report_file.write(
                        f"  Plugin Avg Time Per Iteration: {row['avg_time_in_plugin_per_iter_ms']:.2f} ms\n"
                    )

                if "avg_elapsed_time_per_iter_ms" in available_cols:
                    report_file.write(
                        f"  Plugin Avg Time Per Algorithm: {row['avg_elapsed_time_per_iter_ms']:.2f} ms\n"
                    )

                # Average time per operation if we have both pieces of data
                if (
                    "total_time_ms" in available_cols
                    and "total_algorithms" in available_cols
                    and row["total_algorithms"] > 0
                ):
                    avg_time_per_op = row["total_time_ms"] / row["total_algorithms"]
                    report_file.write(
                        f"  Plugin Avg Time Per Operation: {avg_time_per_op:.2f} ms\n"
                    )
            else:
                report_file.write("Plugin Timing Data: Not available\n")

            report_file.write("\n" + "=" * 80 + "\n\n")

    log_message(f"Plugin report generated in {output_dir}/plugin_report.txt")
