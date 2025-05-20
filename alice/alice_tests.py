# Alice: Testing Script

import argparse
import os
import pandas as pd
import re
import socket
import subprocess
import time
import yaml


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="QKD-IPSec Testing Script (Alice)")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of test iterations to run (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/output",
        help="Directory for storing output files (default: /output)",
    )
    return parser.parse_args()


def setup_logging(output_dir):
    """Setup logging to both console and file"""
    log_file_path = f"{output_dir}/alice_log.txt"

    # Initialize log file with clear formatting
    with open(log_file_path, "w") as log_file:
        log_file.write("StrongSwan QKD Plugin Test - Alice Log\n")
        log_file.write("====================================\n\n")

    log_file = open(log_file_path, "a")

    def log_message(message):
        print(message)  # Keep console output
        log_file.write(f"{message}\n")
        log_file.flush()  # Ensure it's written immediately

    return log_message, log_file


def load_proposals(config_path, args, log_message=None):
    """
    Load cryptographic proposals from the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file
        args: Command line arguments
        log_message: Optional logging function

    Returns:
        tuple: (proposals, esp_proposals, num_iterations)
    """
    # Default values if config loading fails
    default_proposals = [
        "aes128-sha256-ecp256",
        "aes128-sha256-x25519",
        "aes128-sha256-kyber1",
        "aes128-sha256-qkd",
        "aes128-sha256-qkd_kyber1",
    ]
    default_iterations = 3

    # Initialize with defaults
    proposals = default_proposals
    num_iterations = (
        args.iterations if args.iterations is not None else default_iterations
    )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

            # Load proposals
            if "proposals" in config:
                proposals = config["proposals"]

            # Use config file iterations only if not specified via command line
            if args.iterations is None and "test_iterations" in config:
                num_iterations = config.get("test_iterations", default_iterations)

            # Log loading success if log function is provided
            if log_message:
                log_message(f"Loaded proposals from {config_path}: {proposals}")

    except Exception as e:
        error_msg = f"Error loading proposal configuration: {e}. Using defaults."
        print(error_msg)

        # Log the error if log function is provided
        if log_message:
            log_message(error_msg)

    # Always create a copy of proposals for ESP proposals
    esp_proposals = proposals.copy()

    return proposals, esp_proposals, num_iterations


def run_cmd(cmd, capture_output=False, start_new_session=False, input_data=None):
    """Run a command with proper environment sourcing"""
    # Prepend source command to ensure environment is loaded
    full_cmd = f"source /set_env.sh && {cmd}"

    if cmd == "/charon":
        # Use tee to capture the output to the log file
        tee_cmd = f"{full_cmd} 2>&1 | tee -a {OUTPUT_DIR}/alice_log.txt"
        return subprocess.Popen(["bash", "-c", tee_cmd], start_new_session=True)
    # Run through bash to handle the source command
    elif capture_output:
        return subprocess.run(
            ["bash", "-c", full_cmd], capture_output=True, text=True, input=input_data
        )
    elif start_new_session:
        return subprocess.Popen(
            ["bash", "-c", full_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        return subprocess.run(["bash", "-c", full_cmd])


def establish_connection(host, port, num_iterations, log_message):
    """Establish connection with Bob and exchange initial information"""
    # Create socket TCP/IP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    log_message(f"Wait for Bob to connect to port {port}...")
    conn, addr = server_socket.accept()
    log_message(f"Bob is connected with {addr}")

    log_message(f"Sending number of iterations ({num_iterations}) to Bob...")
    conn.send(str(num_iterations).encode())
    time.sleep(0.5)

    return conn, server_socket


def update_config(config_file, prop, esp_prop, log_message):
    """Update the swanctl.conf file with new proposals"""
    log_message("Update 'swanctl.conf'")
    log_message("\tproposals: " + prop)
    log_message("\tesp_proposals: " + esp_prop)
    log_message("\n")

    # Read the configuration file
    with open(config_file, "r") as file:
        config_data = file.read()

    # Regular expression to search and replace
    config_data = re.sub(r"(\bproposals\s*=\s*)[^\s]+", rf"\1{prop}", config_data)
    config_data = re.sub(
        r"(\besp_proposals\s*=\s*)[^\s]+", rf"\1{esp_prop}", config_data
    )

    with open(config_file, "w") as file:
        file.write(config_data)

    log_message("Configuration file updated.")


def run_test_iteration(i, num_iterations, conn, log_message):
    """Run a single test iteration with Bob"""
    log_message(f"Iteration {i}/{num_iterations}")
    log_message("Waiting for Bob to execute 'charon'...")
    data = conn.recv(1024).decode()

    # Add a separator before StrongSwan output
    log_message("\n----- StrongSwan Output Start -----\n")

    log_message("Executing strongSwan...")
    strongswan_proc = run_cmd("/charon", start_new_session=True)
    time.sleep(3)  # Waiting 'charon' to be ready

    log_message("Starting strongSwan SA...")
    init_output = run_cmd("swanctl --initiate --child net", capture_output=True)
    # Log the output of the swanctl command if available
    if init_output.stdout:
        log_message(init_output.stdout)

    time.sleep(3)

    log_message("Stop strongSwan...")
    subprocess.run(["pkill", "-f", "charon"])
    time.sleep(1)

    # Add a separator after StrongSwan output
    log_message("\n----- StrongSwan Output End -----\n")

    log_message("Sending ACK to Bob ...")
    conn.send("0".encode())

    log_message("\n\n")


def capture_and_process_traffic(prop, output_dir, log_message):
    """Capture network traffic and process IKE data for a proposal"""
    ts_res = f"{output_dir}/capture_{prop}.pcap"
    tshark_proc = run_cmd(f"tshark -w {ts_res}", start_new_session=True)
    log_message("Capturing traffic with tshark...")

    # Return the tshark process to be stopped later
    return tshark_proc, ts_res


def process_ike_data(ts_res, output_dir, prop, log_message):
    """Process IKE data from captured traffic"""
    filter_ts = f"{output_dir}/results_{prop}.txt"
    log_message("Extracting time data from IKE_SA_INIT messages...")
    with open(filter_ts, "w") as result_file:
        tshark_output = run_cmd(f"tshark -r {ts_res} -Y isakmp", capture_output=True)
        filter_output = subprocess.run(
            ["grep", "IKE_SA_INIT"],
            input=tshark_output.stdout,
            capture_output=True,
            text=True,
        )
        result_file.write(filter_output.stdout)
    log_message(f"Process completed. Data stored in {filter_ts}")

    initiator_times = []
    responder_times = []
    init_request_count = 0
    resp_response_count = 0

    with open(filter_ts, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) > 1:
                hs_time = float(parts[1])
                if "Initiator Request" in line:
                    initiator_times.append(hs_time)
                    init_request_count += 1
                elif "Responder Response" in line:
                    responder_times.append(hs_time)
                    resp_response_count += 1

    # Computing time differences
    latencies = [resp - init for init, resp in zip(initiator_times, responder_times)]

    return latencies, init_request_count, resp_response_count


def reset_timing_log(log_file_path, log_message):
    try:
        with open(log_file_path, "w") as f:
            f.write(
                "method,create_timestamp,create_microseconds,destroy_timestamp,destroy_microseconds\n"
            )
        log_message(f"Initialized timing log at {log_file_path}")
        return True
    except Exception as e:
        log_message(f"Error initializing timing log: {e}")
        return False


def process_plugin_timing_data(plugin_timing_log, prop, log_message):
    """
    Process plugin timing data from temporary file for each proposal

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
        # Print raw file content for debugging
        with open(plugin_timing_log, "r") as f:
            content = f.read()
            log_message(f"Raw timing log content:\n{content}")
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


def generate_report(
    output_dir, proposals, df_latencies, df_counters, df_plugin_timing, log_message
):
    """Generate the test report"""
    with open(f"{output_dir}/report.txt", "w") as report_file:
        report_file.write("StrongSwan QKD Plugin Performance Test Results\n")
        report_file.write("==============================================\n\n")

        for prop in proposals:
            if prop in df_latencies and not df_latencies[prop].empty:
                avg_latency = df_latencies[prop].mean()
                min_latency = df_latencies[prop].min()
                max_latency = df_latencies[prop].max()
                std_latency = (
                    df_latencies[prop].std() if len(df_latencies[prop]) > 1 else 0
                )
            else:
                avg_latency = min_latency = max_latency = std_latency = "N/A"

            init_count = (
                df_counters.at[0, f"{prop}_init_requests"]
                if f"{prop}_init_requests" in df_counters
                else "N/A"
            )
            resp_count = (
                df_counters.at[0, f"{prop}_resp_responses"]
                if f"{prop}_resp_responses" in df_counters
                else "N/A"
            )

            report_file.write(f"Proposal: {prop}\n")
            report_file.write(f"IKE_SA_INIT Initiator Requests: {init_count}\n")
            report_file.write(f"IKE_SA_INIT Responder Responses: {resp_count}\n")

            if isinstance(avg_latency, float):
                report_file.write(f"Average Latency: {avg_latency:.6f} seconds\n")
                report_file.write(f"Standard Deviation: {std_latency:.6f} seconds\n")
                report_file.write(f"Min Latency: {min_latency:.6f} seconds\n")
                report_file.write(f"Max Latency: {max_latency:.6f} seconds\n")
            else:
                report_file.write(f"Average Latency: {avg_latency}\n")
                report_file.write(f"Standard Deviation: {std_latency}\n")
                report_file.write(f"Min Latency: {min_latency}\n")
                report_file.write(f"Max Latency: {max_latency}\n")

            # Add plugin timing data
            plugin_data = (
                df_plugin_timing[df_plugin_timing["proposal"] == prop]
                if not df_plugin_timing.empty
                else pd.DataFrame()
            )

            if not plugin_data.empty:
                row = plugin_data.iloc[0]

                # Check which columns exist before trying to access them
                available_cols = row.index.tolist()

                if "total_algorithms" in available_cols:
                    report_file.write(
                        f"Plugin Operations Count: {row['total_algorithms']}\n"
                    )

                if "total_time_ms" in available_cols:
                    report_file.write(
                        f"Plugin Sum of Times: {row['total_time_ms']:.2f} ms\n"
                    )
                    if "stddev_time_ms" in available_cols:
                        report_file.write(
                            f"Plugin Sum Std Dev: {row['stddev_time_ms']:.2f} ms\n"
                        )

                # Add the new metric - total plugin time (first creation to last destruction)
                if "total_time_plugin_ms" in available_cols:
                    report_file.write(
                        f"Plugin Total Time (first to last): {row['total_time_plugin_ms']:.2f} ms\n"
                    )
                    if "stddev_plugin_ms" in available_cols:
                        report_file.write(
                            f"Plugin Total Time Std Dev: {row['stddev_plugin_ms']:.2f} ms\n"
                        )

                # Average times with standard deviations
                if "avg_time_in_plugin_per_iter_ms" in available_cols:
                    report_file.write(
                        f"Plugin Avg Time Per Iteration: {row['avg_time_in_plugin_per_iter_ms']:.2f} ms\n"
                    )

                if "avg_elapsed_time_per_iter_ms" in available_cols:
                    report_file.write(
                        f"Plugin Avg Time Per Algorithm: {row['avg_elapsed_time_per_iter_ms']:.2f} ms\n"
                    )

                # Average time per operation if we have both pieces of data
                if (
                    "total_time_ms" in available_cols
                    and "total_algorithms" in available_cols
                    and row["total_algorithms"] > 0
                ):
                    avg_time_per_op = row["total_time_ms"] / row["total_algorithms"]
                    report_file.write(
                        f"Plugin Avg Time Per Operation: {avg_time_per_op:.2f} ms\n"
                    )

                # Calculate plugin contribution to total IKE latency if both are available
                if isinstance(avg_latency, float) and "total_time_ms" in available_cols:
                    avg_latency_ms = avg_latency * 1000  # Convert to ms
                    plugin_percentage = (
                        (row["total_time_ms"] / avg_latency_ms) * 100
                        if avg_latency_ms > 0
                        else 0
                    )
                    report_file.write(
                        f"Plugin Sum Contribution: {plugin_percentage:.2f}% of IKE latency\n"
                    )

                # Calculate plugin total time contribution to IKE latency
                if (
                    isinstance(avg_latency, float)
                    and "total_time_plugin_ms" in available_cols
                ):
                    avg_latency_ms = avg_latency * 1000  # Convert to ms
                    plugin_total_percentage = (
                        (row["total_time_plugin_ms"] / avg_latency_ms) * 100
                        if avg_latency_ms > 0
                        else 0
                    )
                    report_file.write(
                        f"Plugin Total Contribution: {plugin_total_percentage:.2f}% of IKE latency\n"
                    )
            else:
                report_file.write("Plugin Timing Data: Not available\n")

            report_file.write("\n")

    log_message(f"Report generated in {output_dir}/report.txt")


def main():
    # Initialize parameters
    args = parse_arguments()
    HOST = "0.0.0.0"
    PORT = 12345
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    NUM_ITERATIONS = args.iterations
    PLUGIN_TIMING_LOG = "/tmp/plugin_timing.csv"
    CONFIG_FILE = "/etc/swanctl/swanctl.conf"
    PROPOSALS_CONFIG_PATH = "/etc/swanctl/shared/proposals_config.yml"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup logging
    log_message, log_file = setup_logging(OUTPUT_DIR)

    # Load proposals from configuration file
    proposals, esp_proposals, NUM_ITERATIONS = load_proposals(
        PROPOSALS_CONFIG_PATH, args, log_message
    )

    log_message("StrongSwan QKD Plugin Test - Alice Log")
    log_message("====================================")
    log_message(f"Starting test with {NUM_ITERATIONS} iterations per proposal")

    # Initialize data structures
    df_counters = pd.DataFrame()  # For request/response counts
    df_latencies = pd.DataFrame()  # For latency measurements
    df_plugin_timing = pd.DataFrame()

    # Establish connection with Bob
    conn, server_socket = establish_connection(HOST, PORT, NUM_ITERATIONS, log_message)

    # Process each proposal
    for prop, esp_prop in zip(proposals, esp_proposals):
        log_message(f"Testing proposal: {prop}, ESP: {esp_prop}")

        # Update configuration
        update_config(CONFIG_FILE, prop, esp_prop, log_message)

        # Start traffic capture
        tshark_proc, ts_res = capture_and_process_traffic(prop, OUTPUT_DIR, log_message)

        # Store timing data for this proposal
        proposal_iterations = []

        # Run test iterations
        for i in range(1, NUM_ITERATIONS + 1):
            reset_timing_log(PLUGIN_TIMING_LOG, log_message)
            run_test_iteration(i, NUM_ITERATIONS, conn, log_message)
            # Process plugin timing data for this iteration - now separate from run_test_iteration
            iteration_stats = process_plugin_timing_data(
                PLUGIN_TIMING_LOG, f"iteration_{i}", log_message
            )

            # Store iteration timing with proposal information
            iteration_stats["proposal"] = prop
            iteration_stats["iteration"] = i
            proposal_iterations.append(iteration_stats)

        # Stop traffic capture
        log_message("Stop tshark...")
        tshark_proc.terminate()
        time.sleep(2)

        # Process captured data
        latencies, init_request_count, resp_response_count = process_ike_data(
            ts_res, OUTPUT_DIR, prop, log_message
        )

        # Aggregate timing data from all iterations for this proposal
        proposal_summary = aggregate_plugin_timing(
            proposal_iterations, prop, NUM_ITERATIONS, log_message
        )

        # Add to proposal timing dataframe
        df_plugin_timing = pd.concat(
            [df_plugin_timing, pd.DataFrame([proposal_summary])], ignore_index=True
        )

        # Store data
        df_counters.at[0, f"{prop}_init_requests"] = init_request_count
        df_counters.at[0, f"{prop}_resp_responses"] = resp_response_count
        df_latencies[prop] = pd.Series(latencies)

        log_message(f"Completed testing with proposal: {prop}\n\n")

    # Save DataFrames
    counters_file = f"{OUTPUT_DIR}/counters.csv"
    latencies_file = f"{OUTPUT_DIR}/latencies.csv"
    plugin_timing_file = f"{OUTPUT_DIR}/plugin_timing_summary.csv"

    df_counters.to_csv(counters_file, index=False)
    df_latencies.to_csv(latencies_file, index=False)
    df_plugin_timing.to_csv(plugin_timing_file, index=False)

    log_message(f"Counter data stored in '{counters_file}'")
    log_message(f"Latency data stored in '{latencies_file}'")
    log_message(f"Plugin timing data stored in '{plugin_timing_file}'")

    # Generate report
    generate_report(
        OUTPUT_DIR, proposals, df_latencies, df_counters, df_plugin_timing, log_message
    )

    # Cleanup
    log_message("Test completed. Closing connections...")
    log_message(f"Log saved to {OUTPUT_DIR}/alice_log.txt")
    log_file.close()
    conn.close()
    server_socket.close()


if __name__ == "__main__":
    main()
