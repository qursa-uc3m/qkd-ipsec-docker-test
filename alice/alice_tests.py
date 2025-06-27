#!/usr/bin/env python3
# Copyright (C) 2024-2025 Javier Blanco-Romero @fj-blanco (UC3M, QURSA project)
#
# alice/alice_tests.py
# Alice: Testing Script with Raw Plugin Timing Collection and Distributed Mode Support

import argparse
import os
import pandas as pd
import re
import socket
import subprocess
import requests
import time
import yaml
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

from utils.utils import (
    run_cmd,
)  # Remove alice. prefix since you're running from alice/ directory
from data_proc.plugin_proc import (
    collect_raw_plugin_timing_data,
    process_plugin_timing_data,
    aggregate_plugin_timing,
    generate_plugin_report,
)
from data_proc.pcap_proc import (
    process_ike_data,
    generate_pcap_report,
)


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
    parser.add_argument(
        "--remote-ip",
        type=str,
        help="IP address of Bob's machine (for distributed mode)",
    )
    parser.add_argument(
        "--coordination-port",
        type=int,
        default=8080,
        help="Port for coordination with Bob (default: 8080)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run in distributed mode (wait for Bob)",
    )
    return parser.parse_args()


class CoordinationServer:
    """HTTP server for coordinating with Bob"""

    def __init__(self, port, alice_instance):
        self.port = port
        self.alice = alice_instance
        self.server = None
        self.server_thread = None
        self.bob_ready = False
        self.alice_ready = False

    class CoordinationHandler(BaseHTTPRequestHandler):
        def __init__(self, alice_instance, *args, **kwargs):
            self.alice = alice_instance
            super().__init__(*args, **kwargs)

        def do_GET(self):
            if self.path == "/status":
                status = {
                    "alice_ready": self.alice.alice_ready,
                    "bob_ready": self.alice.bob_ready,
                }
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/bob_ready":
                self.alice.bob_ready = True
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "acknowledged"}')
            elif self.path == "/bob_connecting":
                # Bob is ready to connect for next test
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "proceed"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress default HTTP server logging
            pass

    def start(self):
        """Start the coordination server"""
        handler = lambda *args, **kwargs: self.CoordinationHandler(
            self.alice, *args, **kwargs
        )
        self.server = HTTPServer(("0.0.0.0", self.port), handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def stop(self):
        """Stop the coordination server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=5)


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


def capture_and_process_traffic(prop, output_dir, log_message):
    """Capture network traffic and process IKE data for a proposal"""
    ts_res = f"{output_dir}/capture_{prop}.pcap"
    tshark_proc = run_cmd(
        f"tshark -w {ts_res}", start_new_session=True, output_dir=output_dir
    )
    log_message("Capturing traffic with tshark...")

    # Return the tshark process to be stopped later
    return tshark_proc, ts_res


def establish_connection(
    host,
    port,
    num_iterations,
    log_message,
    distributed_mode=False,
    coordination_server=None,
):
    """Establish connection with Bob and exchange initial information"""
    if distributed_mode and coordination_server:
        log_message("=== Distributed Mode Coordination ===")
        log_message("Alice is ready and waiting for Bob...")
        coordination_server.alice.alice_ready = True

        # Wait for Bob to signal ready
        log_message(
            f"Waiting for Bob to connect on coordination port {coordination_server.port}..."
        )
        timeout = 300  # 5 minute timeout
        start_time = time.time()

        while (
            not coordination_server.alice.bob_ready
            and (time.time() - start_time) < timeout
        ):
            time.sleep(1)

        if not coordination_server.alice.bob_ready:
            raise Exception("Timeout waiting for Bob to signal ready")

        log_message("Bob is ready! Proceeding with test coordination...")
        time.sleep(2)  # Brief pause for final setup

    # Create socket TCP/IP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


def run_test_iteration(
    i, num_iterations, conn, log_message, distributed_mode=False, remote_ip=None
):
    """Run a single test iteration with Bob"""
    log_message(f"Iteration {i}/{num_iterations}")

    if distributed_mode and remote_ip:
        # In distributed mode, notify Bob we're ready for this iteration
        try:

            response = requests.post(f"http://{remote_ip}:8080/alice_ready", timeout=10)
            log_message("Notified Bob that Alice is ready for iteration")
        except Exception as e:
            log_message(f"Warning: Could not notify Bob via coordination: {e}")

    log_message("Waiting for Bob to execute 'charon'...")
    data = conn.recv(1024).decode()

    # Add a separator before StrongSwan output
    log_message("\n----- StrongSwan Output Start -----\n")

    log_message("Executing strongSwan...")
    strongswan_proc = run_cmd("/charon", start_new_session=True, output_dir=OUTPUT_DIR)
    time.sleep(2)  # Waiting 'charon' to be ready

    log_message("Starting strongSwan SA...")
    init_output = run_cmd(
        "swanctl --initiate --child net", capture_output=True, output_dir=OUTPUT_DIR
    )
    # Log the output of the swanctl command if available
    if init_output.stdout:
        log_message(init_output.stdout)

    time.sleep(2)

    log_message("Stop strongSwan...")
    subprocess.run(["pkill", "-f", "charon"])
    time.sleep(0.5)

    # Add a separator after StrongSwan output
    log_message("\n----- StrongSwan Output End -----\n")

    log_message("Sending ACK to Bob ...")
    conn.send("0".encode())

    log_message("\n\n")


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

    # Distributed mode detection
    distributed_mode = args.distributed or args.remote_ip is not None
    coordination_port = args.coordination_port

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
    if distributed_mode:
        log_message(f"Running in DISTRIBUTED mode")
        log_message(f"Bob's IP: {args.remote_ip}")
        log_message(f"Coordination port: {coordination_port}")
    else:
        log_message(f"Running in SINGLE MACHINE mode")
    log_message(f"Starting test with {NUM_ITERATIONS} iterations per proposal")

    # Initialize coordination server for distributed mode
    coordination_server = None
    if distributed_mode:
        # Create alice instance with log_message method
        alice_instance = type(
            "obj",
            (object,),
            {"alice_ready": False, "bob_ready": False, "log_message": log_message},
        )()

        coordination_server = CoordinationServer(coordination_port, alice_instance)
        coordination_server.alice = alice_instance
        coordination_server.start()
        log_message(f"Started coordination server on port {coordination_port}")

    # Initialize data structures
    df_counters = pd.DataFrame()  # For request/response counts (backward compatibility)
    df_plugin_timing = pd.DataFrame()  # For aggregated plugin timing
    df_raw_plugin_timing = (
        pd.DataFrame()
    )  # For raw individual plugin timing measurements
    df_bytes = pd.DataFrame()  # For pcap measurements (bytes + detailed latencies)

    try:
        # Establish connection with Bob
        conn, server_socket = establish_connection(
            HOST,
            PORT,
            NUM_ITERATIONS,
            log_message,
            distributed_mode,
            coordination_server,
        )

        # Process each proposal
        for prop, esp_prop in zip(proposals, esp_proposals):
            log_message(f"Testing proposal: {prop}, ESP: {esp_prop}")

            # Update configuration
            update_config(CONFIG_FILE, prop, esp_prop, log_message)

            # Start traffic capture
            tshark_proc, ts_res = capture_and_process_traffic(
                prop, OUTPUT_DIR, log_message
            )

            # Store timing data for this proposal
            proposal_iterations = []

            # Run test iterations
            for i in range(1, NUM_ITERATIONS + 1):
                reset_timing_log(PLUGIN_TIMING_LOG, log_message)
                run_test_iteration(
                    i,
                    NUM_ITERATIONS,
                    conn,
                    log_message,
                    distributed_mode,
                    args.remote_ip,
                )

                # Collect raw plugin timing data for this iteration
                raw_timing_data = collect_raw_plugin_timing_data(
                    PLUGIN_TIMING_LOG, prop, i, log_message
                )

                # Append to the master raw timing DataFrame
                if not raw_timing_data.empty:
                    df_raw_plugin_timing = pd.concat(
                        [df_raw_plugin_timing, raw_timing_data], ignore_index=True
                    )

                # Process plugin timing data for aggregated stats
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

            # Process captured data with tshark-based analysis
            log_message(f"Processing PCAP data for {prop} using tshark...")

            try:
                # Process the IKE data from the captured PCAP file
                combined_stats, detailed_latencies = process_ike_data(
                    ts_res, OUTPUT_DIR, prop, log_message, NUM_ITERATIONS
                )

                # Verify we got valid data
                if combined_stats.get("total_message_count", 0) == 0:
                    log_message(
                        f"WARNING: No IKE packets found for {prop} - check capture"
                    )
                else:
                    log_message(
                        f"Successfully processed {combined_stats['total_message_count']} IKE packets for {prop}"
                    )

            except Exception as e:
                log_message(f"ERROR processing PCAP for {prop}: {e}")
                # Create empty stats to prevent crashes
                combined_stats = {
                    "proposal": prop,
                    "total_ike_bytes": 0,
                    "total_message_count": 0,
                    "total_avg_latency": 0,
                }

                # Add empty stats for all standard exchanges
                exchanges = [
                    "ike_sa_init",
                    "ike_auth",
                    "ike_intermediate",
                    "create_child_sa",
                    "informational",
                ]
                fields = [
                    "count",
                    "bytes",
                    "avg_latency",
                    "min_latency",
                    "max_latency",
                    "std_latency",
                    "latency_count",
                ]

                for exchange in exchanges:
                    for field in fields:
                        combined_stats[f"{exchange}_{field}"] = 0

                # Add empty port stats
                for port in ["500", "4500"]:
                    combined_stats[f"port_{port}_count"] = 0
                    combined_stats[f"port_{port}_bytes"] = 0

            # Store enhanced data (bytes + detailed latencies)
            df_bytes = pd.concat(
                [df_bytes, pd.DataFrame([combined_stats])], ignore_index=True
            )

            # Aggregate timing data from all iterations for this proposal
            proposal_summary = aggregate_plugin_timing(
                proposal_iterations, prop, NUM_ITERATIONS, log_message
            )

            # Add to proposal timing dataframe
            df_plugin_timing = pd.concat(
                [df_plugin_timing, pd.DataFrame([proposal_summary])], ignore_index=True
            )

            log_message(f"Completed testing with proposal: {prop}\n\n")

        # Save DataFrames
        plugin_timing_file = f"{OUTPUT_DIR}/plugin_timing_summary.csv"
        raw_plugin_timing_file = f"{OUTPUT_DIR}/plugin_timing_raw.csv"
        pcap_measurements_file = f"{OUTPUT_DIR}/pcap_measurements.csv"

        df_plugin_timing.to_csv(plugin_timing_file, index=False)
        df_raw_plugin_timing.to_csv(raw_plugin_timing_file, index=False)
        df_bytes.to_csv(pcap_measurements_file, index=False)

        log_message(f"Plugin timing data stored in '{plugin_timing_file}'")
        log_message(f"Raw plugin timing data stored in '{raw_plugin_timing_file}'")
        log_message(
            f"PCAP measurements (bytes + detailed latencies) stored in '{pcap_measurements_file}'"
        )

        # Generate enhanced reports
        generate_plugin_report(OUTPUT_DIR, proposals, df_plugin_timing, log_message)

        generate_pcap_report(
            OUTPUT_DIR, proposals, df_plugin_timing, df_bytes, log_message
        )

        # Cleanup
        log_message("Test completed. Closing connections...")
        log_message(f"Log saved to {OUTPUT_DIR}/alice_log.txt")
        conn.close()
        server_socket.close()

    finally:
        # Always cleanup
        if log_file:
            log_file.close()
        if coordination_server:
            coordination_server.stop()
            log_message("Coordination server stopped")


if __name__ == "__main__":
    main()
