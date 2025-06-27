#!/usr/bin/env python3
"""
Bob testing script for QKD-enabled strongSwan testing environment.
Updated with distributed mode support and coordination.
"""

import subprocess
import time
import socket
import re
import os
import yaml
import json
import requests
import argparse
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

# Constants
DEFAULT_SERVER_IP = "172.30.0.3"  # Alice's IP in Docker network (single machine)
PORT = 12345
OUTPUT_DIR = "/output"
BOB_LOG_FILE = f"{OUTPUT_DIR}/bob_log.txt"
CONFIG_FILE = "/etc/swanctl/swanctl.conf"
PROPOSALS_CONFIG_PATH = "/etc/swanctl/shared/proposals_config.yml"
MAX_RETRIES = 30  # Increased for distributed mode
DEFAULT_ITERATIONS = 3
COORDINATION_PORT = 8080


def parse_arguments():
    """Parse command line arguments for Bob"""
    parser = argparse.ArgumentParser(description="QKD-IPSec Testing Script (Bob)")
    parser.add_argument(
        "--alice-ip",
        type=str,
        help="IP address of Alice's machine (for distributed mode)",
    )
    parser.add_argument(
        "--coordination-port",
        type=int,
        default=COORDINATION_PORT,
        help="Port for Bob's coordination server (default: 8080)",
    )
    parser.add_argument(
        "--alice-coordination-port",
        type=int,
        default=COORDINATION_PORT + 1,
        help="Port for Alice's coordination server (default: 8081)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run in distributed mode (coordinate with Alice)",
    )
    return parser.parse_args()


class BobCoordinationServer:
    """HTTP server for Bob to coordinate with Alice"""

    def __init__(self, port):
        self.port = port
        self.server = None
        self.server_thread = None
        self.alice_ready_for_iteration = False

    class CoordinationHandler(BaseHTTPRequestHandler):
        def __init__(self, bob_instance, *args, **kwargs):
            self.bob = bob_instance
            super().__init__(*args, **kwargs)

        def do_POST(self):
            if self.path == "/alice_ready":
                self.bob.alice_ready_for_iteration = True
                log_to_file("Alice is ready for next iteration")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "acknowledged"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress default HTTP server logging
            pass

    def start(self):
        """Start the coordination server"""
        handler = lambda *args, **kwargs: self.CoordinationHandler(
            self, *args, **kwargs
        )
        self.server = HTTPServer(("0.0.0.0", self.port), handler)
        self.server_thread = Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def stop(self):
        """Stop the coordination server"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=5)


# Function to log messages to the file
def log_to_file(message):
    with open(BOB_LOG_FILE, "a") as log_file:
        log_file.write(f"{message}\n")
    print(message)  # Also print to console


# Function to run a command with proper environment sourcing
def run_cmd(cmd, capture_output=False, start_new_session=False):
    # Prepend source command to ensure environment is loaded
    full_cmd = f"source /set_env.sh && {cmd}"

    if cmd == "/charon":
        # Use tee to capture the output to the log file
        tee_cmd = f"{full_cmd} 2>&1 | tee -a {BOB_LOG_FILE}"
        return subprocess.Popen(["bash", "-c", tee_cmd], start_new_session=True)
    # Run through bash to handle the source command
    elif capture_output:
        return subprocess.run(["bash", "-c", full_cmd], capture_output=True, text=True)
    elif start_new_session:
        return subprocess.Popen(
            ["bash", "-c", full_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    else:
        return subprocess.run(["bash", "-c", full_cmd])


def load_proposals():
    """
    Load the proposals from the configuration file.

    Returns:
        tuple: (proposals, esp_proposals)
    """
    try:
        with open(PROPOSALS_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            proposals = config["proposals"]
            # Check if esp_proposals is defined in config file
            if "esp_proposals" in config:
                esp_proposals = config["esp_proposals"]
            else:
                # Assuming ESP proposals are the same as IKE proposals
                esp_proposals = proposals.copy()
            log_to_file(f"Loaded proposals from {PROPOSALS_CONFIG_PATH}: {proposals}")
            return proposals, esp_proposals
    except Exception as e:
        print(f"Error loading proposal configuration: {e}. Using defaults.")
        log_to_file(f"Error loading proposal configuration: {e}. Using defaults.")
        # Default proposals if loading fails
        proposals = [
            "aes128-sha256-ecp256",
            "aes128-sha256-x25519",
            "aes128-sha256-kyber1",
            "aes128-sha256-qkd",
            "aes128-sha256-qkd_kyber1",
        ]
        esp_proposals = proposals.copy()
        return proposals, esp_proposals


def signal_alice_ready(alice_ip, alice_coordination_port):
    """Signal to Alice that Bob is ready"""
    try:
        url = f"http://{alice_ip}:{alice_coordination_port}/bob_ready"
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            log_to_file("Successfully signaled Alice that Bob is ready")
            return True
        else:
            log_to_file(f"Failed to signal Alice. Status code: {response.status_code}")
            return False
    except Exception as e:
        log_to_file(f"Error signaling Alice: {e}")
        return False


def wait_for_alice_coordination_server(alice_ip, alice_coordination_port, timeout=300):
    """Wait for Alice's coordination server to be available"""
    log_to_file(
        f"Waiting for Alice's coordination server at {alice_ip}:{alice_coordination_port}..."
    )
    start_time = time.time()

    while (time.time() - start_time) < timeout:
        try:
            url = f"http://{alice_ip}:{alice_coordination_port}/status"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                log_to_file("Alice's coordination server is available!")
                return True
        except Exception:
            pass

        elapsed = int(time.time() - start_time)
        if elapsed % 30 == 0:  # Log every 30 seconds
            log_to_file(f"Still waiting for Alice... ({elapsed}s elapsed)")

        time.sleep(5)

    log_to_file(f"Timeout waiting for Alice's coordination server after {timeout}s")
    return False


def wait_for_alice_iteration_ready(alice_ip, coordination_port, timeout=60):
    """Wait for Alice to be ready for the next iteration"""
    try:
        # Check Alice's status
        url = f"http://{alice_ip}:{coordination_port}/status"
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    if status.get("alice_ready", False):
                        return True
            except:
                pass
            time.sleep(1)

        log_to_file("Timeout waiting for Alice to be ready for iteration")
        return False
    except Exception as e:
        log_to_file(f"Error checking Alice status: {e}")
        return False


def connect_to_alice(
    server_ip,
    port,
    max_retries,
    distributed_mode=False,
    alice_ip=None,
    coordination_port=None,
    alice_coordination_port=None,
):
    """Connect to Alice with retry logic and coordination for distributed mode"""

    if distributed_mode and alice_ip:
        log_to_file("=== Distributed Mode Coordination ===")
        log_to_file("Bob starting coordination with Alice...")
        log_to_file(f"Bob's coordination port: {coordination_port}")
        log_to_file(f"Alice's coordination port: {alice_coordination_port}")

        # Start Bob's coordination server first
        bob_coordination = BobCoordinationServer(coordination_port)
        bob_coordination.start()
        log_to_file(f"Started Bob's coordination server on port {coordination_port}")

        # Wait indefinitely for Alice's coordination server to be available
        log_to_file("Waiting for Alice to start her coordination server...")
        log_to_file("Bob will wait indefinitely until Alice appears...")

        wait_count = 0
        while True:
            try:
                url = f"http://{alice_ip}:{alice_coordination_port}/status"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    log_to_file("✓ Alice's coordination server is available!")
                    break
            except Exception:
                pass

            wait_count += 1
            if wait_count % 6 == 0:  # Log every 30 seconds (6 * 5 seconds)
                elapsed_minutes = (wait_count * 5) // 60
                elapsed_seconds = (wait_count * 5) % 60
                log_to_file(
                    f"Still waiting for Alice... ({elapsed_minutes}m {elapsed_seconds}s elapsed)"
                )

            time.sleep(5)

        # Signal Alice that Bob is ready - also wait indefinitely
        log_to_file("Signaling Alice that Bob is ready...")
        while True:
            if signal_alice_ready(alice_ip, alice_coordination_port):
                log_to_file("✓ Successfully signaled Alice that Bob is ready")
                break
            else:
                log_to_file("Failed to signal Alice, retrying in 10 seconds...")
                time.sleep(10)

        log_to_file(
            "Coordination established. Proceeding to connect for data exchange..."
        )
        # Use Alice's IP for connection in distributed mode
        server_ip = alice_ip
    else:
        bob_coordination = None

    # Create socket TCP/IP for data exchange
    log_to_file(f"Connecting to Alice at {server_ip}:{port}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Try to connect with retry - also wait indefinitely in distributed mode
    retries = 0
    connected = False

    while not connected:
        try:
            client_socket.connect((server_ip, port))
            connected = True
            log_to_file("✓ Connected to Alice's test server")
        except socket.error as e:
            retries += 1
            if distributed_mode:
                # In distributed mode, wait indefinitely
                log_to_file(
                    f"Connection attempt {retries} failed: {e}. Retrying in 5 seconds..."
                )
                time.sleep(5)
            else:
                # In single machine mode, use max retries
                if retries >= max_retries:
                    log_to_file("Failed to connect after maximum retries. Exiting.")
                    if bob_coordination:
                        bob_coordination.stop()
                    return None, None, None
                log_to_file(
                    f"Connection attempt {retries}/{max_retries} failed: {e}. Retrying in 5 seconds..."
                )
                time.sleep(5)

    return client_socket, bob_coordination, server_ip


def main():
    """Main function for Bob testing script."""
    # Parse arguments first
    args = parse_arguments()

    # Determine mode and server IP
    distributed_mode = args.distributed or args.alice_ip is not None
    alice_ip = args.alice_ip
    coordination_port = args.coordination_port
    alice_coordination_port = args.alice_coordination_port

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize log file FIRST
    with open(BOB_LOG_FILE, "w") as log_file:
        log_file.write("StrongSwan QKD Plugin Test - Bob Log\n")
        log_file.write("====================================\n\n")

    # NOW add debug logging AFTER log file is initialized
    import sys

    log_to_file(f"DEBUG: Bob received command line arguments: {sys.argv}")
    log_to_file(f"DEBUG: Parsed arguments:")
    log_to_file(f"  - alice_ip: {args.alice_ip}")
    log_to_file(f"  - coordination_port: {args.coordination_port}")
    log_to_file(f"  - alice_coordination_port: {args.alice_coordination_port}")
    log_to_file(f"  - distributed: {args.distributed}")
    log_to_file(f"DEBUG: distributed_mode = {distributed_mode}")

    if distributed_mode:
        server_ip = alice_ip  # Use Alice's IP for connection
        log_to_file(f"Running in DISTRIBUTED mode")
        log_to_file(f"Alice IP: {alice_ip}")
        log_to_file(f"Bob's coordination port: {coordination_port}")
        log_to_file(f"Alice's coordination port: {alice_coordination_port}")
    else:
        server_ip = DEFAULT_SERVER_IP  # Use default Docker network IP
        log_to_file(f"Running in SINGLE MACHINE mode")
        log_to_file(f"Alice IP: {server_ip}")

    # Load proposals
    proposals, esp_proposals = load_proposals()

    # Connect to Alice with coordination
    client_socket, bob_coordination, final_server_ip = connect_to_alice(
        server_ip,
        PORT,
        MAX_RETRIES,
        distributed_mode,
        alice_ip,
        coordination_port,
        alice_coordination_port,
    )

    if not client_socket:
        log_to_file("DEBUG: connect_to_alice returned None, exiting")
        return 1

    try:
        # Receive number of iterations from Alice
        try:
            iterations_str = client_socket.recv(1024).decode()
            NUM_ITERATIONS = int(iterations_str)
            log_to_file(f"Received number of iterations from Alice: {NUM_ITERATIONS}")
        except (ValueError, TypeError) as e:
            log_to_file(f"Error parsing iterations, using default: {e}")
            NUM_ITERATIONS = DEFAULT_ITERATIONS

        # Read initial config file
        with open(CONFIG_FILE, "r") as file:
            config_data = file.read()

        # Process each proposal
        for prop, esp_prop in zip(proposals, esp_proposals):
            log_to_file(f"Testing proposal: {prop}, ESP: {esp_prop}")

            # Update configuration
            config_data = re.sub(
                r"(\bproposals\s*=\s*)[^\s]+", rf"\1{prop}", config_data
            )
            config_data = re.sub(
                r"(\besp_proposals\s*=\s*)[^\s]+", rf"\1{esp_prop}", config_data
            )

            with open(CONFIG_FILE, "w") as file:
                file.write(config_data)

            log_to_file("Configuration file updated.")

            # Run iterations for this proposal
            for i in range(1, NUM_ITERATIONS + 1):
                log_to_file(f"Iteration {i}/{NUM_ITERATIONS}")

                if distributed_mode and alice_ip:
                    # In distributed mode, wait for Alice to be ready for this iteration
                    log_to_file("Waiting for Alice to be ready for this iteration...")
                    if not wait_for_alice_iteration_ready(alice_ip, coordination_port):
                        log_to_file("Warning: Alice not ready, proceeding anyway")

                log_to_file("Executing strongSwan...")

                # Add a separator before StrongSwan output
                log_to_file("\n----- StrongSwan Output Start -----\n")

                # This will capture StrongSwan output directly to bob_log.txt
                strongswan_proc = run_cmd("/charon", start_new_session=True)

                time.sleep(3)  # Waiting 'charon' to be ready

                # Add a separator after StrongSwan output
                log_to_file("\n----- StrongSwan Output End -----\n")

                log_to_file("Send ACK to Alice")
                client_socket.send("0".encode())

                log_to_file("Wait ACK from Alice")
                _ = client_socket.recv(1024).decode()

                log_to_file("Stop strongSwan...")
                subprocess.run(["pkill", "-f", "charon"])
                log_to_file("\n")
                time.sleep(1)

            log_to_file(f"Completed testing with proposal: {prop}\n\n")

        log_to_file("Test completed. Log saved to " + BOB_LOG_FILE)

    finally:
        # Cleanup
        if client_socket:
            client_socket.close()
        if bob_coordination:
            bob_coordination.stop()
            log_to_file("Bob coordination server stopped")

    return 0


if __name__ == "__main__":
    exit(main())
