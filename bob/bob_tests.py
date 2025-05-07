#!/usr/bin/env python3
"""
Bob testing script for QKD-enabled strongSwan testing environment.
"""

import subprocess
import time
import socket
import re
import os
import yaml

# Constants
SERVER_IP = "172.30.0.3"  # Alice's IP in the Docker network
PORT = 12345
OUTPUT_DIR = "/output"
BOB_LOG_FILE = f"{OUTPUT_DIR}/bob_log.txt"
CONFIG_FILE = "/etc/swanctl/swanctl.conf"
PROPOSALS_CONFIG_PATH = "/etc/swanctl/shared/proposals_config.yml"
MAX_RETRIES = 5
DEFAULT_ITERATIONS = 3

# Function to log messages to the file
def log_to_file(message):
    with open(BOB_LOG_FILE, "a") as log_file:
        log_file.write(f"{message}\n")

# Function to run a command with proper environment sourcing
def run_cmd(cmd, capture_output=False, start_new_session=False):
    # Prepend source command to ensure environment is loaded
    full_cmd = f"source /set_env.sh && {cmd}"
    
    if cmd == "/charon":
        # Use tee to capture the output to the log file
        tee_cmd = f"{full_cmd} 2>&1 | tee -a {BOB_LOG_FILE}"
        return subprocess.Popen(["bash", "-c", tee_cmd], 
                               start_new_session=True)
    # Run through bash to handle the source command
    elif capture_output:
        return subprocess.run(["bash", "-c", full_cmd], capture_output=True, text=True)
    elif start_new_session:
        return subprocess.Popen(["bash", "-c", full_cmd], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL, 
                               start_new_session=True)
    else:
        return subprocess.run(["bash", "-c", full_cmd])

def load_proposals():
    """
    Load the proposals from the configuration file.
    
    Returns:
        tuple: (proposals, esp_proposals)
    """
    try:
        with open(PROPOSALS_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            proposals = config['proposals']
            # Check if esp_proposals is defined in config file
            if 'esp_proposals' in config:
                esp_proposals = config['esp_proposals']
            else:
                # Assuming ESP proposals are the same as IKE proposals
                esp_proposals = proposals.copy()
            log_to_file(f"Loaded proposals from {PROPOSALS_CONFIG_PATH}: {proposals}")
            return proposals, esp_proposals
    except Exception as e:
        print(f"Error loading proposal configuration: {e}. Using defaults.")
        print(f"Error loading proposal configuration: {e}. Using defaults.")
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

def main():
    """Main function for Bob testing script."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize log file
    with open(BOB_LOG_FILE, "w") as log_file:
        log_file.write("StrongSwan QKD Plugin Test - Bob Log\n")
        log_file.write("====================================\n\n")
    
    # Load proposals
    proposals, esp_proposals = load_proposals()
    
    # Create socket TCP/IP
    print(f"Connecting to Alice at {SERVER_IP}:{PORT}...")
    log_to_file(f"Connecting to Alice at {SERVER_IP}:{PORT}...")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Try to connect with retry
    retries = 0
    connected = False

    while not connected and retries < MAX_RETRIES:
        try:
            client_socket.connect((SERVER_IP, PORT))
            connected = True
            print("Connected to server")
            log_to_file("Connected to server")
        except socket.error as e:
            retries += 1
            print(f"Connection attempt {retries} failed: {e}. Retrying in 5 seconds...")
            log_to_file(f"Connection attempt {retries} failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

    if not connected:
        print("Failed to connect after maximum retries. Exiting.")
        log_to_file("Failed to connect after maximum retries. Exiting.")
        return 1
        
    # Receive number of iterations from Alice
    try:
        iterations_str = client_socket.recv(1024).decode()
        NUM_ITERATIONS = int(iterations_str)
        print(f"Received number of iterations from Alice: {NUM_ITERATIONS}")
        log_to_file(f"Received number of iterations from Alice: {NUM_ITERATIONS}")
    except (ValueError, TypeError) as e:
        print(f"Error parsing iterations, using default: {e}")
        log_to_file(f"Error parsing iterations, using default: {e}")
        NUM_ITERATIONS = DEFAULT_ITERATIONS

    with open(CONFIG_FILE, "r") as file:
        config_data = file.read()
        
    for prop, esp_prop in zip(proposals, esp_proposals):
        log_to_file(f"Testing proposal: {prop}, ESP: {esp_prop}")
        
        # Regular expression to search and replace
        config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
        config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

        with open(CONFIG_FILE, "w") as file:
            file.write(config_data)

        print("Configuration file updated.")
        log_to_file("Configuration file updated.")

        for i in range(1, NUM_ITERATIONS + 1):
            print(f"Iteration {i}/{NUM_ITERATIONS}")
            log_to_file(f"Iteration {i}/{NUM_ITERATIONS}")

            print("Executing strongSwan...")
            log_to_file("Executing strongSwan...")
            
            # Add a separator before StrongSwan output
            log_to_file("\n----- StrongSwan Output Start -----\n")
            
            # This will capture StrongSwan output directly to bob_log.txt
            strongswan_proc = run_cmd("/charon", start_new_session=True)

            time.sleep(3)  # Waiting 'charon' to be ready
        
            # Add a separator after StrongSwan output
            log_to_file("\n----- StrongSwan Output End -----\n")
            
            print("Send ACK to Alice")
            log_to_file("Send ACK to Alice")
            client_socket.send("0".encode()) 
            
            print("Wait ACK from Alice")
            log_to_file("Wait ACK from Alice")
            _ = client_socket.recv(1024).decode()

            print("Stop strongSwan...")
            log_to_file("Stop strongSwan...")
            subprocess.run(["pkill", "-f", "charon"])
            print("\n\n")
            log_to_file("\n")
            time.sleep(1)
        
        log_to_file(f"Completed testing with proposal: {prop}\n\n")

    client_socket.close()
    print("Test completed. Log saved to " + BOB_LOG_FILE)
    log_to_file("Test completed. Log saved to " + BOB_LOG_FILE)
    return 0

if __name__ == "__main__":
    exit(main())