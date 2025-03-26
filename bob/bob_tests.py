# Bob: Testing Script

import subprocess
import time
import socket
import re
import os

# Use environment variables or configuration values
SERVER_IP = "172.30.0.3"  # Alice's IP in the Docker network
PORT = 12345
OUTPUT_DIR = "/output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

config_file = "/etc/swanctl/swanctl.conf"

proposals = [
    "aes128-sha256-ecp256",
    "aes128-sha256-x25519",
    #"aes128-sha256-kyber1",
    #"aes128-sha256-hqc1",
    "aes128-sha256-qkd", 
    "aes128-sha256-qkd_kyber1",
    #"aes128-sha256-qkd_hqc1",
]

esp_proposals = [
    "aes128-sha256-ecp256",
    "aes128-sha256-x25519",
    #"aes128-sha256-kyber1",
    #"aes128-sha256-hqc1",
    "aes128-sha256-qkd", 
    "aes128-sha256-qkd_kyber1",
    #"aes128-sha256-qkd_hqc1",
]

# Function to run a command with proper environment sourcing
def run_cmd(cmd, capture_output=False, start_new_session=False):
    # Prepend source command to ensure environment is loaded
    full_cmd = f"source /set_env.sh && {cmd}"
    
    # Run through bash to handle the source command
    if capture_output:
        return subprocess.run(["bash", "-c", full_cmd], capture_output=True, text=True)
    elif start_new_session:
        return subprocess.Popen(["bash", "-c", full_cmd], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL, 
                               start_new_session=True)
    else:
        return subprocess.run(["bash", "-c", full_cmd])

# Create socket TCP/IP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to Alice at {SERVER_IP}:{PORT}...")

# Try to connect with retry
max_retries = 5
retries = 0
connected = False

while not connected and retries < max_retries:
    try:
        client_socket.connect((SERVER_IP, PORT))
        connected = True
        print("Connected to server")
    except socket.error as e:
        retries += 1
        print(f"Connection attempt {retries} failed: {e}. Retrying in 5 seconds...")
        time.sleep(5)

if not connected:
    print("Failed to connect after maximum retries. Exiting.")
    exit(1)
    
# Receive number of iterations from Alice
try:
    iterations_str = client_socket.recv(1024).decode()
    NUM_ITERATIONS = int(iterations_str)
    print(f"Received number of iterations from Alice: {NUM_ITERATIONS}")
except (ValueError, TypeError) as e:
    print(f"Error parsing iterations, using default: {e}")
    NUM_ITERATIONS = 3

with open(config_file, "r") as file:
    config_data = file.read()

# Log file for Bob
with open(f"{OUTPUT_DIR}/bob_log.txt", "w") as log_file:
    log_file.write("StrongSwan QKD Plugin Test - Bob Log\n")
    log_file.write("====================================\n\n")
    
    for prop, esp_prop in zip(proposals, esp_proposals):
        log_file.write(f"Testing proposal: {prop}, ESP: {esp_prop}\n")
        
        # Regular expression to search and replace
        config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
        config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

        with open(config_file, "w") as file:
            file.write(config_data)

        print("Configuration file updated.")
        log_file.write("Configuration file updated.\n")

        for i in range(1, NUM_ITERATIONS + 1):
            print(f"Iteration {i}/{NUM_ITERATIONS}")
            log_file.write(f"Iteration {i}/{NUM_ITERATIONS}\n")

            print("Executing strongSwan...")
            log_file.write("Executing strongSwan...\n")
            strongswan_proc = run_cmd("/charon", start_new_session=True)

            time.sleep(3)  # Waiting 'charon' to be ready
        
            print("Send ACK to Alice")
            log_file.write("Send ACK to Alice\n")
            client_socket.send("0".encode()) 
            
            print("Wait ACK from Alice")
            log_file.write("Wait ACK from Alice\n")
            rsp = client_socket.recv(1024).decode()

            print("Stop strongSwan...")
            log_file.write("Stop strongSwan...\n")
            subprocess.run(["pkill", "-f", "charon"])
            print("\n\n")
            log_file.write("\n")
            time.sleep(1)
        
        log_file.write("Completed testing with proposal: " + prop + "\n\n")

client_socket.close()
print("Test completed. Log saved to " + OUTPUT_DIR + "/bob_log.txt")