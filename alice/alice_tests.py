# Alice: Testing Script

import subprocess
import time
import socket
import re
import pandas as pd
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='QKD-IPSec Testing Script (Alice)')
parser.add_argument('--iterations', type=int, default=3, 
                    help='Number of test iterations to run (default: 10)')
args = parser.parse_args()

# Use environment variables or default values
HOST = "0.0.0.0"  
PORT = 12345
OUTPUT_DIR = "/output"
NUM_ITERATIONS = args.iterations

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

# Setup logging to file
log_file_path = f"{OUTPUT_DIR}/alice_log.txt"
log_file = open(log_file_path, "w")

# Function to write to both console and log file
def log_message(message):
    print(message)  # Keep console output
    log_file.write(f"{message}\n")
    log_file.flush()  # Ensure it's written immediately

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to run a command with proper environment sourcing
def run_cmd(cmd, capture_output=False, start_new_session=False, input_data=None):
    # Prepend source command to ensure environment is loaded
    full_cmd = f"source /set_env.sh && {cmd}"
    
    # Run through bash to handle the source command
    if capture_output:
        return subprocess.run(["bash", "-c", full_cmd], capture_output=True, text=True, input=input_data)
    elif start_new_session:
        return subprocess.Popen(["bash", "-c", full_cmd], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL, 
                               start_new_session=True)
    else:
        return subprocess.run(["bash", "-c", full_cmd])


df_counters = pd.DataFrame()  # For request/response counts
df_latencies = pd.DataFrame()  # For latency measurements

# Create socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

log_message(f"Wait for Bob to connect to port {PORT}...")

conn, addr = server_socket.accept()
log_message(f"Bob is connected with {addr}")

log_message(f"Sending number of iterations ({NUM_ITERATIONS}) to Bob...")
conn.send(str(NUM_ITERATIONS).encode())
time.sleep(0.5)

# Read the configuration file
with open(config_file, "r") as file:
    config_data = file.read()

for prop, esp_prop in zip(proposals, esp_proposals):
    
    log_message("Update 'swanctl.conf'")
    log_message("\tproposals: " + prop)
    log_message("\tesp_proposals: " + esp_prop)
    log_message("\n")
    
    # Regular expression to search and replace
    config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
    config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

    with open(config_file, "w") as file:
        file.write(config_data)

    log_message("Configuration file updated.")

    ts_res = f"{OUTPUT_DIR}/capture_{prop}.pcap"
    tshark_proc = run_cmd(f"tshark -w {ts_res}", start_new_session=True)
    log_message("Capturing traffic with tshark...")

    for i in range(1, NUM_ITERATIONS + 1):
        log_message(f"Iteration {i}/{NUM_ITERATIONS}")
        log_message("Waiting for Bob to execute 'charon'...")
        data = conn.recv(1024).decode()
        
        log_message("Executing strongSwan...")
        strongswan_proc = run_cmd("/charon", start_new_session=True)
        time.sleep(3)  # Waiting 'charon' to be ready
        
        log_message("Starting strongSwan SA...")
        run_cmd("swanctl --initiate --child net")

        time.sleep(3)
        
        log_message("Stop strongSwan...")
        subprocess.run(["pkill", "-f", "charon"])
        time.sleep(1)
        
        log_message("Sending ACK to Bob ...")
        conn.send("0".encode())
        
        log_message("\n\n")

    # Getting time info from ".pcap" and save it
    log_message("Stop tshark...")
    tshark_proc.terminate()  
    time.sleep(2)

    filter_ts = f"{OUTPUT_DIR}/results_{prop}.txt"
    log_message("Extracting time data from IKE_SA_INIT messages...")
    with open(filter_ts, "w") as result_file:
        # Use the helper function
        tshark_output = run_cmd(f"tshark -r {ts_res} -Y isakmp", capture_output=True)
        filter_output = subprocess.run(
            ["grep", "IKE_SA_INIT"], input=tshark_output.stdout, capture_output=True, text=True
        )
        result_file.write(filter_output.stdout)
    log_message("Process completed. Data stored in " + filter_ts)

    initiator_times = []
    responder_times = []
    # Add counters for IKE_SA_INIT messages
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
    
    # Store counters in df_counters
    df_counters.at[0, f"{prop}_init_requests"] = init_request_count
    df_counters.at[0, f"{prop}_resp_responses"] = resp_response_count

    # Store latencies in df_latencies
    df_latencies[prop] = latencies

    # Save both DataFrames to separate CSV files
    counters_file = f"{OUTPUT_DIR}/counters.csv"
    latencies_file = f"{OUTPUT_DIR}/latencies.csv"

    df_counters.to_csv(counters_file, index=False)
    df_latencies.to_csv(latencies_file, index=False)

    log_message(f"Counter data stored in '{counters_file}'")
    log_message(f"Latency data stored in '{latencies_file}'")

# Generate a simple report
with open(f"{OUTPUT_DIR}/report.txt", "w") as report_file:
    report_file.write("StrongSwan QKD Plugin Performance Test Results\n")
    report_file.write("==============================================\n\n")
    
    for prop in proposals:
        avg_latency = df_latencies[prop].mean()
        min_latency = df_latencies[prop].min()
        max_latency = df_latencies[prop].max()
        init_count = df_counters.at[0, f"{prop}_init_requests"] if f"{prop}_init_requests" in df_counters else "N/A"
        resp_count = df_counters.at[0, f"{prop}_resp_responses"] if f"{prop}_resp_responses" in df_counters else "N/A"
        
        report_file.write(f"Proposal: {prop}\n")
        report_file.write(f"IKE_SA_INIT Initiator Requests: {init_count}\n")
        report_file.write(f"IKE_SA_INIT Responder Responses: {resp_count}\n")
        report_file.write(f"Average Latency: {avg_latency:.6f} seconds\n")
        report_file.write(f"Min Latency: {min_latency:.6f} seconds\n")
        report_file.write(f"Max Latency: {max_latency:.6f} seconds\n\n")

log_message(f"Report generated in {OUTPUT_DIR}/report.txt")

log_file.close()
conn.close()