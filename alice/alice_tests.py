# Alice: Testing Script

import subprocess
import time
import socket
import re
import pandas as pd
import os
import sys

# Use environment variables or default values
HOST = "0.0.0.0"  
PORT = 12345
OUTPUT_DIR = "/output"

config_file = "/etc/swanctl/swanctl.conf"

proposals = [
    "aes128-sha256-x25519",
    "aes128-sha256-x448"
]

esp_proposals = [
    "aes128-sha256-x25519",
    "aes128-sha256-x448"
]

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


df = pd.DataFrame()

# Create socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Wait for Bob to connect to port {PORT}...")

conn, addr = server_socket.accept()
print(f"Bob is connected with {addr}")

# Read the configuration file
with open(config_file, "r") as file:
    config_data = file.read()

for prop, esp_prop in zip(proposals, esp_proposals):
    
    print("Update 'swanctl.conf'")
    print("\tproposals: " + prop)
    print("\tesp_proposals: " + esp_prop)
    print("\n")
    
    # Regular expression to search and replace
    config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
    config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

    with open(config_file, "w") as file:
        file.write(config_data)

    print("Configuration file updated.")

    ts_res = f"{OUTPUT_DIR}/capture_{prop}.pcap"
    tshark_proc = run_cmd(f"tshark -w {ts_res}", start_new_session=True)
    print("Capturing traffic with tshark...")

    for i in range(1, 4):
        print(f"Iteration {i}")
        print("Waiting for Bob to execute 'charon'...")
        data = conn.recv(1024).decode()
        
        print("Executing strongSwan...")
        strongswan_proc = run_cmd("/charon", start_new_session=True)
        time.sleep(3)  # Waiting 'charon' to be ready
        
        print("Starting strongSwan SA...")
        run_cmd("swanctl --initiate --child net")

        time.sleep(3)
        
        print("Stop strongSwan...")
        subprocess.run(["pkill", "-f", "charon"])
        time.sleep(1)
        
        print("Sending ACK to Bob ...")
        conn.send("0".encode())
        
        print("\n\n")

    # Getting time info from ".pcap" and save it
    print("Stop tshark...")
    tshark_proc.terminate()  
    time.sleep(2)

    filter_ts = f"{OUTPUT_DIR}/results_{prop}.txt"
    print("Extracting time data from IKE_SA_INIT messages...")
    with open(filter_ts, "w") as result_file:
        # Use the helper function
        tshark_output = run_cmd(f"tshark -r {ts_res} -Y isakmp", capture_output=True)
        filter_output = subprocess.run(
            ["grep", "IKE_SA_INIT"], input=tshark_output.stdout, capture_output=True, text=True
        )
        result_file.write(filter_output.stdout)
    print("Process completed. Data stored in " + filter_ts)

    initiator_times = []
    responder_times = []

    with open(filter_ts, "r") as file:
        for line in file:
            parts = line.split()
            if len(parts) > 1:
                hs_time = float(parts[1])  
                if "Initiator Request" in line:
                    initiator_times.append(hs_time)
                elif "Responder Response" in line:
                    responder_times.append(hs_time)

    # Computing time differences
    latencies = [resp - init for init, resp in zip(initiator_times, responder_times)]

    # Add column to CSV
    df[prop] = latencies

output_file = f"{OUTPUT_DIR}/results.csv"
df.to_csv(output_file, index=False)

print(f"Differences stored in '{output_file}'")

# Generate a simple report
with open(f"{OUTPUT_DIR}/report.txt", "w") as report_file:
    report_file.write("StrongSwan QKD Plugin Performance Test Results\n")
    report_file.write("==============================================\n\n")
    
    for prop in proposals:
        avg_latency = df[prop].mean()
        min_latency = df[prop].min()
        max_latency = df[prop].max()
        
        report_file.write(f"Proposal: {prop}\n")
        report_file.write(f"Average Latency: {avg_latency:.6f} seconds\n")
        report_file.write(f"Min Latency: {min_latency:.6f} seconds\n")
        report_file.write(f"Max Latency: {max_latency:.6f} seconds\n\n")

print(f"Report generated in {OUTPUT_DIR}/report.txt")
conn.close()