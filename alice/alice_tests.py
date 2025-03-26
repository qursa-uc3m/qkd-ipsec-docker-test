# Alice: Testing Script

import subprocess
import time
import socket
import re
import pandas as pd
import os
import sys
import argparse

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='QKD-IPSec Testing Script (Alice)')
    parser.add_argument('--iterations', type=int, default=3, 
                        help='Number of test iterations to run (default: 3)')
    return parser.parse_args()

def setup_logging(output_dir):
    """Setup logging to both console and file"""
    log_file_path = f"{output_dir}/alice_log.txt"
    log_file = open(log_file_path, "w")
    
    def log_message(message):
        print(message)  # Keep console output
        log_file.write(f"{message}\n")
        log_file.flush()  # Ensure it's written immediately
    
    return log_message, log_file

def run_cmd(cmd, capture_output=False, start_new_session=False, input_data=None):
    """Run a command with proper environment sourcing"""
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
    config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
    config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

    with open(config_file, "w") as file:
        file.write(config_data)

    log_message("Configuration file updated.")

def run_test_iteration(i, num_iterations, conn, log_message):
    """Run a single test iteration with Bob"""
    log_message(f"Iteration {i}/{num_iterations}")
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
            ["grep", "IKE_SA_INIT"], input=tshark_output.stdout, capture_output=True, text=True
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

def generate_report(output_dir, proposals, df_latencies, df_counters, log_message):
    """Generate the test report"""
    with open(f"{output_dir}/report.txt", "w") as report_file:
        report_file.write("StrongSwan QKD Plugin Performance Test Results\n")
        report_file.write("==============================================\n\n")
        
        for prop in proposals:
            if prop in df_latencies and not df_latencies[prop].empty:
                avg_latency = df_latencies[prop].mean()
                min_latency = df_latencies[prop].min()
                max_latency = df_latencies[prop].max()
            else:
                avg_latency = min_latency = max_latency = "N/A"
                
            init_count = df_counters.at[0, f"{prop}_init_requests"] if f"{prop}_init_requests" in df_counters else "N/A"
            resp_count = df_counters.at[0, f"{prop}_resp_responses"] if f"{prop}_resp_responses" in df_counters else "N/A"
            
            report_file.write(f"Proposal: {prop}\n")
            report_file.write(f"IKE_SA_INIT Initiator Requests: {init_count}\n")
            report_file.write(f"IKE_SA_INIT Responder Responses: {resp_count}\n")
            
            if isinstance(avg_latency, float):
                report_file.write(f"Average Latency: {avg_latency:.6f} seconds\n")
                report_file.write(f"Min Latency: {min_latency:.6f} seconds\n")
                report_file.write(f"Max Latency: {max_latency:.6f} seconds\n\n")
            else:
                report_file.write(f"Average Latency: {avg_latency}\n")
                report_file.write(f"Min Latency: {min_latency}\n")
                report_file.write(f"Max Latency: {max_latency}\n\n")

    log_message(f"Report generated in {output_dir}/report.txt")

def main():
    # Initialize parameters
    args = parse_arguments()
    HOST = "0.0.0.0"  
    PORT = 12345
    OUTPUT_DIR = "/output"
    NUM_ITERATIONS = args.iterations
    QKD_TIMING_LOG = "/tmp/qkd_timing.csv"
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

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup logging
    log_message, log_file = setup_logging(OUTPUT_DIR)
    
    # Initialize data structures
    df_counters = pd.DataFrame()  # For request/response counts
    df_latencies = pd.DataFrame()  # For latency measurements
    
    # Establish connection with Bob
    conn, server_socket = establish_connection(HOST, PORT, NUM_ITERATIONS, log_message)
    
    # Process each proposal
    for prop, esp_prop in zip(proposals, esp_proposals):
        # Update configuration
        update_config(config_file, prop, esp_prop, log_message)
        
        # Start traffic capture
        tshark_proc, ts_res = capture_and_process_traffic(prop, OUTPUT_DIR, log_message)
        
        # Run test iterations
        for i in range(1, NUM_ITERATIONS + 1):
            run_test_iteration(i, NUM_ITERATIONS, conn, log_message)
        
        # Stop traffic capture
        log_message("Stop tshark...")
        tshark_proc.terminate()  
        time.sleep(2)
        
        # Process captured data
        latencies, init_request_count, resp_response_count = process_ike_data(
            ts_res, OUTPUT_DIR, prop, log_message
        )
        
        # Store data
        df_counters.at[0, f"{prop}_init_requests"] = init_request_count
        df_counters.at[0, f"{prop}_resp_responses"] = resp_response_count
        df_latencies[prop] = latencies
        
        # Save DataFrames
        counters_file = f"{OUTPUT_DIR}/counters.csv"
        latencies_file = f"{OUTPUT_DIR}/latencies.csv"
        df_counters.to_csv(counters_file, index=False)
        df_latencies.to_csv(latencies_file, index=False)
        log_message(f"Counter data stored in '{counters_file}'")
        log_message(f"Latency data stored in '{latencies_file}'")
    
    # Generate report
    generate_report(OUTPUT_DIR, proposals, df_latencies, df_counters, log_message)
    
    # Cleanup
    log_file.close()
    conn.close()
    server_socket.close()

if __name__ == "__main__":
    main()