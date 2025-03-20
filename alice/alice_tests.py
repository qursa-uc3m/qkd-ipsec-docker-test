# Alice: Testing Script

import subprocess
import time
import socket
import re
import pandas as pd

HOST = "0.0.0.0"  
PORT = 12345       

config_file = "/etc/swanctl/swanctl.conf"

proposals = [	"aes128-sha256-x25519",
		        "aes128-sha256-x448"	]

esp_proposals = [	"aes128-sha256-x25519",
			        "aes128-sha256-x448"	]

df = pd.DataFrame()

# Create socket TCP/IP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Wait for Bob to connect to port {PORT}...")

conn, addr = server_socket.accept()
print(f"Bob is connected with {addr}")

# Leer el archivo
with open(config_file, "r") as file:
    config_data = file.read()

for prop,esp_prop in zip(proposals,esp_proposals):
    
    print("Update 'swanctl.conf'")
    print("\tproposals: "+prop)
    print("\tesp_proposals: "+esp_prop)
    print("\n")
    
    # Regular expresion to search and replace
    config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
    config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

    with open(config_file, "w") as file:
        file.write(config_data)

    print("Configuration file updated.")

    ts_res = "/tmp/capture.pcap"
    tshark_proc = subprocess.Popen(
        ["sudo", "tshark", "-w", ts_res],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
    print("Capturing traffic with tshark...")

    for i in range(1, 4):
        print(f"Iteration {i}")
        print("Waiting Bob to execute 'charon'...")
        data = conn.recv(1024).decode()
        
        print("Executing strongSwan...")
        strongswan_proc = subprocess.Popen(
            ["sudo", "./charon"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )
        time.sleep(3)  # Waiting 'charon' to be ready
        
        print("Starting strongSwan SA...")
        subprocess.run(["sudo", "swanctl", "--initiate", "--child", "net"])
        
        time.sleep(3)
        
        print("Stop strongSwan...")
        subprocess.run(["sudo", "pkill", "-f", "charon"])
        time.sleep(1)
        
        print("Sending ACK to Bob ...")
        conn.send("0".encode())
        
        print("\n\n")

    # Getting time info from ".pcap" and safe it on ".csv"
    print("Stop tshark...")
    tshark_proc.terminate()  
    time.sleep(2)

    filter_ts = "/tmp/results.txt"
    print("Extracting time data from IKE_SA_INIT messages...")
    with open(filter_ts, "w") as resultado_file:
        tshark_output = subprocess.run(
            ["sudo", "tshark", "-r", ts_res, "-Y", "isakmp"],
            capture_output=True, text=True
        )
        filtro_output = subprocess.run(
            ["grep", "IKE_SA_INIT"], input=tshark_output.stdout, capture_output=True, text=True
        )
        resultado_file.write(filtro_output.stdout)
    print("Process completed. Data stored in " + filter_ts)

    initiator_times = []
    responder_times = []

    output_file = "results.csv"

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

df.to_csv(output_file, index=False)

print(f"Differences stored in '.csv' {output_file}")

