# Bob: Testing Script

import subprocess
import time
import socket
import re

SERVER_IP = "192.168.0.18" # Server IP
PORT = 12345

config_file = "/etc/swanctl/swanctl.conf"

proposals = [	"aes128-sha256-x25519",
		"aes128-sha256-x448"	]
esp_proposals =	[	"aes128-sha256-x25519",
			"aes128-sha256-x448"	]

# Create socket TCP/IP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, PORT))

print("Connected to server")

with open(config_file, "r") as file:
    config_data = file.read()

for prop,esp_prop in zip(proposals,esp_proposals):
    # Regular expresión to search and replace
    config_data = re.sub(r'(\bproposals\s*=\s*)[^\s]+', rf'\1{prop}', config_data)
    config_data = re.sub(r'(\besp_proposals\s*=\s*)[^\s]+', rf'\1{esp_prop}', config_data)

    with open(config_file, "w") as file:
        file.write(config_data)

    print("Configuration file updated.")

    for i in range(1, 4):
        print(f"Iteration {i}")

        print("Executing strongSwan...")
        strongswan_proc = subprocess.Popen(
	    ["sudo", "./charon"],
	    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )
        time.sleep(3) # Waiting 'charon' to be ready
    
        print("Send ACK to Alice")
        client_socket.send("0".encode()) 
        print("Wait ACK from Alice")
        rsp = client_socket.recv(1024).decode()

        print("Stop strongSwan...")
        subprocess.run(["sudo", "pkill", "-f", "charon"])
        print("\n\n")
        time.sleep(1)
