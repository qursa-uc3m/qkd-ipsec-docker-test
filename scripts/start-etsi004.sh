#!/bin/bash

# Create necessary directory
mkdir -p qkd_certs/etsi004

# Function to generate certificates
generate_cert() {
    local CN=$1
    echo "Generating certificates for $CN..."
    # Generate server certificates
    openssl req -x509 -newkey rsa:4096 \
      -keyout qkd_certs/etsi004/server_key_${CN}.pem \
      -out qkd_certs/etsi004/server_cert_${CN}.pem \
      -days 365 -nodes -subj "/CN=$CN"
    # Generate client certificates
    openssl req -x509 -newkey rsa:4096 \
      -keyout qkd_certs/etsi004/client_key_${CN}.pem \
      -out qkd_certs/etsi004/client_cert_${CN}.pem \
      -days 365 -nodes -subj "/CN=$CN"
    # Set permissions
    chmod 644 qkd_certs/etsi004/*.pem
}

# Ensure we have the certificates for ETSI 004
if [ ! -f "qkd_certs/etsi004/server_cert_qkd_server_alice.pem" ] || [ ! -f "qkd_certs/etsi004/server_cert_qkd_server_bob.pem" ]; then
    echo "Generating ETSI 004 certificates directly in qkd_certs/etsi004..."
    
    # Generate certificates for Alice and Bob
    generate_cert qkd_server_alice
    generate_cert qkd_server_bob
    
    echo "Certificates generated in qkd_certs/etsi004/"
    echo "Certificate files in qkd_certs/etsi004:"
    ls -la qkd_certs/etsi004/*.pem
fi

# Set the API version
export ETSI_API_VERSION=004
export QKD_BACKEND=python_client


# Remove and recreate the network
echo "Setting up qkd_network..."
docker network rm qkd_network 2>/dev/null || true
docker network create qkd_network 2>/dev/null || true

# Clone the ETSI QKD 004 repository if not already done
if [ ! -d "etsi-qkd-004" ]; then
    echo "Cloning ETSI QKD 004 repository..."
    git clone https://github.com/QUBIP/etsi-qkd-004.git -b ksid_sync
fi

# Start the QUBIP ETSI 004 infrastructure
echo "Starting QUBIP ETSI 004 infrastructure..."
docker-compose -f qkd-etsi004.yml up -d --build

# Wait for the QUBIP infrastructure to be ready
echo "Waiting for QUBIP servers to be ready..."
sleep 5

echo "ETSI 004 environment is ready!"
