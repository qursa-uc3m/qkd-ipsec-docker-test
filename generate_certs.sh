#!/bin/bash

# Ensure the certs directory exists
mkdir -p certs

# Default CN is "localhost" if not provided
CN=${1:-localhost}

# Generate server certificates with the provided CN in the filename
openssl req -x509 -newkey rsa:4096 \
  -keyout certs/server_key_${CN}.pem \
  -out certs/server_cert_${CN}.pem \
  -days 365 -nodes -subj "/CN=$CN"

# Generate client certificates with the provided CN in the filename
openssl req -x509 -newkey rsa:4096 \
  -keyout certs/client_key_${CN}.pem \
  -out certs/client_cert_${CN}.pem \
  -days 365 -nodes -subj "/CN=$CN"

# Set permissions on the generated certificates
chmod 644 certs/*.pem
