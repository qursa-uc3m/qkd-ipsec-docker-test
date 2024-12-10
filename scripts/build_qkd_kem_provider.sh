#!/bin/bash
set -e

# Create the OpenSSL modules directory if it doesn't exist
mkdir -p /usr/local/lib/ossl-modules

cd /qkd-kem-provider
chmod +x scripts/fullbuild.sh

# Build the provider
./scripts/fullbuild.sh -F

# Copy the provider to the OpenSSL modules directory
cp _build/lib/qkdkemprovider.so /usr/local/lib/ossl-modules/

# Set permissions
chmod 755 /usr/local/lib/ossl-modules/qkdkemprovider.so