#!/bin/bash
set -e

# Create the OpenSSL modules directory if it doesn't exist
mkdir -p /usr/local/lib/ossl-modules

cd /qkd-kem-provider
chmod +x scripts/fullbuild.sh

export LIBOQS_BRANCH="0.12.0"
export OQSPROV_CMAKE_PARAMS="-DQKD_KEY_ID_CH=ON"

# Build the provider
./scripts/fullbuild.sh -F

# Copy the provider to the OpenSSL modules directory
cp _build/lib/qkdkemprovider.so /usr/local/lib/ossl-modules/

# Set permissions
chmod 755 /usr/local/lib/ossl-modules/qkdkemprovider.so