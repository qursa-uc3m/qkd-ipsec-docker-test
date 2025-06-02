#!/bin/bash
set -e

# This script builds the QKD KEM provider with configurable ETSI API version
# Usage: ETSI_API_VERSION=014|004 ./build_qkd_kem_provider.sh

# Set default ETSI API version if not specified
if [ -z "$ETSI_API_VERSION" ] || [ "$ETSI_API_VERSION" != "004" ]; then
    export ETSI_API_VERSION="014"
    echo "Using ETSI API version: 014"
else
    echo "Using ETSI API version: 004"
fi

# Set default QKD initiation mode if not specified
if [ "$ETSI_API_VERSION" = "004" ]; then
    # ETSI 004 always uses QKD_KEY_ID_CH=ON regardless of initiation mode
    QKD_KEY_ID_CH="ON"
    echo "QKD_KEY_ID_CH=ON (ETSI 004 always sends key ID in first message)"
elif [ "$QKD_INITIATION_MODE" = "client" ]; then
    # ETSI 014 with client-initiated mode
    QKD_KEY_ID_CH="ON"
    echo "QKD_KEY_ID_CH=ON (client-initiated: key ID sent in first message)"
else
    # ETSI 014 with server-initiated mode
    QKD_KEY_ID_CH="OFF"
    echo "QKD_KEY_ID_CH=OFF (server-initiated: key ID sent in response message)"
fi

# Create the OpenSSL modules directory if it doesn't exist
mkdir -p /usr/local/lib/ossl-modules

# First, build and install liboqs directly to system paths
LIBOQS_BRANCH="0.12.0"
echo "Building liboqs version $LIBOQS_BRANCH..."
git clone --depth 1 --branch $LIBOQS_BRANCH https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs
cd /tmp/liboqs
mkdir -p build
cd build

cmake -GNinja \
    -DOQS_USE_OPENSSL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_BUILD_TYPE=Release \
    -DOQS_BUILD_ONLY_LIB=ON \
    ..

ninja -j$(nproc)
ninja install
cd /
rm -rf /tmp/liboqs

# Now build the QKD KEM provider
cd /qkd-kem-provider
chmod +x scripts/fullbuild.sh

# Skip liboqs build since we've already installed it
export liboqs_DIR=/usr

# Set provider build parameters
# Add ETSI API version to the parameters
if [ "$ETSI_API_VERSION" = "004" ]; then
    export OQSPROV_CMAKE_PARAMS="-DQKD_KEY_ID_CH=$QKD_KEY_ID_CH -DQKD_BACKEND=${QKD_BACKEND:-simulated} -DETSI_004_API=ON -DETSI_014_API=OFF"
else
    export OQSPROV_CMAKE_PARAMS="-DQKD_KEY_ID_CH=$QKD_KEY_ID_CH -DQKD_BACKEND=${QKD_BACKEND:-simulated} -DETSI_014_API=ON -DETSI_004_API=OFF"
fi

# Build the provider - but only clean the provider build, not liboqs
ETSI_API_VERSION=$ETSI_API_VERSION ./scripts/fullbuild.sh -f

# Copy the provider to the OpenSSL modules directory
cp _build/lib/qkdkemprovider.so /usr/local/lib/ossl-modules/

# Run ldconfig to update the dynamic linker cache
ldconfig

# Set permissions
chmod 755 /usr/local/lib/ossl-modules/qkdkemprovider.so

echo "=============================================="
echo "QKD KEM provider build completed successfully"
echo "=============================================="
echo "Configuration:"
echo "- ETSI API version: $ETSI_API_VERSION"
echo "- QKD initiation mode: $QKD_INITIATION_MODE" 
echo "- QKD_KEY_ID_CH: $QKD_KEY_ID_CH"
echo "- Backend mode: ${QKD_BACKEND:-simulated}"
echo "=============================================="