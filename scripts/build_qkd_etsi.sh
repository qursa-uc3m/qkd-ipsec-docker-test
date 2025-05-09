#!/bin/bash

# Change to QKD ETSI API directory
cd /qkd-etsi-api-c-wrapper
mkdir -p build
cd build

# Configure environment variables for Python client if using ETSI 004
if [ "$ETSI_API_VERSION" = "004" ]; then
    # Install Python client dependencies for ETSI 004
    apt-get update && apt-get install -y python3-dev
    
    # Install the Python client module
    bash /qkd-etsi-api-c-wrapper/scripts/install_python_etsi004_client.sh
    
    # Set backend to python_client for ETSI 004
    BACKEND_FLAG="python_client"
    ETSI004_FLAG="ON"
    ETSI014_FLAG="OFF"
    
    # Set environment variables needed for ETSI 004
    source /qkd-etsi-api-c-wrapper/scripts/etsi004_env.sh
else
    # Set QKD_BACKEND flag based on environment variable
    if [ "$QKD_BACKEND" = "simulated" ]; then
        BACKEND_FLAG="simulated"
    elif [ "$QKD_BACKEND" = "qukaydee" ]; then
        BACKEND_FLAG="qukaydee"
    elif [ "$QKD_BACKEND" = "cerberis-xgr" ]; then
        BACKEND_FLAG="cerberis_xgr"
    else
        echo "[INFO] Unknown QKD_BACKEND: $QKD_BACKEND, defaulting to simulated"
        BACKEND_FLAG="simulated"
    fi
    
    # By default enable ETSI014 and disable ETSI004
    ETSI004_FLAG="OFF"
    ETSI014_FLAG="ON"
fi

echo "[INFO] Using API version: ${ETSI_API_VERSION:-014}, backend: $BACKEND_FLAG"
echo "[INFO] ETSI004: $ETSI004_FLAG, ETSI014: $ETSI014_FLAG"

# Configure CMake with appropriate flags
cmake -DENABLE_ETSI004=$ETSI004_FLAG -DENABLE_ETSI014=$ETSI014_FLAG \
      -DQKD_BACKEND=$BACKEND_FLAG -DQKD_DEBUG_LEVEL=4 -DBUILD_TESTS=ON ..

# Build and install
make -j$(nproc)
make install

echo "[INFO] QKD ETSI API C wrapper installation completed"

# Create symbolic links to ensure proper library loading
ln -sf /usr/local/lib/libqkd-etsi-api-c-wrapper.so /usr/lib/
ldconfig

echo "[INFO] Library paths updated"