#!/bin/bash

cd /qkd-etsi-api-c-wrapper
mkdir -p build
cd build

# Set QKD_BACKEND flag based on environment variable
if [ "$QKD_BACKEND" = "simulated" ]; then
    BACKEND_FLAG=simulated
elif [ "$QKD_BACKEND" = "qukaydee" ]; then
    BACKEND_FLAG=qukaydee
elif [ "$QKD_BACKEND" = "cerberis-xgr" ]; then
    BACKEND_FLAG=cerberis_xgr
else
    echo "[INFO] Unknown QKD_BACKEND: $QKD_BACKEND, defaulting to simulated"
    BACKEND_FLAG=simulated
fi

echo "[INFO] Using QKD_BACKEND=$QKD_BACKEND, setting DQKD_BACKEND=$BACKEND_FLAG"

cmake -DENABLE_ETSI004=OFF -DENABLE_ETSI014=ON -DQKD_BACKEND=$BACKEND_FLAG -DQKD_DEBUG_LEVEL=4 -DBUILD_TESTS=OFF ..

# Build and install
make
make install

echo "[INFO] QKD ETSI API C wrapper installation completed"