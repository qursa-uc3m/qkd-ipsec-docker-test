#!/bin/bash

cd /qkd-etsi-api-c-wrapper
mkdir -p build
cd build
cmake -DENABLE_ETSI004=OFF -DENABLE_ETSI014=ON -DQKD_BACKEND=qukaydee -DQKD_DEBUG_LEVEL=4 -DBUILD_TESTS=OFF ..
make
make install