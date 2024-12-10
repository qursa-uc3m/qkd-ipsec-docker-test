#!/bin/bash

cd /qkd-etsi-api
mkdir -p build
cd build
cmake -DQKD_BACKEND=simulated -DQKD_DEBUG_LEVEL=4 -DBUILD_TESTS=ON ..
make
make install