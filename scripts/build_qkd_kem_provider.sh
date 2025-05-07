#!/bin/bash
set -e
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
ninja
ninja install
cd /
rm -rf /tmp/liboqs

# Now build the QKD KEM provider
cd /qkd-kem-provider
chmod +x scripts/fullbuild.sh

# Skip liboqs build since we've already installed it
export liboqs_DIR=/usr  # Tell the build script to use system installed liboqs
export OQSPROV_CMAKE_PARAMS="-DQKD_KEY_ID_CH=ON"

# Build the provider - but only clean the provider build, not liboqs
./scripts/fullbuild.sh -f

# Copy the provider to the OpenSSL modules directory
cp _build/lib/qkdkemprovider.so /usr/local/lib/ossl-modules/

# Run ldconfig to update the dynamic linker cache
ldconfig

# Set permissions
chmod 755 /usr/local/lib/ossl-modules/qkdkemprovider.so

echo "QKD KEM provider has been successfully built and installed."