#!/bin/bash
set -e

# Use specific version of liboqs
LIBOQS_BRANCH="0.12.0"

echo "Starting liboqs build process version $LIBOQS_BRANCH..."

# Clone liboqs repository with specific tag
git clone --depth 1 --branch $LIBOQS_BRANCH https://github.com/open-quantum-safe/liboqs.git /liboqs
cd /liboqs

# Create build directory
mkdir -p build
cd build

# Configure and build with consistent flags
cmake -GNinja \
      -DOQS_USE_OPENSSL=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=Release \
      -DOQS_BUILD_ONLY_LIB=ON \
      ..

# Build
ninja

# Install the library
ninja install

# Clean up
cd /
rm -rf /liboqs

echo "liboqs version $LIBOQS_BRANCH has been successfully built and installed."