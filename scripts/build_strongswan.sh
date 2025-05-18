#!/bin/bash
set -e

WORK_DIR="/strongswan"

# Use ETSI API version from environment or default to 014
ETSI_API_VERSION=${ETSI_API_VERSION:-014}

echo "Starting strongSwan build process..."
echo "Building with ETSI API version: $ETSI_API_VERSION"

cd "$WORK_DIR"

make clean || true

if [ -f ./autogen.sh ]; then
    echo "Running autogen.sh..."
    ./autogen.sh
else
    echo "autogen.sh not found. Skipping..."
fi

# Set up build flags based on API version
if [ "$ETSI_API_VERSION" = "004" ]; then
    echo "Using ETSI 004 API"
    BUILD_FLAGS="--with-etsi-api=004"
else
    echo "Using ETSI 014 API"
    BUILD_FLAGS="--with-etsi-api=014"
fi

echo "Configuring the build..."
echo "Build flags: $BUILD_FLAGS"

./configure \
    --prefix=/usr \
    --sysconfdir=/etc \
    --with-dev-headers=/usr/include/strongswan \
    --disable-defaults \
    --enable-charon \
    --enable-ikev2 \
    --enable-nonce \
    --enable-random \
    --enable-openssl \
    --enable-oqs \
    --enable-pem \
    --enable-x509 \
    --enable-pubkey \
    --enable-constraints \
    --enable-pki \
    --enable-socket-default \
    --enable-kernel-netlink \
    --enable-swanctl \
    --enable-resolve \
    --enable-eap-identity \
    --enable-eap-md5 \
    --enable-eap-dynamic \
    --enable-eap-tls \
    --enable-updown \
    --enable-vici \
    --enable-silent-rules \
    $BUILD_FLAGS \
    LDFLAGS="-luuid -L/usr/local/lib"

echo "Building strongSwan..."
make -j$(nproc)

echo "Installing strongSwan..."
make install

echo "Checking charon installation..."
ls -la /usr/lib/ipsec/charon || echo "Charon not found in /usr/lib/ipsec/"
ls -la /usr/libexec/ipsec/charon || echo "Charon not found in /usr/libexec/ipsec/"

echo "strongSwan has been successfully built and installed."

# Create a simple marker file to indicate which API version was used
echo "ETSI_API_VERSION=${ETSI_API_VERSION}" > /etc/strongswan-etsi-api-version
echo "Build information saved to /etc/strongswan-etsi-api-version"