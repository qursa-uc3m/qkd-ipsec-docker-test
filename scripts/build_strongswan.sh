#!/bin/bash
set -e

WORK_DIR="/strongswan"

echo "Starting strongSwan build process..."

cd "$WORK_DIR"

make clean || true

if [ -f ./autogen.sh ]; then
    echo "Running autogen.sh..."
    ./autogen.sh
else
    echo "autogen.sh not found. Skipping..."
fi

echo "Configuring the build..."
./configure \
    --prefix=/usr \
    --sysconfdir=/etc \
    --disable-defaults \
    --enable-charon \
    --enable-ikev2 \
    --enable-nonce \
    --enable-random \
    --enable-openssl \
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
    --enable-qkd \
    --enable-silent-rules

echo "Building strongSwan..."
make -j$(nproc)

echo "Installing strongSwan..."
make install

echo "Checking charon installation..."
ls -la /usr/lib/ipsec/charon || echo "Charon not found in /usr/lib/ipsec/"
ls -la /usr/libexec/ipsec/charon || echo "Charon not found in /usr/libexec/ipsec/"

echo "strongSwan has been successfully built and installed."
