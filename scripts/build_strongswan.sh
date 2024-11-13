#!/bin/bash
set -e

# Boolean flag to control whether to clone the repository
CLONE_REPO=true

# Repository information
REPO_URL="https://github.com/qursa-uc3m/strongswan.git"
BRANCH="qkd-nodes"
WORK_DIR="/strongswan"

echo "Starting strongSwan build process..."

if [ "$CLONE_REPO" = true ]; then
    echo "Cloning strongSwan repository..."
    rm -rf "$WORK_DIR"
    git clone -b "$BRANCH" "$REPO_URL" "$WORK_DIR"
fi

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
