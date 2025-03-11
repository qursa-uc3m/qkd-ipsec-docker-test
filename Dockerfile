# Base image
# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Build argument for QKD support
ARG BUILD_QKD_ETSI=true
ARG BUILD_QKD_KEM=true
ARG STRONGSWAN_BRANCH=qkd
ARG STRONGSWAN_REPO=https://github.com/qursa-uc3m/strongswan.git

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENSSL_CONF=/etc/ssl/qkd-kem-openssl.cnf
ENV OPENSSL_MODULES=/usr/local/lib/ossl-modules
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/ossl-modules

# Install build dependencies
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
    nano \
    wget \
    bzip2 \
    make \
    gcc \
    libssl-dev \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \
    gettext \
    libgmp-dev \
    build-essential \
    cmake \
    ninja-build \
    flex \
    bison \
    python3 \
    gperf \
    && rm -rf /var/lib/apt/lists/*

COPY config/openssl.cnf /etc/ssl/qkd-kem-openssl.cnf

# Clone repositories
RUN if [ "$BUILD_QKD_ETSI" = "true" ]; then \
    git clone https://github.com/qursa-uc3m/qkd-etsi-api.git /qkd-etsi-api; \
    fi

RUN if [ "$BUILD_QKD_KEM" = "true" ]; then \
    git clone https://github.com/qursa-uc3m/qkd-kem-provider.git /qkd-kem-provider; \
    fi

COPY scripts/build_*.sh /
RUN chmod +x /build_*.sh

# Build QKD ETSI API if requested
RUN if [ "$BUILD_QKD_ETSI" = "true" ]; then \
    /build_qkd_etsi.sh; \
    fi

# Build QKD KEM provider if requested
RUN if [ "$BUILD_QKD_KEM" = "true" ]; then \
    /build_qkd_kem_provider.sh; \
    fi

# Clone strongSwan repository
RUN git clone -b "$STRONGSWAN_BRANCH" "$STRONGSWAN_REPO" /strongswan;

# Build strongSwan
RUN /build_strongswan.sh

# Create symlink for charon daemon
RUN ln -s /usr/libexec/ipsec/charon /charon

# Expose ports
# 500: IKE
# 4500: NAT-T
EXPOSE 500 4500