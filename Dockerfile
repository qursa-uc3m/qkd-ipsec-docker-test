# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

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
    flex \
    bison \
    python3 \
    gperf \
    libcurl4-openssl-dev \
    libjansson-dev \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/build_strongswan.sh /build_strongswan.sh
RUN chmod +x /build_strongswan.sh

RUN /build_strongswan.sh

RUN ln -s /usr/libexec/ipsec/charon /charon

EXPOSE 500/udp 4500/udp