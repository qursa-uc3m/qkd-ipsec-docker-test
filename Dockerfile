# Base image
FROM ubuntu:22.04

# Build argument for QKD support
ARG BUILD_QKD_ETSI=true
ARG BUILD_QKD_KEM=false
ARG QKD_BACKEND=simulated
ARG ACCOUNT_ID=
ARG ETSI_API_VERSION=014
ARG STRONGSWAN_VERSION=6.0.0beta6

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENSSL_CONF=/etc/ssl/qkd-kem-openssl.cnf
ENV OPENSSL_MODULES=/usr/local/lib/ossl-modules
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/ossl-modules
# Pass build args to env vars
ENV QKD_BACKEND=${QKD_BACKEND}
ENV ACCOUNT_ID=${ACCOUNT_ID}
ENV ETSI_API_VERSION=${ETSI_API_VERSION}
ENV STRONGSWAN_VERSION=${STRONGSWAN_VERSION}

# Install build dependencies and testing tools
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
    uuid-dev \
    net-tools \
    iptables \
    tcpdump \
    curl \
    libcurl4-openssl-dev \
    libjansson-dev \
    && rm -rf /var/lib/apt/lists/*

COPY config/openssl.cnf /etc/ssl/qkd-kem-openssl.cnf

# Create directory for QKD certificates
RUN mkdir -p /qkd_certs

# Copy QKD certificates
# Using a multi-stage build to conditionally copy
COPY qkd_certs /qkd_certs

# Clone and build QKD ETSI API if requested
RUN if [ "$BUILD_QKD_ETSI" = "true" ]; then \
    git clone --depth 1 https://github.com/qursa-uc3m/qkd-etsi-api-c-wrapper.git /qkd-etsi-api-c-wrapper; \
    fi

# Copy build and environment scripts
COPY scripts/build_*.sh /
COPY scripts/set_env.sh /
COPY scripts/add_timing_hooks.py /
RUN chmod +x /build_*.sh /set_env.sh

# Build QKD ETSI API
RUN if [ "$BUILD_QKD_ETSI" = "true" ]; then \
    /build_qkd_etsi.sh; \
    fi

# Build QKD KEM provider if requested
RUN if [ "$BUILD_QKD_KEM" = "true" ]; then \
    git clone --depth 1 https://github.com/qursa-uc3m/qkd-kem-provider.git /qkd-kem-provider && \
    ETSI_API_VERSION=${ETSI_API_VERSION} /build_qkd_kem_provider.sh; \
else \
    /build_liboqs.sh; \
fi

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tshark \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install pandas pyyaml

# Create directory for output
RUN mkdir -p /output

# Clone and build StrongSwan from GitHub using 6.0.0beta6 tag
WORKDIR /
RUN git clone --depth 1 --branch ${STRONGSWAN_VERSION} https://github.com/strongswan/strongswan.git /strongswan

# Apply timing hooks to strongSwan
RUN python3 /add_timing_hooks.py /strongswan

# Build strongSwan using the script
WORKDIR /strongswan
RUN /build_strongswan.sh

# Clone and build external QKD plugins from GitHub
WORKDIR /
RUN git clone --depth 1 https://github.com/qursa-uc3m/qkd-plugins-strongswan.git /qkd-plugins-strongswan

WORKDIR /qkd-plugins-strongswan
RUN autoreconf -i && \
    ./configure \
    --with-strongswan-headers=/usr/include/strongswan \
    --with-plugin-dir=/usr/lib/ipsec/plugins \
    --with-qkd-etsi-api=/usr/local \
    --with-qkd-kem-provider=/usr/local/lib/ossl-modules \
    --with-etsi-api-version=${ETSI_API_VERSION} \
    --enable-qkd \
    $([ "$BUILD_QKD_KEM" = "true" ] && echo "--enable-qkd-kem" || echo "--disable-qkd-kem") \
    && make \
    && make install

# Create symlink for charon daemon
RUN ln -s /usr/libexec/ipsec/charon /charon

# Expose ports
# 500: IKE
# 4500: NAT-T
EXPOSE 500/udp 4500/udp