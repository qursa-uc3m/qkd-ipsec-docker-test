#!/bin/bash

# Base directory for certificates
CERTS_DIR="/qkd_certs"
# Dedicated certificate directories for each API version
ETSI004_CERTS_DIR="${CERTS_DIR}/etsi004"
ETSI014_CERTS_DIR="${CERTS_DIR}/etsi014"

# Determine which ETSI API version to use
ETSI_API_VERSION=${ETSI_API_VERSION:-014}

# Set compiler flags based on ETSI API version - do this once at the beginning
if [ "${ETSI_API_VERSION}" = "004" ]; then
    export CPPFLAGS="-DETSI_004_API"
    export CFLAGS="-DETSI_004_API"
else
    export CPPFLAGS="-DETSI_014_API"
    export CFLAGS="-DETSI_014_API"
fi

if [ "${ETSI_API_VERSION}" = "004" ]; then
    echo "Setting up ETSI 004 environment:"
    
    # Server connection settings for ETSI 004 (original variable names)
    export SERVER_ADDRESS=${SERVER_ADDRESS:-"qkd_server_bob"}
    export SERVER_PORT=${SERVER_PORT:-25576}
    export CLIENT_ADDRESS=${CLIENT_ADDRESS:-"qkd_server_alice"}
    
    # Additional QKD server settings for StrongSwan
    export QKD_SERVER_ALICE=${QKD_SERVER_ALICE:-"qkd_server_alice:25575"}
    export QKD_SERVER_BOB=${QKD_SERVER_BOB:-"qkd_server_bob:25576"}
    
    # Certificate paths for ETSI 004 (original variable names)
    export CLIENT_CERT_PEM=${CLIENT_CERT_PEM:-"${ETSI004_CERTS_DIR}/client_cert_qkd_server_alice.pem"}
    export CLIENT_CERT_KEY=${CLIENT_CERT_KEY:-"${ETSI004_CERTS_DIR}/client_key_qkd_server_alice.pem"}
    export SERVER_CERT_PEM=${SERVER_CERT_PEM:-"${ETSI004_CERTS_DIR}/server_cert_qkd_server_bob.pem"}
    
    # QKD specific parameters (original variable names)
    export KEY_INDEX=0
    export METADATA_SIZE=1024
    export QKD_USE_TLS="true"
    export QKD_CLIENT_VERIFY="true"
    
    # QoS parameters (original variable names)
    export QOS_KEY_CHUNK_SIZE=32
    export QOS_MAX_BPS=40000
    export QOS_MIN_BPS=5000
    export QOS_JITTER=10
    export QOS_PRIORITY=0
    export QOS_TIMEOUT=5000
    export QOS_TTL=3600
    
    # Check if certificate files exist
    if [ ! -f "$CLIENT_CERT_PEM" ] || [ ! -f "$CLIENT_CERT_KEY" ] || [ ! -f "$SERVER_CERT_PEM" ]; then
        echo "WARNING: Some certificate files were not found at the expected paths:"
        echo "  CLIENT_CERT_PEM: $CLIENT_CERT_PEM"
        echo "  CLIENT_CERT_KEY: $CLIENT_CERT_KEY"
        echo "  SERVER_CERT_PEM: $SERVER_CERT_PEM"
        echo ""
        echo "If you've placed certificates elsewhere, please set these variables manually."
    else
        echo "Certificate paths set successfully."
    fi
    
    echo "QKD environment variables set for connecting to $SERVER_ADDRESS:$SERVER_PORT"
    
elif [ "${QKD_BACKEND}" = "qukaydee" ]; then
    echo "Setting up QuKayDee environment (ETSI 014):"
    
    # Certificate configuration with ETSI014_CERTS_DIR
    export QKD_MASTER_CA_CERT_PATH="${ETSI014_CERTS_DIR}/account-${ACCOUNT_ID}-server-ca-qukaydee-com.crt"
    export QKD_SLAVE_CA_CERT_PATH="${ETSI014_CERTS_DIR}/account-${ACCOUNT_ID}-server-ca-qukaydee-com.crt"
    
    export QKD_MASTER_CERT_PATH="${ETSI014_CERTS_DIR}/sae-1.crt"
    export QKD_MASTER_KEY_PATH="${ETSI014_CERTS_DIR}/sae-1.key"
    
    export QKD_SLAVE_CERT_PATH="${ETSI014_CERTS_DIR}/sae-2.crt"
    export QKD_SLAVE_KEY_PATH="${ETSI014_CERTS_DIR}/sae-2.key"
    
    # QuKayDee configuration
    if [ -z "${ACCOUNT_ID}" ]; then
        echo "Warning: ACCOUNT_ID not set. Please set your QuKayDee account ID."
    else
        export QKD_MASTER_KME_HOSTNAME="https://kme-1.acct-${ACCOUNT_ID}.etsi-qkd-api.qukaydee.com"
        export QKD_SLAVE_KME_HOSTNAME="https://kme-2.acct-${ACCOUNT_ID}.etsi-qkd-api.qukaydee.com"
        export QKD_MASTER_SAE="sae-1"
        export QKD_SLAVE_SAE="sae-2"
        
        echo "QKD_MASTER_KME_HOSTNAME=$QKD_MASTER_KME_HOSTNAME"
        echo "QKD_SLAVE_KME_HOSTNAME=$QKD_SLAVE_KME_HOSTNAME"
        echo "QKD_MASTER_SAE=$QKD_MASTER_SAE"
        echo "QKD_SLAVE_SAE=$QKD_SLAVE_SAE"
    fi
    
elif [ "${QKD_BACKEND}" = "cerberis-xgr" ]; then
    echo "Setting up Cerberis-XGR environment (ETSI 014):"
    
    # Certificate configuration with ETSI014_CERTS_DIR
    export QKD_MASTER_CA_CERT_PATH="${ETSI014_CERTS_DIR}/ChrisCA.pem"
    export QKD_SLAVE_CA_CERT_PATH="${ETSI014_CERTS_DIR}/ChrisCA.pem"
    
    export QKD_MASTER_CERT_PATH="${ETSI014_CERTS_DIR}/ETSIA.pem"
    export QKD_MASTER_KEY_PATH="${ETSI014_CERTS_DIR}/ETSIA-key.pem"
    
    export QKD_SLAVE_CERT_PATH="${ETSI014_CERTS_DIR}/ETSIB.pem"
    export QKD_SLAVE_KEY_PATH="${ETSI014_CERTS_DIR}/ETSIB-key.pem"
    
    # Cerberis-XGR configuration
    export QKD_MASTER_KME_HOSTNAME="https://castor.det.uvigo.es:444"
    export QKD_SLAVE_KME_HOSTNAME="https://castor.det.uvigo.es:442"
    export QKD_MASTER_SAE="CONSA"
    export QKD_SLAVE_SAE="CONSB"
    
    echo "QKD_MASTER_KME_HOSTNAME=$QKD_MASTER_KME_HOSTNAME"
    echo "QKD_SLAVE_KME_HOSTNAME=$QKD_SLAVE_KME_HOSTNAME"
    echo "QKD_MASTER_CA_CERT_PATH=$QKD_MASTER_CA_CERT_PATH"
    echo "QKD_SLAVE_CA_CERT_PATH=$QKD_SLAVE_CA_CERT_PATH"
    echo "QKD_MASTER_CERT_PATH=$QKD_MASTER_CERT_PATH"
    echo "QKD_MASTER_KEY_PATH=$QKD_MASTER_KEY_PATH"
    echo "QKD_SLAVE_CERT_PATH=$QKD_SLAVE_CERT_PATH"
    echo "QKD_SLAVE_KEY_PATH=$QKD_SLAVE_KEY_PATH"
    echo ""
    
else
    echo "Using default QKD backend (simulated)"
    # Set default environment variables for simulated mode
    export QKD_MASTER_KME_HOSTNAME="localhost"
    export QKD_SLAVE_KME_HOSTNAME="localhost"
    export QKD_MASTER_SAE="sae-1"
    export QKD_SLAVE_SAE="sae-2"
    
    # For ETSI 004 simulated mode, set required environment variables
    if [ "${ETSI_API_VERSION}" = "004" ]; then
        export SERVER_ADDRESS="localhost"
        export SERVER_PORT=25576
        export CLIENT_ADDRESS="localhost"
        echo "Using ETSI 004 API with simulated backend"
    else
        echo "Using ETSI 014 API with simulated backend"
    fi
fi

# Print all QKD environment variables
echo "QKD Environment Variables:"
env | grep -E "QKD_|ETSI_|SERVER_|CLIENT_|QOS_" | sort