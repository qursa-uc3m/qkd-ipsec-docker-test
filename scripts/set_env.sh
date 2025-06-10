#!/bin/bash

# Base directory for certificates
CERTS_DIR="/qkd_certs"
# Dedicated certificate directories for each API version
ETSI004_CERTS_DIR="${CERTS_DIR}/etsi004"
ETSI014_CERTS_DIR="${CERTS_DIR}/etsi014"

# Determine which ETSI API version to use
ETSI_API_VERSION=${ETSI_API_VERSION:-014}

# Set compiler flags based on ETSI API version
if [ "${ETSI_API_VERSION}" = "004" ]; then
    export CPPFLAGS="-DETSI_004_API"
    export CFLAGS="-DETSI_004_API"
else
    export CPPFLAGS="-DETSI_014_API"
    export CFLAGS="-DETSI_014_API"
fi

if [ "${ETSI_API_VERSION}" = "004" ]; then
    echo "Setting up ETSI 004 environment:"
    
    # Check required URIs are set by Docker Compose
    if [ -z "$QKD_SOURCE_URI" ] || [ -z "$QKD_DEST_URI" ]; then
        echo "ERROR: QKD_SOURCE_URI and QKD_DEST_URI must be set for ETSI 004"
        exit 1
    fi
    
    # Certificate configuration
    # Only set defaults if not already provided by Docker Compose
    CONTAINER_NAME=$(hostname)
    
    if [ "$CONTAINER_NAME" = "alice" ]; then
        export CLIENT_CERT_PEM=${CLIENT_CERT_PEM:-"${ETSI004_CERTS_DIR}/client_cert_qkd_server_alice.pem"}
        export CLIENT_CERT_KEY=${CLIENT_CERT_KEY:-"${ETSI004_CERTS_DIR}/client_key_qkd_server_alice.pem"}
        export SERVER_CERT_PEM=${SERVER_CERT_PEM:-"${ETSI004_CERTS_DIR}/server_cert_qkd_server_alice.pem"}
        echo "Configured as Alice"
    elif [ "$CONTAINER_NAME" = "bob" ]; then
        export CLIENT_CERT_PEM=${CLIENT_CERT_PEM:-"${ETSI004_CERTS_DIR}/client_cert_qkd_server_bob.pem"}
        export CLIENT_CERT_KEY=${CLIENT_CERT_KEY:-"${ETSI004_CERTS_DIR}/client_key_qkd_server_bob.pem"}
        export SERVER_CERT_PEM=${SERVER_CERT_PEM:-"${ETSI004_CERTS_DIR}/server_cert_qkd_server_bob.pem"}
        echo "Configured as Bob"
    else
        echo "WARNING: Unknown container name '$CONTAINER_NAME', using Alice defaults"
        export CLIENT_CERT_PEM=${CLIENT_CERT_PEM:-"${ETSI004_CERTS_DIR}/client_cert_qkd_server_alice.pem"}
        export CLIENT_CERT_KEY=${CLIENT_CERT_KEY:-"${ETSI004_CERTS_DIR}/client_key_qkd_server_alice.pem"}
        export SERVER_CERT_PEM=${SERVER_CERT_PEM:-"${ETSI004_CERTS_DIR}/server_cert_qkd_server_alice.pem"}
    fi
    
    # Check if certificate files exist
    if [ ! -f "$CLIENT_CERT_PEM" ] || [ ! -f "$CLIENT_CERT_KEY" ] || [ ! -f "$SERVER_CERT_PEM" ]; then
        echo "WARNING: Some certificate files were not found:"
        echo "  CLIENT_CERT_PEM: $CLIENT_CERT_PEM"
        echo "  CLIENT_CERT_KEY: $CLIENT_CERT_KEY"
        echo "  SERVER_CERT_PEM: $SERVER_CERT_PEM"
        echo ""
        echo "Available certificates in ${ETSI004_CERTS_DIR}:"
        ls -la "${ETSI004_CERTS_DIR}/" 2>/dev/null || echo "  Directory not found or empty"
    else
        echo "Certificate paths set successfully"
    fi
    
    echo "ETSI 004 Configuration:"
    echo "  Source URI: $QKD_SOURCE_URI"
    echo "  Destination URI: $QKD_DEST_URI"
    echo "  Backend: ${QKD_BACKEND:-python_client}"
    
elif [ "${QKD_BACKEND}" = "qukaydee" ]; then
    echo "Setting up QuKayDee environment (ETSI 014):"
    
    # Certificate configuration
    export QKD_MASTER_CA_CERT_PATH="${ETSI014_CERTS_DIR}/account-${ACCOUNT_ID}-server-ca-qukaydee-com.crt"
    export QKD_SLAVE_CA_CERT_PATH="${ETSI014_CERTS_DIR}/account-${ACCOUNT_ID}-server-ca-qukaydee-com.crt"
    
    export QKD_MASTER_CERT_PATH="${ETSI014_CERTS_DIR}/sae-1.crt"
    export QKD_MASTER_KEY_PATH="${ETSI014_CERTS_DIR}/sae-1.key"
    
    export QKD_SLAVE_CERT_PATH="${ETSI014_CERTS_DIR}/sae-2.crt"
    export QKD_SLAVE_KEY_PATH="${ETSI014_CERTS_DIR}/sae-2.key"
    
    # QuKayDee configuration
    if [ -z "${ACCOUNT_ID}" ]; then
        echo "ERROR: ACCOUNT_ID not set. Please set your QuKayDee account ID."
        exit 1
    fi
    
    export QKD_MASTER_KME_HOSTNAME="https://kme-1.acct-${ACCOUNT_ID}.etsi-qkd-api.qukaydee.com"
    export QKD_SLAVE_KME_HOSTNAME="https://kme-2.acct-${ACCOUNT_ID}.etsi-qkd-api.qukaydee.com"
    export QKD_MASTER_SAE="sae-1"
    export QKD_SLAVE_SAE="sae-2"
    
    echo "QuKayDee Configuration:"
    echo "  Master KME: $QKD_MASTER_KME_HOSTNAME"
    echo "  Slave KME: $QKD_SLAVE_KME_HOSTNAME"
    echo "  Master SAE: $QKD_MASTER_SAE"
    echo "  Slave SAE: $QKD_SLAVE_SAE"
    
elif [ "${QKD_BACKEND}" = "cerberis-xgr" ]; then
    echo "Setting up Cerberis-XGR environment (ETSI 014):"
    
    # Certificate configuration
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
    
    echo "Cerberis-XGR Configuration:"
    echo "  Master KME: $QKD_MASTER_KME_HOSTNAME"
    echo "  Slave KME: $QKD_SLAVE_KME_HOSTNAME"
    
else
    echo "Using simulated QKD backend"
    
    if [ "${ETSI_API_VERSION}" = "014" ]; then
        # ETSI 014 simulated mode requires KME/SAE variables
        export QKD_MASTER_KME_HOSTNAME=${QKD_MASTER_KME_HOSTNAME:-"localhost"}
        export QKD_SLAVE_KME_HOSTNAME=${QKD_SLAVE_KME_HOSTNAME:-"localhost"}
        export QKD_MASTER_SAE=${QKD_MASTER_SAE:-"sae-1"}
        export QKD_SLAVE_SAE=${QKD_SLAVE_SAE:-"sae-2"}
        
        echo "ETSI 014 Simulated Configuration:"
        echo "  Master KME: $QKD_MASTER_KME_HOSTNAME"
        echo "  Slave KME: $QKD_SLAVE_KME_HOSTNAME"
        echo "  Master SAE: $QKD_MASTER_SAE"
        echo "  Slave SAE: $QKD_SLAVE_SAE"
    else
        echo "ETSI 004 simulated mode - using Docker Compose URIs"
    fi
fi

# Print all QKD environment variables
echo ""
echo "QKD Environment Variables:"
env | grep -E "QKD_|ETSI_|CLIENT_CERT_|SERVER_CERT_" | sort