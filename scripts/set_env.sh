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
    
    # QKD URI Configuration
    # Don't override if already set by Docker Compose
    if [ -z "$QKD_SOURCE_URI" ]; then
        export QKD_SOURCE_URI="client://qkd_server_alice:25575"
    fi
    if [ -z "$QKD_DEST_URI" ]; then
        export QKD_DEST_URI="server://qkd_server_bob:25576"
    fi
    
    # QKD Host/Port Configuration (fallback for URI construction)
    export QKD_CLIENT_HOST=${QKD_CLIENT_HOST:-"qkd_server_alice"}
    export QKD_SERVER_HOST=${QKD_SERVER_HOST:-"qkd_server_bob"}
    export QKD_CLIENT_PORT=${QKD_CLIENT_PORT:-"25575"}
    export QKD_SERVER_PORT=${QKD_SERVER_PORT:-"25576"}
    
    # QKD QoS Configuration (for flexible adapter)
    export QKD_KEY_CHUNK_SIZE=${QKD_KEY_CHUNK_SIZE:-32}
    export QKD_TIMEOUT=${QKD_TIMEOUT:-60000}
    export QKD_MAX_BPS=${QKD_MAX_BPS:-40000}
    export QKD_MIN_BPS=${QKD_MIN_BPS:-5000}
    
    # Legacy KME settings for backward compatibility
    export QKD_MASTER_KME_HOSTNAME=${QKD_MASTER_KME_HOSTNAME:-"qkd_server_alice"}
    export QKD_SLAVE_KME_HOSTNAME=${QKD_SLAVE_KME_HOSTNAME:-"qkd_server_bob"}
    export QKD_MASTER_SAE=${QKD_MASTER_SAE:-"sae-1"}
    export QKD_SLAVE_SAE=${QKD_SLAVE_SAE:-"sae-2"}
    
    # Server connection settings for ETSI 004
    export SERVER_ADDRESS=${SERVER_ADDRESS:-"qkd_server_bob"}
    export SERVER_PORT=${SERVER_PORT:-25576}
    export CLIENT_ADDRESS=${CLIENT_ADDRESS:-"qkd_server_alice"}
    
    # Certificate configuration
    # Only set defaults if not already provided by Docker Compose
    CONTAINER_NAME=$(hostname)
    
    if [ "$CONTAINER_NAME" = "alice" ] || [ "$QKD_ROLE" = "alice" ]; then
        # Alice configuration - connects to Alice's server
        export CLIENT_CERT_PEM=${CLIENT_CERT_PEM:-"${ETSI004_CERTS_DIR}/client_cert_qkd_server_alice.pem"}
        export CLIENT_CERT_KEY=${CLIENT_CERT_KEY:-"${ETSI004_CERTS_DIR}/client_key_qkd_server_alice.pem"}
        export SERVER_CERT_PEM=${SERVER_CERT_PEM:-"${ETSI004_CERTS_DIR}/server_cert_qkd_server_alice.pem"}
        echo "Configured as Alice (connects to Alice's server)"
    elif [ "$CONTAINER_NAME" = "bob" ] || [ "$QKD_ROLE" = "bob" ]; then
        # Bob configuration - connects to Bob's server with Bob's certificates
        export CLIENT_CERT_PEM=${CLIENT_CERT_PEM:-"${ETSI004_CERTS_DIR}/client_cert_qkd_server_bob.pem"}
        export CLIENT_CERT_KEY=${CLIENT_CERT_KEY:-"${ETSI004_CERTS_DIR}/client_key_qkd_server_bob.pem"}
        export SERVER_CERT_PEM=${SERVER_CERT_PEM:-"${ETSI004_CERTS_DIR}/server_cert_qkd_server_bob.pem"}
        echo "Configured as Bob (connects to Bob's server)"
    else
        # Default fallback - but don't override if set by Docker Compose
        if [ -z "$CLIENT_CERT_PEM" ]; then
            export CLIENT_CERT_PEM="${ETSI004_CERTS_DIR}/client_cert_qkd_server_alice.pem"
        fi
        if [ -z "$CLIENT_CERT_KEY" ]; then
            export CLIENT_CERT_KEY="${ETSI004_CERTS_DIR}/client_key_qkd_server_alice.pem"
        fi
        if [ -z "$SERVER_CERT_PEM" ]; then
            export SERVER_CERT_PEM="${ETSI004_CERTS_DIR}/server_cert_qkd_server_alice.pem"
        fi
        echo "Using default configuration (respecting Docker Compose overrides)"
    fi
    
    # QKD specific parameters (original variable names)
    export KEY_INDEX=${KEY_INDEX:-0}
    export METADATA_SIZE=${METADATA_SIZE:-1024}
    export QKD_USE_TLS=${QKD_USE_TLS:-"true"}
    export QKD_CLIENT_VERIFY=${QKD_CLIENT_VERIFY:-"true"}
    
    # QoS parameters (original variable names - kept for backward compatibility)
    export QOS_KEY_CHUNK_SIZE=${QKD_KEY_CHUNK_SIZE}
    export QOS_MAX_BPS=${QKD_MAX_BPS}
    export QOS_MIN_BPS=${QKD_MIN_BPS}
    export QOS_JITTER=${QOS_JITTER:-10}
    export QOS_PRIORITY=${QOS_PRIORITY:-0}
    export QOS_TIMEOUT=${QKD_TIMEOUT}
    export QOS_TTL=${QOS_TTL:-3600}
    
    # Check if certificate files exist
    if [ ! -f "$CLIENT_CERT_PEM" ] || [ ! -f "$CLIENT_CERT_KEY" ] || [ ! -f "$SERVER_CERT_PEM" ]; then
        echo "WARNING: Some certificate files were not found at the expected paths:"
        echo "  CLIENT_CERT_PEM: $CLIENT_CERT_PEM"
        echo "  CLIENT_CERT_KEY: $CLIENT_CERT_KEY"
        echo "  SERVER_CERT_PEM: $SERVER_CERT_PEM"
        echo ""
        echo "Available certificates in ${ETSI004_CERTS_DIR}:"
        ls -la "${ETSI004_CERTS_DIR}/" 2>/dev/null || echo "  Directory not found or empty"
        echo ""
        echo "If you've placed certificates elsewhere, please set these variables manually."
    else
        echo "Certificate paths set successfully."
    fi
    
    echo "QKD URI Configuration:"
    echo "  Source URI: $QKD_SOURCE_URI"
    echo "  Destination URI: $QKD_DEST_URI"
    echo "  Client Host: $QKD_CLIENT_HOST:$QKD_CLIENT_PORT"
    echo "  Server Host: $QKD_SERVER_HOST:$QKD_SERVER_PORT"
    echo ""
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
    export QKD_MASTER_KME_HOSTNAME=${QKD_MASTER_KME_HOSTNAME:-"localhost"}
    export QKD_SLAVE_KME_HOSTNAME=${QKD_SLAVE_KME_HOSTNAME:-"localhost"}
    export QKD_MASTER_SAE=${QKD_MASTER_SAE:-"sae-1"}
    export QKD_SLAVE_SAE=${QKD_SLAVE_SAE:-"sae-2"}
        
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