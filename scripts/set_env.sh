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
    echo "Setting up ETSI 004 environment with compilation flags"
else
    export CPPFLAGS="-DETSI_014_API"
    export CFLAGS="-DETSI_014_API"
    echo "Setting up ETSI 014 environment with compilation flags"
fi

if [ "${ETSI_API_VERSION}" = "004" ]; then
    if [ -z "$QKD_SOURCE_URI" ] || [ -z "$QKD_DEST_URI" ]; then
        echo "WARNING: QKD_SOURCE_URI and QKD_DEST_URI not set for ETSI 004"
        echo "This may be expected in distributed mode - proceeding..."
        # Don't exit - let the application handle this
    else
        # Detect container role from existing environment
        if [[ "$QKD_DEST_URI" == *"qkd_server_alice"* ]]; then
            CONTAINER_ROLE="alice"
        elif [[ "$QKD_DEST_URI" == *"qkd_server_bob"* ]]; then
            CONTAINER_ROLE="bob"
        else
            # Check if BENCHMARK_ROLE is set (from Docker Compose)
            if [ -n "$BENCHMARK_ROLE" ]; then
                CONTAINER_ROLE="$BENCHMARK_ROLE"
                echo "Using BENCHMARK_ROLE for container role: $CONTAINER_ROLE"
            else
                echo "WARNING: Cannot determine container role from QKD_DEST_URI: $QKD_DEST_URI"
                echo "Proceeding without role detection - application will handle this"
                CONTAINER_ROLE="unknown"
            fi
        fi
        
        # Validate that certificate paths are set (by Docker Compose) - only if URIs are set
        if [ -n "$CLIENT_CERT_PEM" ] && [ -n "$CLIENT_CERT_KEY" ] && [ -n "$SERVER_CERT_PEM" ]; then
            # Check if certificate files exist
            if [ -f "$CLIENT_CERT_PEM" ] && [ -f "$CLIENT_CERT_KEY" ] && [ -f "$SERVER_CERT_PEM" ]; then
                echo "ETSI 004 Configuration (from Docker Compose):"
                echo "  Container Role: $CONTAINER_ROLE"
                echo "  Source URI: $QKD_SOURCE_URI"
                echo "  Destination URI: $QKD_DEST_URI"
                echo "  Backend: ${QKD_BACKEND:-python_client}"
                echo "  Certificates validated successfully"
            else
                echo "WARNING: Certificate files not found - may be expected in distributed mode"
                echo "  CLIENT_CERT_PEM: $CLIENT_CERT_PEM"
                echo "  CLIENT_CERT_KEY: $CLIENT_CERT_KEY"
                echo "  SERVER_CERT_PEM: $SERVER_CERT_PEM"
            fi
        else
            echo "WARNING: Certificate environment variables not set - may be expected in distributed mode"
        fi
    fi
    
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
env | grep -E "QKD_|ETSI_|CLIENT_CERT_|SERVER_CERT_|BENCHMARK_ROLE" | sort

# Exit successfully - let the application handle any remaining issues
echo "Environment setup completed successfully"