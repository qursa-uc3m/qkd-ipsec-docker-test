#!/bin/bash

# Directory where certificates are stored
CERTS_DIR="/qkd_certs"

# Check if CERBERIS_XGR is enabled
if [ "${QKD_BACKEND}" = "qukaydee" ]; then
    echo "Setting up QuKayDee environment:"
    
    # Certificate configuration
    export QKD_MASTER_CA_CERT_PATH="${CERTS_DIR}/account-${ACCOUNT_ID}-server-ca-qukaydee-com.crt"
    export QKD_SLAVE_CA_CERT_PATH="${CERTS_DIR}/account-${ACCOUNT_ID}-server-ca-qukaydee-com.crt"
    
    export QKD_MASTER_CERT_PATH="${CERTS_DIR}/sae-1.crt"
    export QKD_MASTER_KEY_PATH="${CERTS_DIR}/sae-1.key"
    
    export QKD_SLAVE_CERT_PATH="${CERTS_DIR}/sae-2.crt"
    export QKD_SLAVE_KEY_PATH="${CERTS_DIR}/sae-2.key"
    
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
else
    echo "Using default QKD backend (simulated)"
    # Set default environment variables for simulated mode
    export QKD_MASTER_KME_HOSTNAME="localhost"
    export QKD_SLAVE_KME_HOSTNAME="localhost"
    export QKD_MASTER_SAE="sae-1"
    export QKD_SLAVE_SAE="sae-2"
fi

# Print all QKD environment variables
echo "QKD Environment Variables:"
env | grep "QKD_" | sort