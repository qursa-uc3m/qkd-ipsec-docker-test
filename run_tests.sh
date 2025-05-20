#!/bin/bash

# Test orchestration script for strongSwan QKD testing
set -e  # Exit on error

# Default configuration
ITERATIONS=20
ANALYZE_RESULTS=true
APPLY_NETWORK_CONDITIONS=true
NETWORK_DURATION="15m"  # Duration should exceed your test time
LATENCY=100  # ms
JITTER=0   # ms
PACKET_LOSS=5  # percent

# Get API version from environment or default to 014
ETSI_API_VERSION=${ETSI_API_VERSION:-014}
QKD_BACKEND=${QKD_BACKEND:-cerberis-xgr}

# Generate timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Alice and Bob IP addresses (from your docker-compose.yml)
ALICE_IP="172.30.0.3"
BOB_IP="172.30.0.2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --iterations|-i)
      ITERATIONS="$2"
      shift 2
      ;;
    --no-analyze|-n)
      ANALYZE_RESULTS=false
      shift
      ;;
    --no-network-conditions|-nn)
      APPLY_NETWORK_CONDITIONS=false
      shift
      ;;
    --latency|-l)
      LATENCY="$2"
      shift 2
      ;;
    --jitter|-j)
      JITTER="$2"
      shift 2
      ;;
    --packet-loss|-p)
      PACKET_LOSS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --iterations, -i NUMBER       Number of test iterations (default: 20)"
      echo "  --no-analyze, -n              Skip running analysis after tests"
      echo "  --no-network-conditions, -nn  Skip applying network conditions"
      echo "  --latency, -l NUMBER          Network latency in ms (default: 100)"
      echo "  --jitter, -j NUMBER           Latency jitter in ms (default: 0)"
      echo "  --packet-loss, -p NUMBER      Packet loss percentage (default: 5)"
      echo "  --help, -h                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Define the output directory structure
if [ "$APPLY_NETWORK_CONDITIONS" = true ]; then
  RELATIVE_DIR="${ETSI_API_VERSION}/${QKD_BACKEND}/lat${LATENCY}_jit${JITTER}_loss${PACKET_LOSS}_iter${ITERATIONS}_time_${TIMESTAMP}"
else
  RELATIVE_DIR="${ETSI_API_VERSION}/${QKD_BACKEND}/no_network_conditions_iter${ITERATIONS}_${TIMESTAMP}"
fi

# Host and Docker output directories
OUTPUT_DIR="./results/${RELATIVE_DIR}"
DOCKER_OUTPUT_DIR="/output/${RELATIVE_DIR}"

# Create results directory (both locally and in Docker)
mkdir -p $OUTPUT_DIR
docker exec alice bash -c "mkdir -p ${DOCKER_OUTPUT_DIR}"

# Create test configuration metadata file
cat > "${OUTPUT_DIR}/test_config.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "etsi_api_version": "${ETSI_API_VERSION}",
  "qkd_backend": "${QKD_BACKEND}",
  "iterations": ${ITERATIONS},
  "network_conditions": {
    "applied": ${APPLY_NETWORK_CONDITIONS},
    "latency_ms": ${LATENCY},
    "jitter_ms": ${JITTER},
    "packet_loss_percent": ${PACKET_LOSS}
  }
}
EOF

echo "Running tests with $ITERATIONS iterations..."
echo "Results will be stored in $OUTPUT_DIR/"

# Apply network conditions if enabled
if [ "$APPLY_NETWORK_CONDITIONS" = true ]; then
  echo "Applying network conditions for testing..."
  echo "  - Latency: ${LATENCY}ms with ${JITTER}ms jitter"
  echo "  - Packet loss: ${PACKET_LOSS}%"
  echo "  - Duration: ${NETWORK_DURATION}"
  echo "  - Only affecting traffic between Alice (${ALICE_IP}) and Bob (${BOB_IP})"
  
  # Apply latency to Alice's outgoing traffic to Bob only
  docker run -d --rm -v /var/run/docker.sock:/var/run/docker.sock \
    gaiaadm/pumba netem --tc-image ghcr.io/alexei-led/pumba-alpine-nettools:latest \
    --duration "${NETWORK_DURATION}" \
    --interface eth0 \
    --target "${BOB_IP}" \
    delay --time "${LATENCY}" --jitter "${JITTER}" --distribution normal \
    alice
  
  # Apply latency to Bob's outgoing traffic to Alice only
  docker run -d --rm -v /var/run/docker.sock:/var/run/docker.sock \
    gaiaadm/pumba netem --tc-image ghcr.io/alexei-led/pumba-alpine-nettools:latest \
    --duration "${NETWORK_DURATION}" \
    --interface eth0 \
    --target "${ALICE_IP}" \
    delay --time "${LATENCY}" --jitter "${JITTER}" --distribution normal \
    bob
  
  # Apply packet loss to Alice's outgoing traffic to Bob only
  docker run -d --rm -v /var/run/docker.sock:/var/run/docker.sock \
    gaiaadm/pumba netem --tc-image ghcr.io/alexei-led/pumba-alpine-nettools:latest \
    --duration "${NETWORK_DURATION}" \
    --interface eth0 \
    --target "${BOB_IP}" \
    loss --percent "${PACKET_LOSS}" --correlation 20 \
    alice
  
  # Apply packet loss to Bob's outgoing traffic to Alice only
  docker run -d --rm -v /var/run/docker.sock:/var/run/docker.sock \
    gaiaadm/pumba netem --tc-image ghcr.io/alexei-led/pumba-alpine-nettools:latest \
    --duration "${NETWORK_DURATION}" \
    --interface eth0 \
    --target "${ALICE_IP}" \
    loss --percent "${PACKET_LOSS}" --correlation 20 \
    bob
  
  # Give a moment for network conditions to apply
  echo "Waiting for network conditions to be applied..."
  sleep 5
fi

# Start Bob's test script in the background
docker exec -d bob bash -c "export IS_TLS_SERVER=1 && source /set_env.sh && python3 /etc/swanctl/bob_tests.py > ${DOCKER_OUTPUT_DIR}/bob_log.txt 2>&1"

# Run Alice's test script
docker exec alice bash -c "chmod -R 777 ${DOCKER_OUTPUT_DIR} && source /set_env.sh && python3 /etc/swanctl/alice_tests.py --iterations $ITERATIONS --output-dir ${DOCKER_OUTPUT_DIR}"

echo "Tests completed. Results accessible in $OUTPUT_DIR/"

# Run analysis if not disabled
if [ "$ANALYZE_RESULTS" = true ]; then
  echo "Analyzing results..."
  ANALYSIS_DIR="./analysis/${RELATIVE_DIR}"
  mkdir -p "${ANALYSIS_DIR}"
  
  python3 analyze_results.py "${OUTPUT_DIR}/plugin_timing_summary.csv" "${ANALYSIS_DIR}" --log-scale
  echo "Analysis completed! Results available in ${ANALYSIS_DIR}"
fi

# Optionally, stop containers
read -p "Stop containers? (y/n): " stop_containers
if [ "$stop_containers" = "y" ]; then
    docker-compose down
    echo "Containers stopped."
fi