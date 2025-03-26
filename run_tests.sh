#!/bin/bash

# Test orchestration script for strongSwan QKD testing
set -e  # Exit on error

# Default configuration
ITERATIONS=10
OUTPUT_DIR="results"
ANALYZE_RESULTS=true

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
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --iterations, -i NUMBER  Number of test iterations (default: 10)"
      echo "  --no-analyze, -n         Skip running analysis after tests"
      echo "  --help, -h               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create results directory
mkdir -p $OUTPUT_DIR

echo "Running tests with $ITERATIONS iterations..."
echo "Results will be stored in $OUTPUT_DIR/"

# Start Bob's test script in the background
docker exec -d bob bash -c "source /set_env.sh && python3 /etc/swanctl/bob_tests.py"

# Run Alice's test script
docker exec alice bash -c "source /set_env.sh && python3 /etc/swanctl/alice_tests.py --iterations $ITERATIONS"

echo "Tests completed. Results accesible in $OUTPUT_DIR/."

# Run analysis if not disabled
if [ "$ANALYZE_RESULTS" = true ]; then
  echo "Analyzing results..."
  python3 analyze_results.py "results/latencies.csv" "$OUTPUT_DIR/analysis"
  echo "Analysis completed! Results available in $OUTPUT_DIR/analysis/"
fi

# Optionally, stop containers
read -p "Stop containers? (y/n): " stop_containers
if [ "$stop_containers" = "y" ]; then
    docker-compose down
    echo "Containers stopped."
fi