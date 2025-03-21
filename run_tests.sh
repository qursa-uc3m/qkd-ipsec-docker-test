#!/bin/bash

# Test orchestration script for strongSwan QKD testing
set -e  # Exit on error

# Create results directory
mkdir -p results

echo "Running tests..."
# Start Bob's test script in the background
docker exec -d bob bash -c "source /set_env.sh && python3 /etc/swanctl/bob_tests.py"

# Run Alice's test script
docker exec alice bash -c "source /set_env.sh && python3 /etc/swanctl/alice_tests.py"

echo "Tests completed. Results accesible in results/."
echo "You can process the results by running 'python3 process_results.py.'"

# Optionally, stop containers
read -p "Stop containers? (y/n): " stop_containers
if [ "$stop_containers" = "y" ]; then
    docker-compose down
    echo "Containers stopped."
fi