#!/bin/bash

# Stop and remove containers created by docker-compose
echo "Stopping docker-compose containers..."
docker-compose down

# Stop and remove any Pumba containers that might be running
echo "Stopping Pumba containers..."
docker rm -f $(docker ps -a -q --filter ancestor=gaiaadm/pumba) 2>/dev/null || true

# Stop and remove any other QKD or StrongSwan containers that might be left
echo "Stopping any remaining QKD or StrongSwan containers..."
docker rm -f $(docker ps -a -q --filter name=qkd_) 2>/dev/null || true
docker rm -f $(docker ps -a -q --filter name=alice) 2>/dev/null || true
docker rm -f $(docker ps -a -q --filter name=bob) 2>/dev/null || true
docker rm -f $(docker ps -a -q --filter name=strongswan) 2>/dev/null || true

# Remove all unused images, networks, and volumes
echo "Removing unused Docker resources..."
docker system prune -a --volumes -f

echo "Docker environment cleaned successfully!"