#!/bin/bash
# Colver - Auto-deploy script
# Called by webhook_server.py on GitHub push to master

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "$(date): Starting deployment..."

# Pull latest code
echo "Pulling latest code..."
git pull

# Rebuild and restart container
echo "Rebuilding Docker image..."
docker compose up -d --build

echo "$(date): Deployment complete"
docker compose ps
