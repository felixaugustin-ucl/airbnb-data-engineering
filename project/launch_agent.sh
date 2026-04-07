#!/bin/bash
# launch_agent.sh
# ───────────────
# Starts the lakehouse stack (PostgreSQL, MongoDB, Ollama) and launches
# the interactive MCP agent in one command.
#
# Usage:
#   ./launch_agent.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting lakehouse services..."
docker-compose up -d postgres mongodb ollama

echo "Waiting for databases to be healthy..."
docker-compose up -d --wait postgres mongodb

echo "Launching agent..."
docker-compose run --rm mcp-agent
