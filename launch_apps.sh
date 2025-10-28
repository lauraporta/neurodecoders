#!/bin/bash

# Launch script for MLflow UI
# This script launches MLflow UI for experiment tracking
# Usage: ./launch_apps.sh [PORT]
# Example: ./launch_apps.sh 5001
# Default port: 5002

# Get port from command line argument or use default
PORT=${1:-5002}

echo "🚀 Launching MLflow UI..."

# Check if MLflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "❌ MLflow is not installed. Please install it first:"
    echo "   pip install mlflow"
    exit 1
fi

# Load environment variables from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Construct MLflow tracking URI from environment variables
if [ -n "$POSTGRES_USER" ] && [ -n "$POSTGRES_PASSWORD" ] && [ -n "$POSTGRES_DB" ]; then
    POSTGRES_HOST=${POSTGRES_HOST:-localhost}
    POSTGRES_PORT=${POSTGRES_PORT:-5432}
    export MLFLOW_TRACKING_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
else
    echo "⚠️  Warning: Database credentials not found in .env file"
    echo "    Using default tracking URI"
fi

# Get current hostname
HOSTNAME=$(hostname)
USERNAME=$(whoami)

# Launch MLflow UI
echo "Starting MLflow UI on port ${PORT}..."
echo "📊 MLflow UI will be available at: http://localhost:${PORT}"
echo ""
echo "🌐 For remote access, use SSH port forwarding:"
echo "   ssh -N ${USERNAME}@${HOSTNAME} -J ${USERNAME}@ssh.swc.ucl.ac.uk,${USERNAME}@hpc-gw2 -L ${PORT}:localhost:${PORT}"
echo ""
echo "💡 To stop MLflow UI, press Ctrl+C"
echo ""

# Launch MLflow UI
mlflow ui --host 0.0.0.0 --port ${PORT} --backend-store-uri "${MLFLOW_TRACKING_URI}" --gunicorn-opts "--timeout 3600"

