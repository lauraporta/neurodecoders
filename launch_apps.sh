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
    
    # Check if PostgreSQL is running
    echo "🔍 Checking PostgreSQL service..."
    if ! pg_isready -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} &> /dev/null; then
        echo "⚠️  PostgreSQL is not running. Attempting to start..."
        
        # Check if PGDATA is set, otherwise use default
        if [ -z "$PGDATA" ]; then
            PGDATA="${HOME}/postgres_data"
            echo "ℹ️  PGDATA not set, using default: $PGDATA"
        fi
        
        # Try to start PostgreSQL using pg_ctl (conda environment)
        if [ -d "$PGDATA" ]; then
            LOG_FILE="${HOME}/postgres_logfile.log"
            pg_ctl -D "$PGDATA" -l "$LOG_FILE" start
            sleep 3  # Wait for server to start
            
            # Check again if it's running
            if pg_isready -h ${POSTGRES_HOST} -p ${POSTGRES_PORT} -U ${POSTGRES_USER} &> /dev/null; then
                echo "✅ PostgreSQL service started successfully"
            else
                echo "❌ Failed to start PostgreSQL service"
                echo "   Check log file: $LOG_FILE"
                echo "   Or start manually: pg_ctl -D $PGDATA -l $LOG_FILE start"
                exit 1
            fi
        else
            echo "❌ PostgreSQL data directory not found: $PGDATA"
            echo "   Please initialize PostgreSQL first. See database_migration.md for instructions."
            exit 1
        fi
    else
        echo "✅ PostgreSQL is running"
    fi
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

