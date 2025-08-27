#!/bin/bash

# Launch script for MLflow UI
# This script launches MLflow UI for experiment tracking

echo "🚀 Launching MLflow UI..."

# Check if MLflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "❌ MLflow is not installed. Please install it first:"
    echo "   pip install mlflow"
    exit 1
fi

# Set MLflow tracking URI to local file system
# Read base path from config.yaml
BASE_PATH=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['base_path'])")
export MLFLOW_TRACKING_URI="file:${BASE_PATH}/mlruns"

# Get current hostname
HOSTNAME=$(hostname)
USERNAME=$(whoami)

# Launch MLflow UI
echo "Starting MLflow UI on port 5001..."
echo "📊 MLflow UI will be available at: http://localhost:5001"
echo ""
echo "🌐 For remote access, use SSH port forwarding:"
echo "   ssh -N ${USERNAME}@${HOSTNAME} -J ${USERNAME}@ssh.swc.ucl.ac.uk,${USERNAME}@hpc-gw2 -L 5001:localhost:5001"
echo ""
echo "💡 To stop MLflow UI, press Ctrl+C"
echo ""

# Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri file:${BASE_PATH}/mlruns  --gunicorn-opts "--timeout "3600"
