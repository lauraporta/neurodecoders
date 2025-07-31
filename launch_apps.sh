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
export MLFLOW_TRACKING_URI="file:./mlruns"

# Launch MLflow UI
echo "Starting MLflow UI on port 5000..."
echo "📊 MLflow UI will be available at: http://localhost:5000"
echo ""
echo "🌐 For remote access, use SSH port forwarding:"
echo "   ssh -L 5000:localhost:5000 your-username@gpu-380-18"
echo ""
echo "💡 To stop MLflow UI, press Ctrl+C"
echo ""

# Launch MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
