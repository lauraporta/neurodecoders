#!/bin/bash

# Launch script for neurodecoders Streamlit apps
# This script launches all three apps on different ports

echo "🚀 Launching neurodecoders apps..."

# Function to launch an app on a specific port
launch_app() {
    local app_name=$1
    local app_path=$2
    local port=$3
    
    echo "Starting $app_name on port $port..."
    streamlit run "$app_path" --server.port "$port" --server.headless true &
    echo "✅ $app_name started on http://localhost:$port"
}

# Launch all apps
launch_app "Synthetic Data Generator" "neurodecoders/synthetic/app.py" 8501
launch_app "Neural Encoder" "neurodecoders/encoder/app.py" 8502  
launch_app "Neural Decoder" "neurodecoders/decoder/app.py" 8503

echo ""
echo "🎉 All apps launched successfully!"
echo ""
echo "📱 App URLs:"
echo "   Synthetic Data Generator: http://localhost:8501"
echo "   Neural Encoder:           http://localhost:8502"
echo "   Neural Decoder:           http://localhost:8503"
echo ""
echo "💡 To stop all apps, run: pkill -f streamlit"
echo ""

# Wait for user input to stop
read -p "Press Enter to stop all apps..."
pkill -f streamlit
echo "�� All apps stopped." 