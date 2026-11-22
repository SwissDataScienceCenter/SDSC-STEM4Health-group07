#!/bin/bash
# Launch the bioreactor dashboard
# Usage: ./launch_dashboard.sh [port]

PORT=${1:-8501}
echo "Starting Bioreactor Dashboard on port $PORT..."
echo "Open http://localhost:$PORT in your browser"
cd "$(dirname "$0")"
streamlit run dashboard.py --server.port $PORT --server.headless true 2>&1
