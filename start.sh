#!/bin/bash

# Mosaic Maker Server Startup Script
# This script sets up the virtual environment and starts the server

echo "🎨 Starting Mosaic Maker Server..."

# Change to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check if server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Server is already running on port 8000"
    echo "🔄 Killing existing server..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 2
fi

# Start the server
echo "🚀 Starting server on http://localhost:8000"
echo "📱 Open your browser and navigate to: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""
echo "📝 Server output:"
python simple_server.py