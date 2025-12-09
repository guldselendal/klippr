#!/bin/bash

# Start the Klippr backend server

cd "$(dirname "$0")/backend" || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/bin/uvicorn" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the server
echo "Starting Klippr backend server on http://localhost:8000"
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

