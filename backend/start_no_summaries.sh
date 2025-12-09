#!/bin/bash
# Startup script that disables automatic chapter summary generation on file upload
# Usage: ./start_no_summaries.sh [port]

set -e

# Get port from argument or use default
PORT=${1:-8000}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set environment variable to disable automatic summaries
export DISABLE_AUTO_SUMMARIES=true

# Also disable async pipeline to ensure summaries aren't generated in background
export USE_ASYNC_SUMMARIZATION=false

echo "=========================================="
echo "Starting Klippr Backend (No Auto-Summaries)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Automatic summaries: DISABLED"
echo "  - Async pipeline: DISABLED"
echo "  - Port: $PORT"
echo ""
echo "Note: You can still generate summaries manually via:"
echo "  POST /api/chapters/{chapter_id}/summarize"
echo "  POST /api/documents/{document_id}/summarize-all"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Start uvicorn server (use python3 -m uvicorn for better compatibility)
echo "Starting server..."
python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
