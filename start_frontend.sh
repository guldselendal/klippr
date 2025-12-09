#!/bin/bash

# Start the Klippr frontend development server

cd "$(dirname "$0")/frontend" || exit 1

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Dependencies not found. Installing..."
    npm install
fi

# Start the development server
echo "Starting Klippr frontend on http://localhost:3000"
npm run dev -- --host

