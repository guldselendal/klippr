#!/bin/bash
# Fix OpenAI/httpx version conflict

echo "Fixing OpenAI and httpx version conflict..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Upgrade OpenAI to a compatible version
pip install --upgrade "openai>=1.12.0" "httpx>=0.27.0"

echo "Done! Please restart your backend server."
