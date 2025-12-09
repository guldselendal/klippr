#!/usr/bin/env python3
"""
Startup script that disables automatic chapter summary generation on file upload.
Usage: python3 start_no_summaries.py [port]
"""
import os
import sys
import subprocess

# Set environment variable to disable automatic summaries
os.environ["DISABLE_AUTO_SUMMARIES"] = "true"
os.environ["USE_ASYNC_SUMMARIZATION"] = "false"

# Get port from argument or use default
port = sys.argv[1] if len(sys.argv) > 1 else "8000"

print("=" * 50)
print("Starting Klippr Backend (No Auto-Summaries)")
print("=" * 50)
print()
print("Configuration:")
print("  - Automatic summaries: DISABLED")
print("  - Async pipeline: DISABLED")
print("  - Port:", port)
print()
print("Note: You can still generate summaries manually via:")
print("  POST /api/chapters/{chapter_id}/summarize")
print("  POST /api/documents/{document_id}/summarize-all")
print()
print("Press Ctrl+C to stop the server")
print("=" * 50)
print()

# Start uvicorn server
try:
    # Try to import uvicorn, if not available, use subprocess
    try:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=True)
    except ImportError:
        # Fallback: use subprocess to run uvicorn module
        import subprocess
        print("Note: Running uvicorn via subprocess...")
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", port, "--reload"])
except KeyboardInterrupt:
    print("\nServer stopped by user")
except Exception as e:
    print(f"Error starting server: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure uvicorn is installed: pip install uvicorn")
    print("  2. Activate your virtual environment if you have one")
    print("  3. Try: python3 -m uvicorn main:app --host 0.0.0.0 --port", port)
    sys.exit(1)

