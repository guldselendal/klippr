# Fix Upload Issues - Step by Step

## Issues Found

1. **Backend not running** - Backend server is not accessible
2. **Upload directory missing** - `backend/uploads/` directory doesn't exist or isn't writable

## Fix Steps

### Step 1: Create Upload Directory

```bash
cd backend
mkdir -p uploads
chmod 755 uploads
```

### Step 2: Start Backend Server

```bash
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or if you have a startup script:
```bash
./start_backend.sh
```

### Step 3: Verify Everything Works

```bash
cd backend
python3 debug_upload.py
```

Expected output:
```
✓ Backend is running
✓ Ollama is running
✓ Upload directory 'uploads' exists and is writable
✓ Upload endpoint is working
```

### Step 4: Start Frontend (if not already running)

```bash
cd frontend
npm run dev
```

### Step 5: Test Upload

1. Open browser to `http://localhost:3000`
2. Click "Upload Document"
3. Select a small EPUB or PDF file
4. Monitor backend terminal for any errors

## Common Issues

### Backend Won't Start

**Error**: `ModuleNotFoundError` or import errors
**Fix**: 
```bash
cd backend
pip install -r requirements.txt
```

**Error**: Port 8000 already in use
**Fix**: 
```bash
# Find process using port 8000
lsof -i :8000
# Kill it or use different port
python3 -m uvicorn main:app --reload --port 8001
```

### Upload Directory Permission Issues

**Error**: Permission denied
**Fix**:
```bash
cd backend
chmod 755 uploads
# Or if on macOS/Linux:
sudo chown $(whoami) uploads
```

### Ollama Not Running

**Error**: Cannot connect to Ollama
**Fix**:
```bash
# Start Ollama (if installed)
ollama serve

# Or check if it's running
curl http://localhost:11434/api/tags
```

## Quick Start Script

Create `start_all.sh`:

```bash
#!/bin/bash

# Start Ollama (if not running)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 2
fi

# Create uploads directory
mkdir -p backend/uploads
chmod 755 backend/uploads

# Start backend
cd backend
python3 -m uvicorn main:app --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop all services"
wait
```

