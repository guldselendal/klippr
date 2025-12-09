# Quick Fix: Upload Error

## Problem
Debug script shows:
- ✗ Backend not running
- ✓ Ollama is running  
- ✗ Upload directory check failed (but directory exists)
- ✗ Upload endpoint test failed (because backend isn't running)

## Solution

### Step 1: Start Backend Server

```bash
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or use the startup script:
```bash
./start_backend.sh
```

**Expected output**: 
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 2: Verify Backend is Running

In a new terminal:
```bash
curl http://localhost:8000/health
```

**Expected response**: `{"status":"ok","message":"Backend is running"}`

### Step 3: Test Upload Again

1. Make sure backend is running (Step 1)
2. Open browser to `http://localhost:3000`
3. Try uploading a file
4. Check browser console for detailed error messages

## If Backend Won't Start

### Check Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Check Port Availability
```bash
# Check if port 8000 is in use
lsof -i :8000

# If something is using it, kill it or use different port
python3 -m uvicorn main:app --reload --port 8001
```

### Check Python Version
```bash
python3 --version  # Should be 3.9+
```

## After Starting Backend

Run the debug script again:
```bash
cd backend
python3 debug_upload.py
```

**Expected output**:
```
✓ Backend is running
✓ Ollama is running
✓ Upload directory 'uploads' exists and is writable
✓ Upload endpoint is working
```

## Common Issues

### "ModuleNotFoundError"
**Fix**: Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### "Address already in use"
**Fix**: Kill existing process or use different port
```bash
lsof -i :8000 | grep LISTEN
kill -9 <PID>
```

### Backend starts but uploads still fail
1. Check browser console for specific error
2. Check backend terminal for error logs
3. Verify Ollama is running: `curl http://localhost:11434/api/tags`
4. Try uploading a very small file (< 1 MB) first

