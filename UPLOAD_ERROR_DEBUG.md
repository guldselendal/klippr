# Upload Error Debugging Guide

## Quick Diagnostic Steps

### 1. Check Browser Console
1. Open DevTools (F12)
2. Go to **Console** tab
3. Attempt upload
4. Look for error messages - copy the full error text

### 2. Check Network Tab
1. Go to **Network** tab in DevTools
2. Filter for "upload"
3. Click on the failing request
4. Check:
   - **Status code** (400, 500, 503, 504, timeout?)
   - **Response** tab - what error message is returned?
   - **Headers** - request and response headers

### 3. Check Backend Logs
1. Look at the terminal where backend is running
2. Look for error messages during upload
3. Check for:
   - "Error parsing file"
   - "Ollama timeout"
   - "Cannot connect to Ollama"
   - Any Python tracebacks

### 4. Verify Services
```bash
# Check backend
curl http://localhost:8000/health

# Check Ollama
curl http://localhost:11434/api/tags
```

## Common Error Scenarios

### Error: "Cannot connect to backend server"
- **Cause**: Backend not running
- **Fix**: Start backend: `cd backend && python3 -m uvicorn main:app --reload`

### Error: "Cannot connect to Ollama"
- **Cause**: Ollama not running
- **Fix**: Start Ollama: `ollama serve` (or ensure it's running)

### Error: "Upload timed out"
- **Cause**: File too large or Ollama processing too slow
- **Fix**: 
  - Try smaller file
  - Check Ollama is running
  - Increase timeout (already set to 5 minutes)

### Error: "Error parsing file"
- **Cause**: Corrupted or unsupported file format
- **Fix**: 
  - Verify file is valid EPUB/PDF
  - Try re-downloading or converting the file

### Error: "This file already exists"
- **Cause**: Duplicate filename detection
- **Fix**: Rename the file or delete existing document

### Error: "Only EPUB and PDF files are supported"
- **Cause**: Wrong file type
- **Fix**: Ensure file has `.epub` or `.pdf` extension (case-sensitive)

## What to Report

If upload still fails, provide:
1. **Exact error message** from alert/console
2. **HTTP status code** from Network tab
3. **Response body** from Network tab
4. **Backend logs** showing error
5. **File details**: size, type, filename
6. **Browser console errors** (full stack trace if available)

