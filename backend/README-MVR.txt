# Minimum Viable Run (MVR) Guide

## Environment

**OS**: macOS (darwin 24.6.0) or Linux
**Python**: 3.9+
**CPU**: 4-8 cores typical
**Memory**: 4GB+ recommended

## Build and Run Commands

### 1. Setup Environment
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Database Setup
```bash
# PostgreSQL (recommended)
export DATABASE_URL="postgresql://klippr:klippr@localhost:5432/klippr"
# OR SQLite (legacy)
export DATABASE_URL="sqlite:///./readerz.db"

# Initialize database
python3 -c "from database import init_db; init_db()"
```

### 3. Start Backend Server
```bash
# Ensure async pipeline is DISABLED for MVR (default)
unset USE_ASYNC_SUMMARIZATION  # or set to "false"

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# OR
python3 main.py
```

### 4. Verify Startup
```bash
# Health check
curl http://localhost:8000/health
# Expected: {"status":"ok","message":"Backend is running"}

# Root endpoint
curl http://localhost:8000/
# Expected: {"message":"ReaderZ API is running"}
```

## Critical Flows (MVR)

### Flow 1: Application Startup
**Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`
**Expected Output**:
- "ℹ️  Async pipeline disabled (USE_ASYNC_SUMMARIZATION=false). Using synchronous mode."
- "Upload directory: ..."
- Server starts on port 8000

**Verification**:
- No errors in console
- Health endpoint responds
- Database connection successful

### Flow 2: File Upload and Parsing
**Command**:
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@test.epub"
```

**Expected Behavior**:
- File saved to uploads/
- Chapters extracted
- Document saved to database
- Response includes document_id and chapters

**Verification**:
- File appears in uploads/ directory
- Database contains document and chapters
- Response status 200

### Flow 3: Summary Generation (Synchronous)
**Command**: Same as Flow 2 (automatic during upload)
**Expected Behavior**:
- Summaries generated using ThreadPoolExecutor (3-8 workers)
- Summaries saved to database
- Titles and previews generated

**Verification**:
- Chapter.summary field populated
- Chapter.preview field populated
- No errors in logs

### Flow 4: API Endpoints
**Commands**:
```bash
# Get documents
curl http://localhost:8000/api/documents

# Get chapters
curl http://localhost:8000/api/chapters

# Get single chapter
curl http://localhost:8000/api/chapters/{chapter_id}
```

**Expected Behavior**:
- All endpoints return 200
- JSON responses with expected structure
- No database connection errors

**Verification**:
- Response format matches API spec
- No timeout errors
- Response time < 1s for simple queries

### Flow 5: Shutdown
**Command**: Ctrl+C or SIGTERM
**Expected Behavior**:
- Graceful shutdown
- No hanging threads
- Database connections closed

**Verification**:
- Process exits cleanly
- No "zombie" threads
- Database pool released

## Smoke Tests

### Test 1: Basic Upload
```bash
# Create test file
echo "Test content" > test.txt

# Upload (should fail with proper error)
curl -X POST "http://localhost:8000/api/upload" -F "file=@test.txt"
# Expected: 400 error with message about file type

# Upload valid EPUB/PDF
curl -X POST "http://localhost:8000/api/upload" -F "file=@test.epub"
# Expected: 200 with document_id
```

### Test 2: Database Operations
```bash
# Get documents
curl http://localhost:8000/api/documents | jq '.documents | length'
# Expected: Number > 0 after upload

# Get chapters
curl http://localhost:8000/api/chapters | jq '.chapters | length'
# Expected: Number > 0 after upload
```

### Test 3: Concurrency Limits
```bash
# Check LLM concurrency limiter
curl http://localhost:8000/api/pipeline/status | jq '.llm_concurrency'
# Expected: Shows limits (Ollama: 16, Global: 20 by default)
```

## Troubleshooting

### Issue: Database Connection Error
**Solution**: Check DATABASE_URL, ensure database is running

### Issue: Ollama Not Available
**Solution**: Install Ollama, start service: `ollama serve`

### Issue: Port Already in Use
**Solution**: Change port: `uvicorn main:app --port 8001`

### Issue: Import Errors
**Solution**: Ensure venv is activated, dependencies installed

## Performance Baseline (MVR)

**Target Metrics** (for single file upload with 10 chapters):
- Upload + parsing: < 5s
- Summary generation: 30-120s (depends on LLM)
- API response time (GET): < 100ms (p95)
- Memory usage: < 500MB
- CPU usage: < 50% per core

## Notes

- Async pipeline is **disabled by default** for MVR
- Synchronous mode uses ThreadPoolExecutor with 3-8 workers
- LLM concurrency limited to 16 (Ollama) / 20 (global)
- Database connection pooling: 10 connections (PostgreSQL)

