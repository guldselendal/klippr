# Disable Automatic Summary Generation

This guide explains how to start the backend server with automatic chapter summary generation disabled.

## Quick Start

### Option 1: Using the Shell Script (Linux/macOS)

```bash
cd backend
./start_no_summaries.sh
```

Or specify a custom port:
```bash
./start_no_summaries.sh 8001
```

### Option 2: Using the Python Script (Cross-platform)

```bash
cd backend
python3 start_no_summaries.py
```

Or specify a custom port:
```bash
python3 start_no_summaries.py 8001
```

### Option 3: Manual Environment Variable

```bash
cd backend
export DISABLE_AUTO_SUMMARIES=true
export USE_ASYNC_SUMMARIZATION=false
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## What This Does

When `DISABLE_AUTO_SUMMARIES=true`:

1. **File uploads** will parse and save chapters **without generating summaries**
2. **Chapters are saved** with `summary=None` and `preview=None`
3. **Manual summary generation** is still available via API endpoints

## Manual Summary Generation

Even with automatic summaries disabled, you can still generate summaries manually:

### Generate summary for a single chapter:
```bash
POST /api/chapters/{chapter_id}/summarize
```

### Generate summaries for all chapters in a document:
```bash
POST /api/documents/{document_id}/summarize-all
```

## Environment Variables

- `DISABLE_AUTO_SUMMARIES=true` - Disables automatic summary generation on upload
- `USE_ASYNC_SUMMARIZATION=false` - Also disables async pipeline (recommended when summaries are disabled)

## Use Cases

This mode is useful for:

1. **Fast uploads** - Skip time-consuming summary generation during upload
2. **Testing** - Upload files quickly without waiting for summaries
3. **Selective summarization** - Generate summaries only for specific chapters
4. **Batch processing** - Upload many files first, then generate summaries later

## Example Workflow

```bash
# 1. Start server without auto-summaries
./start_no_summaries.sh

# 2. Upload files (fast, no summaries)
curl -X POST "http://localhost:8000/api/upload" -F "file=@book.epub"

# 3. Generate summaries later when needed
curl -X POST "http://localhost:8000/api/documents/{document_id}/summarize-all"
```

## Re-enabling Automatic Summaries

To re-enable automatic summaries, simply start the server normally:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or unset the environment variable:
```bash
unset DISABLE_AUTO_SUMMARIES
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

