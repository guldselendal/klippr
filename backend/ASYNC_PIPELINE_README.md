# Async Summary Pipeline

## Overview

The async summary pipeline decouples LLM summarization work from database writes, enabling:

- **Incremental persistence**: Chapters are saved as summaries complete (not all at once)
- **True concurrency**: Multiple workers can process summaries while DB writer batches writes
- **Better responsiveness**: Upload endpoint returns immediately, processing happens in background
- **Adaptive scaling**: Worker count adjusts based on load and latency

## Architecture

```
Upload Endpoint
    ↓
Parse Document → Save Document (fast, <100ms)
    ↓
Enqueue Chapters → Summary Queue (bounded: 200)
    ↓
Return Immediately (status: "processing")
    ↓
    ┌─────────────────────────────────────┐
    │  Summary Worker Pool (3-16 workers)│
    │  - Consumes from summary_queue      │
    │  - Generates summaries via LLM      │
    │  - Produces to write_queue          │
    └──────────────┬──────────────────────┘
                   ↓
    ┌─────────────────────────────────────┐
    │  Single DB Writer Thread            │
    │  - Consumes from write_queue         │
    │  - Batches: 50 rows OR 500ms        │
    │  - Commits: Short transactions      │
    └─────────────────────────────────────┘
```

## Features

### 1. Producer/Consumer Pattern
- **Summary Queue**: Bounded queue (200 items) for chapters awaiting summarization
- **Write Queue**: Bounded queue (500 items) for chapters ready to be written
- **Backpressure**: Workers block if write queue is full, preventing memory issues

### 2. Batched Database Writes
- Commits every **50 rows** OR every **500ms**, whichever comes first
- Uses PostgreSQL `ON CONFLICT DO UPDATE` for idempotent writes
- Falls back to `bulk_insert_mappings` for SQLite compatibility

### 3. Adaptive Worker Scaling
- Starts with 3 workers (configurable via `SUMMARY_MAX_WORKERS`)
- Scales up to 16 workers based on:
  - Queue depth (more queued = more workers)
  - LLM latency (higher latency = more workers)
  - CPU cores (target: 2-3× cores worth of work in flight)
- Scales down if DB write queue is saturated (backpressure)

### 4. Graceful Shutdown
- Sends "poison pills" to workers to stop gracefully
- Flushes remaining batches before shutdown
- Configurable timeout for thread joins

## Configuration

### Environment Variables

```bash
# Enable/disable async pipeline (default: true)
USE_ASYNC_SUMMARIZATION=true

# Initial worker count (default: 3)
SUMMARY_MAX_WORKERS=3

# Skip pipeline initialization (for testing)
SKIP_PIPELINE_INIT=false
```

### Feature Flag

The pipeline can be toggled via `USE_ASYNC_SUMMARIZATION`:

- **`true`**: Uses async pipeline (recommended)
- **`false`**: Falls back to synchronous mode (original behavior)

## Usage

### Upload Endpoint

The `/api/upload` endpoint now returns immediately:

```json
{
  "message": "File uploaded and parsing started. Summaries are being generated in the background.",
  "document_id": "uuid",
  "chapters": [
    {
      "id": "chapter-uuid",
      "title": "Chapter 1",
      "summary": null,
      "preview": null,
      "status": "processing"
    }
  ],
  "status": "processing",
  "summary_queue_size": 10,
  "write_queue_size": 5
}
```

### Pipeline Status Endpoint

Check pipeline status:

```bash
GET /api/pipeline/status
```

Response:
```json
{
  "async_enabled": true,
  "summary_queue_size": 10,
  "write_queue_size": 5,
  "summary_queue_maxsize": 200,
  "write_queue_maxsize": 500
}
```

## Monitoring

### Logs

The pipeline logs key events:

```
Summary worker 0 started
DB writer started
Pipeline initialized with 3 workers
Enqueueing 50 chapters for async summary generation...
✓ Enqueued 50 chapters. Summaries will be generated in background.
DB: Committed batch of 50 chapters in 45.23 ms
Scaled up to 6 workers (optimal: 6)
```

### Metrics

Track these metrics for performance:

- **Queue depths**: `summary_queue_size`, `write_queue_size`
- **Worker count**: Number of active worker threads
- **Batch commit time**: Time to commit batches (should be <100ms)
- **LLM latency**: p50/p95/p99 latency from `llm_latencies` deque

## Performance Benefits

### Before (Synchronous)
- Upload endpoint blocks for 2-5 minutes (waiting for all summaries)
- DB session held open during entire LLM phase
- Single transaction for all chapters
- No incremental progress

### After (Async)
- Upload endpoint returns in <1 second
- DB writes happen incrementally (every 50 rows or 500ms)
- Multiple concurrent transactions (better for PostgreSQL)
- Users see progress as chapters complete

### Expected Improvements

- **Upload response time**: 2-5 minutes → <1 second (100-300× faster)
- **First chapter visible**: 15-30 seconds → 15-30 seconds (same, but non-blocking)
- **DB write contention**: Eliminated (batched writes, no long transactions)
- **Throughput**: 3-6× improvement (adaptive workers, better concurrency)

## Troubleshooting

### Pipeline Not Starting

Check logs for:
```
✓ Async summary pipeline enabled
Pipeline initialized with 3 workers
```

If you see warnings, check:
- `USE_ASYNC_SUMMARIZATION` is set to `true`
- `summary_pipeline.py` is importable
- No import errors in logs

### Workers Not Processing

Check queue sizes:
```bash
curl http://localhost:8000/api/pipeline/status
```

If `summary_queue_size` is high but not decreasing:
- Check Ollama is running
- Check worker threads are alive
- Check for errors in logs

### Database Writes Slow

If `write_queue_size` is high:
- Check PostgreSQL is running
- Check connection pool isn't exhausted
- Check for lock contention (shouldn't happen with batched writes)

### Memory Issues

If memory usage is high:
- Reduce `SUMMARY_MAX_WORKERS` (default: 3)
- Reduce queue sizes (in `summary_pipeline.py`)
- Check for memory leaks in LLM calls

## Rollback

To disable async pipeline:

1. Set environment variable:
   ```bash
   export USE_ASYNC_SUMMARIZATION=false
   ```

2. Restart backend server

3. System will use synchronous mode (original behavior)

## Future Enhancements

- [ ] WebSocket endpoint for real-time progress updates
- [ ] Resume processing on restart (persist queue state)
- [ ] Priority queue for important documents
- [ ] Metrics endpoint with Prometheus format
- [ ] Circuit breaker for LLM failures

