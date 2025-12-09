# Ollama Timeout Fix - Final Implementation

## Executive Summary

**Root Cause**: Client-side 120s timeout is insufficient when 8+ parallel workers compete for Ollama GPU/VRAM, causing model reloads and queueing delays that push individual chunk processing beyond the timeout threshold.

**Fix Applied**: Increased adaptive timeout (180-600s), added retry logic (3 attempts with exponential backoff), reduced parallelism (8→3 workers), reduced chunk size (3000→2000 chars), and added connection pooling.

## Reasoning

- **Timeout too low**: 120s insufficient when 8 parallel workers × 4 chunks = 32 concurrent requests saturate Ollama
- **No retries**: Timeouts fail immediately; transient model reloads cause permanent failures
- **Over-parallelization**: 8 workers exceed VRAM capacity, triggering model evictions/reloads
- **Chunk size**: 3000 chars ≈ 900-1000 tokens; under load, generation exceeds 120s
- **Token throughput**: phi3:mini ~20-40 tokens/sec; 2000 token max × parallel load = 50-100s+ per chunk

## Findings

| Aspect | Current | Issue |
|--------|---------|-------|
| **Timeout** | 120s fixed | Too short for chunks under load |
| **Retries** | None | Transient failures cause permanent loss |
| **Workers** | 8 default | Overwhelms Ollama, causes reloads |
| **Chunk size** | 3000 chars | Large chunks take longer under contention |
| **Connection** | New per request | No pooling, higher overhead |

## Root-Cause Hypotheses (Ranked)

### 1. Client Timeout Too Low + High Concurrency (High Confidence)
- **Evidence**: 120s timeout; 8+ workers; timeouts on chunks 5-7 (mid-batch)
- **Fix**: Adaptive timeout (180-600s) + retry logic

### 2. No Retry Logic (High Confidence)
- **Evidence**: Timeouts fail immediately; no retry mechanism
- **Fix**: 3 retries with exponential backoff

### 3. Over-Parallelization (Medium-High Confidence)
- **Evidence**: 8 workers default; timeouts correlate with parallel load
- **Fix**: Reduce to 3 workers default

### 4. Chunk Size (Low-Medium Confidence)
- **Evidence**: 3000 char chunks + prompt ≈ 1000 tokens
- **Fix**: Reduce to 2000 chars for faster processing

## Quick Mitigations Applied

### ✅ 1. Adaptive Timeout with Retry Logic
**File**: `backend/llm_provider.py`
- Timeout: 180-600s (adaptive based on chunk size)
- Retries: 3 attempts with exponential backoff
- Retryable: Timeouts, connection errors, rate limits, server errors

### ✅ 2. Reduced Parallelism
**Files**: `backend/main.py`, `backend/summarizer_parallel.py`, `backend/summarizer.py`
- Default workers: 8 → 3
- Max workers: 16 → 6 (for chunk processing)

### ✅ 3. Reduced Chunk Size
**File**: `backend/summarizer.py`
- Chunk size: 3000 → 2000 chars
- Overlap: 500 → 400 chars

### ✅ 4. Connection Pooling
**File**: `backend/llm_provider.py`
- HTTPAdapter with connection pooling
- Pool size: 10 connections, max 20
- Built-in retry for HTTP errors

## Durable Fixes (Future Enhancements)

### 1. Request Queue with Rate Limiting
- Implement `OllamaRateLimiter` class
- Limit concurrent requests to 3-4
- Add minimum interval between requests

### 2. Model Warm-up on Startup
- Pre-warm Ollama model on backend startup
- Keep model loaded in memory
- Reduce first-request latency

### 3. Streaming Responses
- Enable streaming for long generations
- Detect stalls earlier
- Keep connection alive

## Test Plan

### Minimal Reproduction
1. Upload document with 3 chapters (~10KB each, ~4 chunks each)
2. Set `SUMMARY_MAX_WORKERS=8` (to test old behavior)
3. **Expected**: Timeouts occur
4. Set `SUMMARY_MAX_WORKERS=3` (new default)
5. **Expected**: Zero timeouts, all chunks succeed

### Load Test
1. Upload document with 10+ chapters
2. Monitor Ollama logs for model reloads
3. Track per-chunk latency (p50, p95, p99)
4. **Pass criteria**: p95 < 180s, timeout rate < 1%

### Retry Test
1. Temporarily reduce timeout to 60s to force failures
2. Verify retries occur and succeed
3. **Pass criteria**: All chunks succeed after retries

## Monitoring

### Metrics to Track
- **Per-request**: tokens_in, tokens_out, duration_ms, retry_count, timeout_flag
- **Aggregate**: p50/p95/p99 latency, timeout_rate, retry_rate, concurrent_requests
- **System**: Ollama model reloads, VRAM usage, queue depth

### Alert Thresholds
- **Timeout rate > 5%**: Alert immediately
- **p95 latency > 300s**: Warning
- **Concurrent requests > 6**: Warning
- **Retry rate > 20%**: Investigate Ollama health

### Log Lines Added
```python
# Retry messages
"Retrying Ollama request (attempt X/3, waited Ys)..."

# Timeout failures
"Ollama timeout: Request took too long after 3 attempts (timeout=Zs)"
```

## Final Recommendation

**Applied Configuration**:

```python
# Timeout: Adaptive 180-600s based on chunk size
timeout=(5, base_timeout)  # base_timeout = max(180, min(600, tokens*2+60))

# Retries: 3 attempts with exponential backoff
max_retries=3
backoff = (2^attempt) + random(0,1) seconds

# Parallelism: Reduced to 3 workers
SUMMARY_MAX_WORKERS=3  # Default

# Chunk size: Reduced to 2000 chars
CHUNK_SIZE=2000
OVERLAP=400
```

**Expected Results**:
- Timeout rate: 10-20% → <1%
- Throughput: ~20% slower but much more reliable
- Latency: More consistent, fewer outliers

**Configuration Override**:
```bash
# If you have more GPU/VRAM, can increase workers
export SUMMARY_MAX_WORKERS=4

# For testing, can reduce timeout
# (but not recommended for production)
```

## Files Modified

1. `backend/llm_provider.py` - Retry logic, adaptive timeout, connection pooling
2. `backend/summarizer.py` - Reduced chunk size, reduced workers
3. `backend/summarizer_parallel.py` - Reduced default workers
4. `backend/main.py` - Reduced default workers (3 instances)

All changes are backward compatible and can be overridden via environment variables.

