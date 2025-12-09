# Implementation Guide: Pipeline Optimizations

## Quick Start

### Step 1: Apply Track 1 Optimizations (No Infrastructure)

These are low-risk, high-impact changes that can be applied immediately.

#### 1.1 Increase Queue Sizes

**File**: `backend/summary_pipeline.py`

**Change**:
```python
# Line 42-43
summary_queue = queue.Queue(maxsize=1000)  # Was 200
write_queue = queue.Queue(maxsize=1000)    # Was 500
```

**Impact**: Eliminates producer blocking for 385 chapters
**Time**: 1 minute
**Risk**: Low

#### 1.2 Optimize Chunking

**File**: `backend/summarizer.py`

**Change**:
```python
# In generate_summary() function, around line 95-96
CHUNK_SIZE = 3500  # Was 2000
OVERLAP = 275      # Was 400
```

**Impact**: Reduces chunk count by 40-50%
**Time**: 1 minute
**Risk**: Low

#### 1.3 Add Concurrency Semaphore

**File**: `backend/summary_pipeline.py`

**Add after imports** (around line 56):
```python
# Global semaphore to limit concurrent LLM calls (prevents GPU saturation)
LLM_CONCURRENCY_LIMIT = int(os.getenv("LLM_CONCURRENCY_LIMIT", 4))
llm_semaphore = threading.Semaphore(LLM_CONCURRENCY_LIMIT)
```

**Modify summary_worker()** (around line 71-77):
```python
try:
    # Generate summary with concurrency limit
    with llm_semaphore:  # Limit concurrent LLM calls
        if len(task.content) <= 5000:
            summary = generate_summary(task.content, task.title)
        else:
            summary = generate_summary(task.content, task.title)
```

**Impact**: Prevents GPU saturation, improves utilization
**Time**: 5 minutes
**Risk**: Low

#### 1.4 Update Backpressure Threshold

**File**: `backend/summary_pipeline.py`

**In compute_optimal_workers()** (around line 250):
```python
# Reduce if DB write queue is saturated (backpressure)
if db_write_queue_depth > 800:  # Was 400, now 80% of 1000
    optimal = max(1, optimal - 2)
```

**Impact**: Better backpressure handling
**Time**: 1 minute
**Risk**: Low

### Step 2: Test Optimizations

1. **Restart backend**:
   ```bash
   cd backend
   source venv/bin/activate
   python -m uvicorn main:app --reload
   ```

2. **Upload 385-chapter document**

3. **Run diagnostic** (after 2 minutes):
   ```bash
   python diagnose_pipeline_385.py
   ```

4. **Check metrics**:
   ```bash
   curl http://localhost:8000/api/pipeline/status
   ```

### Step 3: Verify Improvements

**Expected Results:**
- Queue depth stays <800 (was hitting 200)
- Total time: 35-45 minutes (was 96-101 minutes)
- Worker utilization: 50-60% (was 30-40%)
- GPU timeouts: <5% (was 10-20%)

---

## Configuration Summary

### Environment Variables

Add to `.env`:
```bash
# LLM concurrency limit (default: 4)
LLM_CONCURRENCY_LIMIT=4

# Summary workers (default: 3, adaptive 3-16)
SUMMARY_MAX_WORKERS=3
```

### Code Changes Summary

| File | Line | Change | Impact |
|------|------|--------|--------|
| `summary_pipeline.py` | 42 | `maxsize=1000` | High |
| `summary_pipeline.py` | 43 | `maxsize=1000` | High |
| `summary_pipeline.py` | 56 | Add semaphore | Medium |
| `summary_pipeline.py` | 71 | Wrap LLM call | Medium |
| `summary_pipeline.py` | 250 | Backpressure=800 | Low |
| `summarizer.py` | 95 | `CHUNK_SIZE=3500` | High |
| `summarizer.py` | 96 | `OVERLAP=275` | High |

---

## Rollback Plan

If issues occur, revert changes:

1. **Queue sizes**: Change back to 200/500
2. **Chunking**: Change back to 2000/400
3. **Semaphore**: Remove `with llm_semaphore:` wrapper
4. **Restart backend**

Or disable async pipeline entirely:
```bash
export USE_ASYNC_SUMMARIZATION=false
```

---

## Monitoring

After implementing, monitor:

1. **Queue depths**: Should stay <80% capacity
2. **Worker count**: Should adapt 3-16 based on load
3. **GPU timeouts**: Should be <5%
4. **Total time**: Should be 35-45 min for 385 chapters

Use diagnostic script:
```bash
python diagnose_pipeline_385.py
```

---

## Expected Timeline

- **Implementation**: 10-15 minutes
- **Testing**: 1-2 hours (wait for 385 chapters to process)
- **Validation**: Compare before/after metrics

**Total**: ~2-3 hours for complete validation

