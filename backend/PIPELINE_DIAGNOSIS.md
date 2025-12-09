# Pipeline Diagnosis: Zero Progress with Queued Items

## Section 1: Reasoning

### Symptom Mapping to Hypotheses

**Symptom 1: `Active workers: 0` (metrics) but `3 alive` (threads)**
- **H1.1**: Workers never call `worker_metrics.record_completion()` or `record_failure()` because they're stuck before reaching that code (confidence: 0.7)
- **H1.2**: Workers crash immediately after startup with unhandled exception, leaving threads in "alive" state but not executing (confidence: 0.2)
- **H1.3**: Metrics update only happens in adaptive manager, which requires 5+ latencies to run; workers stuck before generating any latencies (confidence: 0.1)

**Symptom 2: `Completed tasks: 0, Failed tasks: 0`**
- **H2.1**: Workers never process items because they're blocked in `queue.get()` or immediately after (confidence: 0.4)
- **H2.2**: Workers process items but crash before calling `worker_metrics.record_completion/failure()`, and exception handler also fails (confidence: 0.3)
- **H2.3**: Workers are stuck in LLM call that blocks indefinitely, even with timeout wrapper (confidence: 0.3)

**Symptom 3: `Summary queue: 7 items, Write queue: 0 items`**
- **H3.1**: Workers dequeue items but hang in LLM call before enqueueing to write_queue (confidence: 0.6)
- **H3.2**: Workers never dequeue items because `queue.get(timeout=1)` is somehow blocking forever (confidence: 0.2)
- **H3.3**: Workers dequeue and process, but exception occurs before `write_queue.put()`, and exception handler also fails (confidence: 0.2)

**Symptom 4: No worker log output ("Got task", "Processing chapter", etc.)**
- **H4.1**: Workers never reach the `print()` statements because they crash or hang before `queue.get()` returns (confidence: 0.5)
- **H4.2**: Logs are buffered and not flushed, or stdout is redirected/not visible (confidence: 0.2)
- **H4.3**: Workers are in a different process (uvicorn worker process) and logs go to different stream (confidence: 0.3)

## Section 2: Most Likely Root Causes (Ranked)

### 1. Pipeline Not Initialized in FastAPI Startup (Confidence: 0.8)
**Evidence**: 
- `initialize_pipeline()` is only called in `test_10_chapter_upload.py`, not in `main.py`
- No FastAPI lifespan context manager found
- If app runs via `uvicorn main:app`, pipeline never starts
- Workers would never exist in the serving process

**Impact**: Workers don't exist → queue fills but nothing processes

### 2. ThreadPoolExecutor Timeout Ineffective for Blocking I/O (Confidence: 0.7)
**Evidence**:
- `_call_llm_with_timeout()` uses `ThreadPoolExecutor` with `future.result(timeout=300)`
- Underlying `requests.post()` may block in C extension code (urllib3/ssl)
- Python's `ThreadPoolExecutor` timeout only works if the call is interruptible
- If `requests.post()` is in blocking syscall, timeout won't interrupt it

**Impact**: Workers hang indefinitely in LLM calls, never completing or failing

### 3. Workers Crash Silently Before Logging (Confidence: 0.5)
**Evidence**:
- Workers are daemon threads; exceptions may not be visible
- Exception handler at line 399 catches but only prints; if print fails or is buffered, no visibility
- If import fails or global variable access fails, worker dies immediately

**Impact**: Workers start, crash on first task, but no error visible

### 4. Missing `task_done()` in Exception Paths (Confidence: 0.3)
**Evidence**:
- `summary_queue.task_done()` only called at line 395, after successful processing
- If exception occurs before line 395, `task_done()` never called
- Queue may think items are still being processed, but workers are dead

**Impact**: Queue thinks items are in-flight, but workers are stuck/dead

## Section 3: Experiments to Confirm

### E1: Verify Pipeline Initialization
**Code to add**:
```python
# In main.py, add after line 57 (after init_db())
if USE_ASYNC_PIPELINE:
    from summary_pipeline import initialize_pipeline
    print("Initializing async pipeline...")
    initialize_pipeline()
    print("Async pipeline initialized")
```

**Expected outcomes**:
- **If H1 confirmed**: After adding, workers appear and start processing
- **If H1 falsified**: Workers still don't process (check E2)

### E2: Thread Dump and Queue Health
**Code to add** (create `backend/diagnose_pipeline.py`):
```python
import sys
import threading
import traceback
from summary_pipeline import summary_queue, write_queue, worker_threads

def dump_threads(tag="snapshot"):
    frames = sys._current_frames()
    print(f"\n=== Thread Dump ({tag}) ===")
    for t in threading.enumerate():
        f = frames.get(t.ident)
        print(f"\n- {t.name} (alive={t.is_alive()}, daemon={t.daemon})")
        if f:
            traceback.print_stack(f)

def qstats(name, q):
    try:
        return {
            "name": name,
            "qsize": q.qsize(),
            "empty": q.empty(),
            "full": q.full()
        }
    except NotImplementedError:
        return {"name": name, "qsize": "na"}

# Run diagnostics
dump_threads("diagnosis")
print("\n=== Queue Stats ===")
print(qstats("summary", summary_queue))
print(qstats("write", write_queue))
print(f"\nWorker threads: {len(worker_threads)}")
print(f"Alive workers: {len([t for t in worker_threads if t.is_alive()])}")
```

**Expected outcomes**:
- **If workers exist**: See 3+ worker threads with stack traces showing `queue.get()` or LLM call
- **If workers missing**: No worker threads in dump → confirms H1
- **If workers stuck**: Stack shows `requests.post()` or `future.result()` → confirms H2

### E3: Add Robust Worker Logging with Flush
**Code to modify** (in `summary_pipeline.py`, line 270):
```python
def summary_worker(worker_id: int):
    import sys
    sys.stdout.flush()  # Ensure output is visible
    print(f"Summary worker {worker_id} started", flush=True)
    
    global ollama_latencies, gemini_latencies, ollama_error_count
    
    iteration = 0
    while not shutdown_event.is_set():
        iteration += 1
        if iteration % 10 == 0:
            print(f"Worker {worker_id}: Still alive, iteration {iteration}, queue size: {summary_queue.qsize()}", flush=True)
        
        try:
            print(f"Worker {worker_id}: Attempting to get task (timeout=1)...", flush=True)
            task = summary_queue.get(timeout=1)
            print(f"Worker {worker_id}: Got task for chapter {task.chapter_id[:8]}", flush=True)
            # ... rest of code
```

**Expected outcomes**:
- **If workers processing**: See "Got task" messages every 1-2 seconds
- **If workers stuck**: See "Still alive" but no "Got task" → queue.get() blocking
- **If workers dead**: No output after "started" → confirms H3

### E4: Test LLM Call with Explicit Timeout
**Code to add** (temporary test in worker):
```python
# In summary_worker, before LLM call:
print(f"Worker {worker_id}: About to call LLM (provider={provider})", flush=True)
try:
    # Add signal-based timeout as backup
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("LLM call exceeded timeout")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60s hard timeout
    
    summary = generate_summary_with_routing(...)
    
    signal.alarm(0)  # Cancel alarm
except TimeoutError as e:
    print(f"Worker {worker_id}: Hard timeout triggered: {e}", flush=True)
    raise
```

**Expected outcomes**:
- **If ThreadPoolExecutor timeout works**: See normal timeout after 300s
- **If signal timeout needed**: See "Hard timeout triggered" after 60s → confirms H2

## Section 4: Minimal Fix (Immediate)

### Fix 1: Initialize Pipeline in FastAPI Startup
**File**: `backend/main.py`
**Location**: After line 57 (`init_db()`)

```python
# After init_db()
if USE_ASYNC_PIPELINE:
    try:
        from summary_pipeline import initialize_pipeline
        print("Initializing async summary pipeline...")
        initialize_pipeline()
        print("✓ Async pipeline initialized")
    except Exception as e:
        print(f"⚠️  Failed to initialize async pipeline: {e}")
        import traceback
        traceback.print_exc()
        USE_ASYNC_PIPELINE = False  # Fallback to sync mode
```

### Fix 2: Ensure task_done() Always Called
**File**: `backend/summary_pipeline.py`
**Location**: Modify worker loop to use try/finally

```python
# Replace lines 274-395 with:
while not shutdown_event.is_set():
    task = None
    try:
        task = summary_queue.get(timeout=1)
        if task is None:
            break
        
        print(f"Worker {worker_id}: Got task for chapter {task.chapter_id[:8]}", flush=True)
        
        # ... existing processing code ...
        
    except queue.Empty:
        continue
    except Exception as e:
        print(f"Worker {worker_id} error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if provider:
            provider_metrics[provider].add_error(str(type(e).__name__))
        worker_metrics.record_failure()
    finally:
        if task is not None:
            summary_queue.task_done()
```

### Fix 3: Add Signal-Based Timeout Backup
**File**: `backend/summary_pipeline.py`
**Location**: In `_call_llm_with_timeout()` function

```python
def _call_llm_with_timeout(prompt: str, system_prompt: str, provider: str, model: str, timeout: int = 300) -> str:
    """Wrapper to call LLM with timeout (ThreadPoolExecutor + signal backup)"""
    import signal
    import os
    
    # Signal-based timeout (Unix only)
    if hasattr(signal, 'SIGALRM'):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"LLM call exceeded {timeout}s timeout")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_llm, prompt=prompt, system_prompt=system_prompt, provider=provider, model=model)
            try:
                result = future.result(timeout=timeout).strip()
                return result
            except FutureTimeoutError:
                executor.shutdown(wait=False)
                raise TimeoutError(f"LLM call to {provider} exceeded {timeout}s timeout")
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
```

**Trade-offs**:
- Signal-based timeout only works on Unix (macOS/Linux), not Windows
- May interfere with other signal handlers
- ThreadPoolExecutor timeout should work for most cases; signal is backup

## Section 5: Hardening and Ops Guardrails

### 5.1: Health Check Endpoint
**File**: `backend/main.py`
**Add endpoint**:

```python
@app.get("/api/pipeline/health")
async def pipeline_health():
    """Health check for async pipeline"""
    if not USE_ASYNC_PIPELINE:
        return {"status": "disabled", "use_async": False}
    
    from summary_pipeline import (
        summary_queue, write_queue, worker_threads,
        get_all_metrics
    )
    from pipeline_metrics import get_all_metrics
    
    metrics = get_all_metrics()
    alive_workers = len([t for t in worker_threads if t.is_alive()])
    
    health = {
        "status": "healthy" if alive_workers > 0 else "unhealthy",
        "use_async": True,
        "workers": {
            "alive": alive_workers,
            "total": len(worker_threads),
            "active": metrics.get('workers', {}).get('active_workers', 0)
        },
        "queues": {
            "summary": {
                "size": summary_queue.qsize(),
                "maxsize": summary_queue.maxsize
            },
            "write": {
                "size": write_queue.qsize(),
                "maxsize": write_queue.maxsize
            }
        },
        "metrics": metrics
    }
    
    # Determine health status
    if alive_workers == 0 and summary_queue.qsize() > 0:
        health["status"] = "critical"
        health["issue"] = "No workers alive but items in queue"
    elif alive_workers == 0:
        health["status"] = "unhealthy"
        health["issue"] = "No workers alive"
    
    return health
```

### 5.2: Watchdog Thread
**File**: `backend/summary_pipeline.py`
**Add function**:

```python
def pipeline_watchdog():
    """Watchdog that monitors pipeline health and restarts workers if needed"""
    import time
    while not shutdown_event.is_set():
        time.sleep(30)  # Check every 30s
        
        alive_workers = len([t for t in worker_threads if t.is_alive()])
        queue_size = summary_queue.qsize()
        
        # If no workers but items in queue, restart
        if alive_workers == 0 and queue_size > 0:
            print(f"⚠️  WATCHDOG: No workers alive but {queue_size} items in queue. Restarting...")
            try:
                # Clean up dead threads
                global worker_threads
                worker_threads = [t for t in worker_threads if t.is_alive()]
                
                # Start new workers
                initial_workers = int(os.getenv("SUMMARY_MAX_WORKERS", 3))
                for i in range(initial_workers):
                    t = threading.Thread(target=summary_worker, args=(i,), daemon=True)
                    t.start()
                    worker_threads.append(t)
                
                print(f"✓ WATCHDOG: Restarted {initial_workers} workers")
            except Exception as e:
                print(f"❌ WATCHDOG: Failed to restart workers: {e}")
                import traceback
                traceback.print_exc()
```

**Start in `initialize_pipeline()`**:
```python
# After starting adaptive manager
watchdog_thread = threading.Thread(target=pipeline_watchdog, daemon=True)
watchdog_thread.start()
print("Pipeline watchdog thread started")
```

### 5.3: Circuit Breaker for LLM Calls
**File**: `backend/summary_pipeline.py`
**Add at module level**:

```python
# Circuit breaker state
llm_circuit_breaker = {
    'ollama': {'failures': 0, 'last_failure': 0, 'open': False},
    'gemini': {'failures': 0, 'last_failure': 0, 'open': False}
}

def check_circuit_breaker(provider: str) -> bool:
    """Check if circuit breaker is open for provider"""
    import time
    cb = llm_circuit_breaker[provider]
    
    # Reset after 60s
    if time.time() - cb['last_failure'] > 60:
        cb['failures'] = 0
        cb['open'] = False
    
    # Open after 3 consecutive failures
    if cb['failures'] >= 3:
        cb['open'] = True
        return False
    
    return True

def record_circuit_breaker_failure(provider: str):
    """Record failure for circuit breaker"""
    import time
    cb = llm_circuit_breaker[provider]
    cb['failures'] += 1
    cb['last_failure'] = time.time()
    if cb['failures'] >= 3:
        cb['open'] = True
        print(f"⚠️  Circuit breaker OPEN for {provider} (3 failures)")
```

**Use in worker**:
```python
# Before LLM call
if not check_circuit_breaker(provider):
    print(f"Worker {worker_id}: Circuit breaker open for {provider}, skipping")
    # Fallback to other provider or skip
    continue

try:
    summary = generate_summary_with_routing(...)
except Exception as e:
    record_circuit_breaker_failure(provider)
    raise
```

### 5.4: Structured Logging
**Replace print() with logging**:
```python
import logging
logger = logging.getLogger("summary_pipeline")
logger.setLevel(logging.INFO)

# In worker:
logger.info(f"Worker {worker_id} started")
logger.debug(f"Worker {worker_id}: Got task", extra={"chapter_id": task.chapter_id[:8]})
logger.error(f"Worker {worker_id} error", exc_info=True)
```

## Section 6: Follow-up Questions

1. **How is the FastAPI app started?**
   - Command: `uvicorn main:app --workers 1 --reload false`?
   - Or `python main.py`?
   - Or via Docker/gunicorn?

2. **Are there any logs from worker startup?**
   - Should see "Summary worker X started" messages
   - If missing, workers never started

3. **Is Ollama/Gemini accessible?**
   - Can you `curl http://localhost:11434/api/tags`?
   - Is `GEMINI_API_KEY` set if using Gemini?

4. **What Python version and OS?**
   - Signal-based timeout requires Unix
   - Thread behavior may differ by version

5. **Are there any import errors at startup?**
   - Check for `ImportError` or `ModuleNotFoundError` when importing `summary_pipeline`

## Immediate Action Plan

1. **Add Fix 1** (pipeline initialization in FastAPI) - **HIGHEST PRIORITY**
2. **Run E2** (thread dump) to see current state
3. **Add Fix 2** (ensure task_done() always called)
4. **Add health endpoint** for monitoring
5. **Run test again** and observe logs

The most likely issue is **Fix 1**: pipeline not initialized in FastAPI startup. This would explain all symptoms perfectly.

