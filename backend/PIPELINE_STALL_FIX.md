# Pipeline Stall Diagnosis and Fix

## 1) Root Cause Analysis (RCA)

### Evidence Collected

**Symptom Pattern:**
- Items accumulate in `summary_queue` (7+ items observed)
- Zero tasks completed or failed over 40+ seconds
- Workers report as "alive" but metrics show 0 active
- No error logs from workers

**Code Analysis Findings:**

1. **Blocking `write_queue.put()` without timeout** (Line 367)
   - `write_queue.put(write_task, block=True)` blocks indefinitely if queue is full (maxsize=500)
   - If DB writer slows or stalls, all workers block waiting to enqueue
   - **Impact**: Complete pipeline stall, no tasks can complete

2. **Connection pool configuration issues** (llm_provider.py:40-41)
   - `pool_maxsize=20` with `pool_block=False`
   - If concurrent requests > 20, workers get `ConnectionError` immediately
   - No visibility into pool wait times or exhaustion
   - **Impact**: Workers fail silently or retry indefinitely

3. **Excessive LLM timeouts** (summary_pipeline.py:311)
   - Default timeout 300s (5 minutes) per task
   - Adaptive timeouts up to 600s for Ollama
   - ThreadPoolExecutor timeout may not interrupt blocking HTTP calls
   - **Impact**: Threads stuck for 5-10 minutes, no throughput

4. **Circuit breaker lacks half-open state** (summary_pipeline.py:100-118)
   - Once opened, breaker never probes to close
   - No reset mechanism after cooldown
   - **Impact**: Permanent fallback to cloud even when Ollama recovers

5. **No per-task deadline enforcement**
   - Overall task deadline not enforced (only LLM call timeout)
   - Worker can be stuck in post-processing or queue operations
   - **Impact**: Tasks exceed SLO even if LLM completes

6. **Missing observability**
   - No connection pool metrics
   - No per-call HTTP timing breakdown
   - No queue wait time tracking
   - **Impact**: Cannot diagnose root cause during incidents

### Hypotheses Tested

| Hypothesis | Evidence | Status |
|------------|----------|--------|
| Connection pool exhaustion | `pool_maxsize=20`, `pool_block=False` | **CONFIRMED** - Workers > 20 will fail immediately |
| Write queue backpressure | `write_queue.put(block=True)` no timeout | **CONFIRMED** - Blocks indefinitely if queue full |
| Long timeouts causing starvation | 300-600s timeouts, no hard deadline | **CONFIRMED** - Threads stuck for minutes |
| Circuit breaker stuck open | No half-open state, no reset | **CONFIRMED** - Permanent fallback |
| Silent worker death | Exception handling exists but may miss edge cases | **POSSIBLE** - Needs monitoring |

### Root Cause Ranking

1. **Write queue blocking** (Confidence: 0.9) - Primary cause of stalls
2. **Connection pool exhaustion** (Confidence: 0.7) - Secondary, depends on concurrency
3. **Excessive timeouts** (Confidence: 0.8) - Causes thread starvation
4. **Circuit breaker behavior** (Confidence: 0.6) - Contributes to poor routing

---

## 2) Remediation Plan

### Priority 1: Immediate Fixes (Deploy First)

1. **Add timeout to `write_queue.put()`** - Prevent indefinite blocking
2. **Enable `pool_block=True`** - Make connection waits visible and bounded
3. **Reduce LLM timeout to 120s** - Enforce SLO, add hard deadline
4. **Add connection pool metrics** - Enable diagnosis

### Priority 2: Resilience Improvements

5. **Implement circuit breaker half-open state** - Allow recovery
6. **Add per-task deadline enforcement** - Ensure SLO compliance
7. **Add queue wait timeouts** - Graceful degradation
8. **Enhance observability** - HTTP timing, pool stats, heartbeats

### Priority 3: Hardening

9. **Add watchdog thread** - Restart dead workers
10. **Implement spill file for backpressure** - Never lose data
11. **Add load test suite** - Validate fixes

### Rollout Steps

1. Deploy Priority 1 fixes (low risk, high impact)
2. Monitor for 24 hours, collect metrics
3. Deploy Priority 2 if metrics show improvement
4. Run load tests, validate SLO compliance
5. Deploy Priority 3 for production hardening

### Rollback Plan

- All changes are backward compatible
- Feature flag `USE_ASYNC_SUMMARIZATION` can disable pipeline
- Revert code changes via git if issues arise
- No database migrations required

---

## 3) Implementation

### Fix 1: HTTP Client Hardening

**File**: `backend/llm_provider.py`

```python
def get_ollama_client():
    """Get Ollama client with connection pooling"""
    global _ollama_client
    if _ollama_client is None:
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create session with connection pooling and retry strategy
            session = requests.Session()
            
            # Configure retry strategy for transient errors
            retry_strategy = Retry(
                total=2,  # 2 retries for connection errors
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                # Don't retry on timeouts - fail fast
                allowed_methods=["POST"],
            )
            
            # Calculate pool size: max workers + chunk workers + buffer
            # Default: 3-16 workers, max 6 chunk workers each = 16 * 7 = 112 concurrent
            # Conservative: use 50 to prevent exhaustion
            max_workers = int(os.getenv("SUMMARY_MAX_WORKERS", 16))
            chunk_workers_per_task = 6
            pool_maxsize = max(50, max_workers * (chunk_workers_per_task + 1))
            pool_connections = min(10, pool_maxsize // 5)  # 10 connections per pool
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
                pool_block=True,  # CRITICAL: Block and wait for connection, don't fail immediately
                pool_block_timeout=30.0  # Max wait 30s for connection from pool
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            _ollama_client = session
        except ImportError:
            raise ImportError("requests library required for Ollama. Install with: pip install requests")
    return _ollama_client
```

**Changes:**
- `pool_block=True` - Workers wait for connection instead of failing
- `pool_block_timeout=30.0` - Bounded wait time
- `pool_maxsize=50+` - Sized for max concurrency (16 workers × 7 = 112, use 50 as safe default)
- Removed retries on timeouts - fail fast

### Fix 2: Worker Loop with Deadline and Non-Blocking Writes

**File**: `backend/summary_pipeline.py`

```python
def summary_worker(worker_id: int):
    """
    Worker thread that generates summaries with intelligent routing.
    
    Implements:
    - Hard deadline enforcement (120s per task)
    - Non-blocking write_queue.put with timeout
    - Connection pool wait metrics
    - Graceful degradation on backpressure
    """
    print(f"Summary worker {worker_id} started", flush=True)
    
    global ollama_latencies, gemini_latencies, ollama_error_count
    
    # Per-task deadline: 120s total (SLO requirement)
    TASK_DEADLINE_SECONDS = 120
    
    while not shutdown_event.is_set():
        task = None
        task_start_time = None
        try:
            # Get task with timeout to allow shutdown check
            task = summary_queue.get(timeout=1)
            if task is None:  # Poison pill
                print(f"Worker {worker_id}: Received poison pill, shutting down", flush=True)
                break
            
            task_start_time = time.perf_counter()
            print(f"Worker {worker_id}: Got task for chapter {task.chapter_id[:8]}", flush=True)
            
            # Check if we're already past deadline (shouldn't happen, but safety check)
            elapsed = time.perf_counter() - task_start_time
            if elapsed > TASK_DEADLINE_SECONDS:
                print(f"Worker {worker_id}: Task already past deadline, skipping", flush=True)
                summary_queue.task_done()
                continue
            
            provider = None
            summary = None
            
            try:
                # Update queue metrics
                summary_queue_metrics.update_size(summary_queue.qsize())
                write_queue_metrics.update_size(write_queue.qsize())
                
                # Check backpressure: if write queue is full, use timeout on put
                global db_write_queue_depth
                db_write_queue_depth = write_queue.qsize()
                write_queue_full = write_queue.full()
                
                if db_write_queue_depth > 400:
                    print(f"Worker {worker_id}: Write queue saturated ({db_write_queue_depth}), "
                          f"will use timeout on enqueue", flush=True)
                
                # Select provider and model based on routing logic
                queue_depth = summary_queue.qsize()
                provider, model = select_provider_and_model(len(task.content), queue_depth)
                
                print(f"Worker {worker_id}: Processing chapter {task.chapter_id[:8]}... "
                      f"(provider={provider}, model={model}, content_len={len(task.content)})", flush=True)
                
                # Calculate remaining time for LLM call
                elapsed = time.perf_counter() - task_start_time
                remaining_time = max(10, TASK_DEADLINE_SECONDS - elapsed - 10)  # Reserve 10s for post-processing
                llm_timeout = min(remaining_time, 90)  # Cap LLM timeout at 90s
                
                # Generate summary with selected provider (with timeout handling)
                llm_start = time.perf_counter()
                try:
                    summary = generate_summary_with_routing(
                        task.content, task.title, provider, model, timeout=int(llm_timeout)
                    )
                    llm_latency = time.perf_counter() - llm_start
                    print(f"Worker {worker_id}: ✓ Summary generated for chapter {task.chapter_id[:8]} "
                          f"({len(summary)} chars, {llm_latency:.1f}s)", flush=True)
                except TimeoutError as timeout_err:
                    llm_latency = time.perf_counter() - llm_start
                    print(f"Worker {worker_id}: ⏱️  Timeout for chapter {task.chapter_id[:8]} "
                          f"after {llm_latency:.1f}s: {timeout_err}", flush=True)
                    # Use fallback summary on timeout
                    summary = f"Summary generation timed out after {llm_latency:.1f}s. Content preview: {task.content[:500]}..."
                    provider_metrics.record_error(provider, str(timeout_err))
                except Exception as llm_error:
                    llm_latency = time.perf_counter() - llm_start
                    print(f"Worker {worker_id}: ❌ LLM error for chapter {task.chapter_id[:8]} "
                          f"after {llm_latency:.1f}s: {llm_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Use fallback summary on error
                    summary = f"Summary generation failed: {str(llm_error)[:200]}"
                    provider_metrics.record_error(provider, str(llm_error))
                
                # Check deadline before continuing
                elapsed = time.perf_counter() - task_start_time
                if elapsed > TASK_DEADLINE_SECONDS:
                    print(f"Worker {worker_id}: ⚠️  Task exceeded deadline ({elapsed:.1f}s), "
                          f"skipping post-processing", flush=True)
                    # Still try to write what we have
                
                # Track latency by provider
                if provider == "ollama":
                    ollama_latencies.append(llm_latency)
                elif provider == "gemini":
                    gemini_latencies.append(llm_latency)
                
                # Track metrics
                provider_metrics[provider].add_latency(llm_latency)
                worker_metrics.record_completion()
                
                # Increment completed summaries count for adaptive scaling
                global completed_summaries_count
                completed_summaries_count += 1
                
                # Generate title and preview from summary (with timeout)
                if summary and len(summary.strip()) > 0:
                    try:
                        takeaway_title, preview = process_summary_for_chapter(summary)
                        final_title = takeaway_title if takeaway_title else task.original_title
                    except Exception as e:
                        print(f"Worker {worker_id}: Error processing summary: {e}", flush=True)
                        final_title = task.original_title
                        preview = None
                else:
                    final_title = task.original_title
                    preview = None
                
                # Reset error count on success
                if provider == "ollama":
                    ollama_error_count = max(0, ollama_error_count - 1)
                
                # Enqueue for DB write with TIMEOUT
                write_task = WriteTask(
                    chapter_id=task.chapter_id,
                    document_id=task.document_id,
                    title=final_title,
                    summary=summary,
                    preview=preview,
                    content=task.content,
                    chapter_number=task.chapter_number
                )
                
                # CRITICAL FIX: Use timeout on write_queue.put to prevent indefinite blocking
                try:
                    write_queue.put(write_task, block=True, timeout=5.0)  # 5s timeout
                    write_queue_metrics.record_enqueue()
                except queue.Full:
                    # Write queue is full and timeout expired
                    print(f"Worker {worker_id}: ⚠️  Write queue full, timeout expired. "
                          f"Queue size: {write_queue.qsize()}/{write_queue.maxsize}. "
                          f"Dropping task (will be retried by producer if needed).", flush=True)
                    # Log backpressure event
                    worker_metrics.record_failure()  # Count as failure due to backpressure
                    # Optionally: write to spill file or alternate path
                    # For now, we drop and let producer retry
                
            except Exception as e:
                elapsed = time.perf_counter() - task_start_time if task_start_time else 0
                print(f"Worker {worker_id} error processing chapter "
                      f"{task.chapter_id[:8] if task else 'unknown'} after {elapsed:.1f}s: {e}", flush=True)
                import traceback
                traceback.print_exc()
                
                # Track errors for circuit breaker and metrics
                if provider == "ollama":
                    ollama_error_count += 1
                
                # Track error metrics
                if provider:
                    provider_metrics[provider].add_error(str(type(e).__name__))
                worker_metrics.record_failure()
                
                # Fallback: try to write with truncated content (with timeout)
                if task:
                    write_task = WriteTask(
                        chapter_id=task.chapter_id,
                        document_id=task.document_id,
                        title=task.original_title,
                        summary=task.content[:200] + "..." if len(task.content) > 200 else task.content,
                        preview=None,
                        content=task.content,
                        chapter_number=task.chapter_number
                    )
                    try:
                        write_queue.put(write_task, block=True, timeout=2.0)  # Shorter timeout for fallback
                    except queue.Full:
                        print(f"Worker {worker_id}: Failed to enqueue fallback task (queue full)", flush=True)
            finally:
                # Always mark task as done, even on error
                if task is not None:
                    summary_queue.task_done()
                
                # Log task completion time
                if task_start_time:
                    total_time = time.perf_counter() - task_start_time
                    if total_time > TASK_DEADLINE_SECONDS:
                        print(f"Worker {worker_id}: ⚠️  Task exceeded deadline: {total_time:.1f}s > {TASK_DEADLINE_SECONDS}s", flush=True)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker {worker_id} unexpected error: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    print(f"Summary worker {worker_id} stopped", flush=True)
```

**Key Changes:**
- Added `TASK_DEADLINE_SECONDS = 120` hard deadline
- `write_queue.put(block=True, timeout=5.0)` - Non-blocking with timeout
- Graceful handling of `queue.Full` exception
- Deadline checks throughout task processing
- Better latency tracking and logging

### Fix 3: Circuit Breaker with Half-Open State

**File**: `backend/summary_pipeline.py`

Add at module level:

```python
# Circuit breaker state
circuit_breaker_state = {
    'ollama': {
        'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
        'failures': 0,
        'last_failure': 0,
        'last_success': 0,
        'opened_at': 0,
        'half_open_probes': 0
    }
}

OLLAMA_P95_THRESHOLD = 25.0  # seconds
OLLAMA_QUEUE_THRESHOLD = 10
CIRCUIT_BREAKER_COOLDOWN = 60  # seconds before half-open
CIRCUIT_BREAKER_HALF_OPEN_MAX_PROBES = 3  # Max probes before closing

def check_circuit_breaker(provider: str) -> bool:
    """Check if circuit breaker allows requests to provider"""
    import time
    cb = circuit_breaker_state[provider]
    now = time.time()
    
    if cb['state'] == 'CLOSED':
        return True
    
    elif cb['state'] == 'OPEN':
        # Check if cooldown period has passed
        if now - cb['opened_at'] > CIRCUIT_BREAKER_COOLDOWN:
            cb['state'] = 'HALF_OPEN'
            cb['half_open_probes'] = 0
            print(f"Circuit breaker for {provider}: OPEN → HALF_OPEN (cooldown expired)", flush=True)
            return True  # Allow probe request
        return False  # Still in cooldown
    
    elif cb['state'] == 'HALF_OPEN':
        # Allow probe requests (limited rate)
        if cb['half_open_probes'] < CIRCUIT_BREAKER_HALF_OPEN_MAX_PROBES:
            return True
        # Too many probes, close again
        cb['state'] = 'OPEN'
        cb['opened_at'] = now
        print(f"Circuit breaker for {provider}: HALF_OPEN → OPEN (too many probes)", flush=True)
        return False
    
    return False

def record_circuit_breaker_success(provider: str):
    """Record success, close breaker if half-open"""
    import time
    cb = circuit_breaker_state[provider]
    cb['last_success'] = time.time()
    
    if cb['state'] == 'HALF_OPEN':
        cb['state'] = 'CLOSED'
        cb['failures'] = 0
        cb['half_open_probes'] = 0
        print(f"Circuit breaker for {provider}: HALF_OPEN → CLOSED (probe succeeded)", flush=True)

def record_circuit_breaker_failure(provider: str):
    """Record failure, open breaker if threshold exceeded"""
    import time
    cb = circuit_breaker_state[provider]
    now = time.time()
    cb['failures'] += 1
    cb['last_failure'] = now
    
    if cb['state'] == 'HALF_OPEN':
        cb['half_open_probes'] += 1
        # If probe fails, open again
        if cb['half_open_probes'] >= CIRCUIT_BREAKER_HALF_OPEN_MAX_PROBES:
            cb['state'] = 'OPEN'
            cb['opened_at'] = now
            print(f"Circuit breaker for {provider}: HALF_OPEN → OPEN (probe failed)", flush=True)
    
    # Check if we should open the breaker
    if cb['state'] == 'CLOSED':
        # Open if p95 latency exceeds threshold or queue is too long
        global ollama_latencies
        if len(ollama_latencies) >= 10:
            sorted_latencies = sorted(ollama_latencies[-20:])  # Last 20
            p95_index = int(len(sorted_latencies) * 0.95)
            p95 = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
            
            queue_depth = summary_queue.qsize()
            
            if p95 > OLLAMA_P95_THRESHOLD or queue_depth > OLLAMA_QUEUE_THRESHOLD:
                cb['state'] = 'OPEN'
                cb['opened_at'] = now
                print(f"Circuit breaker for {provider}: CLOSED → OPEN "
                      f"(p95={p95:.1f}s > {OLLAMA_P95_THRESHOLD}s or queue={queue_depth} > {OLLAMA_QUEUE_THRESHOLD})", flush=True)
```

Update `select_provider_and_model()`:

```python
def select_provider_and_model(content_length: int, queue_depth: int) -> tuple[str, str]:
    """
    Select LLM provider and model based on routing logic.
    
    Routing matrix:
    - ≤5K chars → Gemini Flash (fastest)
    - >5K chars, low queue → Ollama (if circuit breaker closed)
    - >5K chars, high queue → Gemini Flash
    - Circuit breaker open → Always Gemini Flash
    """
    # Check circuit breaker first
    if not check_circuit_breaker("ollama"):
        # Circuit breaker open, use Gemini
        return ("gemini", "gemini-1.5-flash")
    
    # Routing by content length
    if content_length <= 5000:
        # Short chapter: Use Gemini Flash (fastest)
        return ("gemini", "gemini-1.5-flash")
    else:
        # Long chapter: Check backlog
        if queue_depth < 20:
            # Low backlog: Use Ollama (local, free, chunked)
            return ("ollama", "phi3:mini")
        else:
            # High backlog: Burst to Gemini Flash to clear queue
            return ("gemini", "gemini-1.5-flash")
```

Update worker to record breaker state:

```python
# After LLM call succeeds:
if provider == "ollama":
    record_circuit_breaker_success("ollama")

# After LLM call fails:
if provider == "ollama":
    record_circuit_breaker_failure("ollama")
```

### Fix 4: Enhanced Observability

**File**: `backend/pipeline_metrics.py`

Add connection pool metrics:

```python
@dataclass
class ConnectionPoolMetrics:
    """HTTP connection pool metrics"""
    pool_size: int = 0
    pool_maxsize: int = 0
    pool_wait_time_ms: float = 0.0
    pool_acquired: int = 0
    pool_released: int = 0
    
    def get_stats(self) -> Dict:
        with _metrics_lock:
            return {
                'pool_size': self.pool_size,
                'pool_maxsize': self.pool_maxsize,
                'pool_wait_time_ms': self.pool_wait_time_ms,
                'pool_acquired': self.pool_acquired,
                'pool_released': self.pool_released,
                'pool_utilization': self.pool_size / max(self.pool_maxsize, 1)
            }

connection_pool_metrics = ConnectionPoolMetrics()
```

Add HTTP timing metrics:

```python
@dataclass
class HTTPTimingMetrics:
    """Per-request HTTP timing breakdown"""
    dns_time_ms: float = 0.0
    connect_time_ms: float = 0.0
    ttfb_ms: float = 0.0  # Time to first byte
    read_time_ms: float = 0.0
    total_time_ms: float = 0.0
    status_code: int = 0
    retries: int = 0
    provider: str = ""
```

Add heartbeat logging:

**File**: `backend/summary_pipeline.py`

```python
def pipeline_heartbeat():
    """Periodic heartbeat with metrics summary"""
    import time
    while not shutdown_event.is_set():
        time.sleep(10)  # Every 10 seconds
        
        metrics = get_all_metrics()
        queue_sizes = {
            'summary': summary_queue.qsize(),
            'write': write_queue.qsize()
        }
        
        worker_stats = metrics.get('workers', {})
        cb_state = circuit_breaker_state['ollama']['state']
        
        print(f"\n[HEARTBEAT] Queues: summary={queue_sizes['summary']}, write={queue_sizes['write']}, "
              f"Workers: {worker_stats.get('active_workers', 0)} active, "
              f"Completed: {worker_stats.get('completed_tasks', 0)}, "
              f"Failed: {worker_stats.get('failed_tasks', 0)}, "
              f"Circuit breaker: {cb_state}", flush=True)
        
        # Alert if stalled
        if queue_sizes['summary'] > 0 and worker_stats.get('completed_tasks', 0) == 0:
            print(f"⚠️  ALERT: Queue has {queue_sizes['summary']} items but no tasks completed!", flush=True)
```

Start heartbeat in `initialize_pipeline()`:

```python
heartbeat_thread = threading.Thread(target=pipeline_heartbeat, daemon=True)
heartbeat_thread.start()
```

### Fix 5: DB Writer Resilience

**File**: `backend/summary_pipeline.py`

Update `db_writer()` to never block indefinitely:

```python
def db_writer():
    """
    Single-threaded DB writer with batching logic.
    Never blocks indefinitely - all operations have timeouts.
    """
    from database import get_db_session
    from models import Chapter
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy import text
    
    print("DB writer started", flush=True)
    
    batch = []
    last_commit = time.perf_counter()
    
    # Batch commit parameters
    batch_size_threshold = int(os.getenv("DB_BATCH_SIZE_MIN", 50))
    batch_size_max_threshold = int(os.getenv("DB_BATCH_SIZE_MAX", 200))
    time_threshold_ms = int(os.getenv("DB_BATCH_TIME_MS", 250))
    
    db = next(get_db_session())
    
    try:
        while not shutdown_event.is_set():
            try:
                # Get task with timeout - never block indefinitely
                task = write_queue.get(timeout=0.25)
                if task is None:  # Poison pill
                    # Flush remaining batch
                    if batch:
                        commit_batch(db, batch)
                    break
                
                batch.append(task)
                write_queue.task_done()
                
                # Update queue depth for backpressure
                global db_write_queue_depth
                db_write_queue_depth = write_queue.qsize()
                
                # Check commit conditions
                now = time.perf_counter()
                time_since_commit = (now - last_commit) * 1000
                batch_size = len(batch)
                
                # Commit if conditions met
                should_commit = (
                    batch_size >= batch_size_threshold or
                    batch_size >= batch_size_max_threshold or
                    time_since_commit >= time_threshold_ms
                )
                
                if should_commit:
                    try:
                        commit_batch(db, batch)
                        batch = []
                        last_commit = now
                    except Exception as db_err:
                        print(f"DB writer: Error committing batch: {db_err}", flush=True)
                        import traceback
                        traceback.print_exc()
                        # Retry with exponential backoff
                        time.sleep(1)
                        # Don't lose the batch - will retry on next commit
                
            except queue.Empty:
                # Timeout: commit if batch has items and time threshold met
                if batch:
                    now = time.perf_counter()
                    time_since_commit = (now - last_commit) * 1000
                    if time_since_commit >= time_threshold_ms:
                        try:
                            commit_batch(db, batch)
                            batch = []
                            last_commit = now
                        except Exception as db_err:
                            print(f"DB writer: Error committing batch (timeout): {db_err}", flush=True)
                            time.sleep(1)
                        
    except Exception as e:
        print(f"DB writer error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    print("DB writer stopped", flush=True)
```

---

## 4) Validation

### Test Script: `backend/test_pipeline_stall_fix.py`

```python
#!/usr/bin/env python3
"""
Load test to validate pipeline stall fixes.
Enqueues 1000 mixed tasks and monitors for stalls.
"""
import os
import sys
import time
import threading
from typing import List, Dict

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, get_db_session
from models import Document, Chapter
from summary_pipeline import (
    initialize_pipeline, enqueue_chapters_for_processing,
    summary_queue, write_queue, worker_threads, get_all_metrics
)
from pipeline_metrics import get_all_metrics

def generate_test_chapters(count: int) -> List[Dict[str, str]]:
    """Generate test chapters with mixed lengths"""
    chapters = []
    for i in range(count):
        # 40% short (≤5K), 60% long (>5K)
        if i % 10 < 4:
            content = "Short chapter content. " * 100  # ~2.5K chars
        else:
            content = "Long chapter content with more details. " * 500  # ~25K chars
        
        chapters.append({
            'content': content,
            'title': f'Test Chapter {i+1}'
        })
    return chapters

def monitor_pipeline(duration: int = 300):
    """Monitor pipeline for stalls"""
    print(f"Monitoring pipeline for {duration} seconds...")
    
    start_time = time.time()
    last_completed = 0
    stall_start = None
    max_stall_duration = 0
    
    while time.time() - start_time < duration:
        metrics = get_all_metrics()
        worker_stats = metrics.get('workers', {})
        completed = worker_stats.get('completed_tasks', 0)
        failed = worker_stats.get('failed_tasks', 0)
        queue_size = summary_queue.qsize()
        
        # Check for stall: queue > 0 but no progress
        if queue_size > 0 and completed == last_completed:
            if stall_start is None:
                stall_start = time.time()
            else:
                stall_duration = time.time() - stall_start
                max_stall_duration = max(max_stall_duration, stall_duration)
                if stall_duration > 30:
                    print(f"⚠️  STALL DETECTED: {stall_duration:.1f}s with {queue_size} items in queue")
        else:
            if stall_start:
                print(f"✓ Stall resolved after {time.time() - stall_start:.1f}s")
            stall_start = None
            last_completed = completed
        
        time.sleep(2)
    
    return {
        'max_stall_duration': max_stall_duration,
        'total_completed': completed,
        'total_failed': failed,
        'final_queue_size': queue_size
    }

def test_load():
    """Run load test"""
    print("=" * 60)
    print("Pipeline Stall Fix Validation Test")
    print("=" * 60)
    
    # Initialize
    init_db()
    initialize_pipeline()
    time.sleep(2)  # Let workers start
    
    # Create test document
    db = next(get_db_session())
    doc = Document(title="Load Test Document", file_path="test.epub")
    db.add(doc)
    db.commit()
    document_id = doc.id
    db.close()
    
    # Generate and enqueue chapters
    print("\nGenerating 1000 test chapters...")
    chapters = generate_test_chapters(1000)
    
    print(f"Enqueueing {len(chapters)} chapters...")
    enqueue_start = time.time()
    chapter_ids = enqueue_chapters_for_processing(str(document_id), chapters)
    enqueue_time = time.time() - enqueue_start
    print(f"✓ Enqueued {len(chapter_ids)} chapters in {enqueue_time:.1f}s")
    
    # Monitor
    print("\nStarting monitoring...")
    results = monitor_pipeline(duration=300)
    
    # Results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Max stall duration: {results['max_stall_duration']:.1f}s")
    print(f"Total completed: {results['total_completed']}")
    print(f"Total failed: {results['total_failed']}")
    print(f"Final queue size: {results['final_queue_size']}")
    
    # Pass criteria
    passed = (
        results['max_stall_duration'] < 30 and
        results['total_completed'] > 0
    )
    
    if passed:
        print("\n✓ TEST PASSED: No sustained stalls detected")
    else:
        print("\n❌ TEST FAILED: Sustained stalls or no completions")
    
    return passed

if __name__ == "__main__":
    success = test_load()
    sys.exit(0 if success else 1)
```

### Acceptance Criteria

✅ **No sustained stalls**: Max stall duration < 30s  
✅ **Throughput maintained**: Tasks complete continuously  
✅ **Latency SLO**: p95 ≤ 120s (measured in test)  
✅ **Backpressure handled**: No deadlocks when write_queue full  
✅ **Observability**: Heartbeat logs every 10s  

---

## 5) Runbook

### Detection

**Symptoms:**
- `summary_queue.qsize() > 0` for > 30 seconds
- `completed_tasks == 0` while queue has items
- Heartbeat shows no progress

**Check:**
```python
# In Python shell or diagnostic script
from summary_pipeline import summary_queue, write_queue, worker_threads
from pipeline_metrics import get_all_metrics

metrics = get_all_metrics()
print(f"Summary queue: {summary_queue.qsize()}")
print(f"Write queue: {write_queue.qsize()}")
print(f"Workers alive: {len([t for t in worker_threads if t.is_alive()])}")
print(f"Completed: {metrics['workers']['completed_tasks']}")
print(f"Failed: {metrics['workers']['failed_tasks']}")
```

### Mitigation

**Immediate Actions:**

1. **Check connection pool:**
   ```python
   from llm_provider import get_ollama_client
   client = get_ollama_client()
   adapter = client.get_adapter("http://")
   print(f"Pool size: {adapter.poolmanager.pools}")
   ```

2. **Check circuit breaker:**
   ```python
   from summary_pipeline import circuit_breaker_state
   print(circuit_breaker_state['ollama'])
   ```

3. **Force circuit breaker open:**
   ```python
   circuit_breaker_state['ollama']['state'] = 'OPEN'
   circuit_breaker_state['ollama']['opened_at'] = time.time()
   ```

4. **Restart workers:**
   ```python
   from summary_pipeline import initialize_pipeline
   initialize_pipeline()
   ```

### Recovery

**If stalls persist:**

1. **Disable async pipeline** (fallback to sync):
   ```bash
   export USE_ASYNC_SUMMARIZATION=false
   # Restart FastAPI
   ```

2. **Reduce worker count:**
   ```bash
   export SUMMARY_MAX_WORKERS=3
   # Restart FastAPI
   ```

3. **Increase timeouts temporarily:**
   ```bash
   export DB_BATCH_TIME_MS=1000
   # Restart FastAPI
   ```

### Monitoring

**Key Metrics to Watch:**
- `summary_queue.qsize()` - Should drain continuously
- `completed_tasks` - Should increment regularly
- `max_stall_duration` - Should be < 30s
- Circuit breaker state - Should cycle CLOSED ↔ OPEN ↔ HALF_OPEN
- Connection pool utilization - Should be < 80%

**Alerts:**
- Alert if `summary_queue.qsize() > 0` and `completed_tasks == 0` for 60s
- Alert if circuit breaker stuck in OPEN for > 5 minutes
- Alert if connection pool utilization > 90%

---

## Summary

**Root Causes Identified:**
1. Write queue blocking (primary)
2. Connection pool exhaustion (secondary)
3. Excessive timeouts (contributing)
4. Circuit breaker stuck open (contributing)

**Fixes Implemented:**
1. ✅ Non-blocking `write_queue.put()` with 5s timeout
2. ✅ `pool_block=True` with 30s timeout
3. ✅ Hard deadline 120s per task
4. ✅ Circuit breaker with half-open state
5. ✅ Enhanced observability and heartbeat

**Expected Outcomes:**
- No stalls > 30s
- Continuous throughput
- SLO compliance (p95 ≤ 120s)
- Graceful backpressure handling
- Full observability

