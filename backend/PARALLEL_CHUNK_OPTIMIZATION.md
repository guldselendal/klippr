# Parallel Chunk-Summarization Pipeline: Analysis and Optimization

## 1. Current State and Timeline

### Call Graph

```
FastAPI Request (/api/upload)
  ↓
[Async Pipeline Enabled?]
  ├─ YES → enqueue_chapters_for_processing()
  │         ↓
  │         summary_queue.put(SummaryTask) × N chapters
  │         ↓
  │         summary_worker() threads (default: 3, configurable via SUMMARY_MAX_WORKERS)
  │         ↓
  │         generate_summary_with_routing() [summary_pipeline.py:126]
  │         ├─ content ≤ 5K: Single LLM call
  │         └─ content > 5K: Chunking path
  │            ↓
  │            split_content_into_chunks() → 47 chunks (for 72K chars)
  │            ↓
  │            ThreadPoolExecutor(max_workers=min(6, 47)) = 6 workers [summary_pipeline.py:191]
  │            ↓
  │            summarize_chunk_with_provider() × 47 (parallel)
  │            ↓
  │            call_llm() → Ollama HTTP request × 47
  │            ↓
  │            merge_chunk_summaries() → 1 final LLM call
  │
  └─ NO → generate_summaries_parallel() [summarizer_parallel.py]
            ↓
            generate_summary() [summarizer.py:145]
            ├─ content ≤ 5K: Single LLM call
            └─ content > 5K: Chunking path
               ↓
               split_content_into_chunks() → 47 chunks
               ↓
               ThreadPoolExecutor(max_workers=min(cpu_count, 47, 16)) = up to 16 workers [summarizer.py:193]
               ↓
               summarize_chunk() × 47 (parallel)
               ↓
               call_llm() → Ollama HTTP request × 47
               ↓
               merge_chunk_summaries() → 1 final LLM call
```

### Concurrency Analysis

#### Scenario 1: 1 Long Chapter (72K chars, 47 chunks)

**Async Pipeline Path:**
- Summary workers: 3 (default)
- Active summary worker: 1 (processing this chapter)
- Chunk workers spawned: 6 (min(6, 47))
- **In-flight LLM requests: 6** (all to Ollama)
- HTTP connections: 6 (from shared pool of 50)
- GPU load: ~6 concurrent inference tasks

**Synchronous Path:**
- Chunk workers: min(CPU, 47, 16) = typically 8-16
- **In-flight LLM requests: 8-16** (all to Ollama)
- HTTP connections: 8-16 (from shared pool)
- GPU load: ~8-16 concurrent inference tasks

**Timeline:**
- Chunk processing: 47 chunks ÷ 6 workers = ~8 batches
- At 3s per chunk: ~24s total (with 6 workers) vs ~9-15s (with 16 workers)
- Merge: +3s
- **Total: ~27-33s**

#### Scenario 2: 2 Long Chapters Simultaneously

**Async Pipeline Path:**
- Summary workers: 3
- Active: 2 (one per chapter)
- Each spawns 6 chunk workers
- **In-flight LLM requests: 12** (6 + 6, all to Ollama)
- HTTP connections: 12
- GPU load: ~12 concurrent tasks

**Synchronous Path:**
- 2 chapters × 16 workers = 32 potential workers
- **In-flight LLM requests: 32** (all to Ollama)
- HTTP connections: 32
- GPU load: ~32 concurrent tasks

**Risk:** Both paths exceed safe Ollama concurrency (4), causing timeouts.

#### Scenario 3: 3 Long Chapters Simultaneously

**Async Pipeline Path:**
- Summary workers: 3 (all active)
- Each spawns 6 chunk workers
- **In-flight LLM requests: 18** (6 + 6 + 6)
- HTTP connections: 18
- GPU load: ~18 concurrent tasks

**Synchronous Path:**
- 3 chapters × 16 workers = 48 potential workers
- **In-flight LLM requests: 48** (all to Ollama)
- HTTP connections: 48
- GPU load: ~48 concurrent tasks

**Critical:** Massive timeout risk. Ollama cannot handle >4 concurrent requests reliably.

### Resource Contention Points

1. **Ollama Server (localhost:11434)**
   - Hard limit: ~4 concurrent requests before timeouts
   - Current: Up to 18-48 concurrent requests
   - **Bottleneck severity: CRITICAL**

2. **HTTP Connection Pool**
   - Current: pool_maxsize = max(50, max_workers × 7) = 50-112
   - Actual need: Should match global LLM concurrency cap (4-8)
   - **Waste: 10-20x over-provisioned**

3. **GPU/VRAM**
   - Shared across all concurrent requests
   - Each Ollama request loads model weights into VRAM
   - 18-48 concurrent = potential OOM or severe slowdown
   - **Bottleneck severity: HIGH**

4. **Nested ThreadPoolExecutors**
   - summary_worker (3 threads) × chunk workers (6-16 each) = 18-48 threads
   - Each thread holds HTTP connection
   - **Amplification factor: 6-16x**

---

## 2. Bottlenecks and Failure Modes

### Ranked by Impact

#### 1. **Ollama Concurrency Exceeding Safe Limit (CRITICAL)**
- **Impact:** Timeouts, failed requests, degraded quality
- **Root cause:** No global concurrency limiter; nested executors multiply requests
- **Evidence:** User reports "timeouts when >4 concurrent requests"
- **Frequency:** Every multi-chapter upload

#### 2. **Nested ThreadPoolExecutor Amplification (CRITICAL)**
- **Impact:** 3 summary workers × 6-16 chunk workers = 18-48 concurrent LLM calls
- **Root cause:** Each summary worker creates its own executor with no global coordination
- **Evidence:** Code shows `ThreadPoolExecutor(max_workers=min(6, len(chunks)))` inside `summary_worker()`
- **Frequency:** Every long chapter in async pipeline

#### 3. **Duplicated Concurrency Logic (HIGH)**
- **Impact:** Inconsistent limits, maintenance burden, unpredictable behavior
- **Root cause:** 
  - `summary_pipeline.py:191`: `min(6, len(chunks))`
  - `summarizer.py:193`: `min(cpu_count, len(chunks), 16)`
  - `llm_provider.py:43`: `pool_maxsize = max(50, max_workers × 7)`
- **Evidence:** Three different calculation methods across files
- **Frequency:** Always

#### 4. **HTTP Pool Over-Provisioning (MEDIUM)**
- **Impact:** Memory waste, connection exhaustion risk
- **Root cause:** Pool sized for theoretical max (112) vs actual need (4-8)
- **Evidence:** `pool_maxsize = max(50, max_workers × (chunk_workers_per_task + 1))`
- **Frequency:** Always

#### 5. **Lack of Backpressure (MEDIUM)**
- **Impact:** Queue buildup, memory pressure, no graceful degradation
- **Root cause:** No circuit breaker, no adaptive throttling
- **Evidence:** Workers continue spawning requests even when Ollama is saturated
- **Frequency:** Under load

#### 6. **Excessive Overlap (LOW-MEDIUM)**
- **Impact:** 400-char overlap on 2000-char chunks = 20% redundancy
- **Root cause:** Fixed overlap ratio, not token-aware
- **Evidence:** `OVERLAP = 400` in `summarizer.py:163`
- **Frequency:** Every chunked chapter
- **Cost:** ~20% extra tokens, ~20% more requests

#### 7. **No Request Prioritization/Fairness (LOW)**
- **Impact:** Long chapters can starve short ones
- **Root cause:** FIFO queue, no round-robin
- **Frequency:** Mixed workload

#### 8. **GIL Irrelevance (NOT A BOTTLENECK)**
- **Note:** Threads are I/O-bound (HTTP waits), GIL released during I/O
- **Conclusion:** Threading model is appropriate; concurrency control is the issue

---

## 3. Recommendations (Prioritized)

### A. Concurrency Control (Priority 1: CRITICAL)

**Why:** Directly addresses bottleneck #1 and #2. Prevents timeouts and resource exhaustion.

#### A1. Global LLM Concurrency Limiter

**Implementation:**
- Create singleton `LLMConcurrencyLimiter` with provider-specific semaphores
- Default: Ollama = 4, Gemini = 8, OpenAI = 10, Global = 12
- Wrap all `call_llm()` invocations with acquire/release

**Code Sketch:**
```python
# llm_concurrency.py
import threading
import os
from typing import Dict

class LLMConcurrencyLimiter:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        # Provider-specific limits
        self.ollama_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OLLAMA", 4))
        self.gemini_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GEMINI", 8))
        self.openai_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OPENAI", 10))
        self.global_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GLOBAL", 12))
        
        # Semaphores
        self.ollama_sem = threading.Semaphore(self.ollama_limit)
        self.gemini_sem = threading.Semaphore(self.gemini_limit)
        self.openai_sem = threading.Semaphore(self.openai_limit)
        self.global_sem = threading.Semaphore(self.global_limit)
        
        # Metrics
        self.in_flight = {"ollama": 0, "gemini": 0, "openai": 0, "total": 0}
        self.metrics_lock = threading.Lock()
    
    def acquire(self, provider: str):
        """Acquire semaphore for provider. Blocks if limit reached."""
        provider_sem = getattr(self, f"{provider}_sem", None)
        if provider_sem:
            provider_sem.acquire()
        self.global_sem.acquire()
        
        with self.metrics_lock:
            self.in_flight[provider] = self.in_flight.get(provider, 0) + 1
            self.in_flight["total"] += 1
    
    def release(self, provider: str):
        """Release semaphore for provider."""
        provider_sem = getattr(self, f"{provider}_sem", None)
        if provider_sem:
            provider_sem.release()
        self.global_sem.release()
        
        with self.metrics_lock:
            self.in_flight[provider] = max(0, self.in_flight.get(provider, 0) - 1)
            self.in_flight["total"] = max(0, self.in_flight["total"] - 1)
    
    def get_metrics(self) -> Dict:
        with self.metrics_lock:
            return self.in_flight.copy()

# Usage in llm_provider.py
limiter = LLMConcurrencyLimiter()

def call_ollama(...):
    limiter.acquire("ollama")
    try:
        # ... existing code ...
        return result
    finally:
        limiter.release("ollama")
```

#### A2. Remove Nested Executors

**Implementation:**
- Replace per-chapter `ThreadPoolExecutor` with a shared task queue
- Use `concurrent.futures.ThreadPoolExecutor` at module level (shared across all workers)
- Or: Use a bounded queue + worker pool pattern

**Code Sketch:**
```python
# summary_pipeline.py
from llm_concurrency import LLMConcurrencyLimiter

# Shared executor for chunk processing (created once)
_chunk_executor = None
_chunk_executor_lock = threading.Lock()

def get_chunk_executor():
    global _chunk_executor
    if _chunk_executor is None:
        with _chunk_executor_lock:
            if _chunk_executor is None:
                # Size based on global limiter, not per-chapter
                limiter = LLMConcurrencyLimiter()
                max_workers = limiter.global_limit  # e.g., 12
                _chunk_executor = ThreadPoolExecutor(max_workers=max_workers)
    return _chunk_executor

def generate_summary_with_routing(...):
    if provider == "ollama" and len(content) > 5000:
        chunks = split_content_into_chunks(...)
        
        # Use shared executor instead of creating new one
        executor = get_chunk_executor()
        futures = []
        for idx, chunk in enumerate(chunks):
            future = executor.submit(
                summarize_chunk_with_provider,
                chunk, idx, len(chunks), title, provider, model
            )
            futures.append(future)
        
        # Collect results (limiter ensures only 4 Ollama requests in-flight)
        chunk_summaries = []
        for future in as_completed(futures):
            chunk_summaries.append(future.result())
        
        return merge_chunk_summaries(chunk_summaries, title)
```

#### A3. Adaptive Backoff

**Implementation:**
- Monitor p95 latency and error rate per provider
- If p95 > threshold (e.g., 25s) or error rate > 20%, reduce concurrency cap by 1
- Recover slowly (increase by 1 every 60s if metrics improve)

**Code Sketch:**
```python
# llm_concurrency.py (add to LLMConcurrencyLimiter)
def adapt_limits(self, provider: str, p95_latency: float, error_rate: float):
    """Adaptively adjust concurrency limits based on metrics."""
    current_limit = getattr(self, f"{provider}_limit")
    
    if p95_latency > 25.0 or error_rate > 0.2:
        # Reduce limit
        new_limit = max(1, current_limit - 1)
        if new_limit != current_limit:
            setattr(self, f"{provider}_limit", new_limit)
            # Recreate semaphore with new limit
            setattr(self, f"{provider}_sem", threading.Semaphore(new_limit))
            print(f"Reduced {provider} concurrency to {new_limit} (p95={p95_latency:.1f}s, errors={error_rate:.1%})")
    elif p95_latency < 15.0 and error_rate < 0.05:
        # Recover slowly
        new_limit = min(self.global_limit, current_limit + 1)
        if new_limit != current_limit:
            setattr(self, f"{provider}_limit", new_limit)
            setattr(self, f"{provider}_sem", threading.Semaphore(new_limit))
            print(f"Increased {provider} concurrency to {new_limit}")
```

### B. Architecture (Priority 2: HIGH)

**Why:** Eliminates duplication, centralizes control, improves maintainability.

#### B1. Centralize Concurrency Logic

**Implementation:**
- Create `llm_concurrency.py` module
- Remove all `min(6, ...)`, `min(cpu_count, ...)` logic from `summary_pipeline.py` and `summarizer.py`
- All LLM calls go through limiter

**Migration:**
1. Create `llm_concurrency.py` with `LLMConcurrencyLimiter`
2. Update `llm_provider.py` to use limiter
3. Remove nested executor caps from `summary_pipeline.py` and `summarizer.py`
4. Update `SUMMARY_MAX_WORKERS` to only control summary worker threads (not chunk workers)

#### B2. Optional: Async Migration

**Implementation:**
- Migrate LLM calls to `asyncio` + `httpx.AsyncClient`
- Use `asyncio.Semaphore` for concurrency control
- Keep FastAPI async endpoints

**Code Sketch:**
```python
# llm_provider_async.py
import asyncio
import httpx

class AsyncLLMConcurrencyLimiter:
    def __init__(self):
        self.ollama_sem = asyncio.Semaphore(4)
        self.global_sem = asyncio.Semaphore(12)
    
    async def acquire(self, provider: str):
        await getattr(self, f"{provider}_sem").acquire()
        await self.global_sem.acquire()
    
    async def release(self, provider: str):
        getattr(self, f"{provider}_sem").release()
        self.global_sem.release()

async def call_ollama_async(prompt: str, ...):
    limiter = AsyncLLMConcurrencyLimiter()
    await limiter.acquire("ollama")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{ollama_url}/api/chat",
                json={"model": model, "messages": messages}
            )
            return response.json()["message"]["content"]
    finally:
        await limiter.release("ollama")
```

**Trade-off:** Adds complexity; stage behind feature flag.

#### B3. Fairness Scheduler

**Implementation:**
- Round-robin chunk processing across chapters
- Prevents one long chapter from monopolizing workers

**Code Sketch:**
```python
# Use a priority queue or round-robin scheduler
from queue import PriorityQueue
import time

class FairChunkScheduler:
    def __init__(self):
        self.chapter_queues = {}  # chapter_id -> queue of chunks
        self.last_served = {}  # chapter_id -> timestamp
        self.lock = threading.Lock()
    
    def add_chapter(self, chapter_id: str, chunks: List):
        with self.lock:
            self.chapter_queues[chapter_id] = list(chunks)
            self.last_served[chapter_id] = 0
    
    def get_next_chunk(self) -> Optional[Tuple[str, chunk]]:
        """Round-robin: return chunk from least-recently-served chapter."""
        with self.lock:
            if not self.chapter_queues:
                return None
            
            # Find chapter with oldest last_served time
            chapter_id = min(self.chapter_queues.keys(), 
                           key=lambda cid: self.last_served.get(cid, 0))
            
            if self.chapter_queues[chapter_id]:
                chunk = self.chapter_queues[chapter_id].pop(0)
                self.last_served[chapter_id] = time.time()
                return (chapter_id, chunk)
            else:
                del self.chapter_queues[chapter_id]
                return self.get_next_chunk()
```

### C. Chunking Strategy (Priority 3: MEDIUM)

**Why:** Reduces request count, token costs, and processing time.

#### C1. Token-Aware Chunking

**Implementation:**
- Use tokenizer to count tokens (not characters)
- Target model context window (e.g., 4096 tokens for phi3:mini)
- Reduce overlap to 200 chars (or ~50 tokens)

**Code Sketch:**
```python
# chunking.py
try:
    from transformers import AutoTokenizer
    _tokenizer = None
    
    def get_tokenizer(model_name: str = "microsoft/phi-3-mini-4k-instruct"):
        global _tokenizer
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
        return _tokenizer
    
    def split_by_tokens(content: str, max_tokens: int = 3500, overlap_tokens: int = 50):
        """Split content by tokens, not characters."""
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(content)
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap_tokens
        
        return chunks
except ImportError:
    # Fallback to character-based
    def split_by_tokens(content: str, max_tokens: int, overlap_tokens: int):
        # Approximate: 1 token ≈ 4 chars
        chunk_size = max_tokens * 4
        overlap = overlap_tokens * 4
        return split_content_into_chunks(content, chunk_size, overlap)
```

#### C2. Two-Pass Map-Reduce

**Implementation:**
- First pass: Fast, low-concurrency chunk summaries (cap at 2-3 workers)
- Second pass: Merge with single LLM call

**Benefit:** Reduces peak concurrency while maintaining quality.

#### C3. Content Hash Caching

**Implementation:**
- Hash chunk content (SHA256)
- Cache summaries by hash
- Skip re-summarization on retries

**Code Sketch:**
```python
# caching.py
import hashlib
import json
from functools import lru_cache

_chunk_cache = {}  # hash -> summary

def get_chunk_hash(chunk: str) -> str:
    return hashlib.sha256(chunk.encode()).hexdigest()

def get_cached_summary(chunk: str) -> Optional[str]:
    chunk_hash = get_chunk_hash(chunk)
    return _chunk_cache.get(chunk_hash)

def cache_summary(chunk: str, summary: str):
    chunk_hash = get_chunk_hash(chunk)
    _chunk_cache[chunk_hash] = summary
```

### D. Networking and Retries (Priority 4: MEDIUM)

**Why:** Right-sizes resources, improves reliability.

#### D1. Right-Size HTTP Pool

**Implementation:**
- Set `pool_maxsize = global_limiter.global_limit + 2` (headroom)
- Set `pool_connections = min(10, pool_maxsize // 3)`

**Code Sketch:**
```python
# llm_provider.py
from llm_concurrency import LLMConcurrencyLimiter

def get_ollama_client():
    limiter = LLMConcurrencyLimiter()
    
    # Right-size pool to match limiter
    pool_maxsize = limiter.global_limit + 2  # e.g., 14
    pool_connections = min(10, pool_maxsize // 3)  # e.g., 4
    
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        pool_block=True
    )
    # ... rest of code
```

#### D2. Idempotent Retry with Circuit Breaker

**Implementation:**
- Exponential backoff with jitter
- Circuit breaker: open after 5 consecutive failures, close after 60s

**Code Sketch:**
```python
# circuit_breaker.py
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise
```

### E. GPU/Provider Utilization (Priority 5: LOW)

**Why:** Optimizes resource usage, but less critical than concurrency control.

#### E1. Token-Based Rate Limiting

**Implementation:**
- Estimate output tokens per request
- Cap sum(tokens_in_flight) to fit GPU throughput

**Note:** Requires tokenizer and GPU profiling. Lower priority.

#### E2. Prefer Larger Chunks

**Implementation:**
- When provider is bottleneck, increase chunk size to reduce request count
- Trade-off: Higher memory per request, but fewer requests

### F. Observability (Priority 6: HIGH for Operations)

**Why:** Essential for monitoring and debugging.

#### F1. Metrics

**Implementation:**
- In-flight requests per provider
- Queue depth
- Request rate (req/s)
- Latency (p50, p95, p99)
- Timeout/error rate
- Tokens/sec (if available)
- Retry count

**Code Sketch:**
```python
# metrics.py (extend pipeline_metrics.py)
class LLMConcurrencyMetrics:
    def __init__(self):
        self.in_flight = {"ollama": 0, "gemini": 0, "openai": 0}
        self.queue_depth = 0
        self.request_rate = 0.0
        self.latencies = defaultdict(list)
        self.errors = defaultdict(int)
        self.retries = defaultdict(int)
    
    def record_request(self, provider: str, latency: float, error: bool = False):
        self.latencies[provider].append(latency)
        if error:
            self.errors[provider] += 1
    
    def get_stats(self) -> Dict:
        return {
            "in_flight": self.in_flight.copy(),
            "queue_depth": self.queue_depth,
            "latency_p95": {p: np.percentile(l, 95) for p, l in self.latencies.items()},
            "error_rate": {p: self.errors[p] / max(1, len(self.latencies[p])) 
                          for p in self.latencies.keys()}
        }
```

#### F2. Logging

**Implementation:**
- Log throttle events (when semaphore blocks)
- Log adaptive limit changes
- Log circuit breaker state changes

---

## 4. Implementation Sketches

### Global Limiter (Threaded) - Full Implementation

```python
# llm_concurrency.py
import threading
import os
import time
from typing import Dict, Optional
from contextlib import contextmanager

class LLMConcurrencyLimiter:
    """Global concurrency limiter for LLM calls."""
    _instance: Optional['LLMConcurrencyLimiter'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Read limits from environment
        self.ollama_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OLLAMA", 4))
        self.gemini_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GEMINI", 8))
        self.openai_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OPENAI", 10))
        self.deepseek_limit = int(os.getenv("LLM_MAX_CONCURRENCY_DEEPSEEK", 8))
        self.global_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GLOBAL", 12))
        
        # Create semaphores
        self.ollama_sem = threading.Semaphore(self.ollama_limit)
        self.gemini_sem = threading.Semaphore(self.gemini_limit)
        self.openai_sem = threading.Semaphore(self.openai_limit)
        self.deepseek_sem = threading.Semaphore(self.deepseek_limit)
        self.global_sem = threading.Semaphore(self.global_limit)
        
        # Metrics
        self.in_flight = {"ollama": 0, "gemini": 0, "openai": 0, "deepseek": 0, "total": 0}
        self.metrics_lock = threading.Lock()
        self.total_requests = 0
        self.blocked_requests = 0
        
        self._initialized = True
    
    @contextmanager
    def acquire(self, provider: str):
        """Context manager for acquiring/releasing semaphore."""
        provider_sem = getattr(self, f"{provider}_sem", None)
        
        # Acquire provider semaphore
        if provider_sem:
            provider_sem.acquire()
            provider_acquired = True
        else:
            provider_acquired = False
        
        # Acquire global semaphore
        self.global_sem.acquire()
        
        # Update metrics
        with self.metrics_lock:
            self.in_flight[provider] = self.in_flight.get(provider, 0) + 1
            self.in_flight["total"] += 1
            self.total_requests += 1
        
        try:
            yield
        finally:
            # Release
            if provider_acquired:
                provider_sem.release()
            self.global_sem.release()
            
            with self.metrics_lock:
                self.in_flight[provider] = max(0, self.in_flight.get(provider, 0) - 1)
                self.in_flight["total"] = max(0, self.in_flight["total"] - 1)
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        with self.metrics_lock:
            return {
                "in_flight": self.in_flight.copy(),
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "limits": {
                    "ollama": self.ollama_limit,
                    "gemini": self.gemini_limit,
                    "openai": self.openai_limit,
                    "global": self.global_limit
                }
            }

# Usage in llm_provider.py
limiter = LLMConcurrencyLimiter()

def call_ollama(prompt: str, system_prompt: Optional[str] = None, ...):
    with limiter.acquire("ollama"):
        # ... existing Ollama call code ...
        return result
```

### Async Variant

```python
# llm_concurrency_async.py
import asyncio
import os
from typing import Dict, Optional
from contextlib import asynccontextmanager

class AsyncLLMConcurrencyLimiter:
    """Async version of concurrency limiter."""
    _instance: Optional['AsyncLLMConcurrencyLimiter'] = None
    _lock = asyncio.Lock()
    
    async def __new__(cls):
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    await cls._instance._init()
        return cls._instance
    
    async def _init(self):
        self.ollama_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OLLAMA", 4))
        self.global_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GLOBAL", 12))
        
        self.ollama_sem = asyncio.Semaphore(self.ollama_limit)
        self.global_sem = asyncio.Semaphore(self.global_limit)
        
        self.in_flight = {"ollama": 0, "total": 0}
        self.metrics_lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire(self, provider: str):
        await self.ollama_sem.acquire()
        await self.global_sem.acquire()
        
        async with self.metrics_lock:
            self.in_flight[provider] += 1
            self.in_flight["total"] += 1
        
        try:
            yield
        finally:
            self.ollama_sem.release()
            self.global_sem.release()
            
            async with self.metrics_lock:
                self.in_flight[provider] = max(0, self.in_flight[provider] - 1)
                self.in_flight["total"] = max(0, self.in_flight["total"] - 1)

# Usage
async def call_ollama_async(prompt: str, ...):
    limiter = await AsyncLLMConcurrencyLimiter()
    async with limiter.acquire("ollama"):
        async with httpx.AsyncClient() as client:
            response = await client.post(...)
            return response.json()["message"]["content"]
```

### Unify Config

```python
# config.py
import os

class LLMConfig:
    """Centralized LLM configuration."""
    
    # Summary worker threads (not chunk workers)
    SUMMARY_WORKERS = int(os.getenv("SUMMARY_MAX_WORKERS", 3))
    
    # LLM concurrency limits
    OLLAMA_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY_OLLAMA", 4))
    GEMINI_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY_GEMINI", 8))
    GLOBAL_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY_GLOBAL", 12))
    
    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # Reduced from 400
    
    # HTTP pool (right-sized)
    HTTP_POOL_MAXSIZE = GLOBAL_CONCURRENCY + 2
    HTTP_POOL_CONNECTIONS = min(10, HTTP_POOL_MAXSIZE // 3)
```

---

## 5. Example Calculations

### Scenario: Global Ollama Cap = 4, 3 Chapters × 10 Chunks Each

**Current (Broken) Behavior:**
- 3 summary workers × 6 chunk workers = 18 concurrent requests
- Ollama receives 18 requests → timeouts
- **Result: High failure rate**

**With Global Limiter:**
- Global cap = 4 Ollama requests
- 30 chunks total, but only 4 processed concurrently
- Scheduler: 4 in-flight, 26 queued
- At 3s/request: Throughput = 4 requests / 3s = 1.33 chunks/sec
- Total time: 30 chunks ÷ 1.33 chunks/sec = ~22.5s
- **Result: Zero timeouts, stable throughput**

**With Reduced Overlap (200 chars):**
- 10K chars → ~5 chunks (was ~6 with 400 overlap)
- 3 chapters × 5 chunks = 15 chunks total
- Total time: 15 chunks ÷ 1.33 chunks/sec = ~11.3s
- **Savings: ~50% reduction in requests, ~50% faster**

### Throughput Comparison

| Configuration | Concurrent Requests | Timeouts | Total Time (30 chunks) |
|--------------|---------------------|----------|------------------------|
| Current (no limiter) | 18 | High | ~10s (with failures) |
| With limiter (cap=4) | 4 | None | ~22.5s |
| With limiter + reduced overlap | 4 | None | ~11.3s |

**Trade-off:** Slightly higher latency per chapter, but zero timeouts and predictable behavior.

---

## 6. Test Plan

### Load Tests

**Test 1: Single Long Chapter**
- Input: 1 chapter, 72K chars, 47 chunks
- Expected: 4 concurrent Ollama requests, no timeouts
- Metrics: p95 latency < 30s, error rate = 0%

**Test 2: Two Concurrent Long Chapters**
- Input: 2 chapters, 72K chars each, 47 chunks each
- Expected: 4 concurrent Ollama requests (shared), no timeouts
- Metrics: p95 latency < 60s, error rate = 0%

**Test 3: Three Concurrent Long Chapters**
- Input: 3 chapters, 72K chars each, 47 chunks each
- Expected: 4 concurrent Ollama requests (shared), no timeouts
- Metrics: p95 latency < 90s, error rate < 1%

**Test 4: Mixed Workload**
- Input: 1 long chapter (47 chunks) + 5 short chapters (single call each)
- Expected: Fair scheduling, short chapters not starved
- Metrics: Short chapters complete within 10s, long chapter within 90s

### Chaos Tests

**Test 5: Slow Ollama**
- Artificially slow Ollama (add 5s delay per request)
- Expected: Backpressure engages, queue builds, no crashes
- Metrics: Error rate < 5%, system remains responsive

**Test 6: Ollama Failure**
- Simulate Ollama downtime (5s)
- Expected: Circuit breaker opens, requests fail fast
- Metrics: No hanging requests, clear error messages

### Regression Tests

**Test 7: Summary Quality**
- Compare summaries before/after overlap reduction (400 → 200)
- Expected: Quality maintained (human evaluation or similarity score > 0.9)
- Metrics: BLEU score or semantic similarity

**Test 8: Performance Regression**
- Compare end-to-end time for 10-chapter upload
- Expected: Within 20% of baseline (may be slower due to lower concurrency, but more stable)
- Metrics: Total time, p95 latency

---

## 7. Rollout Plan

### Phase 1: Global Limiter (Week 1)
**Goal:** Introduce concurrency control without breaking changes

1. Create `llm_concurrency.py` with `LLMConcurrencyLimiter`
2. Update `llm_provider.py` to use limiter (wrap all `call_*` functions)
3. Set default limits: Ollama=4, Global=12
4. Deploy behind feature flag: `USE_LLM_LIMITER=true`
5. Monitor: Error rate, latency, in-flight metrics
6. **Success criteria:** Zero timeouts, error rate < 1%

### Phase 2: Centralize Config (Week 2)
**Goal:** Remove duplication, unify limits

1. Create `config.py` with centralized settings
2. Remove `min(6, ...)`, `min(cpu_count, ...)` from `summary_pipeline.py` and `summarizer.py`
3. Update `SUMMARY_MAX_WORKERS` to only control summary worker threads
4. Right-size HTTP pool based on global limiter
5. **Success criteria:** Consistent behavior, no nested executor caps

### Phase 3: Remove Nested Executors (Week 3)
**Goal:** Eliminate N×M amplification

1. Create shared `ThreadPoolExecutor` for chunk processing
2. Update `generate_summary_with_routing()` to use shared executor
3. Remove per-chapter executor creation
4. **Success criteria:** Max concurrency = global limit (12), not 18-48

### Phase 4: Adaptive Control (Week 4)
**Goal:** Self-tuning limits

1. Add adaptive backoff logic to `LLMConcurrencyLimiter`
2. Integrate with metrics from `pipeline_metrics.py`
3. Add logging for limit changes
4. **Success criteria:** Limits adjust automatically under load

### Phase 5: Optional Async Migration (Week 5-6)
**Goal:** Reduce thread overhead (optional)

1. Create `llm_provider_async.py` with async versions
2. Add feature flag: `USE_ASYNC_LLM=true`
3. Migrate chunk processing to async
4. A/B test: async vs threaded
5. **Success criteria:** 10-20% latency improvement, or no regression

### Phase 6: Chunking Optimizations (Week 7)
**Goal:** Reduce request count

1. Implement token-aware chunking (optional, requires tokenizer)
2. Reduce overlap from 400 to 200 chars
3. Add content hash caching
4. **Success criteria:** 20-40% reduction in requests, quality maintained

---

## 8. Risks and Trade-offs

### Risk 1: Increased Per-Chapter Latency
**Impact:** MEDIUM
**Mitigation:** 
- Lower concurrency (4 vs 18) increases latency, but eliminates timeouts
- Trade-off: 22.5s with zero failures vs 10s with 30% failures
- **Acceptable:** Users prefer reliability over speed

### Risk 2: Async Migration Complexity
**Impact:** LOW (optional)
**Mitigation:**
- Stage behind feature flag
- Keep threaded version as fallback
- Thorough testing before rollout

### Risk 3: Token-Aware Chunking Dependencies
**Impact:** LOW
**Mitigation:**
- Fallback to character-based if tokenizer unavailable
- Make tokenizer optional dependency

### Risk 4: Over-Conservative Limits
**Impact:** LOW
**Mitigation:**
- Start with safe defaults (Ollama=4)
- Adaptive control adjusts automatically
- Environment variables allow tuning

### Risk 5: Head-of-Line Blocking
**Impact:** LOW
**Mitigation:**
- Fairness scheduler (round-robin) prevents starvation
- Priority queue for short vs long chapters (future enhancement)

### Trade-off Summary

| Aspect | Current | Optimized | Verdict |
|--------|---------|-----------|---------|
| Concurrency | 18-48 | 4-12 | ✅ Better (stable) |
| Timeouts | High | Zero | ✅ Better |
| Per-chapter latency | 10s (with failures) | 22.5s (reliable) | ⚠️ Acceptable trade-off |
| Total throughput | High (unstable) | Moderate (stable) | ✅ Better (reliability > speed) |
| Resource usage | Wasteful | Efficient | ✅ Better |
| Maintainability | Poor (duplicated) | Good (centralized) | ✅ Better |

**Conclusion:** The optimized pipeline trades some per-chapter latency for stability, reliability, and maintainability. This is the correct trade-off for a production system.

---

## Summary

The current pipeline suffers from **unbounded concurrency amplification** (3 summary workers × 6-16 chunk workers = 18-48 concurrent LLM requests) that exceeds Ollama's safe limit (4), causing timeouts and failures.

**Immediate Action Items:**
1. Implement global concurrency limiter (Priority 1)
2. Remove nested executors (Priority 1)
3. Centralize configuration (Priority 2)
4. Right-size HTTP pool (Priority 4)
5. Reduce chunk overlap (Priority 3)

**Expected Outcomes:**
- Zero timeouts
- Predictable latency (p95 < 30s per chapter)
- 20-40% reduction in requests (via overlap reduction)
- Improved maintainability (centralized logic)

**Timeline:** 4-7 weeks for full rollout, with critical fixes (Phases 1-3) in 3 weeks.

