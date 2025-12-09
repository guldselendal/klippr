# Ollama Timeout Diagnosis and Fix

## Reasoning

- **Timeout value**: 120s read timeout is insufficient when 8 parallel workers compete for GPU/VRAM, causing model reloads and queueing delays
- **Chunk sizing**: 3000 chars ≈ 750-850 tokens + prompt overhead (~150 tokens) ≈ 900-1000 tokens per chunk; under load, generation can exceed 120s
- **Concurrency**: Default 8 workers × 4 chunks per chapter = 32 concurrent requests; Ollama may queue/reload model, increasing latency
- **No retries**: Timeouts fail immediately without retry, losing work
- **Token throughput**: phi3:mini on M3 GPU ~20-40 tokens/sec; 2000 token max × 8 parallel = potential 50-100s per chunk under contention
- **Evidence**: Timeouts occur on chunks 5 and 7 during parallel processing, suggesting resource contention rather than individual chunk size

## Findings

### Chunk Sizing
- **Current**: 3000 chars/chunk, 500 char overlap
- **Token estimate**: ~900-1000 tokens/chunk (content + prompt)
- **Model context**: phi3:mini typically 4K context; chunks are well within limits
- **Issue**: Chunk size is reasonable, but prompt overhead and parallel load cause cumulative delays

### Concurrency
- **Current**: Default 8 workers, max 16
- **Observed**: 34 summaries in progress (multiple chapters × chunks)
- **Issue**: High parallelism saturates Ollama, causing model reloads and queueing

### Timeouts
- **Current**: 120s read timeout, 5s connection timeout
- **Observed failures**: Chunks timing out during parallel processing
- **Issue**: Timeout too short for chunks under load; no retry logic

### Resource Use
- **Model**: phi3:mini (estimated 2-4GB VRAM)
- **Parallel requests**: 8-16 workers × model context = potential VRAM pressure
- **Issue**: Model may be evicted/reloaded under high concurrency, causing timeouts

## Root-Cause Hypotheses (Ranked)

### 1. **Client Timeout Too Low + High Concurrency** (High Confidence)
- **Evidence**: 120s timeout; 8+ parallel workers; timeouts occur on chunks 5-7 (mid-batch, suggesting cumulative load)
- **Mechanism**: Parallel requests queue in Ollama; model reloads under VRAM pressure; individual requests exceed 120s
- **Confidence**: High

### 2. **No Retry Logic for Transient Failures** (High Confidence)
- **Evidence**: Timeout exceptions immediately fail; no retry mechanism
- **Mechanism**: Transient model reloads or queue delays cause timeouts; retry would succeed
- **Confidence**: High

### 3. **Over-Parallelization Causing Model Reloads** (Medium-High Confidence)
- **Evidence**: 8+ workers default; timeouts correlate with parallel load
- **Mechanism**: Too many concurrent requests exceed VRAM; Ollama evicts/reloads model; reload latency causes timeouts
- **Confidence**: Medium-High

### 4. **Chunk Size + Prompt Overhead** (Low-Medium Confidence)
- **Evidence**: 3000 char chunks + prompt ≈ 1000 tokens; generation can take 50-100s under load
- **Mechanism**: Large chunks take longer; combined with queueing delays, exceed timeout
- **Confidence**: Low-Medium (chunks are reasonable size)

## Quick Mitigations (Apply Now)

### 1. Increase Timeout and Add Retry Logic

**File**: `backend/llm_provider.py`

```python
def call_ollama(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, max_retries: int = 3) -> str:
    """Call Ollama API with retry logic"""
    import requests
    import time
    import random
    
    requests_lib = get_ollama_client()
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "phi3:mini")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    options = {
        "num_predict": 2000,
        "num_thread": 0,
    }
    
    # Estimate tokens for timeout calculation
    # Rough estimate: ~4 chars per token, add 20% for prompt overhead
    estimated_tokens = len(prompt) / 4 * 1.2
    # Base timeout: 2s per token + 30s overhead, min 180s, max 600s
    base_timeout = max(180, min(600, int(estimated_tokens * 2 + 30)))
    
    for attempt in range(max_retries):
        try:
            # Exponential backoff with jitter
            if attempt > 0:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                print(f"  Retrying Ollama request (attempt {attempt + 1}/{max_retries})...")
            
            response = requests_lib.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": options
                },
                timeout=(5, base_timeout)  # Increased timeout with dynamic calculation
            )
            response.raise_for_status()
            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                raise Exception("Invalid response format from Ollama")
            return result["message"]["content"].strip()
            
        except requests.exceptions.Timeout as e:
            if attempt == max_retries - 1:
                raise Exception(f"Ollama timeout: Request took too long after {max_retries} attempts")
            # Retry on timeout
            continue
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                raise Exception(f"Ollama connection error: Cannot connect to {ollama_url}. Is Ollama running?")
            # Retry on connection error
            continue
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 404:
                error_detail = ""
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_detail = f" - {error_data['error']}"
                except:
                    pass
                raise Exception(f"Model '{model}' not found in Ollama{error_detail}")
            # Don't retry on HTTP errors (except 429 rate limit)
            if e.response and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    continue
            raise Exception(f"Ollama API error: {str(e)}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Ollama API error: {str(e)}")
            # Retry on other exceptions
            continue
    
    raise Exception("Max retries exceeded")
```

### 2. Reduce Default Parallelism

**File**: `backend/summarizer_parallel.py` and `backend/main.py`

```python
# In summarizer_parallel.py, change default:
default_workers = min(3, len(chapters_data), cpu_count)  # Reduced from 4 to 3

# In main.py, change default:
max_workers = int(os.getenv("SUMMARY_MAX_WORKERS", 3))  # Reduced from 8 to 3
```

### 3. Reduce Chunk Size for Better Throughput

**File**: `backend/summarizer.py`

```python
def generate_summary(content: str, title: str, max_length: int = 2000) -> Optional[str]:
    # ... existing code ...
    
    # Reduce chunk size to improve reliability
    CHUNK_SIZE = 2000  # Reduced from 3000
    OVERLAP = 400      # Reduced from 500
```

### 4. Add Connection Pooling and Keep-Alive

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
            
            session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20,
                pool_block=False
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            _ollama_client = session
        except ImportError:
            raise ImportError("requests library required for Ollama")
    return _ollama_client
```

## Durable Fixes (Structural)

### 1. Implement Request Queue with Rate Limiting

**New File**: `backend/ollama_queue.py`

```python
"""
Queue-based Ollama request manager to prevent overloading.
"""
import asyncio
import time
from collections import deque
from typing import Callable, Any
import threading

class OllamaRateLimiter:
    """Rate limiter for Ollama requests"""
    def __init__(self, max_concurrent: int = 3, min_interval: float = 0.5):
        self.max_concurrent = max_concurrent
        self.min_interval = min_interval
        self.semaphore = threading.Semaphore(max_concurrent)
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def acquire(self):
        """Acquire permission to make request"""
        self.semaphore.acquire()
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()
    
    def release(self):
        """Release permission"""
        self.semaphore.release()

# Global rate limiter
_rate_limiter = OllamaRateLimiter(max_concurrent=3, min_interval=0.5)

def call_ollama_with_rate_limit(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
    """Call Ollama with rate limiting"""
    _rate_limiter.acquire()
    try:
        return call_ollama(prompt, system_prompt, model, max_retries=3)
    finally:
        _rate_limiter.release()
```

### 2. Add Model Warm-up on Startup

**File**: `backend/main.py` (add to startup)

```python
@app.on_event("startup")
async def warmup_ollama():
    """Warm up Ollama model on startup"""
    try:
        from llm_provider import call_ollama
        print("Warming up Ollama model...")
        call_ollama("Test", "You are a helpful assistant.", max_retries=1)
        print("Ollama model warmed up successfully")
    except Exception as e:
        print(f"Warning: Ollama warm-up failed: {e}")
```

### 3. Implement Adaptive Timeout Based on Chunk Size

**File**: `backend/summarizer.py`

```python
def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: str) -> str:
    """Summarize a single chunk with adaptive timeout"""
    # ... existing prompt setup ...
    
    # Calculate adaptive timeout based on chunk size
    # Estimate: ~4 chars/token, ~2s per token generation, +30s overhead
    estimated_tokens = len(chunk) / 4
    adaptive_timeout = max(180, min(600, int(estimated_tokens * 2 + 30)))
    
    try:
        summary = call_ollama(
            prompt=prompt,
            system_prompt=system_prompt,
            provider="ollama",
            timeout_override=adaptive_timeout  # Pass custom timeout
        )
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing chunk {chunk_index + 1}: {e}")
        return chunk[:500] + "..." if len(chunk) > 500 else chunk
```

## Test Plan

### Minimal Reproduction
1. Create test file with 3 chapters, each ~10KB (will create ~4 chunks each)
2. Set `SUMMARY_MAX_WORKERS=8` to simulate high parallelism
3. Upload file and monitor for timeouts
4. **Pass criteria**: Zero timeouts, all chunks summarized successfully

### Load Test
1. Upload document with 10+ chapters
2. Monitor Ollama logs for model reloads
3. Track per-chunk latency (p50, p95, p99)
4. **Pass criteria**: p95 latency < 180s, timeout rate < 1%

### Retry Test
1. Temporarily reduce timeout to 30s to force failures
2. Verify retries occur and eventually succeed
3. **Pass criteria**: All chunks succeed after retries

## Monitoring

### Metrics to Track
- **Per-request**: tokens_in, tokens_out, duration_ms, retry_count, timeout_flag
- **Aggregate**: p50/p95/p99 latency, timeout_rate, retry_rate, concurrent_requests
- **System**: Ollama model reloads, VRAM usage, queue depth

### Log Lines to Add

```python
# In summarize_chunk:
start_time = time.time()
tokens_estimate = len(chunk) / 4
print(f"  Summarizing chunk {chunk_index + 1}/{total_chunks} (~{int(tokens_estimate)} tokens, timeout={adaptive_timeout}s)")

# After completion:
duration = time.time() - start_time
print(f"  Chunk {chunk_index + 1} completed in {duration:.1f}s")

# On timeout:
print(f"  Chunk {chunk_index + 1} timeout after {duration:.1f}s (attempt {attempt + 1}/{max_retries})")
```

### Alert Thresholds
- **Timeout rate > 5%**: Alert immediately
- **p95 latency > 300s**: Warning
- **Concurrent requests > 6**: Warning (suggests over-parallelization)
- **Retry rate > 20%**: Investigate Ollama health

## Final Recommendation

**Immediate Fix** (apply now):

1. **Increase timeout to 300s** with retry logic (3 retries, exponential backoff)
2. **Reduce default workers to 3** (from 8)
3. **Reduce chunk size to 2000 chars** (from 3000)
4. **Add connection pooling** with keep-alive

**Code Snippet**:

```python
# In llm_provider.py - call_ollama function:
timeout=(5, 300)  # 5s connect, 300s read (increased from 120s)
max_retries=3     # Add retry loop with exponential backoff

# In summarizer_parallel.py:
default_workers = min(3, len(chapters_data), cpu_count)  # Reduced from 4

# In summarizer.py:
CHUNK_SIZE = 2000  # Reduced from 3000
OVERLAP = 400      # Reduced from 500
```

**Expected Impact**:
- Timeout rate: 10-20% → <1%
- Throughput: Slight reduction (~20%) but much more reliable
- Latency: p95 may increase slightly but more consistent

