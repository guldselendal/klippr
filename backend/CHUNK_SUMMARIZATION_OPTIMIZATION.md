# Chapter Chunk Summarization Pipeline Optimization Plan

## Executive Summary

This document provides a comprehensive optimization plan for reducing p95 end-to-end latency of the chapter chunk summarization pipeline while preserving quality and maintaining cost guardrails. The current pipeline processes 8k-20k word chapters through chunking, parallel summarization, and aggregation.

**Target**: ≥40% reduction in p95 E2E latency with ≤5% quality delta and ≤10% cost increase.

---

## A) Diagnosis and Baseline

### Current Pipeline Architecture

```
Chapter Input (8k-20k words)
  ↓
[Preprocess] → Content extraction, normalization
  ↓
[Chunking] → Split into 2000-char chunks with 400-char overlap
  ↓
[Parallel Chunk Summarization] → ThreadPoolExecutor (max 16 workers)
  ↓
[Merge/Reduce] → Sequential LLM call to combine chunk summaries
  ↓
[Post-process] → Sentence counting, truncation (10-12 sentences)
  ↓
[Persist] → Database write (batched)
```

### Current Implementation Details

**Chunking Strategy:**
- Fixed chunk size: 2000 characters
- Overlap: 400 characters (~20%)
- Sentence-boundary aware splitting
- Threshold: 5000 chars triggers chunking

**Concurrency:**
- ThreadPoolExecutor with max_workers = min(CPU_count, chunk_count, 16)
- Global concurrency limiter: 4 concurrent Ollama requests
- No rate-limit awareness beyond semaphore
- Nested executors (pipeline executor + chunk executor)

**Prompts:**
- Chunk prompt: ~150 tokens (verbose instructions)
- Merge prompt: ~200 tokens (includes all chunk summaries)
- System prompts: ~30 tokens each
- No structured output format

**Models:**
- All chunks: Same model (phi3:mini or configured)
- Merge step: Same model
- No tiered model strategy

### Bottleneck Analysis

**Top 2 Bottlenecks by p95 Contribution:**

1. **Sequential Merge Step (Critical Path)**
   - **Contribution**: ~30-40% of p95 latency
   - **Time**: 3-8 seconds (depends on chunk count)
   - **Sensitivity**: High - blocks final output
   - **Root Cause**: Must wait for all chunks, then sequential merge call
   - **Current**: No parallelization, no caching, processes all chunk summaries

2. **Chunk Summarization Parallelization Limits**
   - **Contribution**: ~25-35% of p95 latency
   - **Time**: Limited by slowest chunk (p95 of chunk times)
   - **Sensitivity**: High - determines merge start time
   - **Root Cause**: 
     - Global concurrency cap (4) too low for many chunks
     - No token-aware chunk sizing (inefficient context usage)
     - Verbose prompts increase token costs and latency
     - No caching (reprocesses identical chunks)

**Additional Bottlenecks:**

3. **Prompt Verbosity** (~15-20% of latency)
   - Chunk prompts: 150+ tokens
   - Merge prompt: 200+ tokens + all chunk summaries
   - No structured output reduces parsing overhead but increases generation

4. **No Caching** (~0% current, but 30-90% potential savings)
   - Identical chunks re-summarized
   - No resume capability for partial failures
   - No content-hash based deduplication

5. **Fixed Chunk Size** (~10-15% inefficiency)
   - Character-based, not token-based
   - May underutilize model context (e.g., 2000 chars ≈ 500 tokens, but model supports 2k-4k tokens)
   - Overlap may be excessive (400 chars ≈ 100 tokens)

### Estimated Current p95 E2E Latency

**For 15k-word chapter (~60k chars, ~30 chunks):**
- Chunking: 0.1s
- Parallel chunk summarization: 45-60s (p95 of slowest chunk, with 4 concurrent)
- Merge: 5-8s
- Post-process: 0.1s
- Persist: 0.01s

**Current p95 E2E: ~60-70 seconds**

---

## B) Quick Wins (1 Day Implementation)

### QW1: Compress Chunk Prompts (10-25% latency reduction)

**Change:**
- Reduce chunk prompt from ~150 tokens to ~60 tokens
- Use terse, structured instructions
- Request compact JSON output

**Implementation:**
```python
CHUNK_PROMPT_COMPACT = """Summarize section {idx}/{total} of "{title}".

Output JSON:
{{"summary":"<=120 words","key_points":["<=5 bullets"],"entities":["optional"]}}

Content:
{chunk}"""
```

**Impact:**
- **Latency**: 10-25% reduction (fewer tokens to generate)
- **Cost**: 15-30% token savings
- **Quality**: Minimal impact (structured output may improve consistency)

### QW2: Parallelize with Bounded Concurrency (25-50% p95 reduction)

**Change:**
- Increase effective concurrency for chunk processing
- Use shared executor pool across all chapters
- Implement token-aware rate limiting

**Implementation:**
```python
# Shared executor for all chunk tasks
CHUNK_EXECUTOR = ThreadPoolExecutor(max_workers=32)  # Higher than global limiter

# Token-aware concurrency (estimate tokens per request)
def estimate_tokens(text):
    return len(text) // 4  # Rough estimate

# Rate limit by tokens/min, not just requests
TOKEN_RATE_LIMIT = 100000  # tokens/min
```

**Impact:**
- **Latency**: 25-50% p95 reduction (better parallelization)
- **Cost**: Neutral (same total tokens)
- **Quality**: No impact

### QW3: Cache Per-Chunk Summaries (30-90% savings for re-runs)

**Change:**
- Cache chunk summaries by content_hash + prompt_hash + model_version
- Skip summarization for cached chunks
- TTL: 30 days, invalidate on model/prompt change

**Implementation:**
```python
import hashlib
import json
from functools import lru_cache

def get_chunk_cache_key(chunk, prompt_template, model):
    content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
    return f"chunk_summary:{content_hash}:{prompt_hash}:{model}"

# Check cache before LLM call
cache_key = get_chunk_cache_key(chunk, CHUNK_PROMPT_COMPACT, model)
cached = cache.get(cache_key)
if cached:
    return cached
```

**Impact:**
- **Latency**: 30-90% reduction for re-runs/edits
- **Cost**: 30-90% savings
- **Quality**: No impact (same cached result)

### QW4: Use Smaller Model for Chunks, Larger for Merge (20-40% cost/time)

**Change:**
- Chunks: phi3:mini (fast, cheap)
- Merge: gpt-oss:20b or gemini-1.5-flash (better quality for aggregation)

**Implementation:**
```python
def summarize_chunk(chunk, ...):
    return call_llm(..., provider="ollama", model="phi3:mini")

def merge_chunk_summaries(summaries, ...):
    return call_llm(..., provider="gemini", model="gemini-1.5-flash")
```

**Impact:**
- **Latency**: 20-40% reduction (faster chunk processing)
- **Cost**: 20-40% savings (cheaper chunks)
- **Quality**: Minimal impact (merge step uses better model)

### QW5: Reduce Overlap (10-15% token savings)

**Change:**
- Reduce overlap from 400 chars to 200 chars
- Use sentence-boundary aware overlap (only overlap at sentence boundaries)

**Impact:**
- **Latency**: 5-10% reduction (fewer tokens)
- **Cost**: 10-15% savings
- **Quality**: Minimal impact (200 chars still provides context)

**Combined Quick Wins Estimated Impact:**
- **p95 Latency Reduction**: 40-60%
- **Cost Reduction**: 25-40%
- **Quality Impact**: ≤2% (structured output may improve consistency)

---

## C) Detailed Recommendations by Layer

### C1) Architecture and Flow

#### C1.1: Fan-Out/Fan-In with Bounded Concurrency

**Current Issue**: Nested ThreadPoolExecutors create unbounded concurrency amplification.

**Solution**: Single shared executor with token-aware rate limiting.

```python
# Global chunk executor (shared across all chapters)
CHUNK_EXECUTOR = ThreadPoolExecutor(max_workers=64)

# Token-aware semaphore
class TokenRateLimiter:
    def __init__(self, tokens_per_min=100000):
        self.tokens_per_min = tokens_per_min
        self.tokens_used = 0
        self.window_start = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, estimated_tokens):
        with self.lock:
            now = time.time()
            if now - self.window_start > 60:
                self.tokens_used = 0
                self.window_start = now
            
            while self.tokens_used + estimated_tokens > self.tokens_per_min:
                time.sleep(0.1)  # Wait for window reset
            
            self.tokens_used += estimated_tokens
```

**Benefits:**
- Prevents concurrency explosion
- Respects token-based rate limits
- Better resource utilization

#### C1.2: Asynchronous I/O with Overlapped Compute

**Current Issue**: Synchronous LLM calls block threads.

**Solution**: Use asyncio + httpx for async HTTP calls.

```python
import asyncio
import httpx

async def summarize_chunk_async(chunk, ...):
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{ollama_url}/api/chat",
            json={"model": model, "messages": messages},
        )
        return response.json()["message"]["content"]

# Process chunks concurrently
async def process_chunks_async(chunks):
    tasks = [summarize_chunk_async(chunk, ...) for chunk in chunks]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Benefits:**
- Better thread utilization
- Lower memory overhead
- Natural backpressure via asyncio.Semaphore

#### C1.3: Resumable Pipeline with Checkpointing

**Current Issue**: No resume capability for partial failures.

**Solution**: Store chunk summaries in database with status.

```python
class ChunkSummary(Base):
    __tablename__ = "chunk_summaries"
    id = Column(String, primary_key=True)
    chapter_id = Column(String, ForeignKey("chapters.id"))
    chunk_index = Column(Integer)
    content_hash = Column(String, index=True)
    summary = Column(Text)
    status = Column(String)  # "pending", "completed", "failed"
    created_at = Column(DateTime)

# Resume logic
def resume_chapter_summarization(chapter_id):
    completed = db.query(ChunkSummary).filter(
        ChunkSummary.chapter_id == chapter_id,
        ChunkSummary.status == "completed"
    ).all()
    
    pending_chunks = get_pending_chunks(chapter_id, completed)
    # Process only pending chunks
```

**Benefits:**
- Idempotent retries
- Partial progress preservation
- Fault tolerance

### C2) Chunking and Aggregation

#### C2.1: Token-Aware Chunking

**Current Issue**: Character-based chunking underutilizes model context.

**Solution**: Use tokenizer to size chunks by tokens.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

def split_by_tokens(content, target_tokens=1500, overlap_tokens=200):
    tokens = tokenizer.encode(content)
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + target_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap_tokens
    
    return chunks
```

**Benefits:**
- Better context utilization (60-80% vs 30-40%)
- Fewer chunks for same content
- More predictable token counts

#### C2.2: Tree-Style Aggregation

**Current Issue**: Sequential merge of all chunks is slow.

**Solution**: Hierarchical tree reduction (branching factor 4-8).

```python
def tree_reduce(summaries, branching_factor=4):
    """Recursively merge summaries in a tree structure."""
    if len(summaries) <= branching_factor:
        return merge_summaries(summaries)  # Final merge
    
    # Split into groups
    groups = [summaries[i:i+branching_factor] 
              for i in range(0, len(summaries), branching_factor)]
    
    # Merge each group in parallel
    merged_groups = await asyncio.gather(*[
        merge_summaries_async(group) for group in groups
    ])
    
    # Recursively reduce
    return await tree_reduce(merged_groups, branching_factor)
```

**Benefits:**
- Parallel merge steps
- O(log n) depth vs O(1) sequential
- 40-60% faster merge for 30+ chunks

#### C2.3: Adaptive Chunking

**Current Issue**: Fixed chunk size doesn't adapt to content density.

**Solution**: Larger chunks for narrative, smaller for dense technical.

```python
def adaptive_chunk_size(content, base_size=1500):
    # Detect content type
    sentence_length = avg_sentence_length(content)
    technical_density = count_technical_terms(content) / len(content)
    
    if technical_density > 0.1:  # Dense technical
        return base_size * 0.7  # Smaller chunks
    elif sentence_length > 20:  # Narrative prose
        return base_size * 1.3  # Larger chunks
    else:
        return base_size
```

**Benefits:**
- Better quality for dense content
- Fewer chunks for narrative
- Adaptive to content characteristics

### C3) Prompts and Outputs

#### C3.1: Compact Structured Output

**Current Issue**: Verbose prompts and free-form text output.

**Solution**: JSON schema with tight constraints.

```python
CHUNK_PROMPT_STRUCTURED = """Summarize section {idx}/{total}.

Output JSON only:
{{
  "summary": "<=120 words",
  "key_points": ["<=5 bullets"],
  "entities": ["optional named entities"]
}}

Content:
{chunk}"""

# Parser
def parse_chunk_summary(response):
    import json
    # Extract JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"summary": response[:120]}  # Fallback
```

**Benefits:**
- 40-60% token reduction
- Faster parsing
- More consistent output

#### C3.2: Compress Merge Input

**Current Issue**: Merge prompt includes full chunk summaries.

**Solution**: Use only key_points and entities for merge.

```python
MERGE_PROMPT_COMPACT = """Merge {count} section summaries into one chapter summary.

Sections:
{key_points_only}

Output: 10-12 sentences, <=2000 chars."""

# Extract only key points
key_points = [chunk["key_points"] for chunk in chunk_summaries]
```

**Benefits:**
- 50-70% token reduction in merge step
- Faster merge generation
- Focus on essential information

#### C3.3: Early Termination

**Current Issue**: Models generate beyond required length.

**Solution**: Stop sequences and max_tokens.

```python
options = {
    "num_predict": 500,  # Cap output tokens
    "stop": ["\n\n\n", "---", "###"],  # Stop sequences
    "temperature": 0.1  # Low temperature for consistency
}
```

**Benefits:**
- 10-20% latency reduction
- Token savings
- More predictable output length

### C4) Models and Parameters

#### C4.1: Tiered Model Strategy

**Current Issue**: Same model for all steps.

**Solution**: Fast model for chunks, better model for merge.

```python
CHUNK_MODEL = "phi3:mini"  # Fast, cheap
MERGE_MODEL = "gemini-1.5-flash"  # Better quality, still fast

# Or use larger model only for final merge
if len(chunks) > 20:
    MERGE_MODEL = "gpt-oss:20b"  # Better for complex aggregation
```

**Benefits:**
- 20-40% cost/time savings
- Better quality for aggregation
- Optimal resource allocation

#### C4.2: Model-Specific Optimization

**Current Issue**: Generic parameters for all models.

**Solution**: Model-specific temperature, stop sequences, max_tokens.

```python
MODEL_CONFIGS = {
    "phi3:mini": {
        "temperature": 0.1,
        "max_tokens": 500,
        "stop": ["\n\n\n"]
    },
    "gemini-1.5-flash": {
        "temperature": 0.0,
        "max_output_tokens": 500,
        "stop_sequences": ["---"]
    }
}
```

**Benefits:**
- Optimal settings per model
- Better quality/consistency
- Reduced latency

### C5) Caching and Reuse

#### C5.1: Content-Hash Based Caching

**Current Issue**: No caching of chunk summaries.

**Solution**: Cache by content_hash + prompt_hash + model_version.

```python
class ChunkCache:
    def __init__(self, ttl_days=30):
        self.cache = {}  # Or use Redis/DB
        self.ttl = ttl_days * 86400
    
    def get(self, content_hash, prompt_hash, model):
        key = f"{content_hash}:{prompt_hash}:{model}"
        entry = self.cache.get(key)
        if entry and time.time() - entry["timestamp"] < self.ttl:
            return entry["summary"]
        return None
    
    def set(self, content_hash, prompt_hash, model, summary):
        key = f"{content_hash}:{prompt_hash}:{model}"
        self.cache[key] = {
            "summary": summary,
            "timestamp": time.time()
        }
```

**Benefits:**
- 30-90% savings for re-runs
- Faster processing
- Cost reduction

#### C5.2: Resume from Checkpoints

**Current Issue**: Must restart from scratch on failure.

**Solution**: Store chunk summaries in DB, resume incomplete chapters.

```python
def resume_chapter(chapter_id):
    # Load completed chunks
    completed = load_chunk_summaries(chapter_id)
    
    # Identify missing chunks
    all_chunks = get_chunks_for_chapter(chapter_id)
    missing = [c for c in all_chunks if c.id not in completed]
    
    # Process only missing chunks
    return process_chunks(missing)
```

**Benefits:**
- Fault tolerance
- Partial progress preservation
- Idempotent retries

### C6) I/O and Infrastructure

#### C6.1: Connection Pooling

**Current Issue**: New connections for each request.

**Solution**: Maintain warmed connection pools.

```python
# Already implemented in llm_provider.py
# Ensure pool_maxsize matches concurrency limits
pool_maxsize = global_limit + 2  # Headroom
```

**Benefits:**
- Reduced connection overhead
- Better throughput
- Lower latency

#### C6.2: Batch API Requests (if supported)

**Current Issue**: One request per chunk.

**Solution**: Batch multiple chunks in single request (if API supports).

```python
# If Ollama supports batch requests
def batch_summarize_chunks(chunks):
    messages = [
        {"role": "user", "content": chunk_prompt.format(chunk=c)}
        for c in chunks
    ]
    # Single API call for multiple chunks
    return call_ollama_batch(messages)
```

**Benefits:**
- Reduced API overhead
- Better throughput
- Lower latency

### C7) Reliability and Error Handling

#### C7.1: Circuit Breakers

**Current Issue**: No protection against failing providers.

**Solution**: Circuit breaker pattern.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.state = "closed"  # closed, open, half-open
        self.last_failure = None
    
    def call(self, func):
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen()
        
        try:
            result = func()
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "open"
            raise
```

**Benefits:**
- Fast failure detection
- Automatic recovery
- Prevents cascading failures

#### C7.2: Retry with Exponential Backoff

**Current Issue**: Simple retries may overwhelm system.

**Solution**: Jittered exponential backoff.

```python
def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait)
```

**Benefits:**
- Prevents thundering herd
- Better success rate
- Graceful degradation

---

## D) Implementation Notes

### D1) Concurrency Control

```python
# Global concurrency limiter (already exists)
from llm_concurrency import LLMConcurrencyLimiter

limiter = LLMConcurrencyLimiter()

# Token-aware rate limiter
class TokenRateLimiter:
    def __init__(self, tokens_per_min=100000):
        self.tokens_per_min = tokens_per_min
        self.tokens_used = 0
        self.window_start = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, estimated_tokens):
        with self.lock:
            now = time.time()
            if now - self.window_start > 60:
                self.tokens_used = 0
                self.window_start = now
            
            while self.tokens_used + estimated_tokens > self.tokens_per_min:
                time.sleep(0.1)
            
            self.tokens_used += estimated_tokens

# Combined limiter
async def summarize_chunk_with_limits(chunk, ...):
    estimated_tokens = len(chunk) // 4
    
    # Acquire both limiters
    with limiter.acquire("ollama"):
        await token_limiter.acquire(estimated_tokens)
        return await summarize_chunk_async(chunk, ...)
```

### D2) Caching Implementation

```python
# Redis or in-memory cache
import redis
import hashlib
import json

cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(chunk, prompt_template, model):
    content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
    return f"chunk:{content_hash}:{prompt_hash}:{model}"

def get_cached_summary(chunk, prompt_template, model):
    key = get_cache_key(chunk, prompt_template, model)
    cached = cache.get(key)
    if cached:
        return json.loads(cached)
    return None

def cache_summary(chunk, prompt_template, model, summary):
    key = get_cache_key(chunk, prompt_template, model)
    cache.setex(key, 30 * 86400, json.dumps(summary))  # 30 days TTL
```

### D3) Tree Reducer

```python
async def tree_reduce_async(summaries, branching_factor=4, model="gemini-1.5-flash"):
    """Hierarchical tree reduction of summaries."""
    if len(summaries) <= branching_factor:
        # Final merge
        return await merge_summaries_async(summaries, model)
    
    # Split into groups
    groups = [summaries[i:i+branching_factor] 
              for i in range(0, len(summaries), branching_factor)]
    
    # Merge each group in parallel
    merged_groups = await asyncio.gather(*[
        merge_summaries_async(group, model) for group in groups
    ])
    
    # Recursively reduce
    return await tree_reduce_async(merged_groups, branching_factor, model)
```

---

## E) Experiment Plan

### E1) Metrics to Track

**Latency Metrics:**
- p50, p95, p99 E2E latency (chapter input → final summary)
- Per-stage latency: chunking, chunk summarization, merge, post-process
- Queue wait times
- Retry counts and backoff times

**Quality Metrics:**
- Sentence count (target: 10-12)
- Summary length (target: 1800-2000 chars)
- ROUGE-L score vs human reference (if available)
- Factuality check (spot-check for hallucinations)

**Cost Metrics:**
- Tokens per chapter (input + output)
- Cost per chapter
- Cache hit rate
- API call counts

**Throughput Metrics:**
- Chapters processed per hour
- Queue depth
- Worker utilization

### E2) Test Matrix

| Test Case | Chapter Size | Chunk Count | Model Config | Expected p95 |
|-----------|--------------|-------------|--------------|--------------|
| Baseline | 15k words | 30 chunks | phi3:mini all | 60-70s |
| Quick Wins | 15k words | 30 chunks | Compact prompts, tiered | 35-45s |
| Full Opt | 15k words | 20 chunks (token-aware) | Tiered + tree reduce | 25-35s |
| Large Chapter | 20k words | 40 chunks | Full optimization | 40-50s |
| Small Chapter | 8k words | 1 chunk (no chunking) | Direct summary | 8-12s |

### E3) Dataset

- **Size**: 50 chapters (mix of 8k, 15k, 20k words)
- **Diversity**: Narrative, technical, mixed content
- **Ground Truth**: Human-written summaries for 10 chapters (for quality metrics)

### E4) Acceptance Criteria

**Primary:**
- p95 E2E latency reduction: ≥40%
- Quality delta: ≤5% (ROUGE-L or sentence count variance)
- Cost increase: ≤10%

**Secondary:**
- p50 latency reduction: ≥30%
- Cache hit rate: ≥50% for re-runs
- Error rate: ≤1%

### E5) Phased Rollout

**Phase 1: Quick Wins (Week 1)**
- Deploy compact prompts
- Enable caching
- Increase concurrency limits
- **Target**: 30-40% latency reduction

**Phase 2: Architecture (Week 2)**
- Migrate to async I/O
- Implement tree reduction
- Add checkpointing
- **Target**: Additional 20-30% reduction

**Phase 3: Advanced (Week 3)**
- Token-aware chunking
- Tiered models
- Adaptive chunking
- **Target**: Final 10-20% reduction

**Phase 4: Monitoring (Week 4)**
- Full metrics collection
- A/B testing
- Quality validation
- **Target**: Validate all acceptance criteria

---

## F) Target State and Expected p95

### Recommended Target State

**Architecture:**
- Async I/O with asyncio + httpx
- Token-aware chunking (1500 tokens/chunk, 200 token overlap)
- Tree reduction (branching factor 4)
- Tiered models (phi3:mini for chunks, gemini-1.5-flash for merge)
- Content-hash caching with 30-day TTL
- Checkpointing for resume capability

**Configuration:**
- Chunk model: phi3:mini (fast, cheap)
- Merge model: gemini-1.5-flash (quality, speed)
- Global concurrency: 16 Ollama, 32 Gemini
- Token rate limit: 100k tokens/min
- Chunk size: 1500 tokens (adaptive)
- Overlap: 200 tokens

**Expected p95 E2E Latency: 25-35 seconds** (for 15k-word chapter)

**Breakdown:**
- Chunking: 0.1s
- Parallel chunk summarization: 15-20s (p95, with 16 concurrent)
- Tree merge (2 levels): 5-8s
- Post-process: 0.1s
- Persist: 0.01s

**Total: ~25-35s** (vs 60-70s baseline)

### Trade-offs

**Latency:**
- ✅ 50-60% reduction in p95
- ✅ Better parallelization
- ✅ Faster merge with tree reduction

**Cost:**
- ✅ 25-40% reduction (smaller model for chunks, caching)
- ⚠️ Slight increase if using larger model for merge (offset by caching)

**Quality:**
- ✅ Structured output improves consistency
- ✅ Tiered models maintain/improve quality
- ⚠️ Compact prompts may reduce detail (mitigated by structured output)

**Complexity:**
- ⚠️ More complex architecture (async, tree reduction)
- ⚠️ Additional caching layer
- ✅ Better fault tolerance (checkpointing)

**Maintainability:**
- ✅ Clear separation of concerns
- ✅ Configurable parameters
- ⚠️ More code to maintain

---

## Reasoning Summary

### Assumptions
1. Chapters are 8k-20k words (60k-150k chars)
2. Current p95 E2E: 60-70 seconds
3. Quality must be preserved (10-12 sentences, comprehensive)
4. Cost guardrails exist (budget per chapter)
5. Infrastructure supports async I/O and connection pooling

### Key Constraints
1. **Quality**: Must maintain factuality, no hallucinations
2. **Cost**: Budget per chapter must not be exceeded
3. **Resumability**: Must support idempotent retries
4. **Rate Limits**: Token and request limits must be respected

### Why Each Change Reduces E2E Time

1. **Compact Prompts**: Fewer tokens → faster generation → 10-25% reduction
2. **Parallelization**: Better concurrency → faster chunk processing → 25-50% reduction
3. **Caching**: Skip LLM calls for cached chunks → 30-90% reduction (re-runs)
4. **Tiered Models**: Faster chunks → 20-40% reduction
5. **Tree Reduction**: Parallel merge steps → 40-60% faster merge
6. **Token-Aware Chunking**: Fewer chunks → faster overall → 10-15% reduction
7. **Async I/O**: Better resource utilization → 10-20% reduction
8. **Early Termination**: Stop at required length → 10-20% reduction

### Critical Path Optimization

The critical path is: **chunk processing → merge → post-process**

Optimizations target:
- **Chunk processing**: Parallelization, caching, faster model
- **Merge**: Tree reduction, compact input, better model
- **Post-process**: Minimal (already fast)

By optimizing the critical path, we achieve the target 40-60% p95 reduction.

---

## Implementation Priority

1. **Immediate (Day 1)**: Quick wins (QW1-QW5)
2. **Week 1**: Caching, checkpointing, compact prompts
3. **Week 2**: Async I/O, tree reduction, tiered models
4. **Week 3**: Token-aware chunking, adaptive chunking
5. **Week 4**: Monitoring, validation, fine-tuning

---

## Conclusion

This optimization plan provides a comprehensive roadmap for reducing p95 E2E latency by 40-60% while maintaining quality and controlling costs. The phased approach allows for incremental improvements with validation at each stage.

**Expected Outcome**: p95 E2E latency of 25-35 seconds (down from 60-70 seconds) for a 15k-word chapter, with ≤5% quality impact and ≤10% cost increase.

