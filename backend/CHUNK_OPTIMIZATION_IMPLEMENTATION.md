# Chunk Summarization Optimization - Implementation Steps

This document provides concrete, step-by-step implementation instructions for the optimization plan.

---

## Phase 1: Quick Wins (Day 1) - 30-40% Latency Reduction

### Step 1.1: Compress Chunk Prompts

**File**: `backend/prompt_utils.py`

**Action**: Add compact chunk prompt template

```python
# Add after existing prompts
CHUNK_PROMPT_COMPACT = """Summarize section {idx}/{total} of "{title}".

Output JSON:
{{"summary":"<=120 words","key_points":["<=5 bullets"],"entities":["optional"]}}

Content:
{chunk}"""

CHUNK_SYSTEM_COMPACT = "Output valid JSON only. Be concise."
```

**File**: `backend/summarizer.py`

**Action**: Update `summarize_chunk()` function

```python
def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: str) -> str:
    from prompt_utils import CHUNK_PROMPT_COMPACT, CHUNK_SYSTEM_COMPACT
    
    prompt = CHUNK_PROMPT_COMPACT.format(
        idx=chunk_index + 1,
        total=total_chunks,
        title=title,
        chunk=chunk
    )
    
    try:
        response = call_llm(
            prompt=prompt,
            system_prompt=CHUNK_SYSTEM_COMPACT,
            provider="ollama"
        )
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("summary", response[:120])  # Fallback to first 120 words
        return response[:120]  # Fallback
    except Exception as e:
        print(f"Error summarizing chunk {chunk_index + 1}: {e}")
        return chunk[:500] + "..."
```

**File**: `backend/summary_pipeline.py`

**Action**: Update `summarize_chunk_with_provider()` similarly

**Test**: Run `test_prompt_output.py` to verify JSON parsing works

---

### Step 1.2: Reduce Overlap

**File**: `backend/summarizer.py`

**Action**: Update chunking parameters

```python
# In generate_summary() function, change:
CHUNK_SIZE = 2000  # Keep same
OVERLAP = 200      # Change from 400 to 200

# Also update split_content_into_chunks() to ensure overlap happens at sentence boundaries
def split_content_into_chunks(content: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    # ... existing code ...
    # When moving start position, ensure it's at sentence boundary
    start = end - overlap
    # Find nearest sentence boundary
    for i in range(start, max(start - 100, 0), -1):
        if i < len(content) and content[i] in '.!?\n':
            start = i + 1
            break
```

**Test**: Verify chunk count decreases by ~10-15% for same content

---

### Step 1.3: Implement Tiered Models

**File**: `backend/summarizer.py`

**Action**: Update model selection

```python
# At top of file
CHUNK_MODEL = os.getenv("CHUNK_MODEL", "phi3:mini")  # Fast for chunks
MERGE_MODEL = os.getenv("MERGE_MODEL", "gemini-1.5-flash")  # Better for merge

# In summarize_chunk()
def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: str) -> str:
    # ... existing code ...
    summary = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        provider="ollama",
        model=CHUNK_MODEL  # Use fast model
    )

# In merge_chunk_summaries()
def merge_chunk_summaries(chunk_summaries: List[str], title: str) -> str:
    # ... existing code ...
    merged_summary = call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        provider="gemini",  # Use better model for merge
        model=MERGE_MODEL
    )
```

**File**: `backend/summary_pipeline.py`

**Action**: Update `generate_summary_with_routing()` similarly

**Test**: Verify chunks use phi3:mini, merge uses gemini-1.5-flash

---

### Step 1.4: Increase Concurrency for Chunks

**File**: `backend/llm_concurrency.py`

**Action**: Increase Ollama concurrency limit

```python
# In __init__()
self.ollama_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OLLAMA", "16"))  # Increase from 4 to 16
```

**File**: `backend/summarizer.py`

**Action**: Update max_workers calculation

```python
# In generate_summary()
max_workers = min(16, len(chunks), 16)  # Allow up to 16 concurrent chunk requests
```

**Test**: Monitor concurrency metrics to ensure no overload

---

### Step 1.5: Add Compact Merge Prompt

**File**: `backend/prompt_utils.py`

**Action**: Add compact merge prompt that uses only key_points

```python
MERGE_PROMPT_COMPACT = """Merge {count} section summaries into one chapter summary (10-12 sentences, <=2000 chars).

Sections:
{key_points}

Output: Comprehensive summary covering all sections."""

# Helper function to extract key points
def extract_key_points_for_merge(chunk_summaries: List[dict]) -> str:
    """Extract only key_points from chunk summaries for merge."""
    sections = []
    for i, chunk_data in enumerate(chunk_summaries):
        if isinstance(chunk_data, dict):
            key_points = chunk_data.get("key_points", [])
            sections.append(f"Section {i+1}: {'; '.join(key_points[:5])}")
        else:
            # Fallback for non-JSON summaries
            sections.append(f"Section {i+1}: {chunk_data[:100]}")
    return "\n".join(sections)
```

**File**: `backend/summarizer.py`

**Action**: Update `merge_chunk_summaries()`

```python
def merge_chunk_summaries(chunk_summaries: List[str], title: str) -> str:
    from prompt_utils import MERGE_PROMPT_COMPACT, COMBINE_SYSTEM, extract_key_points_for_merge
    
    # Parse chunk summaries if they're JSON
    parsed_summaries = []
    for summary in chunk_summaries:
        import json
        import re
        json_match = re.search(r'\{.*\}', summary, re.DOTALL)
        if json_match:
            try:
                parsed_summaries.append(json.loads(json_match.group()))
            except:
                parsed_summaries.append({"summary": summary})
        else:
            parsed_summaries.append({"summary": summary})
    
    # Extract key points for compact merge
    key_points = extract_key_points_for_merge(parsed_summaries)
    
    prompt = MERGE_PROMPT_COMPACT.format(
        count=len(chunk_summaries),
        key_points=key_points
    )
    
    # ... rest of merge logic ...
```

**Test**: Verify merge prompt is 50-70% shorter

---

## Phase 2: Caching and Checkpointing (Week 1)

### Step 2.1: Create Chunk Cache Module

**File**: `backend/chunk_cache.py` (NEW)

**Action**: Create caching module

```python
"""
Chunk summary caching by content hash.
"""
import hashlib
import json
import time
from typing import Optional, Dict
import os

# Use Redis if available, otherwise in-memory dict
try:
    import redis
    USE_REDIS = True
    cache_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True
    )
except ImportError:
    USE_REDIS = False
    _memory_cache: Dict[str, Dict] = {}


def get_cache_key(chunk: str, prompt_template: str, model: str) -> str:
    """Generate cache key from content hash, prompt hash, and model."""
    content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
    return f"chunk_summary:{content_hash}:{prompt_hash}:{model}"


def get_cached_summary(chunk: str, prompt_template: str, model: str) -> Optional[dict]:
    """Get cached chunk summary if available."""
    key = get_cache_key(chunk, prompt_template, model)
    
    if USE_REDIS:
        cached = cache_client.get(key)
        if cached:
            return json.loads(cached)
    else:
        entry = _memory_cache.get(key)
        if entry and time.time() - entry["timestamp"] < entry["ttl"]:
            return entry["data"]
    
    return None


def cache_summary(chunk: str, prompt_template: str, model: str, summary: dict, ttl_days: int = 30):
    """Cache chunk summary."""
    key = get_cache_key(chunk, prompt_template, model)
    ttl_seconds = ttl_days * 86400
    
    if USE_REDIS:
        cache_client.setex(key, ttl_seconds, json.dumps(summary))
    else:
        _memory_cache[key] = {
            "data": summary,
            "timestamp": time.time(),
            "ttl": ttl_seconds
        }


def invalidate_cache(prompt_template: str = None, model: str = None):
    """Invalidate cache entries matching prompt or model."""
    # Implementation depends on cache backend
    if USE_REDIS:
        # Use pattern matching to delete keys
        pattern = "chunk_summary:*"
        if model:
            pattern = f"chunk_summary:*:*:{model}"
        for key in cache_client.scan_iter(match=pattern):
            cache_client.delete(key)
    else:
        # Clear memory cache if prompt/model changed
        _memory_cache.clear()
```

---

### Step 2.2: Integrate Caching into Summarizer

**File**: `backend/summarizer.py`

**Action**: Add cache checks before LLM calls

```python
from chunk_cache import get_cached_summary, cache_summary
from prompt_utils import CHUNK_PROMPT_COMPACT

def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: str) -> str:
    model = CHUNK_MODEL
    prompt_template = CHUNK_PROMPT_COMPACT
    
    # Check cache first
    cached = get_cached_summary(chunk, prompt_template, model)
    if cached:
        print(f"  Cache hit for chunk {chunk_index + 1}")
        return cached.get("summary", "")
    
    # Generate summary
    prompt = prompt_template.format(
        idx=chunk_index + 1,
        total=total_chunks,
        title=title,
        chunk=chunk
    )
    
    try:
        response = call_llm(...)
        # Parse and cache
        summary_data = parse_chunk_response(response)
        cache_summary(chunk, prompt_template, model, summary_data)
        return summary_data.get("summary", response[:120])
    except Exception as e:
        # ... error handling ...
```

---

### Step 2.3: Add Checkpointing Table

**File**: `backend/models.py`

**Action**: Add ChunkSummary model

```python
class ChunkSummary(Base):
    __tablename__ = "chunk_summaries"
    
    id = Column(String, primary_key=True)
    chapter_id = Column(String, ForeignKey("chapters.id"), index=True)
    chunk_index = Column(Integer)
    content_hash = Column(String, index=True)
    summary = Column(Text)
    status = Column(String)  # "pending", "completed", "failed"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

**File**: `backend/migrate_add_chunk_summaries.py` (NEW)

**Action**: Create migration script

```python
#!/usr/bin/env python3
"""Migration to add chunk_summaries table."""
from database import engine, Base
from models import ChunkSummary

def migrate():
    """Create chunk_summaries table."""
    ChunkSummary.__table__.create(bind=engine, checkfirst=True)
    print("✓ chunk_summaries table created")

if __name__ == "__main__":
    migrate()
```

---

### Step 2.4: Implement Resume Logic

**File**: `backend/summarizer.py`

**Action**: Add resume capability

```python
def resume_chapter_summarization(chapter_id: str, chunks: List[str], title: str) -> List[str]:
    """Resume chapter summarization from checkpoints."""
    from database import get_db_session
    from models import ChunkSummary
    import hashlib
    
    db = next(get_db_session())
    try:
        # Load completed chunks
        completed = db.query(ChunkSummary).filter(
            ChunkSummary.chapter_id == chapter_id,
            ChunkSummary.status == "completed"
        ).all()
        
        completed_hashes = {c.content_hash: c.summary for c in completed}
        
        # Identify missing chunks
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
            
            if chunk_hash in completed_hashes:
                # Use cached summary
                chunk_summaries.append(completed_hashes[chunk_hash])
            else:
                # Generate new summary
                summary = summarize_chunk(chunk, idx, len(chunks), title)
                
                # Save checkpoint
                chunk_summary = ChunkSummary(
                    id=str(uuid.uuid4()),
                    chapter_id=chapter_id,
                    chunk_index=idx,
                    content_hash=chunk_hash,
                    summary=summary,
                    status="completed"
                )
                db.add(chunk_summary)
                chunk_summaries.append(summary)
        
        db.commit()
        return chunk_summaries
    finally:
        db.close()
```

---

## Phase 3: Architecture Improvements (Week 2)

### Step 3.1: Implement Token-Aware Rate Limiter

**File**: `backend/token_rate_limiter.py` (NEW)

**Action**: Create token-based rate limiter

```python
"""
Token-aware rate limiter for LLM API calls.
"""
import time
import threading
from typing import Optional

class TokenRateLimiter:
    """Rate limiter based on tokens per minute."""
    
    def __init__(self, tokens_per_min: int = 100000):
        self.tokens_per_min = tokens_per_min
        self.tokens_used = 0
        self.window_start = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, estimated_tokens: int, timeout: Optional[float] = None):
        """Acquire tokens, blocking if necessary."""
        start_time = time.time()
        
        with self.lock:
            while True:
                now = time.time()
                
                # Reset window if 60 seconds passed
                if now - self.window_start >= 60:
                    self.tokens_used = 0
                    self.window_start = now
                
                # Check if we can proceed
                if self.tokens_used + estimated_tokens <= self.tokens_per_min:
                    self.tokens_used += estimated_tokens
                    return
                
                # Check timeout
                if timeout and (now - start_time) >= timeout:
                    raise TimeoutError("Token rate limit timeout")
                
                # Wait a bit before retrying
                time.sleep(0.1)
    
    def get_metrics(self) -> dict:
        """Get current rate limiter metrics."""
        with self.lock:
            return {
                "tokens_used": self.tokens_used,
                "tokens_per_min": self.tokens_per_min,
                "tokens_remaining": max(0, self.tokens_per_min - self.tokens_used),
                "window_elapsed": time.time() - self.window_start
            }
```

---

### Step 3.2: Integrate Token Limiter

**File**: `backend/llm_provider.py`

**Action**: Add token limiter to LLM calls

```python
from token_rate_limiter import TokenRateLimiter

# Global token limiter
_token_limiter = None

def get_token_limiter():
    global _token_limiter
    if _token_limiter is None:
        tokens_per_min = int(os.getenv("LLM_TOKEN_RATE_LIMIT", "100000"))
        _token_limiter = TokenRateLimiter(tokens_per_min)
    return _token_limiter

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token)."""
    return len(text) // 4

# In call_ollama()
def call_ollama(prompt: str, system_prompt: Optional[str] = None, ...):
    limiter = LLMConcurrencyLimiter()
    token_limiter = get_token_limiter()
    
    # Estimate tokens
    estimated_tokens = estimate_tokens(prompt)
    if system_prompt:
        estimated_tokens += estimate_tokens(system_prompt)
    estimated_tokens += 500  # Output estimate
    
    with limiter.acquire("ollama"):
        token_limiter.acquire(estimated_tokens, timeout=30)
        # ... existing LLM call ...
```

---

### Step 3.3: Implement Tree Reduction

**File**: `backend/summarizer.py`

**Action**: Add tree reduction function

```python
async def tree_reduce_async(chunk_summaries: List[str], title: str, branching_factor: int = 4) -> str:
    """Hierarchical tree reduction of chunk summaries."""
    from prompt_utils import MERGE_PROMPT_COMPACT, COMBINE_SYSTEM
    import asyncio
    
    if len(chunk_summaries) <= branching_factor:
        # Final merge - use synchronous call
        return merge_chunk_summaries(chunk_summaries, title)
    
    # Split into groups
    groups = [chunk_summaries[i:i+branching_factor] 
              for i in range(0, len(chunk_summaries), branching_factor)]
    
    # Merge each group in parallel (using ThreadPoolExecutor for now)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    merged_groups = []
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = {
            executor.submit(merge_chunk_summaries, group, title): group
            for group in groups
        }
        
        for future in as_completed(futures):
            try:
                merged_groups.append(future.result())
            except Exception as e:
                print(f"Error merging group: {e}")
                # Fallback: use first summary from group
                merged_groups.append(futures[future][0])
    
    # Recursively reduce
    return await tree_reduce_async(merged_groups, title, branching_factor)

# Update merge_chunk_summaries() to use tree reduction for large sets
def merge_chunk_summaries(chunk_summaries: List[str], title: str) -> str:
    if len(chunk_summaries) > 8:
        # Use tree reduction for large sets
        import asyncio
        return asyncio.run(tree_reduce_async(chunk_summaries, title))
    
    # ... existing merge logic for small sets ...
```

---

### Step 3.4: Add Early Termination

**File**: `backend/llm_provider.py`

**Action**: Update model options for early termination

```python
# In call_ollama()
options = {
    "num_predict": 500,  # Cap output tokens (reduced from 2000)
    "num_thread": 0,
    "temperature": TEMPERATURE.get("ollama", 0.1),
    "stop": STOP_SEQUENCES.get("ollama", [])  # Already implemented
}

# In call_gemini()
generation_config = {
    "temperature": TEMPERATURE.get("gemini", 0.1),
    "max_output_tokens": 500,  # Add cap
    "stop_sequences": STOP_SEQUENCES.get("gemini", [])
}
```

---

## Phase 4: Advanced Optimizations (Week 3)

### Step 4.1: Implement Token-Aware Chunking

**File**: `backend/tokenizer_utils.py` (NEW)

**Action**: Create tokenizer utility

```python
"""
Token-aware chunking utilities.
"""
import os

# Try to use transformers tokenizer, fallback to character-based
try:
    from transformers import AutoTokenizer
    _tokenizer = None
    
    def get_tokenizer():
        global _tokenizer
        if _tokenizer is None:
            model_name = os.getenv("TOKENIZER_MODEL", "microsoft/phi-3-mini-4k-instruct")
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
        return _tokenizer
    
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        tokenizer = get_tokenizer()
        return len(tokenizer.encode(text))
    
    def split_by_tokens(content: str, target_tokens: int = 1500, overlap_tokens: int = 200) -> List[str]:
        """Split content by tokens instead of characters."""
        tokenizer = get_tokenizer()
        tokens = tokenizer.encode(content)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + target_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap_tokens
            if start >= len(tokens):
                break
        
        return chunks

except ImportError:
    # Fallback to character-based estimation
    def count_tokens(text: str) -> int:
        """Estimate tokens (4 chars per token)."""
        return len(text) // 4
    
    def split_by_tokens(content: str, target_tokens: int = 1500, overlap_tokens: int = 200) -> List[str]:
        """Fallback: split by characters with token estimation."""
        target_chars = target_tokens * 4
        overlap_chars = overlap_tokens * 4
        return split_content_into_chunks(content, target_chars, overlap_chars)
```

**File**: `backend/summarizer.py`

**Action**: Update chunking to use token-aware splitting

```python
from tokenizer_utils import split_by_tokens, count_tokens

def generate_summary(content: str, title: str, max_length: int = 2000) -> Optional[str]:
    # ... existing code ...
    
    if len(content) > CHUNK_THRESHOLD:
        # Use token-aware chunking
        chunks = split_by_tokens(content, target_tokens=1500, overlap_tokens=200)
        print(f"  Split into {len(chunks)} chunks (token-aware, ~1500 tokens/chunk)")
        # ... rest of processing ...
```

---

### Step 4.2: Implement Adaptive Chunking

**File**: `backend/summarizer.py`

**Action**: Add adaptive chunk size logic

```python
def detect_content_type(content: str) -> dict:
    """Detect content characteristics for adaptive chunking."""
    sentences = content.split('.')
    avg_sentence_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
    
    # Count technical terms (simple heuristic)
    technical_terms = ['algorithm', 'function', 'method', 'class', 'variable', 
                       'parameter', 'implementation', 'architecture', 'protocol']
    technical_count = sum(1 for term in technical_terms if term.lower() in content.lower())
    technical_density = technical_count / max(len(content.split()), 1)
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "technical_density": technical_density,
        "is_technical": technical_density > 0.001,
        "is_narrative": avg_sentence_length > 20
    }

def adaptive_chunk_size(content: str, base_tokens: int = 1500) -> int:
    """Adapt chunk size based on content characteristics."""
    characteristics = detect_content_type(content)
    
    if characteristics["is_technical"]:
        return int(base_tokens * 0.7)  # Smaller chunks for dense technical
    elif characteristics["is_narrative"]:
        return int(base_tokens * 1.3)  # Larger chunks for narrative
    else:
        return base_tokens

# Update generate_summary()
def generate_summary(content: str, title: str, max_length: int = 2000) -> Optional[str]:
    # ... existing code ...
    
    if len(content) > CHUNK_THRESHOLD:
        # Adaptive chunk size
        chunk_size_tokens = adaptive_chunk_size(content, base_tokens=1500)
        chunks = split_by_tokens(content, target_tokens=chunk_size_tokens, overlap_tokens=200)
        # ... rest of processing ...
```

---

## Phase 5: Monitoring and Validation (Week 4)

### Step 5.1: Add Detailed Metrics

**File**: `backend/pipeline_metrics.py`

**Action**: Add chunk-level metrics

```python
# Add to existing metrics
class ChunkMetrics:
    def __init__(self):
        self.chunk_times = []
        self.merge_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.token_counts = []
    
    def record_chunk_time(self, duration: float):
        self.chunk_times.append(duration)
    
    def record_merge_time(self, duration: float):
        self.merge_times.append(duration)
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def record_tokens(self, tokens: int):
        self.token_counts.append(tokens)
    
    def get_stats(self) -> dict:
        return {
            "chunk_p50": np.percentile(self.chunk_times, 50) if self.chunk_times else 0,
            "chunk_p95": np.percentile(self.chunk_times, 95) if self.chunk_times else 0,
            "merge_p50": np.percentile(self.merge_times, 50) if self.merge_times else 0,
            "merge_p95": np.percentile(self.merge_times, 95) if self.merge_times else 0,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "avg_tokens_per_chunk": np.mean(self.token_counts) if self.token_counts else 0
        }

chunk_metrics = ChunkMetrics()
```

---

### Step 5.2: Instrument Summarizer

**File**: `backend/summarizer.py`

**Action**: Add metrics collection

```python
from pipeline_metrics import chunk_metrics
from tokenizer_utils import count_tokens

def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: str) -> str:
    start_time = time.time()
    
    # ... existing summarization logic ...
    
    duration = time.time() - start_time
    chunk_metrics.record_chunk_time(duration)
    chunk_metrics.record_tokens(count_tokens(chunk))
    
    # Check cache
    cached = get_cached_summary(...)
    if cached:
        chunk_metrics.record_cache_hit()
    else:
        chunk_metrics.record_cache_miss()
    
    return summary

def merge_chunk_summaries(chunk_summaries: List[str], title: str) -> str:
    start_time = time.time()
    
    # ... existing merge logic ...
    
    duration = time.time() - start_time
    chunk_metrics.record_merge_time(duration)
    
    return merged_summary
```

---

### Step 5.3: Create Test Script

**File**: `backend/test_chunk_optimization.py` (NEW)

**Action**: Create comprehensive test script

```python
#!/usr/bin/env python3
"""
Test script for chunk summarization optimizations.
"""
import time
import statistics
from summarizer import generate_summary

def test_chapter_summarization(content: str, title: str, iterations: int = 5):
    """Test summarization with metrics."""
    times = []
    
    for i in range(iterations):
        start = time.time()
        summary = generate_summary(content, title)
        duration = time.time() - start
        times.append(duration)
        
        print(f"Iteration {i+1}: {duration:.2f}s, summary length: {len(summary)}")
    
    print(f"\nStatistics:")
    print(f"  Mean: {statistics.mean(times):.2f}s")
    print(f"  Median: {statistics.median(times):.2f}s")
    print(f"  p95: {statistics.quantiles(times, n=20)[18]:.2f}s")
    
    return times

# Test with different chapter sizes
if __name__ == "__main__":
    # 15k word chapter
    content_15k = "..."  # Your test content
    test_chapter_summarization(content_15k, "Test Chapter", iterations=5)
```

---

## Implementation Checklist

### Phase 1: Quick Wins (Day 1)
- [x] Step 1.1: Compress chunk prompts
- [x] Step 1.2: Reduce overlap to 200 chars
- [ ] Step 1.3: Implement tiered models
- [x] Step 1.4: Increase Ollama concurrency to 16
- [x] Step 1.5: Add compact merge prompt
- [ ] Test: Run `test_prompt_output.py`
- [ ] Test: Verify 30-40% latency reduction

### Phase 2: Caching (Week 1)
- [x] Step 2.1: Create `chunk_cache.py`
- [x] Step 2.2: Integrate caching into summarizer
- [x] Step 2.3: Add ChunkSummary model
- [ ] Step 2.4: Run migration script
- [ ] Step 2.5: Implement resume logic
- [ ] Test: Verify cache hits work
- [ ] Test: Verify resume from checkpoints

### Phase 3: Architecture (Week 2)
- [ ] Step 3.1: Create `token_rate_limiter.py`
- [ ] Step 3.2: Integrate token limiter
- [ ] Step 3.3: Implement tree reduction
- [ ] Step 3.4: Add early termination
- [ ] Test: Verify tree reduction works
- [ ] Test: Monitor token rate limiting

### Phase 4: Advanced (Week 3)
- [ ] Step 4.1: Create `tokenizer_utils.py`
- [ ] Step 4.2: Implement token-aware chunking
- [ ] Step 4.3: Implement adaptive chunking
- [ ] Test: Verify fewer chunks with token-aware
- [ ] Test: Verify adaptive sizing works

### Phase 5: Monitoring (Week 4)
- [ ] Step 5.1: Add chunk metrics
- [ ] Step 5.2: Instrument summarizer
- [ ] Step 5.3: Create test script
- [ ] Test: Run full test suite
- [ ] Validate: 40%+ latency reduction
- [ ] Validate: ≤5% quality impact

---

## Environment Variables

Add to `.env`:

```bash
# Chunk optimization
CHUNK_MODEL=phi3:mini
MERGE_MODEL=gemini-1.5-flash
LLM_MAX_CONCURRENCY_OLLAMA=16
LLM_MAX_CONCURRENCY_GEMINI=32
LLM_TOKEN_RATE_LIMIT=100000

# Caching
REDIS_HOST=localhost
REDIS_PORT=6379

# Tokenizer (optional)
TOKENIZER_MODEL=microsoft/phi-3-mini-4k-instruct
```

---

## Testing Strategy

1. **Unit Tests**: Test each component individually
2. **Integration Tests**: Test full pipeline with test chapters
3. **Performance Tests**: Measure latency before/after
4. **Quality Tests**: Verify sentence count, length, factuality
5. **Load Tests**: Test with multiple concurrent chapters

---

## Rollback Plan

If issues arise:
1. Disable optimizations via feature flags
2. Revert to previous prompt templates
3. Disable caching if causing issues
4. Reduce concurrency limits if overloaded

Feature flags:
```python
USE_COMPACT_PROMPTS = os.getenv("USE_COMPACT_PROMPTS", "true").lower() == "true"
USE_CHUNK_CACHE = os.getenv("USE_CHUNK_CACHE", "true").lower() == "true"
USE_TREE_REDUCE = os.getenv("USE_TREE_REDUCE", "true").lower() == "true"
USE_TOKEN_AWARE = os.getenv("USE_TOKEN_AWARE", "true").lower() == "true"
```

