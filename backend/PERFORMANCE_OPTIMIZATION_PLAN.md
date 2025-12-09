# Time-Optimized Summary Generation & Storage Plan

## 1) Bottlenecks and Reasoning

### Critical Path Analysis

The current pipeline follows: **Parse → Chunk (if >5K chars) → Summarize (parallel) → Generate Titles/Previews → Persist (bulk insert)**. The primary bottleneck is **SQLite single-writer contention** combined with **synchronous blocking writes** that hold the database lock during the entire summarization phase.

**Time Breakdown (50 chapters, 4-10K chars each, typical case):**

- **Parse**: 200-500ms (I/O bound, single-threaded) - **<2% of total**
- **Chunking**: 50-100ms (CPU bound, in-memory) - **<1% of total**
- **Summarization**: 8,000-15,000ms (LLM bound, 3 workers) - **80-90% of total**
  - Short chapters (≤5K): 15-30s each → 250-500s total with 3 workers → **83-166s wall-clock**
  - Long chapters (>5K, chunked): 30-60s per chapter → 500-1000s total → **166-333s wall-clock**
- **Title/Preview Generation**: 2,000-4,000ms (LLM bound, 3 workers) - **10-15% of total**
- **DB Persistence**: 100-300ms (I/O bound, single transaction) - **1-2% of total**
  - **BUT**: Currently blocks entire request until ALL summaries complete

**Root Cause**: The system waits for ALL summaries to complete before writing ANY data, causing:
1. **Writer starvation**: DB lock held by single transaction waiting for 50+ LLM calls
2. **Memory pressure**: All summaries held in memory until batch write
3. **No incremental progress**: User sees "Saving to database" only after 2-5 minutes of LLM work
4. **Ollama queue saturation**: 3 workers compete for GPU/VRAM, causing timeouts when >4 workers

**SQLite Single-Writer Interaction**: With WAL mode, reads can proceed during writes, but only ONE writer can commit at a time. The current bulk insert holds the write lock for 100-300ms, but the **request handler holds the DB session open for the entire 2-5 minute summarization phase**, preventing other operations and creating perceived contention.

### Quantified Bottlenecks

- **LLM latency dominates** (80-90%): 3 workers × 15-30s per chapter = 83-166s for 50 chapters
- **Synchronous blocking** (5-10% overhead): DB session held open during LLM calls
- **Commit fsync** (1-2% but visible): 100-300ms commit time feels slow after 2-5min wait
- **Worker contention** (10-20% underutilization): Ollama timeouts when >3 workers, but 3 workers underutilize GPU during I/O waits

---

## 2) Proposed Architecture

### Producer/Consumer Design

```
┌─────────────┐
│   Upload    │
│  Endpoint   │
└──────┬──────┘
       │
       ├─ Parse chapters → Insert document (fast, <100ms)
       │
       ├─ Enqueue chapters to summary queue
       │
       └─ Return immediately (document_id, status: "processing")
           │
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  Summary Worker Pool (Adaptive, 3-16 workers)      │
    │  - Consumes from: summary_queue (bounded, size=200)│
    │  - Routes by chapter length:                        │
    │    • ≤5K chars → Fast provider (Gemini Flash)        │
    │    • >5K chars → Chunk → Parallel → Combine        │
    │  - Produces to: write_queue (bounded, size=500)   │
    └──────┬──────────────────────────────────────────────┘
           │
           │ (summary, title, preview, chapter_id)
           │
    ┌──────▼──────────────────────────────────────────────┐
    │  Single DB Writer Thread                            │
    │  - Consumes from: write_queue                       │
    │  - Batches: 50-200 rows OR 250-500ms, whichever   │
    │  - Commits: Short transactions (<100ms each)         │
    │  - Backpressure: Pause workers if queue >400       │
    └─────────────────────────────────────────────────────┘
```

### Exact Parameters

**Queue Sizes:**
- `summary_queue`: 200 items (bounded, blocks producers if full)
- `write_queue`: 500 items (bounded, triggers backpressure at 400)

**Batch Commit Policy:**
- **Batch size**: 50-200 rows (whichever fills first)
- **Time threshold**: 250-500ms (whichever comes first)
- **Commit frequency**: Every batch OR every 500ms, whichever first
- **Rationale**: Balance between write lock contention (smaller batches) and commit overhead (larger batches)

**Worker Scaling:**
- **Initial**: 3 workers (safe default)
- **Adaptive**: Scale to 6-12 based on LLM p95 latency and CPU load
- **Hard cap**: 16 workers (existing constraint)
- **Chunk workers**: Max 6 (reduced from current to prevent DB queue saturation)

---

## 3) SQLite Settings and Schema Changes

### PRAGMA Configuration (Execute Once Per Connection)

```python
PRAGMA journal_mode=WAL;              # Concurrent reads during writes
PRAGMA synchronous=NORMAL;             # Balance speed/durability (was already set)
PRAGMA busy_timeout=30000;             # Wait 30s for lock (increased from 5s)
PRAGMA temp_store=MEMORY;              # Temp tables in RAM (was already set)
PRAGMA mmap_size=67108864;             # 64MB memory-mapped I/O (NEW)
PRAGMA page_size=8192;                 # 8KB pages (optimal for modern SSDs)
PRAGMA cache_size=-100000;             # 100MB cache (was already set)
PRAGMA wal_autocheckpoint=2000;        # Less frequent checkpoints (was already set)
PRAGMA checkpoint_fullfsync=OFF;       # Faster fsync (dev only, was already set)
PRAGMA foreign_keys=ON;                # Keep FK enforcement (was already set)
```

**Rationale**: `mmap_size` and `page_size` optimize for sequential writes. `busy_timeout=30000` allows writer queue to drain without errors.

### Schema Adjustments

**No schema changes required** - existing `chapters` table supports batched writes. However, add **idempotency support**:

```sql
-- Add unique constraint for idempotent upserts (if not exists)
CREATE UNIQUE INDEX IF NOT EXISTS ix_chapters_id ON chapters(id);

-- Ensure document_id index exists (already present)
-- CREATE INDEX IF NOT EXISTS ix_chapters_document_id ON chapters(document_id);
```

**Upsert Pattern (SQLAlchemy):**

```python
# Use ON CONFLICT for idempotent writes
from sqlalchemy.dialects.sqlite import insert

stmt = insert(Chapter).values(chapters_dicts)
stmt = stmt.on_conflict_do_update(
    index_elements=['id'],
    set_={
        'summary': stmt.excluded.summary,
        'preview': stmt.excluded.preview,
        'title': stmt.excluded.title
    }
)
db.execute(stmt)
```

**Impact**: Enables retry/resume without duplicate rows.

---

## 4) Concurrency and Adaptive Scaling

### Adaptive Worker Policy

**Target**: `outstanding_requests × p95_latency ≈ 2-3× core_count`

**Algorithm:**
```python
def compute_optimal_workers(p95_latency_s: float, cpu_count: int, queue_depth: int) -> int:
    """
    Adaptive worker scaling based on observed latency and system load.
    
    Args:
        p95_latency_s: 95th percentile LLM response time (seconds)
        cpu_count: CPU cores available
        queue_depth: Current summary_queue depth
    
    Returns:
        Optimal worker count (1-16)
    """
    # Target: Keep 2-3× CPU cores worth of work in flight
    target_in_flight = (2.5 * cpu_count)
    
    # If latency is high, we need more workers to maintain throughput
    # If latency is low, fewer workers suffice
    workers_by_latency = target_in_flight / max(p95_latency_s, 1.0)
    
    # Scale up if queue is backing up
    if queue_depth > 50:
        workers_by_queue = min(16, queue_depth // 10)
    else:
        workers_by_queue = 3
    
    # Take the higher of the two, but cap at 16
    optimal = min(16, max(3, int(max(workers_by_latency, workers_by_queue))))
    
    # Reduce if DB write queue is saturated (backpressure)
    if db_write_queue_depth > 400:
        optimal = max(1, optimal - 2)  # Reduce by 2 to let writer catch up
    
    return optimal
```

**Update Frequency**: Recompute every 10 completed summaries or every 30 seconds.

**Chunk Worker Policy:**
- **Default**: 4 workers (reduced from 6 to prevent DB queue saturation)
- **Scale down**: If `write_queue_depth > 300`, reduce to 2 workers
- **Scale up**: If `write_queue_depth < 100` and `p95_latency < 10s`, increase to 6

### Chunking Policy

**Dynamic Chunk Sizes Based on Model Context:**

```python
def compute_chunk_size(model_name: str, content_length: int) -> Tuple[int, int]:
    """
    Compute optimal chunk size and overlap based on model context window.
    
    Args:
        model_name: LLM model identifier
        content_length: Total content length in characters
    
    Returns:
        (chunk_size, overlap) tuple
    """
    # Model context windows (approximate, in tokens, ~4 chars/token)
    context_windows = {
        'phi3:mini': 4000,      # ~16K chars
        'gemini-1.5-flash': 1000000,  # ~4M chars (huge)
        'gpt-4-turbo': 128000,  # ~512K chars
        'deepseek-chat': 32000, # ~128K chars
    }
    
    context_tokens = context_windows.get(model_name, 4000)
    context_chars = context_tokens * 4  # Rough estimate
    
    # Use 70% of context for input, 30% for output
    max_chunk_size = int(context_chars * 0.7)
    
    # For fast models (Gemini), use larger chunks to reduce overhead
    if 'gemini' in model_name.lower() or 'flash' in model_name.lower():
        chunk_size = min(8000, max_chunk_size)  # Up to 8K chars
        overlap = 300  # Slightly higher overlap for better continuity
    else:
        # Conservative for local models
        chunk_size = min(2000, max_chunk_size)  # Current: 2K chars
        overlap = 250  # Reduced from 400 for speed
    
    # For very long content, increase chunk size to reduce number of chunks
    if content_length > 50000:  # 50K+ chars
        chunk_size = min(chunk_size * 1.5, max_chunk_size)
        overlap = int(overlap * 1.2)
    
    return (int(chunk_size), int(overlap))
```

**Rationale**: Larger chunks for fast models reduce HTTP overhead and chunk merge cost. Smaller overlap (200-300) is sufficient unless domain-specific jargon density is high.

---

## 5) Model Routing for Speed

### Routing Matrix

| Chapter Length | Backlog | Provider | Model | Timeout | Rationale |
|----------------|---------|----------|-------|---------|-----------|
| ≤5,000 chars | Any | **Gemini 1.5 Flash** | `gemini-1.5-flash` | 30s | Fastest, cheap, good quality |
| >5,000 chars | <20 queued | **Ollama** | `phi3:mini` | 180-600s | Local, no cost, chunked |
| >5,000 chars | ≥20 queued | **Gemini 1.5 Flash** | `gemini-1.5-flash` | 60s | Burst to cloud to clear backlog |
| Any | Ollama p95 >25s | **Gemini 1.5 Flash** | `gemini-1.5-flash` | 30-60s | Circuit breaker: switch if local slow |
| Any | Ollama queue >10 | **Gemini 1.5 Flash** | `gemini-1.5-flash` | 30-60s | Burst to cloud when local saturated |

### Timeout Values

```python
TIMEOUTS = {
    'gemini-1.5-flash': {
        'connection': 5,
        'read': 30,  # Fast model, short timeout
        'retries': 2
    },
    'phi3:mini': {
        'connection': 5,
        'read': 180,  # Adaptive 180-600s based on content length
        'retries': 3
    },
    'gpt-4-turbo': {
        'connection': 5,
        'read': 120,
        'retries': 2
    },
    'deepseek-chat': {
        'connection': 5,
        'read': 90,
        'retries': 2
    }
}
```

### Circuit Breaker Thresholds

```python
CIRCUIT_BREAKER = {
    'ollama_p95_threshold_s': 25,      # Switch if p95 > 25s
    'ollama_queue_threshold': 10,      # Switch if queue > 10
    'ollama_error_rate_threshold': 0.2, # Switch if >20% errors
    'cooldown_seconds': 60             # Wait 60s before retrying Ollama
}
```

### Cost Guardrails (if cloud enabled)

```python
COST_LIMITS = {
    'gemini_flash_per_1k_tokens': 0.075,  # $0.075 per 1M input tokens
    'max_daily_cost_usd': 5.0,            # Hard limit: $5/day
    'max_per_request_cost_usd': 0.50,     # Hard limit: $0.50 per upload
    'warn_at_cost_usd': 3.0               # Warn at $3/day
}
```

**Implementation**: Track token usage per provider, reject requests if limits exceeded.

---

## 6) Prompt Templates

### Full Summary Prompt (10-12 Sentences, ≤2,000 chars)

```python
FULL_SUMMARY_PROMPT = """You are summarizing a chapter from a book. Generate a comprehensive summary that meets these EXACT requirements:

1. Write EXACTLY 10-12 complete sentences (no more, no less).
2. Total length: 1,800-2,000 characters (strict maximum: 2,400 characters).
3. Include EVERY key concept, terminology, and takeaway mentioned in the text.
4. Cover ALL main points, arguments, and conclusions.
5. Be thorough and complete - nothing important should be omitted.

Chapter Title: {title}

Chapter Content:
{content}

Generate the summary now. Count your sentences and ensure you write exactly 10-12 sentences."""

FULL_SUMMARY_SYSTEM = """You are an expert at creating detailed, comprehensive summaries. You always write exactly 10-12 sentences and stay within 2,000 characters. You never omit important concepts or terminology."""
```

### 3-Sentence Preview Prompt

```python
PREVIEW_PROMPT = """Extract the 3 most important sentences from this summary that best represent the chapter's main takeaways. These sentences should:
1. Be self-contained and understandable without context
2. Cover the most critical concepts
3. Be exactly 3 sentences (no more, no less)

Summary:
{summary}

Provide exactly 3 sentences:"""

PREVIEW_SYSTEM = """You extract exactly 3 key sentences from summaries. You are concise and precise."""
```

### Combine Prompt (for Chunked Chapters)

```python
COMBINE_PROMPT = """You are merging summaries from multiple sections of a chapter into a single, comprehensive summary. Requirements:

1. Write EXACTLY 10-12 complete sentences (no more, no less).
2. Total length: 1,800-2,000 characters (strict maximum: 2,400 characters).
3. Include ALL key concepts, terminology, and takeaways from ALL sections.
4. Eliminate redundancy - do not repeat the same point multiple times.
5. Ensure smooth flow and coherence between concepts from different sections.

Chapter Title: {title}

Section Summaries:
{chunk_summaries}

Generate the combined summary. Count your sentences and ensure exactly 10-12 sentences."""

COMBINE_SYSTEM = """You merge multiple summaries into one coherent, comprehensive summary. You eliminate redundancy while ensuring complete coverage. You always write exactly 10-12 sentences."""
```

### Early-Stop Logic

```python
def should_stop_generation(text: str, target_sentences: int = 12, max_chars: int = 2000) -> bool:
    """Check if generation should stop early."""
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences >= target_sentences:
        return True
    if len(text) >= max_chars:
        return True
    return False

# Provider-specific stop sequences
STOP_SEQUENCES = {
    'ollama': ['\n\n\n', '---', '###'],  # Ollama-specific
    'gemini': ['\n\n\n'],
    'openai': ['\n\n\n', '---'],
    'deepseek': ['\n\n\n']
}

# Temperature settings for deterministic output
TEMPERATURE = {
    'ollama': 0.2,      # Low temperature for consistency
    'gemini': 0.1,      # Very low for exact sentence count
    'openai': 0.3,
    'deepseek': 0.2
}
```

---

## 7) Pseudocode for Pipeline

### Producer/Consumer with DB Writer

```python
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class SummaryTask:
    chapter_id: str
    document_id: str
    content: str
    title: str
    chapter_number: int

@dataclass
class WriteTask:
    chapter_id: str
    document_id: str
    title: str
    summary: str
    preview: str
    content: str
    chapter_number: int

# Global queues
summary_queue = queue.Queue(maxsize=200)  # Bounded: blocks if full
write_queue = queue.Queue(maxsize=500)   # Bounded: triggers backpressure

# Metrics for adaptive scaling
llm_latencies = deque(maxlen=100)  # Track last 100 latencies
db_write_queue_depth = 0
summary_queue_depth = 0

def summary_worker(worker_id: int):
    """Worker thread that generates summaries."""
    while True:
        try:
            task = summary_queue.get(timeout=1)
            if task is None:  # Poison pill
                break
            
            t0 = time.perf_counter()
            
            # Route by chapter length
            if len(task.content) <= 5000:
                # Short: use fast provider
                provider = route_to_fast_provider()
                summary = generate_summary_single_pass(task.content, task.title, provider)
            else:
                # Long: chunk and combine
                chunks = split_content_into_chunks(task.content, ...)
                chunk_summaries = summarize_chunks_parallel(chunks, max_workers=4)
                summary = combine_chunk_summaries(chunk_summaries, task.title)
            
            # Generate title and preview
            title, preview = generate_title_and_preview(summary)
            
            latency = time.perf_counter() - t0
            llm_latencies.append(latency)
            
            # Enqueue for DB write
            write_task = WriteTask(
                chapter_id=task.chapter_id,
                document_id=task.document_id,
                title=title,
                summary=summary,
                preview=preview,
                content=task.content,
                chapter_number=task.chapter_number
            )
            
            # Backpressure: block if write queue is full
            write_queue.put(write_task, block=True)
            
            summary_queue.task_done()
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            # Fallback: truncate and write
            write_task = WriteTask(
                chapter_id=task.chapter_id,
                document_id=task.document_id,
                title=task.title,
                summary=task.content[:200] + "...",
                preview=None,
                content=task.content,
                chapter_number=task.chapter_number
            )
            write_queue.put(write_task)

def db_writer():
    """Single-threaded DB writer with batching."""
    batch = []
    last_commit = time.perf_counter()
    batch_size_threshold = 50  # Commit every 50 rows
    time_threshold_ms = 500    # Or every 500ms
    
    db = next(get_db_session())
    
    try:
        while True:
            try:
                # Get task with timeout to allow periodic commits
                task = write_queue.get(timeout=0.5)
                if task is None:  # Poison pill
                    # Flush remaining batch
                    if batch:
                        commit_batch(db, batch)
                    break
                
                batch.append(task)
                write_queue.task_done()
                
                # Check commit conditions
                now = time.perf_counter()
                time_since_commit = (now - last_commit) * 1000
                
                if len(batch) >= batch_size_threshold or time_since_commit >= time_threshold_ms:
                    commit_batch(db, batch)
                    batch = []
                    last_commit = now
                
                # Update queue depth for backpressure
                db_write_queue_depth = write_queue.qsize()
                
            except queue.Empty:
                # Timeout: commit if batch has items and time threshold met
                if batch:
                    now = time.perf_counter()
                    time_since_commit = (now - last_commit) * 1000
                    if time_since_commit >= time_threshold_ms:
                        commit_batch(db, batch)
                        batch = []
                        last_commit = now
                        
    except Exception as e:
        print(f"DB writer error: {e}")
        db.rollback()
    finally:
        db.close()

def commit_batch(db, batch: List[WriteTask]):
    """Commit a batch of writes using bulk insert."""
    if not batch:
        return
    
    t0 = time.perf_counter()
    
    # Prepare dicts for bulk insert
    chapters_dicts = [{
        'id': task.chapter_id,
        'document_id': task.document_id,
        'title': task.title,
        'content': task.content,
        'summary': task.summary,
        'preview': task.preview,
        'chapter_number': task.chapter_number
    } for task in batch]
    
    # Use ON CONFLICT for idempotency
    from sqlalchemy.dialects.sqlite import insert
    stmt = insert(Chapter).values(chapters_dicts)
    stmt = stmt.on_conflict_do_update(
        index_elements=['id'],
        set_={
            'summary': stmt.excluded.summary,
            'preview': stmt.excluded.preview,
            'title': stmt.excluded.title
        }
    )
    
    db.execute(stmt)
    db.commit()
    
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"DB: Committed batch of {len(batch)} chapters in {elapsed:.2f} ms")

def adaptive_worker_manager():
    """Adjust worker count based on metrics."""
    while True:
        time.sleep(30)  # Recompute every 30s
        
        if len(llm_latencies) < 10:
            continue  # Not enough data
        
        p95_latency = sorted(llm_latencies)[int(len(llm_latencies) * 0.95)]
        cpu_count = os.cpu_count() or 4
        queue_depth = summary_queue.qsize()
        
        optimal_workers = compute_optimal_workers(p95_latency, cpu_count, queue_depth)
        
        current_workers = len(worker_threads)
        
        if optimal_workers > current_workers:
            # Scale up
            for i in range(optimal_workers - current_workers):
                t = threading.Thread(target=summary_worker, args=(current_workers + i,))
                t.start()
                worker_threads.append(t)
        elif optimal_workers < current_workers:
            # Scale down (graceful: stop accepting new tasks)
            for i in range(current_workers - optimal_workers):
                summary_queue.put(None)  # Poison pill

# Startup
worker_threads = []
db_writer_thread = threading.Thread(target=db_writer, daemon=True)
db_writer_thread.start()

adaptive_manager_thread = threading.Thread(target=adaptive_worker_manager, daemon=True)
adaptive_manager_thread.start()

# Initial workers
for i in range(3):
    t = threading.Thread(target=summary_worker, args=(i,))
    t.start()
    worker_threads.append(t)
```

---

## 8) Fault Tolerance and Idempotency

### Content Hashing for Deduplication

```python
import hashlib

def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Add hash column to chapters table (migration)
# ALTER TABLE chapters ADD COLUMN content_hash TEXT;
# CREATE INDEX IF NOT EXISTS ix_chapters_content_hash ON chapters(content_hash);

def check_duplicate_summary(db, document_id: str, content_hash: str) -> Optional[str]:
    """Check if summary already exists for this content."""
    existing = db.query(Chapter).filter(
        Chapter.document_id == document_id,
        Chapter.content_hash == content_hash
    ).first()
    return existing.summary if existing and existing.summary else None
```

### Idempotent Upserts

```python
# Use ON CONFLICT for idempotent writes (see pseudocode section 7)
# Key: (document_id, chapter_id) or just (id)

# Retry policy for DB busy errors
def retry_db_operation(func, max_retries=3, backoff_base=0.1):
    """Retry DB operation with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlalchemy.exc.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                wait = backoff_base * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(wait)
                continue
            raise
```

### Partial Progress Persistence

```python
# Track progress in database
# ALTER TABLE documents ADD COLUMN summary_progress INTEGER DEFAULT 0;
# ALTER TABLE documents ADD COLUMN summary_total INTEGER DEFAULT 0;

def update_progress(db, document_id: str, completed: int, total: int):
    """Update summary generation progress."""
    db.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(summary_progress=completed, summary_total=total)
    )
    db.commit()

# Resume on restart: query chapters without summaries for a document
def resume_summarization(document_id: str):
    """Resume summarization for a document."""
    db = next(get_db_session())
    chapters = db.query(Chapter).filter(
        Chapter.document_id == document_id,
        (Chapter.summary == None) | (Chapter.summary == '')
    ).all()
    
    for chapter in chapters:
        task = SummaryTask(
            chapter_id=chapter.id,
            document_id=document_id,
            content=chapter.content,
            title=chapter.title,
            chapter_number=chapter.chapter_number
        )
        summary_queue.put(task)
```

### Backoff and Jitter

```python
def exponential_backoff_with_jitter(attempt: int, base: float = 1.0) -> float:
    """Exponential backoff with jitter."""
    wait = base * (2 ** attempt)
    jitter = random.uniform(0, wait * 0.1)  # 10% jitter
    return wait + jitter

# Apply to:
# - LLM retries: base=1.0s, max=30s
# - DB busy retries: base=0.1s, max=5s
# - Queue full backpressure: base=0.5s, max=10s
```

---

## 9) Observability and Success Metrics

### Metrics to Track

```python
METRICS = {
    # Throughput
    'summaries_per_minute': 0,
    'chapters_processed_total': 0,
    'chapters_failed_total': 0,
    
    # Latency (p50, p95, p99)
    'llm_latency_p50_ms': 0,
    'llm_latency_p95_ms': 0,
    'llm_latency_p99_ms': 0,
    'db_write_latency_p50_ms': 0,
    'db_write_latency_p95_ms': 0,
    
    # Queue depths
    'summary_queue_depth': 0,
    'write_queue_depth': 0,
    
    # Error rates
    'llm_timeout_rate': 0,
    'llm_error_rate': 0,
    'db_busy_rate': 0,
    
    # Worker utilization
    'active_workers': 0,
    'optimal_workers': 0,
    'worker_utilization_pct': 0,
    
    # Provider usage
    'ollama_requests': 0,
    'gemini_requests': 0,
    'cloud_cost_usd': 0.0
}

def log_metrics():
    """Log metrics every 60 seconds."""
    while True:
        time.sleep(60)
        print(f"METRICS: {json.dumps(METRICS, indent=2)}")
```

### A/B Testing Plan

**Phase 1 (Day 1)**: Deploy producer/consumer with 3 workers, measure baseline
- **Control**: Current system (synchronous, 3 workers)
- **Treatment**: New system (async, 3 workers, batched writes)
- **Metric**: End-to-end time for 50-chapter upload

**Phase 2 (Day 2-3)**: Enable adaptive scaling (3-8 workers)
- **Metric**: Throughput (summaries/min) and p95 latency

**Phase 3 (Day 4-5)**: Enable model routing (Gemini for short chapters)
- **Metric**: Time to first summary and total completion time

### 3-Step Rollout

1. **Step 1**: Deploy producer/consumer architecture with fixed 3 workers, keep current system as fallback
   - **Feature flag**: `USE_ASYNC_SUMMARIZATION=false` (default: false)
   - **Rollback**: Set flag to false, system reverts to synchronous

2. **Step 2**: Enable adaptive scaling (3-12 workers) after 24h of stable operation
   - **Feature flag**: `ENABLE_ADAPTIVE_WORKERS=true`
   - **Rollback**: Set to false, reverts to fixed 3 workers

3. **Step 3**: Enable model routing (Gemini for short chapters) after 48h
   - **Feature flag**: `ENABLE_MODEL_ROUTING=true`
   - **Rollback**: Set to false, reverts to Ollama-only

### Fallback Switch

```python
# In main.py upload endpoint
if os.getenv("USE_ASYNC_SUMMARIZATION", "false").lower() == "true":
    # New async path
    enqueue_chapters_for_async_processing(document_id, chapters_data)
    return {"status": "processing", "document_id": document_id}
else:
    # Old synchronous path (fallback)
    summaries = generate_summaries_parallel(chapters_data, max_workers=3)
    # ... existing code ...
```

---

## 10) Estimated Speedup and Day-1 Checklist

### Expected Speedups

| Change | Speedup | Rationale |
|--------|---------|-----------|
| **WAL + batched commits** | 2-5× faster DB writes | Reduce commit overhead from 100-300ms to 20-60ms per batch |
| **Single-writer queue** | Eliminates N-way contention | No more DB lock held during LLM calls |
| **Adaptive workers (3→8)** | +20-40% utilization | Better GPU/CPU utilization, fewer timeouts |
| **Model routing (Gemini for short)** | 3-5× faster for ≤5K chapters | Gemini Flash: 2-5s vs Ollama: 15-30s |
| **Reduced chunk overlap (400→250)** | 10-15% faster chunking | Less redundant processing |
| **Early-stop generation** | 5-10% faster LLM calls | Stop at 12 sentences instead of max tokens |
| **Total expected** | **3-6× faster end-to-end** | Combined: 2-5min → 30-60s for 50 chapters |

### Day-1 Implementation Checklist

**Phase 1: Infrastructure (2-3 hours)**
- [x] Add PRAGMA settings to `database.py` (mmap_size, page_size, busy_timeout)
- [x] Create `summary_queue` and `write_queue` (bounded queues)
- [x] Implement `db_writer()` thread with batching logic
- [x] Add idempotent upsert pattern (ON CONFLICT)
- [x] Add content_hash column migration
- [x] Test DB writer with synthetic data (50-200 row batches)

**Phase 2: Producer/Consumer (3-4 hours)**
- [x] Implement `summary_worker()` thread
- [x] Implement `adaptive_worker_manager()` thread
- [x] Add metrics tracking (latency, queue depth, errors)
- [x] Add feature flag `USE_ASYNC_SUMMARIZATION`
- [x] Modify upload endpoint to enqueue chapters
- [x] Test with 10-chapter upload

**Phase 3: Model Routing (2-3 hours)**
- [x] Implement routing logic (chapter length → provider)
- [x] Add Gemini 1.5 Flash integration (if not exists)
- [x] Implement circuit breaker (Ollama p95 >25s → switch)
- [x] Add cost tracking and guardrails
- [x] Test routing with mixed chapter lengths

**Phase 4: Prompt Optimization (1-2 hours)**
- [x] Update prompts with exact sentence count requirements
- [x] Add early-stop logic (sentence counting)
- [x] Add provider-specific stop sequences
- [ ] Test prompt output (verify 10-12 sentences)

**Phase 5: Testing & Validation (2-3 hours)**
- [x] Run diagnostic script (`diagnose_db_performance.py`)
- [ ] Test with 50-chapter upload (measure end-to-end time)
- [ ] Verify metrics logging
- [ ] Test fallback (disable feature flag)
- [ ] Test resume/resume (restart with partial progress)

**Total Estimated Time**: 10-15 hours (1.5-2 days)

### Validation Checklist

- [ ] **Baseline**: Measure current system (50 chapters): _____ seconds
- [ ] **After Phase 1**: DB write time per batch: <60ms for 50 rows
- [ ] **After Phase 2**: End-to-end time (50 chapters): <90 seconds (target: 50% improvement)
- [ ] **After Phase 3**: Short chapters (≤5K) use Gemini: <10s per chapter
- [ ] **After Phase 4**: All summaries have 10-12 sentences (verify 10 random samples)
- [ ] **Metrics**: p95 latency <20s, queue depth <100, error rate <5%
- [ ] **Rollback**: Feature flag works, system reverts to synchronous mode

### Success Criteria

- **End-to-end time**: 50 chapters in <90 seconds (vs current 2-5 minutes)
- **First summary**: Available in <15 seconds (vs current 15-30s)
- **DB write time**: <60ms per batch of 50 rows
- **Error rate**: <5% (timeouts, DB busy, etc.)
- **Worker utilization**: 60-80% (vs current 30-40%)

---

## Implementation Files

1. **`backend/summary_pipeline.py`** - Producer/consumer architecture
2. **`backend/summary_worker.py`** - Worker threads and adaptive scaling
3. **`backend/db_writer.py`** - Batched DB writer
4. **`backend/model_router.py`** - Provider routing logic
5. **`backend/prompts.py`** - Optimized prompt templates
6. **`backend/metrics.py`** - Observability and logging
7. **`backend/migrations/add_content_hash.py`** - Schema migration

---

## Notes

- **Risk**: Producer/consumer adds complexity. Mitigation: Feature flag for gradual rollout.
- **Risk**: Model routing increases cloud costs. Mitigation: Cost guardrails and daily limits.
- **Risk**: Adaptive scaling may cause worker thrashing. Mitigation: 30s update frequency, hysteresis (scale up faster than down).

---

**End of Plan**

