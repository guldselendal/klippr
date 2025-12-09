"""
Async summary generation pipeline using producer/consumer pattern.
Decouples LLM work from database writes for better concurrency.
"""
import queue
import threading
import time
import os
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import deque
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from summarizer import generate_summaries_parallel, process_summaries_for_titles_and_previews
from summarizer import generate_summary, process_summary_for_chapter, merge_chunk_summaries
from pipeline_metrics import (
    provider_metrics, summary_queue_metrics, write_queue_metrics,
    worker_metrics, db_metrics, get_all_metrics
)
from prompt_utils import (
    FULL_SUMMARY_PROMPT_TEMPLATE, FULL_SUMMARY_SYSTEM,
    COMBINE_PROMPT_TEMPLATE, COMBINE_SYSTEM,
    count_sentences, truncate_to_sentences
)


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


@dataclass
class SummaryTask:
    """Task for summary generation"""
    chapter_id: str
    document_id: str
    content: str
    title: str
    chapter_number: int
    original_title: str


@dataclass
class WriteTask:
    """Task for database write"""
    chapter_id: str
    document_id: str
    title: str
    summary: Optional[str]
    preview: Optional[str]
    content: str
    chapter_number: int


# Global queues
summary_queue = queue.Queue(maxsize=200)  # Bounded: blocks if full
write_queue = queue.Queue(maxsize=500)   # Bounded: triggers backpressure at 400

# Metrics for adaptive scaling
llm_latencies = deque(maxlen=100)  # Track last 100 latencies
ollama_latencies = deque(maxlen=50)  # Track Ollama-specific latencies
gemini_latencies = deque(maxlen=50)  # Track Gemini-specific latencies
db_write_queue_depth = 0
summary_queue_depth = 0
ollama_error_count = 0  # Track Ollama errors for circuit breaker
completed_summaries_count = 0  # Track completed summaries for adaptive scaling
last_adaptive_check = time.time()  # Track last adaptive scaling check

# Worker management
worker_threads = []
db_writer_thread = None
adaptive_manager_thread = None
shutdown_event = threading.Event()


def select_provider_and_model(content_length: int, queue_depth: int) -> tuple[str, str]:
    """
    Select LLM provider and model based on routing matrix.
    
    Routing logic:
    - ≤5K chars → Gemini Flash (fastest)
    - >5K chars, <20 queued → Ollama (local, free)
    - >5K chars, ≥20 queued → Gemini Flash (burst to cloud)
    - Circuit breaker: Switch to Gemini if Ollama p95 >25s or queue >10
    
    Returns:
        (provider, model) tuple
    """
    global ollama_latencies, ollama_error_count
    
    # Circuit breaker thresholds
    OLLAMA_P95_THRESHOLD = 25.0  # seconds
    OLLAMA_QUEUE_THRESHOLD = 10
    OLLAMA_ERROR_RATE_THRESHOLD = 0.2
    
    # Calculate Ollama p95 latency if we have data
    ollama_p95 = None
    if len(ollama_latencies) >= 10:
        sorted_latencies = sorted(ollama_latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        ollama_p95 = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
    
    # Circuit breaker: Switch to Gemini if Ollama is slow or saturated
    if ollama_p95 and ollama_p95 > OLLAMA_P95_THRESHOLD:
        return ("gemini", "gemini-1.5-flash")
    
    if queue_depth > OLLAMA_QUEUE_THRESHOLD:
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


def generate_summary_with_routing(content: str, title: str, provider: str, model: str, timeout: int = 300) -> str:
    """
    Generate summary using specified provider and model.
    Handles chunking for long content with Ollama.
    
    Args:
        content: Chapter content
        title: Chapter title
        provider: LLM provider name
        model: Model name
        timeout: Maximum time in seconds for LLM call (default: 300s = 5min)
    
    Returns:
        Generated summary
    
    Raises:
        TimeoutError: If LLM call exceeds timeout
        Exception: Other LLM errors
    """
    from llm_provider import call_llm
    from summarizer import split_content_into_chunks, merge_chunk_summaries
    
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200  # Reduced from 250 to 200 for optimization (Step 1.2)
    
    def _call_llm_with_timeout(prompt: str, system_prompt: str, provider: str, model: str) -> str:
        """Wrapper to call LLM with timeout"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(call_llm, prompt=prompt, system_prompt=system_prompt, provider=provider, model=model)
            try:
                return future.result(timeout=timeout).strip()
            except FutureTimeoutError:
                executor.shutdown(wait=False)
                raise TimeoutError(f"LLM call to {provider} exceeded {timeout}s timeout")
    
    # Short content: single pass
    if len(content) <= 5000:
        prompt = FULL_SUMMARY_PROMPT_TEMPLATE.format(title=title, content=content)
        
        system_prompt = FULL_SUMMARY_SYSTEM
        
        try:
            summary = _call_llm_with_timeout(prompt, system_prompt, provider, model)
            # Post-process to ensure exactly 10-12 sentences
            sentence_count = count_sentences(summary)
            if sentence_count > 12:
                summary = truncate_to_sentences(summary, max_sentences=12)
            elif sentence_count < 10 and len(summary) < 2000:
                # If too few sentences but room to grow, note it but don't fail
                print(f"Warning: Summary has only {sentence_count} sentences (target: 10-12)")
            return summary
        except TimeoutError:
            raise
        except Exception as e:
            print(f"Error generating summary with {provider}: {e}")
            raise
    
    # Long content: chunk and process in parallel (only for Ollama, Gemini handles long content natively)
    if provider == "ollama":
        chunks = split_content_into_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Summarize chunks in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        chunk_summaries = []
        # Increased to 16 to match Ollama concurrency limit (Step 1.4)
        with ThreadPoolExecutor(max_workers=min(16, len(chunks))) as executor:
            for idx, chunk in enumerate(chunks):
                future = executor.submit(
                    summarize_chunk_with_provider,
                    chunk, idx, len(chunks), title, provider, model
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    chunk_summaries.append(future.result())
                except Exception as e:
                    print(f"Error summarizing chunk: {e}")
                    chunk_summaries.append("")  # Empty summary for failed chunk
        
        # Merge chunk summaries
        return merge_chunk_summaries(chunk_summaries, title)
    else:
        # Gemini can handle long content directly
        prompt = FULL_SUMMARY_PROMPT_TEMPLATE.format(title=title, content=content)
        
        system_prompt = FULL_SUMMARY_SYSTEM
        
        try:
            summary = _call_llm_with_timeout(prompt, system_prompt, provider, model)
            # Post-process to ensure exactly 10-12 sentences
            sentence_count = count_sentences(summary)
            if sentence_count > 12:
                summary = truncate_to_sentences(summary, max_sentences=12)
            elif sentence_count < 10 and len(summary) < 2000:
                # If too few sentences but room to grow, note it but don't fail
                print(f"Warning: Summary has only {sentence_count} sentences (target: 10-12)")
            return summary
        except TimeoutError:
            raise
        except Exception as e:
            print(f"Error generating summary with {provider}: {e}")
            raise


def summarize_chunk_with_provider(chunk: str, chunk_index: int, total_chunks: int, title: str, provider: str, model: str) -> str:
    """Summarize a single chunk using specified provider with compact prompt and caching."""
    from llm_provider import call_llm
    from prompt_utils import CHUNK_PROMPT_COMPACT, CHUNK_SYSTEM_COMPACT, parse_chunk_summary_response
    from chunk_cache import get_cached_summary, cache_summary
    
    # Use compact prompt for optimization (Step 1.1)
    prompt_template = CHUNK_PROMPT_COMPACT
    
    # Check cache first (Step 2.2)
    cached = get_cached_summary(chunk, prompt_template, model)
    if cached:
        print(f"  Cache hit for chunk {chunk_index + 1}", flush=True)
        return cached.get("summary", "")
    
    prompt = prompt_template.format(
        idx=chunk_index + 1,
        total=total_chunks,
        title=title,
        chunk=chunk
    )
    
    try:
        response = call_llm(prompt=prompt, system_prompt=CHUNK_SYSTEM_COMPACT, provider=provider, model=model)
        
        # Parse JSON response
        parsed = parse_chunk_summary_response(response.strip())
        
        # Cache the parsed summary (Step 2.2)
        cache_summary(chunk, prompt_template, model, parsed)
        
        return parsed.get("summary", response[:120] if len(response) > 120 else response)
    except Exception as e:
        print(f"Error summarizing chunk {chunk_index + 1}: {e}")
        return chunk[:500] + "..." if len(chunk) > 500 else chunk


def summary_worker(worker_id: int):
    """
    Worker thread that generates summaries with intelligent routing.
    
    Implements routing matrix:
    - Routes by chapter length (≤5K → Gemini Flash, >5K → Ollama or Gemini based on backlog)
    - Circuit breaker: Switches to Gemini if Ollama is slow or saturated
    - Handles chunking for long content with Ollama
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
            
            provider = None  # Initialize for exception handling
            
            try:
                # Update queue metrics
                summary_queue_metrics.update_size(summary_queue.qsize())
                write_queue_metrics.update_size(write_queue.qsize())
                
                # Check backpressure: pause if write queue is too full
                global db_write_queue_depth
                db_write_queue_depth = write_queue.qsize()
                if db_write_queue_depth > 400:
                    print(f"Worker {worker_id}: Write queue saturated ({db_write_queue_depth}), waiting...", flush=True)
                    time.sleep(0.5)
                    summary_queue.task_done()
                    continue
                
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
                    summary = generate_summary_with_routing(task.content, task.title, provider, model, timeout=int(llm_timeout))
                    llm_latency = time.perf_counter() - llm_start
                    print(f"Worker {worker_id}: ✓ Summary generated for chapter {task.chapter_id[:8]} "
                          f"({len(summary)} chars, {llm_latency:.1f}s)", flush=True)
                except TimeoutError as timeout_err:
                    llm_latency = time.perf_counter() - llm_start
                    print(f"Worker {worker_id}: ⏱️  Timeout for chapter {task.chapter_id[:8]} "
                          f"after {llm_latency:.1f}s: {timeout_err}", flush=True)
                    # Use fallback summary on timeout
                    summary = f"Summary generation timed out. Content preview: {task.content[:500]}..."
                    provider_metrics.record_error(provider, str(timeout_err))
                except Exception as llm_error:
                    llm_latency = time.perf_counter() - llm_start
                    print(f"Worker {worker_id}: ❌ LLM error for chapter {task.chapter_id[:8]} "
                          f"after {llm_latency:.1f}s: {llm_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Use fallback summary on error
                    summary = f"Summary generation failed. Content preview: {task.content[:500]}..."
                    provider_metrics.record_error(provider, str(llm_error))
                
                # Check deadline before continuing
                elapsed = time.perf_counter() - task_start_time
                if elapsed > TASK_DEADLINE_SECONDS:
                    print(f"Worker {worker_id}: ⚠️  Task exceeded deadline ({elapsed:.1f}s), "
                          f"skipping post-processing", flush=True)
                    # Still try to write what we have
                
                # Track latency by provider
                llm_latency = time.perf_counter() - llm_start
                llm_latencies.append(llm_latency)
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
                
                # Generate title and preview from summary
                if summary and len(summary.strip()) > 0:
                    takeaway_title, preview = process_summary_for_chapter(summary)
                    final_title = takeaway_title if takeaway_title else task.original_title
                else:
                    final_title = task.original_title
                    preview = None
                
                # Reset error count on success
                if provider == "ollama":
                    ollama_error_count = max(0, ollama_error_count - 1)
                
                # Enqueue for DB write
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
                print(f"Worker {worker_id} error processing chapter {task.chapter_id if task else 'unknown'}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                
                # Track errors for circuit breaker and metrics
                if provider == "ollama":
                    ollama_error_count += 1
                
                # Track error metrics
                if provider:
                    provider_metrics[provider].add_error(str(type(e).__name__))
                worker_metrics.record_failure()
                
                # Fallback: write with truncated content
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


def db_writer():
    """
    Single-threaded DB writer with batching logic.
    
    Batch commit policy:
    - Batch size: 50-200 rows (commits when batch reaches 50, or up to 200 if time threshold not met)
    - Time threshold: 250-500ms (commits every 250ms minimum, or up to 500ms if batch not full)
    - Commits when either condition is met (whichever comes first)
    """
    from database import get_db_session
    from models import Chapter
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy import text
    
    print("DB writer started")
    
    batch = []
    last_commit = time.perf_counter()
    
    # Batch commit parameters (matching plan requirements)
    batch_size_min = 50   # Minimum batch size before committing
    batch_size_max = 200  # Maximum batch size (commit before reaching this)
    time_threshold_min_ms = 250  # Minimum time between commits (250ms)
    time_threshold_max_ms = 500  # Maximum time between commits (500ms)
    
    # Use configurable thresholds from environment or defaults
    batch_size_threshold = int(os.getenv("DB_BATCH_SIZE_MIN", batch_size_min))
    batch_size_max_threshold = int(os.getenv("DB_BATCH_SIZE_MAX", batch_size_max))
    time_threshold_ms = int(os.getenv("DB_BATCH_TIME_MS", time_threshold_min_ms))
    
    # Ensure thresholds are within valid ranges
    batch_size_threshold = max(batch_size_min, min(batch_size_threshold, batch_size_max))
    batch_size_max_threshold = max(batch_size_min, min(batch_size_max_threshold, batch_size_max))
    time_threshold_ms = max(time_threshold_min_ms, min(time_threshold_ms, time_threshold_max_ms))
    
    db = next(get_db_session())
    
    try:
        while not shutdown_event.is_set():
            try:
                # Get task with timeout to allow periodic commits
                # Use shorter timeout to check time threshold more frequently
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
                
                # Commit if:
                # 1. Batch reached minimum size (50) OR
                # 2. Batch reached maximum size (200) OR
                # 3. Time threshold met (250-500ms)
                should_commit = (
                    batch_size >= batch_size_threshold or
                    batch_size >= batch_size_max_threshold or
                    time_since_commit >= time_threshold_ms
                )
                
                if should_commit:
                    commit_batch(db, batch)
                    batch = []
                    last_commit = now
                
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
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()
        print("DB writer stopped")


def commit_batch(db, batch: List[WriteTask]):
    """
    Commit a batch of writes using idempotent upsert pattern (ON CONFLICT).
    
    Supports both PostgreSQL and SQLite with proper ON CONFLICT handling.
    This enables retry/resume without duplicate rows.
    """
    if not batch:
        return
    
    from models import Chapter
    from database import IS_POSTGRESQL, IS_SQLITE
    
    t0 = time.perf_counter()
    
    # Prepare dicts for bulk insert with content_hash
    chapters_dicts = [{
        'id': task.chapter_id,
        'document_id': task.document_id,
        'title': task.title,
        'content': task.content,
        'summary': task.summary,
        'preview': task.preview,
        'content_hash': compute_content_hash(task.content),  # Compute hash for deduplication
        'chapter_number': task.chapter_number
    } for task in batch]
    
    # Use ON CONFLICT for idempotency (database-specific implementation)
    try:
        if IS_POSTGRESQL:
            # PostgreSQL: Use PostgreSQL-specific insert with ON CONFLICT
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            
            stmt = pg_insert(Chapter).values(chapters_dicts)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'summary': stmt.excluded.summary,
                    'preview': stmt.excluded.preview,
                    'title': stmt.excluded.title,
                    'content': stmt.excluded.content,
                    'content_hash': stmt.excluded.content_hash,
                    'chapter_number': stmt.excluded.chapter_number
                }
            )
            db.execute(stmt)
            db.commit()
            
        elif IS_SQLITE:
            # SQLite: Use SQLite-specific insert with ON CONFLICT
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            
            stmt = sqlite_insert(Chapter).values(chapters_dicts)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                'summary': stmt.excluded.summary,
                'preview': stmt.excluded.preview,
                'title': stmt.excluded.title,
                    'content': stmt.excluded.content,
                    'content_hash': stmt.excluded.content_hash,
                    'chapter_number': stmt.excluded.chapter_number
                }
                )
            db.execute(stmt)
            db.commit()
        else:
            # Fallback: Use bulk_insert_mappings (not idempotent, but works)
            # This should not happen if database is properly configured
            print("Warning: Unknown database type, using non-idempotent bulk insert")
            db.bulk_insert_mappings(Chapter, chapters_dicts)
            db.commit()
            
    except Exception as e:
        print(f"Error in commit_batch: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    
    elapsed = (time.perf_counter() - t0) * 1000
    
    # Track database metrics
    db_metrics.record_batch(len(batch), elapsed)
    write_queue_metrics.record_dequeue()
    
    print(f"DB: Committed batch of {len(batch)} chapters in {elapsed:.2f} ms (idempotent upsert)")


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
    global db_write_queue_depth
    
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


def adaptive_worker_manager():
    """
    Adaptive worker manager that adjusts worker count based on metrics.
    
    Recomputes every 10 completed summaries OR every 30 seconds, whichever comes first.
    Scales workers based on:
    - LLM p95 latency
    - CPU core count
    - Queue depth
    - DB write queue backpressure
    """
    global worker_threads, completed_summaries_count, last_adaptive_check
    
    print("Adaptive worker manager started")
    
    ADAPTIVE_CHECK_INTERVAL = 30.0  # Maximum seconds between checks
    ADAPTIVE_CHECK_COUNT = 10  # Check after this many completed summaries
    
    while not shutdown_event.is_set():
        try:
            # Check if we should recompute (every 10 summaries OR every 30 seconds)
            time_since_check = time.time() - last_adaptive_check
            should_check = (
                completed_summaries_count >= ADAPTIVE_CHECK_COUNT or
                time_since_check >= ADAPTIVE_CHECK_INTERVAL
            )
            
            if not should_check:
                # Sleep briefly and check again
                time.sleep(1)
                continue
            
            # Reset counters
            completed_summaries_count = 0
            last_adaptive_check = time.time()
            
            # Need at least some data to make decisions
            if len(llm_latencies) < 5:
                print(f"Adaptive manager: Insufficient data ({len(llm_latencies)} latencies), waiting...")
                time.sleep(5)
                continue
            
            # Calculate p95 latency
            sorted_latencies = sorted(llm_latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
            
            cpu_count = os.cpu_count() or 4
            queue_depth = summary_queue.qsize()
            
            optimal_workers = compute_optimal_workers(p95_latency, cpu_count, queue_depth)
            
            # Clean up dead threads (update global list)
            global worker_threads
            worker_threads = [t for t in worker_threads if t.is_alive()]
            current_workers = len(worker_threads)
            
            # Update worker metrics
            worker_metrics.update_active_workers(current_workers)
        
            if optimal_workers > current_workers:
                # Scale up: add new workers
                workers_to_add = optimal_workers - current_workers
                for i in range(workers_to_add):
                    worker_id = current_workers + i
                    t = threading.Thread(target=summary_worker, args=(worker_id,), daemon=True)
                t.start()
                worker_threads.append(t)
                worker_metrics.total_workers_created += 1
                print(f"✓ Scaled up: {current_workers} → {optimal_workers} workers "
                      f"(p95: {p95_latency:.2f}s, queue: {queue_depth}, CPU: {cpu_count})")
                
            elif optimal_workers < current_workers and current_workers > 3:
                # Scale down: send poison pills to excess workers
                workers_to_remove = current_workers - optimal_workers
                for _ in range(workers_to_remove):
                    summary_queue.put(None)  # Poison pill
                print(f"✓ Scaled down: {current_workers} → {optimal_workers} workers "
                      f"(p95: {p95_latency:.2f}s, queue: {queue_depth})")
                
            else:
                # No change needed
                if current_workers != optimal_workers:
                    print(f"ℹ️  Workers: {current_workers} (optimal: {optimal_workers}, "
                          f"p95: {p95_latency:.2f}s, queue: {queue_depth})")
            
            # Brief sleep before next check
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in adaptive worker manager: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)  # Wait before retrying
    
    print("Adaptive worker manager stopped")


def enqueue_chapters_for_processing(document_id: str, chapters_data: List[Dict[str, str]]) -> List[str]:
    """
    Enqueue chapters for async processing.
    
    Args:
        document_id: Document ID
        chapters_data: List of dicts with 'content' and 'title' keys
    
    Returns:
        List of chapter IDs
    """
    chapter_ids = []
    
    for idx, chapter_data in enumerate(chapters_data):
        chapter_id = str(uuid.uuid4())
        chapter_ids.append(chapter_id)
        
        task = SummaryTask(
            chapter_id=chapter_id,
            document_id=document_id,
            content=chapter_data.get('content', ''),
            title=chapter_data.get('title', f'Chapter {idx + 1}'),
            chapter_number=idx + 1,
            original_title=chapter_data.get('title', f'Chapter {idx + 1}')
        )
        
        # Enqueue (will block if queue is full)
        summary_queue.put(task)
        summary_queue_metrics.record_enqueue()
    
    return chapter_ids


def initialize_pipeline():
    """Initialize the async pipeline with initial workers."""
    global worker_threads, db_writer_thread, adaptive_manager_thread
    
    # Don't reinitialize if already initialized
    if db_writer_thread is not None and db_writer_thread.is_alive():
        print("Pipeline already initialized, skipping...")
        return
    
    # Start DB writer (single thread)
    db_writer_thread = threading.Thread(target=db_writer, daemon=True)
    db_writer_thread.start()
    print("DB writer thread started")
    
    # Start adaptive manager
    adaptive_manager_thread = threading.Thread(target=adaptive_worker_manager, daemon=True)
    adaptive_manager_thread.start()
    print("Adaptive worker manager thread started")
    
    # Start initial workers (default: 3)
    initial_workers = int(os.getenv("SUMMARY_MAX_WORKERS", 3))
    worker_threads = []  # Reset list
    for i in range(initial_workers):
        try:
            t = threading.Thread(target=summary_worker, args=(i,), daemon=True)
            t.start()
            worker_threads.append(t)
            # Give thread a moment to start and print its message
            time.sleep(0.1)
        except Exception as e:
                print(f"Error starting worker {i}: {e}")
                import traceback
                traceback.print_exc()
        finally:
    
    # Verify workers started
            alive_count = len([t for t in worker_threads if t.is_alive()])
    print(f"Pipeline initialized with {alive_count}/{initial_workers} workers alive")
    
    # Update worker metrics immediately
    worker_metrics.update_active_workers(alive_count)
    
    if alive_count == 0:
        print("⚠️  WARNING: No workers are alive after initialization!")
        print("   Workers may have crashed on startup. Check for errors above.")


def shutdown_pipeline():
    """Gracefully shutdown the pipeline."""
    global shutdown_event
    
    print("Shutting down pipeline...")
    shutdown_event.set()
    
    # Send poison pills to workers
    for _ in worker_threads:
        summary_queue.put(None)
    
    # Send poison pill to DB writer
    write_queue.put(None)
    
    # Wait for threads to finish (with timeout)
    for t in worker_threads:
        t.join(timeout=5)
    
    if db_writer_thread:
        db_writer_thread.join(timeout=5)
    
    if adaptive_manager_thread:
        adaptive_manager_thread.join(timeout=2)
    
    print("Pipeline shutdown complete")


# Initialize on import (only if not in test mode)
if os.getenv("SKIP_PIPELINE_INIT", "false").lower() != "true":
    initialize_pipeline()

