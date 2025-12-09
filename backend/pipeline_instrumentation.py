"""
Lightweight instrumentation for async summary pipeline performance analysis.
Captures per-stage timings, queue depths, and provider metrics.
"""
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json


@dataclass
class StageTiming:
    """Timing for a single stage of processing"""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def complete(self, metadata: Optional[Dict] = None):
        """Mark stage as complete and calculate duration"""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if metadata:
            self.metadata.update(metadata)


@dataclass
class ChapterMetrics:
    """Metrics for a single chapter"""
    chapter_id: str
    document_id: str
    chapter_number: int
    content_length: int
    is_chunked: bool = False
    num_chunks: int = 0
    
    # Stage timings
    parse_time_ms: Optional[float] = None
    chunk_time_ms: Optional[float] = None
    map_llm_time_ms: Optional[float] = None  # Time for all chunk summaries
    reduce_time_ms: Optional[float] = None  # Time to combine chunks
    title_preview_time_ms: Optional[float] = None
    db_enqueue_time_ms: Optional[float] = None
    db_write_time_ms: Optional[float] = None
    
    # Queue wait times
    summary_queue_wait_ms: Optional[float] = None
    write_queue_wait_ms: Optional[float] = None
    
    # Provider info
    provider: Optional[str] = None
    retries: int = 0
    timeout: bool = False
    error: Optional[str] = None
    
    # Timestamps
    enqueued_at: Optional[float] = None
    processing_started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def total_time_ms(self) -> float:
        """Calculate total processing time"""
        if self.completed_at and self.enqueued_at:
            return (self.completed_at - self.enqueued_at) * 1000
        return 0.0


class PipelineMetrics:
    """Central metrics collector for pipeline performance"""
    
    def __init__(self):
        self.chapter_metrics: Dict[str, ChapterMetrics] = {}
        self.queue_depth_samples: deque = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.worker_counts: deque = deque(maxlen=3600)
        self.provider_stats: Dict[str, Dict] = defaultdict(lambda: {
            'calls': 0,
            'retries': 0,
            'timeouts': 0,
            'errors': 0,
            'latencies': deque(maxlen=1000)
        })
        self.db_write_stats: deque = deque(maxlen=1000)  # Batch write timings
        self.circuit_breaker_events: List[Dict] = []
        
        self._lock = threading.Lock()
        self._sampling_thread = None
        self._sampling_active = False
        
    def start_sampling(self, summary_queue, write_queue, worker_threads_getter):
        """Start background thread to sample queue depths and worker counts"""
        self._sampling_active = True
        
        def sample_loop():
            while self._sampling_active:
                timestamp = time.time()
                summary_depth = summary_queue.qsize()
                write_depth = write_queue.qsize()
                worker_count = len(worker_threads_getter()) if callable(worker_threads_getter) else 0
                
                self.queue_depth_samples.append({
                    'timestamp': timestamp,
                    'summary_queue': summary_depth,
                    'write_queue': write_depth,
                    'workers': worker_count
                })
                
                time.sleep(1)  # Sample every 1 second
        
        self._sampling_thread = threading.Thread(target=sample_loop, daemon=True)
        self._sampling_thread.start()
    
    def stop_sampling(self):
        """Stop background sampling"""
        self._sampling_active = False
        if self._sampling_thread:
            self._sampling_thread.join(timeout=2)
    
    def record_chapter_enqueued(self, chapter_id: str, document_id: str, 
                                chapter_number: int, content_length: int):
        """Record when a chapter is enqueued"""
        with self._lock:
            if chapter_id not in self.chapter_metrics:
                self.chapter_metrics[chapter_id] = ChapterMetrics(
                    chapter_id=chapter_id,
                    document_id=document_id,
                    chapter_number=chapter_number,
                    content_length=content_length
                )
            self.chapter_metrics[chapter_id].enqueued_at = time.perf_counter()
            self.chapter_metrics[chapter_id].summary_queue_wait_ms = 0
    
    def record_chapter_processing_started(self, chapter_id: str):
        """Record when processing actually starts"""
        with self._lock:
            if chapter_id in self.chapter_metrics:
                self.chapter_metrics[chapter_id].processing_started_at = time.perf_counter()
                if self.chapter_metrics[chapter_id].enqueued_at:
                    wait_time = (self.chapter_metrics[chapter_id].processing_started_at - 
                               self.chapter_metrics[chapter_id].enqueued_at) * 1000
                    self.chapter_metrics[chapter_id].summary_queue_wait_ms = wait_time
    
    def record_chunking(self, chapter_id: str, num_chunks: int, chunk_time_ms: float):
        """Record chunking stage"""
        with self._lock:
            if chapter_id in self.chapter_metrics:
                self.chapter_metrics[chapter_id].is_chunked = True
                self.chapter_metrics[chapter_id].num_chunks = num_chunks
                self.chapter_metrics[chapter_id].chunk_time_ms = chunk_time_ms
    
    def record_map_llm(self, chapter_id: str, provider: str, duration_ms: float, 
                       retries: int = 0, timeout: bool = False, error: Optional[str] = None):
        """Record LLM call for chunk summarization"""
        with self._lock:
            if chapter_id in self.chapter_metrics:
                if self.chapter_metrics[chapter_id].map_llm_time_ms:
                    self.chapter_metrics[chapter_id].map_llm_time_ms += duration_ms
                else:
                    self.chapter_metrics[chapter_id].map_llm_time_ms = duration_ms
                self.chapter_metrics[chapter_id].provider = provider
                self.chapter_metrics[chapter_id].retries += retries
                if timeout:
                    self.chapter_metrics[chapter_id].timeout = True
                if error:
                    self.chapter_metrics[chapter_id].error = error
            
            # Update provider stats
            self.provider_stats[provider]['calls'] += 1
            self.provider_stats[provider]['retries'] += retries
            if timeout:
                self.provider_stats[provider]['timeouts'] += 1
            if error:
                self.provider_stats[provider]['errors'] += 1
            self.provider_stats[provider]['latencies'].append(duration_ms)
    
    def record_reduce(self, chapter_id: str, duration_ms: float):
        """Record chunk combination stage"""
        with self._lock:
            if chapter_id in self.chapter_metrics:
                self.chapter_metrics[chapter_id].reduce_time_ms = duration_ms
    
    def record_title_preview(self, chapter_id: str, duration_ms: float):
        """Record title/preview generation"""
        with self._lock:
            if chapter_id in self.chapter_metrics:
                self.chapter_metrics[chapter_id].title_preview_time_ms = duration_ms
    
    def record_db_write(self, chapter_id: str, duration_ms: float):
        """Record database write time"""
        with self._lock:
            if chapter_id in self.chapter_metrics:
                self.chapter_metrics[chapter_id].db_write_time_ms = duration_ms
                self.chapter_metrics[chapter_id].completed_at = time.perf_counter()
    
    def record_db_batch_write(self, batch_size: int, duration_ms: float):
        """Record batch database write"""
        self.db_write_stats.append({
            'timestamp': time.time(),
            'batch_size': batch_size,
            'duration_ms': duration_ms
        })
    
    def record_circuit_breaker_event(self, provider: str, event_type: str, reason: str):
        """Record circuit breaker state change"""
        self.circuit_breaker_events.append({
            'timestamp': time.time(),
            'provider': provider,
            'event_type': event_type,  # 'open', 'close', 'half_open'
            'reason': reason
        })
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        with self._lock:
            chapters = list(self.chapter_metrics.values())
            completed = [c for c in chapters if c.completed_at]
            
            if not completed:
                return {
                    'total_chapters': len(chapters),
                    'completed_chapters': 0,
                    'message': 'No chapters completed yet'
                }
            
            # Calculate percentiles
            def percentile(values, p):
                if not values:
                    return 0
                sorted_vals = sorted(values)
                idx = int(len(sorted_vals) * p / 100)
                return sorted_vals[min(idx, len(sorted_vals) - 1)]
            
            total_times = [c.total_time_ms() for c in completed]
            map_times = [c.map_llm_time_ms or 0 for c in completed if c.map_llm_time_ms]
            reduce_times = [c.reduce_time_ms or 0 for c in completed if c.reduce_time_ms]
            title_times = [c.title_preview_time_ms or 0 for c in completed if c.title_preview_time_ms]
            db_times = [c.db_write_time_ms or 0 for c in completed if c.db_write_time_ms]
            queue_waits = [c.summary_queue_wait_ms or 0 for c in completed if c.summary_queue_wait_ms]
            
            # Provider statistics
            provider_summary = {}
            for provider, stats in self.provider_stats.items():
                latencies = list(stats['latencies'])
                provider_summary[provider] = {
                    'calls': stats['calls'],
                    'retries': stats['retries'],
                    'timeouts': stats['timeouts'],
                    'errors': stats['errors'],
                    'p50_ms': percentile(latencies, 50) if latencies else 0,
                    'p95_ms': percentile(latencies, 95) if latencies else 0,
                    'p99_ms': percentile(latencies, 99) if latencies else 0,
                }
            
            # Queue depth statistics
            queue_samples = list(self.queue_depth_samples)
            if queue_samples:
                summary_depths = [s['summary_queue'] for s in queue_samples]
                write_depths = [s['write_queue'] for s in queue_samples]
                worker_counts = [s['workers'] for s in queue_samples]
            else:
                summary_depths = write_depths = worker_counts = []
            
            # DB write statistics
            db_batches = list(self.db_write_stats)
            batch_sizes = [b['batch_size'] for b in db_batches]
            batch_durations = [b['duration_ms'] for b in db_batches]
            
            return {
                'total_chapters': len(chapters),
                'completed_chapters': len(completed),
                'in_progress': len(chapters) - len(completed),
                
                # Timing statistics (ms)
                'total_time': {
                    'p50': percentile(total_times, 50),
                    'p95': percentile(total_times, 95),
                    'p99': percentile(total_times, 99),
                    'mean': sum(total_times) / len(total_times) if total_times else 0,
                    'min': min(total_times) if total_times else 0,
                    'max': max(total_times) if total_times else 0,
                },
                'map_llm_time': {
                    'p50': percentile(map_times, 50) if map_times else 0,
                    'p95': percentile(map_times, 95) if map_times else 0,
                    'mean': sum(map_times) / len(map_times) if map_times else 0,
                },
                'reduce_time': {
                    'p50': percentile(reduce_times, 50) if reduce_times else 0,
                    'p95': percentile(reduce_times, 95) if reduce_times else 0,
                    'mean': sum(reduce_times) / len(reduce_times) if reduce_times else 0,
                },
                'title_preview_time': {
                    'p50': percentile(title_times, 50) if title_times else 0,
                    'p95': percentile(title_times, 95) if title_times else 0,
                    'mean': sum(title_times) / len(title_times) if title_times else 0,
                },
                'db_write_time': {
                    'p50': percentile(db_times, 50) if db_times else 0,
                    'p95': percentile(db_times, 95) if db_times else 0,
                    'mean': sum(db_times) / len(db_times) if db_times else 0,
                },
                'queue_wait_time': {
                    'p50': percentile(queue_waits, 50) if queue_waits else 0,
                    'p95': percentile(queue_waits, 95) if queue_waits else 0,
                    'mean': sum(queue_waits) / len(queue_waits) if queue_waits else 0,
                },
                
                # Queue depth statistics
                'queue_depths': {
                    'summary_queue': {
                        'max': max(summary_depths) if summary_depths else 0,
                        'mean': sum(summary_depths) / len(summary_depths) if summary_depths else 0,
                        'p95': percentile(summary_depths, 95) if summary_depths else 0,
                    },
                    'write_queue': {
                        'max': max(write_depths) if write_depths else 0,
                        'mean': sum(write_depths) / len(write_depths) if write_depths else 0,
                        'p95': percentile(write_depths, 95) if write_depths else 0,
                    },
                },
                
                # Worker statistics
                'workers': {
                    'max': max(worker_counts) if worker_counts else 0,
                    'mean': sum(worker_counts) / len(worker_counts) if worker_counts else 0,
                },
                
                # Provider statistics
                'providers': provider_summary,
                
                # DB batch statistics
                'db_batches': {
                    'total': len(db_batches),
                    'avg_size': sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
                    'avg_duration_ms': sum(batch_durations) / len(batch_durations) if batch_durations else 0,
                    'p95_duration_ms': percentile(batch_durations, 95) if batch_durations else 0,
                },
                
                # Circuit breaker events
                'circuit_breaker_events': len(self.circuit_breaker_events),
            }
    
    def export_report(self, filename: Optional[str] = None) -> str:
        """Export metrics as JSON report"""
        stats = self.get_statistics()
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'circuit_breaker_events': self.circuit_breaker_events[-100:],  # Last 100 events
        }
        
        json_str = json.dumps(report, indent=2)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str


# Global metrics instance
metrics = PipelineMetrics()

