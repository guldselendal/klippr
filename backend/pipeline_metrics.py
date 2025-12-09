"""
Metrics tracking for the async summary pipeline.

Tracks:
- Latency metrics (p50, p95, p99, mean) by provider
- Queue depths (summary_queue, write_queue)
- Error counts and rates by provider
- Worker metrics (active workers, throughput)
- Database write metrics (batch sizes, commit times)
"""
import time
from typing import Dict, Optional, List
from collections import deque, defaultdict
from dataclasses import dataclass, field
from threading import Lock

# Thread-safe metrics storage
_metrics_lock = Lock()


@dataclass
class LatencyMetrics:
    """Latency metrics for a provider."""
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    errors: deque = field(default_factory=lambda: deque(maxlen=50))
    total_requests: int = 0
    total_errors: int = 0
    
    def add_latency(self, latency: float):
        """Add a latency measurement."""
        with _metrics_lock:
            self.latencies.append(latency)
            self.total_requests += 1
    
    def add_error(self, error_type: str):
        """Add an error."""
        with _metrics_lock:
            self.errors.append({
                'type': error_type,
                'timestamp': time.time()
            })
            self.total_errors += 1
    
    def get_stats(self) -> Dict:
        """Get latency statistics."""
        with _metrics_lock:
            if not self.latencies:
                return {
                    'count': 0,
                    'mean': 0.0,
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            
            sorted_latencies = sorted(self.latencies)
            count = len(sorted_latencies)
            
            return {
                'count': count,
                'mean': sum(sorted_latencies) / count,
                'p50': sorted_latencies[int(count * 0.50)] if count > 0 else 0.0,
                'p95': sorted_latencies[int(count * 0.95)] if count > 0 else sorted_latencies[-1],
                'p99': sorted_latencies[int(count * 0.99)] if count > 0 else sorted_latencies[-1],
                'min': sorted_latencies[0],
                'max': sorted_latencies[-1]
            }
    
    def get_error_rate(self) -> float:
        """Get error rate (errors per total requests)."""
        with _metrics_lock:
            if self.total_requests == 0:
                return 0.0
            return self.total_errors / self.total_requests


@dataclass
class QueueMetrics:
    """Queue depth metrics."""
    current_size: int = 0
    max_size: int = 0
    total_enqueued: int = 0
    total_dequeued: int = 0
    
    def update_size(self, size: int):
        """Update current queue size."""
        with _metrics_lock:
            self.current_size = size
            self.max_size = max(self.max_size, size)
    
    def record_enqueue(self):
        """Record an enqueue operation."""
        with _metrics_lock:
            self.total_enqueued += 1
    
    def record_dequeue(self):
        """Record a dequeue operation."""
        with _metrics_lock:
            self.total_dequeued += 1
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        with _metrics_lock:
            return {
                'current_size': self.current_size,
                'max_size': self.max_size,
                'total_enqueued': self.total_enqueued,
                'total_dequeued': self.total_dequeued,
                'pending': self.total_enqueued - self.total_dequeued
            }


@dataclass
class WorkerMetrics:
    """Worker pool metrics."""
    active_workers: int = 0
    total_workers_created: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    def update_active_workers(self, count: int):
        """Update active worker count."""
        with _metrics_lock:
            self.active_workers = count
    
    def record_completion(self):
        """Record a completed task."""
        with _metrics_lock:
            self.completed_tasks += 1
    
    def record_failure(self):
        """Record a failed task."""
        with _metrics_lock:
            self.failed_tasks += 1
    
    def get_stats(self) -> Dict:
        """Get worker statistics."""
        with _metrics_lock:
            total = self.completed_tasks + self.failed_tasks
            success_rate = (self.completed_tasks / total * 100) if total > 0 else 0.0
            
            return {
                'active_workers': self.active_workers,
                'total_workers_created': self.total_workers_created,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': success_rate,
                'total_tasks': total
            }


@dataclass
class ConnectionPoolMetrics:
    """HTTP connection pool metrics."""
    pool_size: int = 0
    pool_maxsize: int = 0
    pool_wait_time_ms: float = 0.0
    pool_acquired: int = 0
    pool_released: int = 0
    pool_timeouts: int = 0
    
    def update_pool_size(self, size: int, maxsize: int):
        """Update pool size information."""
        with _metrics_lock:
            self.pool_size = size
            self.pool_maxsize = maxsize
    
    def record_acquire(self, wait_time_ms: float = 0.0):
        """Record a connection acquisition."""
        with _metrics_lock:
            self.pool_acquired += 1
            if wait_time_ms > 0:
                self.pool_wait_time_ms = max(self.pool_wait_time_ms, wait_time_ms)
    
    def record_release(self):
        """Record a connection release."""
        with _metrics_lock:
            self.pool_released += 1
    
    def record_timeout(self):
        """Record a pool timeout."""
        with _metrics_lock:
            self.pool_timeouts += 1
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        with _metrics_lock:
            utilization = (self.pool_size / max(self.pool_maxsize, 1)) * 100 if self.pool_maxsize > 0 else 0.0
            return {
                'pool_size': self.pool_size,
                'pool_maxsize': self.pool_maxsize,
                'pool_utilization_pct': utilization,
                'pool_wait_time_ms': self.pool_wait_time_ms,
                'pool_acquired': self.pool_acquired,
                'pool_released': self.pool_released,
                'pool_timeouts': self.pool_timeouts,
                'pool_in_use': self.pool_acquired - self.pool_released
            }


@dataclass
class DatabaseMetrics:
    """Database write metrics."""
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=50))
    commit_times: deque = field(default_factory=lambda: deque(maxlen=50))
    total_batches: int = 0
    total_rows_written: int = 0
    total_commits: int = 0
    
    def record_batch(self, batch_size: int, commit_time_ms: float):
        """Record a batch commit."""
        with _metrics_lock:
            self.batch_sizes.append(batch_size)
            self.commit_times.append(commit_time_ms)
            self.total_batches += 1
            self.total_rows_written += batch_size
            self.total_commits += 1
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with _metrics_lock:
            avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
            avg_commit_time = sum(self.commit_times) / len(self.commit_times) if self.commit_times else 0
            
            sorted_commit_times = sorted(self.commit_times) if self.commit_times else []
            p95_commit_time = sorted_commit_times[int(len(sorted_commit_times) * 0.95)] if sorted_commit_times else 0.0
            
            return {
                'total_batches': self.total_batches,
                'total_rows_written': self.total_rows_written,
                'total_commits': self.total_commits,
                'avg_batch_size': avg_batch_size,
                'avg_commit_time_ms': avg_commit_time,
                'p95_commit_time_ms': p95_commit_time,
                'min_batch_size': min(self.batch_sizes) if self.batch_sizes else 0,
                'max_batch_size': max(self.batch_sizes) if self.batch_sizes else 0
            }


# Global metrics instances
provider_metrics: Dict[str, LatencyMetrics] = defaultdict(lambda: LatencyMetrics())
summary_queue_metrics = QueueMetrics()
write_queue_metrics = QueueMetrics()
worker_metrics = WorkerMetrics()
db_metrics = DatabaseMetrics()
connection_pool_metrics = ConnectionPoolMetrics()


def get_all_metrics() -> Dict:
    """
    Get all pipeline metrics.
    
    Returns:
        Dictionary containing all metrics
    """
    return {
        'latency': {
            provider: metrics.get_stats()
            for provider, metrics in provider_metrics.items()
        },
        'errors': {
            provider: {
                'total_errors': metrics.total_errors,
                'error_rate': metrics.get_error_rate(),
                'recent_errors': len(metrics.errors)
            }
            for provider, metrics in provider_metrics.items()
        },
        'queues': {
            'summary_queue': summary_queue_metrics.get_stats(),
            'write_queue': write_queue_metrics.get_stats()
        },
        'workers': worker_metrics.get_stats(),
        'database': db_metrics.get_stats(),
        'connection_pool': connection_pool_metrics.get_stats(),
        'cost': _get_cost_stats(),
        'timestamp': time.time()
    }


def _get_cost_stats() -> Dict:
    """Get cost statistics if cost tracking is available."""
    try:
        from cost_tracking import cost_tracker
        return cost_tracker.get_stats()
    except ImportError:
        return {
            'total_daily_cost_usd': 0.0,
            'note': 'Cost tracking not available'
        }


def reset_metrics():
    """Reset all metrics (useful for testing)."""
    global provider_metrics, summary_queue_metrics, write_queue_metrics, worker_metrics, db_metrics, connection_pool_metrics
    
    with _metrics_lock:
        provider_metrics.clear()
        summary_queue_metrics = QueueMetrics()
        write_queue_metrics = QueueMetrics()
        worker_metrics = WorkerMetrics()
        db_metrics = DatabaseMetrics()
        connection_pool_metrics = ConnectionPoolMetrics()

