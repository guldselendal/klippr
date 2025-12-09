"""
Global concurrency limiter for LLM API calls.

Prevents overwhelming LLM providers (especially Ollama) with too many
concurrent requests, which causes timeouts and failures.

Usage:
    from llm_concurrency import LLMConcurrencyLimiter
    
    limiter = LLMConcurrencyLimiter()
    with limiter.acquire("ollama"):
        # Make LLM call here
        result = call_ollama(...)
"""
import threading
import os
import time
from typing import Dict, Optional
from contextlib import contextmanager


class LLMConcurrencyLimiter:
    """
    Singleton global concurrency limiter for LLM calls.
    
    Provides provider-specific and global limits to prevent resource exhaustion.
    Default limits:
    - Ollama: 4 concurrent requests (safe limit to avoid timeouts)
    - Gemini: 8 concurrent requests
    - OpenAI: 10 concurrent requests
    - DeepSeek: 8 concurrent requests
    - Global: 12 concurrent requests (across all providers)
    
    Limits can be configured via environment variables:
    - LLM_MAX_CONCURRENCY_OLLAMA
    - LLM_MAX_CONCURRENCY_GEMINI
    - LLM_MAX_CONCURRENCY_OPENAI
    - LLM_MAX_CONCURRENCY_DEEPSEEK
    - LLM_MAX_CONCURRENCY_GLOBAL
    """
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
        if getattr(self, '_initialized', False):
            return
        
        # Read limits from environment variables
        # Reduced Ollama limit from 16 to 8 for MVR (local instances handle 4-8 efficiently)
        self.ollama_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OLLAMA", 8))
        self.gemini_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GEMINI", 8))
        self.openai_limit = int(os.getenv("LLM_MAX_CONCURRENCY_OPENAI", 10))
        self.deepseek_limit = int(os.getenv("LLM_MAX_CONCURRENCY_DEEPSEEK", 8))
        # Global limit adjusted for reduced Ollama concurrency
        self.global_limit = int(os.getenv("LLM_MAX_CONCURRENCY_GLOBAL", 12))
        
        # Create semaphores for each provider
        self.ollama_sem = threading.Semaphore(self.ollama_limit)
        self.gemini_sem = threading.Semaphore(self.gemini_limit)
        self.openai_sem = threading.Semaphore(self.openai_limit)
        self.deepseek_sem = threading.Semaphore(self.deepseek_limit)
        self.global_sem = threading.Semaphore(self.global_limit)
        
        # Metrics tracking
        self.in_flight = {
            "ollama": 0,
            "gemini": 0,
            "openai": 0,
            "deepseek": 0,
            "total": 0
        }
        self.metrics_lock = threading.Lock()
        self.total_requests = 0
        self.blocked_requests = 0
        self.blocked_time_total = 0.0
        
        self._initialized = True
        
        # Log initialization
        print(f"LLM Concurrency Limiter initialized:")
        print(f"  Ollama: {self.ollama_limit} concurrent requests")
        print(f"  Gemini: {self.gemini_limit} concurrent requests")
        print(f"  OpenAI: {self.openai_limit} concurrent requests")
        print(f"  DeepSeek: {self.deepseek_limit} concurrent requests")
        print(f"  Global: {self.global_limit} concurrent requests (across all providers)")
    
    @contextmanager
    def acquire(self, provider: str):
        """
        Context manager for acquiring/releasing concurrency semaphore.
        
        Args:
            provider: Provider name ("ollama", "gemini", "openai", "deepseek")
        
        Usage:
            with limiter.acquire("ollama"):
                result = call_ollama(...)
        
        Blocks if the provider or global limit is reached.
        """
        provider_sem = getattr(self, f"{provider}_sem", None)
        
        # Track blocking time
        block_start = time.perf_counter()
        
        # Acquire provider-specific semaphore
        if provider_sem:
            provider_sem.acquire()
            provider_acquired = True
        else:
            provider_acquired = False
            print(f"Warning: Unknown provider '{provider}', no provider-specific limit applied")
        
        # Acquire global semaphore
        self.global_sem.acquire()
        
        # Calculate blocking time
        block_time = time.perf_counter() - block_start
        if block_time > 0.1:  # Log if blocked for more than 100ms
            with self.metrics_lock:
                self.blocked_requests += 1
                self.blocked_time_total += block_time
            print(f"⚠️  LLM concurrency limiter blocked for {block_time:.2f}s (provider={provider}, "
                  f"in_flight={self.in_flight.get(provider, 0)})")
        
        # Update metrics
        with self.metrics_lock:
            self.in_flight[provider] = self.in_flight.get(provider, 0) + 1
            self.in_flight["total"] += 1
            self.total_requests += 1
        
        try:
            yield
        finally:
            # Release semaphores
            if provider_acquired:
                provider_sem.release()
            self.global_sem.release()
            
            # Update metrics
            with self.metrics_lock:
                self.in_flight[provider] = max(0, self.in_flight.get(provider, 0) - 1)
                self.in_flight["total"] = max(0, self.in_flight["total"] - 1)
    
    def get_metrics(self) -> Dict:
        """
        Get current concurrency metrics.
        
        Returns:
            Dictionary with in-flight counts, limits, and statistics
        """
        with self.metrics_lock:
            avg_block_time = (
                self.blocked_time_total / self.blocked_requests
                if self.blocked_requests > 0 else 0.0
            )
            
            return {
                "in_flight": self.in_flight.copy(),
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "avg_block_time_seconds": avg_block_time,
                "limits": {
                    "ollama": self.ollama_limit,
                    "gemini": self.gemini_limit,
                    "openai": self.openai_limit,
                    "deepseek": self.deepseek_limit,
                    "global": self.global_limit
                }
            }
    
    def get_in_flight(self, provider: Optional[str] = None) -> int:
        """
        Get current number of in-flight requests.
        
        Args:
            provider: Provider name, or None for total across all providers
        
        Returns:
            Number of in-flight requests
        """
        with self.metrics_lock:
            if provider:
                return self.in_flight.get(provider, 0)
            return self.in_flight.get("total", 0)

