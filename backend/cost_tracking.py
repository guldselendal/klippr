"""
Cost tracking and guardrails for LLM API calls.

Tracks token usage and costs per provider, enforces daily and per-request limits.
"""
import os
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from threading import Lock
from datetime import datetime, timedelta

# Thread-safe cost tracking
_cost_lock = Lock()

# Cost per 1M tokens (input + output combined, approximate)
COST_PER_1M_TOKENS = {
    'gemini-1.5-flash': 0.075,  # $0.075 per 1M tokens
    'gpt-4-turbo-preview': 10.0,  # $10 per 1M tokens (approximate)
    'deepseek-chat': 0.14,  # $0.14 per 1M tokens (approximate)
    'ollama': 0.0,  # Local, no cost
    'phi3:mini': 0.0,  # Local, no cost
}

# Cost limits (from environment or defaults)
MAX_DAILY_COST_USD = float(os.getenv("MAX_DAILY_COST_USD", "5.0"))
MAX_PER_REQUEST_COST_USD = float(os.getenv("MAX_PER_REQUEST_COST_USD", "0.50"))
WARN_AT_COST_USD = float(os.getenv("WARN_AT_COST_USD", "3.0"))


@dataclass
class CostTracker:
    """Tracks costs per provider and enforces limits."""
    daily_costs: Dict[str, float] = field(default_factory=dict)  # provider -> cost
    request_costs: Dict[str, float] = field(default_factory=dict)  # request_id -> cost
    daily_reset_time: float = field(default_factory=lambda: time.time())
    total_daily_cost: float = 0.0
    total_requests: int = 0
    
    def reset_daily_if_needed(self):
        """Reset daily costs if a new day has started."""
        now = time.time()
        if now - self.daily_reset_time > 86400:  # 24 hours
            self.daily_costs.clear()
            self.total_daily_cost = 0.0
            self.daily_reset_time = now
    
    def estimate_cost(self, provider: str, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        """
        Estimate cost for a request.
        
        Args:
            provider: Provider name
            model: Model name
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens (default: 0)
        
        Returns:
            Estimated cost in USD
        """
        # Use model name or provider as key
        cost_key = model if model in COST_PER_1M_TOKENS else provider
        cost_per_1m = COST_PER_1M_TOKENS.get(cost_key, 0.0)
        
        if cost_per_1m == 0.0:
            return 0.0  # Free (local models)
        
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1_000_000) * cost_per_1m
        return cost
    
    def record_request(self, provider: str, model: str, input_tokens: int, output_tokens: int = 0, request_id: Optional[str] = None) -> float:
        """
        Record a request and return its cost.
        
        Args:
            provider: Provider name
            model: Model name
            input_tokens: Input tokens (estimated)
            output_tokens: Output tokens (estimated)
            request_id: Optional request ID for per-request tracking
        
        Returns:
            Cost in USD
        """
        with _cost_lock:
            self.reset_daily_if_needed()
            
            cost = self.estimate_cost(provider, model, input_tokens, output_tokens)
            
            # Track daily costs
            provider_key = f"{provider}:{model}"
            if provider_key not in self.daily_costs:
                self.daily_costs[provider_key] = 0.0
            self.daily_costs[provider_key] += cost
            self.total_daily_cost += cost
            self.total_requests += 1
            
            # Track per-request costs
            if request_id:
                if request_id not in self.request_costs:
                    self.request_costs[request_id] = 0.0
                self.request_costs[request_id] += cost
            
            return cost
    
    def check_daily_limit(self) -> tuple[bool, str]:
        """
        Check if daily limit would be exceeded.
        
        Returns:
            (allowed, message) tuple
        """
        with _cost_lock:
            self.reset_daily_if_needed()
            
            if self.total_daily_cost >= MAX_DAILY_COST_USD:
                return False, f"Daily cost limit exceeded: ${self.total_daily_cost:.2f} >= ${MAX_DAILY_COST_USD:.2f}"
            
            if self.total_daily_cost >= WARN_AT_COST_USD:
                return True, f"Warning: Daily cost approaching limit: ${self.total_daily_cost:.2f} / ${MAX_DAILY_COST_USD:.2f}"
            
            return True, ""
    
    def check_request_limit(self, estimated_cost: float) -> tuple[bool, str]:
        """
        Check if per-request limit would be exceeded.
        
        Args:
            estimated_cost: Estimated cost for the request
        
        Returns:
            (allowed, message) tuple
        """
        if estimated_cost > MAX_PER_REQUEST_COST_USD:
            return False, f"Per-request cost limit exceeded: ${estimated_cost:.2f} > ${MAX_PER_REQUEST_COST_USD:.2f}"
        
        return True, ""
    
    def get_stats(self) -> Dict:
        """Get cost statistics."""
        with _cost_lock:
            self.reset_daily_if_needed()
            
            return {
                'total_daily_cost_usd': self.total_daily_cost,
                'max_daily_cost_usd': MAX_DAILY_COST_USD,
                'warn_at_cost_usd': WARN_AT_COST_USD,
                'max_per_request_cost_usd': MAX_PER_REQUEST_COST_USD,
                'total_requests': self.total_requests,
                'provider_costs': dict(self.daily_costs),
                'daily_reset_time': datetime.fromtimestamp(self.daily_reset_time).isoformat(),
                'cost_percentage': (self.total_daily_cost / MAX_DAILY_COST_USD * 100) if MAX_DAILY_COST_USD > 0 else 0.0
            }


# Global cost tracker instance
cost_tracker = CostTracker()


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Rough estimate: ~4 characters per token.
    """
    return len(text) // 4


def record_llm_cost(provider: str, model: str, input_text: str, output_text: str = "", request_id: Optional[str] = None) -> float:
    """
    Record cost for an LLM call.
    
    Args:
        provider: Provider name
        model: Model name
        input_text: Input text
        output_text: Output text (optional)
        request_id: Optional request ID
    
    Returns:
        Cost in USD
    """
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text) if output_text else 0
    
    return cost_tracker.record_request(provider, model, input_tokens, output_tokens, request_id)

