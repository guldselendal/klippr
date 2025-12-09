#!/usr/bin/env python3
"""
Concurrency diagnosis script for the summary pipeline.

Measures baseline performance and identifies bottlenecks.
"""
import os
import sys
import time
import threading
import multiprocessing
from collections import deque
from typing import Dict, List
import json

# Suppress async pipeline initialization
os.environ["USE_ASYNC_SUMMARIZATION"] = "false"
os.environ["SKIP_PIPELINE_INIT"] = "true"

# Import after setting env vars
from llm_concurrency import LLMConcurrencyLimiter
from llm_provider import call_llm
from prompt_utils import CHUNK_PROMPT_COMPACT, CHUNK_SYSTEM_COMPACT


def get_system_info() -> Dict:
    """Get system hardware and runtime info."""
    cpu_count = multiprocessing.cpu_count()
    
    # Try to get CPU model (Unix)
    cpu_model = "Unknown"
    try:
        if sys.platform == "darwin":
            import subprocess
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                  capture_output=True, text=True)
            cpu_model = result.stdout.strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
    except:
        pass
    
    return {
        "cpu_model": cpu_model,
        "cpu_cores": cpu_count,
        "python_version": sys.version,
        "platform": sys.platform
    }


def measure_llm_call_latency(provider: str, model: str, num_calls: int = 10) -> Dict:
    """Measure LLM call latency with different concurrency levels."""
    test_prompt = "Summarize this in one sentence: Python is a programming language."
    
    results = {
        "sequential": [],
        "concurrent_4": [],
        "concurrent_8": [],
        "concurrent_16": []
    }
    
    # Sequential baseline
    print(f"Measuring sequential {provider} calls...")
    for i in range(num_calls):
        start = time.perf_counter()
        try:
            call_llm(prompt=test_prompt, system_prompt="You are a helpful assistant.", 
                    provider=provider, model=model)
            latency = (time.perf_counter() - start) * 1000
            results["sequential"].append(latency)
        except Exception as e:
            print(f"  Error in sequential call {i+1}: {e}")
            results["sequential"].append(None)
    
    # Concurrent with different levels
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    for concurrency in [4, 8, 16]:
        print(f"Measuring {concurrency} concurrent {provider} calls...")
        start_batch = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    call_llm,
                    prompt=test_prompt,
                    system_prompt="You are a helpful assistant.",
                    provider=provider,
                    model=model
                )
                for _ in range(num_calls)
            ]
            
            latencies = []
            for future in as_completed(futures):
                try:
                    future.result()
                    latency = (time.perf_counter() - start_batch) * 1000
                    latencies.append(latency)
                except Exception as e:
                    print(f"  Error in concurrent call: {e}")
        
        results[f"concurrent_{concurrency}"] = latencies
    
    # Calculate statistics
    def calc_stats(values):
        if not values:
            return {}
        valid = [v for v in values if v is not None]
        if not valid:
            return {}
        sorted_vals = sorted(valid)
        n = len(sorted_vals)
        return {
            "mean": sum(valid) / n,
            "p50": sorted_vals[n // 2],
            "p95": sorted_vals[int(n * 0.95)] if n > 0 else None,
            "p99": sorted_vals[int(n * 0.99)] if n > 0 else None,
            "min": min(valid),
            "max": max(valid),
            "count": n
        }
    
    return {
        "sequential": calc_stats(results["sequential"]),
        "concurrent_4": calc_stats(results["concurrent_4"]),
        "concurrent_8": calc_stats(results["concurrent_8"]),
        "concurrent_16": calc_stats(results["concurrent_16"])
    }


def measure_concurrency_limiter_impact() -> Dict:
    """Measure impact of concurrency limiter on blocking."""
    limiter = LLMConcurrencyLimiter()
    
    metrics_before = limiter.get_metrics()
    
    # Simulate high concurrency
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    test_prompt = "Say hello."
    num_requests = 30
    concurrency = 20
    
    blocking_times = []
    start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                lambda: limiter.acquire("ollama").__enter__() or 
                        (time.sleep(0.1), limiter.acquire("ollama").__exit__(None, None, None))
            )
            for _ in range(num_requests)
        ]
        
        for future in as_completed(futures):
            try:
                future.result()
            except:
                pass
    
    total_time = time.perf_counter() - start
    metrics_after = limiter.get_metrics()
    
    return {
        "total_requests": metrics_after["total_requests"],
        "blocked_requests": metrics_after["blocked_requests"],
        "avg_block_time": metrics_after["avg_block_time_seconds"],
        "total_time_seconds": total_time,
        "requests_per_second": num_requests / total_time if total_time > 0 else 0
    }


def count_active_threads() -> int:
    """Count active threads."""
    return threading.active_count()


def diagnose_nested_executors() -> Dict:
    """Simulate nested ThreadPoolExecutor scenario."""
    print("Simulating nested ThreadPoolExecutor scenario...")
    
    # Simulate: 4 summary workers, each processing 10 chunks with 16 workers
    num_summary_workers = 4
    chunks_per_worker = 10
    workers_per_chunk = 16
    
    def dummy_chunk_work():
        time.sleep(0.01)  # Simulate chunk processing
        return "summary"
    
    def summary_worker():
        with ThreadPoolExecutor(max_workers=workers_per_chunk) as executor:
            futures = [executor.submit(dummy_chunk_work) for _ in range(chunks_per_worker)]
            return [f.result() for f in futures]
    
    start = time.perf_counter()
    max_threads = 0
    
    with ThreadPoolExecutor(max_workers=num_summary_workers) as outer_executor:
        futures = [outer_executor.submit(summary_worker) for _ in range(num_summary_workers)]
        
        # Monitor thread count
        def monitor_threads():
            nonlocal max_threads
            while any(not f.done() for f in futures):
                max_threads = max(max_threads, threading.active_count())
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
        monitor_thread.start()
        
        results = [f.result() for f in futures]
        monitor_thread.join(timeout=1)
    
    total_time = time.perf_counter() - start
    
    return {
        "max_threads_observed": max_threads,
        "expected_max_threads": num_summary_workers * workers_per_chunk + 10,  # +10 for overhead
        "total_time_seconds": total_time,
        "theoretical_max_concurrency": num_summary_workers * workers_per_chunk
    }


def main():
    """Run concurrency diagnosis."""
    print("=" * 80)
    print("CONCURRENCY DIAGNOSIS")
    print("=" * 80)
    print()
    
    system_info = get_system_info()
    print(f"System: {system_info['cpu_model']}")
    print(f"CPU Cores: {system_info['cpu_cores']}")
    print(f"Platform: {system_info['platform']}")
    print()
    
    # Check Ollama availability
    print("Checking Ollama availability...")
    try:
        test_result = call_llm(
            prompt="Say hello.",
            system_prompt="You are helpful.",
            provider="ollama",
            model="phi3:mini"
        )
        print(f"✓ Ollama is available (test response: {test_result[:50]}...)")
    except Exception as e:
        print(f"✗ Ollama not available: {e}")
        print("  Skipping LLM latency tests")
        return
    
    print()
    
    # Measure LLM latency with different concurrency
    print("=" * 80)
    print("LLM LATENCY MEASUREMENT")
    print("=" * 80)
    llm_latency = measure_llm_call_latency("ollama", "phi3:mini", num_calls=5)
    print(json.dumps(llm_latency, indent=2))
    print()
    
    # Measure concurrency limiter impact
    print("=" * 80)
    print("CONCURRENCY LIMITER IMPACT")
    print("=" * 80)
    limiter_impact = measure_concurrency_limiter_impact()
    print(json.dumps(limiter_impact, indent=2))
    print()
    
    # Diagnose nested executors
    print("=" * 80)
    print("NESTED EXECUTOR DIAGNOSIS")
    print("=" * 80)
    nested_diag = diagnose_nested_executors()
    print(json.dumps(nested_diag, indent=2))
    print()
    
    # Compile report
    report = {
        "baseline": {
            "env": system_info,
            "metrics": {
                "llm_latency": llm_latency,
                "limiter_impact": limiter_impact,
                "nested_executors": nested_diag
            }
        },
        "quick_triage": [
            "Cap Ollama concurrency to 4-8 (recently increased to 16 may be too high)",
            "Eliminate nested ThreadPoolExecutors - use shared executor pool",
            "Align worker count with CPU cores for CPU-bound work",
            "Consider multiprocessing for CPU-bound chunk processing (bypasses GIL)"
        ]
    }
    
    print("=" * 80)
    print("QUICK TRIAGE RECOMMENDATIONS")
    print("=" * 80)
    for i, rec in enumerate(report["quick_triage"], 1):
        print(f"{i}. {rec}")
    print()
    
    # Save report
    with open("concurrency_diagnosis.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Full report saved to concurrency_diagnosis.json")


if __name__ == "__main__":
    main()

