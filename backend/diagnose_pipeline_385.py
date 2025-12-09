#!/usr/bin/env python3
"""
Diagnostic script for analyzing 385-chapter pipeline performance.
Run this after uploading a document with 385 chapters.
"""
import time
import json
from datetime import datetime
from pipeline_instrumentation import metrics

def analyze_385_chapter_performance():
    """Analyze pipeline performance for 385 chapters"""
    print("=" * 80)
    print("385-Chapter Pipeline Performance Analysis")
    print("=" * 80)
    print()
    
    # Wait for some processing to complete
    print("Collecting metrics... (wait 2 minutes for data)")
    time.sleep(120)
    
    # Get statistics
    stats = metrics.get_statistics()
    
    print("\n" + "=" * 80)
    print("1. CAPACITY ANALYSIS")
    print("=" * 80)
    
    total = stats['total_chapters']
    completed = stats['completed_chapters']
    in_progress = stats['in_progress']
    
    print(f"Total chapters: {total}")
    print(f"Completed: {completed} ({completed/total*100:.1f}%)")
    print(f"In progress: {in_progress} ({in_progress/total*100:.1f}%)")
    
    if completed == 0:
        print("\n⚠️  No chapters completed yet. Wait longer and re-run.")
        return
    
    # Calculate throughput
    if stats['total_time']['mean'] > 0:
        throughput = 60 / (stats['total_time']['mean'] / 1000)  # chapters per minute
        print(f"\nThroughput: {throughput:.2f} chapters/minute")
        
        # Estimate remaining time
        remaining = total - completed
        if throughput > 0:
            eta_minutes = remaining / throughput
            print(f"Estimated time remaining: {eta_minutes:.1f} minutes")
            print(f"Estimated total time: {(completed/throughput + eta_minutes):.1f} minutes")
    
    print("\n" + "=" * 80)
    print("2. TIMING BREAKDOWN (milliseconds)")
    print("=" * 80)
    
    print(f"\nTotal Time (per chapter):")
    print(f"  p50: {stats['total_time']['p50']:.1f} ms")
    print(f"  p95: {stats['total_time']['p95']:.1f} ms")
    print(f"  p99: {stats['total_time']['p99']:.1f} ms")
    print(f"  mean: {stats['total_time']['mean']:.1f} ms")
    
    print(f"\nMap LLM Time (chunk summarization):")
    print(f"  p50: {stats['map_llm_time']['p50']:.1f} ms")
    print(f"  p95: {stats['map_llm_time']['p95']:.1f} ms")
    print(f"  mean: {stats['map_llm_time']['mean']:.1f} ms")
    
    print(f"\nReduce Time (chunk combination):")
    print(f"  p50: {stats['reduce_time']['p50']:.1f} ms")
    print(f"  p95: {stats['reduce_time']['p95']:.1f} ms")
    print(f"  mean: {stats['reduce_time']['mean']:.1f} ms")
    
    print(f"\nTitle/Preview Time:")
    print(f"  p50: {stats['title_preview_time']['p50']:.1f} ms")
    print(f"  p95: {stats['title_preview_time']['p95']:.1f} ms")
    print(f"  mean: {stats['title_preview_time']['mean']:.1f} ms")
    
    print(f"\nDB Write Time:")
    print(f"  p50: {stats['db_write_time']['p50']:.1f} ms")
    print(f"  p95: {stats['db_write_time']['p95']:.1f} ms")
    print(f"  mean: {stats['db_write_time']['mean']:.1f} ms")
    
    print(f"\nQueue Wait Time:")
    print(f"  p50: {stats['queue_wait_time']['p50']:.1f} ms")
    print(f"  p95: {stats['queue_wait_time']['p95']:.1f} ms")
    print(f"  mean: {stats['queue_wait_time']['mean']:.1f} ms")
    
    print("\n" + "=" * 80)
    print("3. QUEUE DYNAMICS")
    print("=" * 80)
    
    sq = stats['queue_depths']['summary_queue']
    wq = stats['queue_depths']['write_queue']
    
    print(f"\nSummary Queue:")
    print(f"  Max depth: {sq['max']}")
    print(f"  Mean depth: {sq['mean']:.1f}")
    print(f"  p95 depth: {sq['p95']:.1f}")
    if sq['max'] >= 190:  # Close to maxsize=200
        print(f"  ⚠️  WARNING: Queue reached {sq['max']}/200 (95%+ capacity)")
        print(f"     Producer likely blocked. Increase maxsize!")
    
    print(f"\nWrite Queue:")
    print(f"  Max depth: {wq['max']}")
    print(f"  Mean depth: {wq['mean']:.1f}")
    print(f"  p95 depth: {wq['p95']:.1f}")
    if wq['max'] >= 380:  # Close to backpressure threshold
        print(f"  ⚠️  WARNING: Write queue reached {wq['max']}/500 (76%+ capacity)")
        print(f"     Backpressure may have triggered. Check DB writer performance!")
    
    print("\n" + "=" * 80)
    print("4. WORKER UTILIZATION")
    print("=" * 80)
    
    workers = stats['workers']
    print(f"Max workers: {workers['max']}")
    print(f"Mean workers: {workers['mean']:.1f}")
    
    if workers['mean'] < 2.5:
        print(f"  ⚠️  WARNING: Low worker count. Adaptive scaling may not be working.")
    
    print("\n" + "=" * 80)
    print("5. PROVIDER STATISTICS")
    print("=" * 80)
    
    for provider, provider_stats in stats['providers'].items():
        print(f"\n{provider}:")
        print(f"  Calls: {provider_stats['calls']}")
        print(f"  Retries: {provider_stats['retries']} ({provider_stats['retries']/max(provider_stats['calls'],1)*100:.1f}%)")
        print(f"  Timeouts: {provider_stats['timeouts']} ({provider_stats['timeouts']/max(provider_stats['calls'],1)*100:.1f}%)")
        print(f"  Errors: {provider_stats['errors']} ({provider_stats['errors']/max(provider_stats['calls'],1)*100:.1f}%)")
        print(f"  p50 latency: {provider_stats['p50_ms']:.1f} ms")
        print(f"  p95 latency: {provider_stats['p95_ms']:.1f} ms")
        print(f"  p99 latency: {provider_stats['p99_ms']:.1f} ms")
        
        if provider_stats['p95_ms'] > 25000:  # 25 seconds
            print(f"  ⚠️  WARNING: p95 latency >25s. Circuit breaker may trigger.")
        if provider_stats['timeouts'] / max(provider_stats['calls'], 1) > 0.2:
            print(f"  ⚠️  WARNING: Timeout rate >20%. Circuit breaker may trigger.")
    
    print("\n" + "=" * 80)
    print("6. DATABASE WRITE STATISTICS")
    print("=" * 80)
    
    db = stats['db_batches']
    print(f"Total batches: {db['total']}")
    print(f"Average batch size: {db['avg_size']:.1f}")
    print(f"Average duration: {db['avg_duration_ms']:.1f} ms")
    print(f"p95 duration: {db['p95_duration_ms']:.1f} ms")
    
    if db['p95_duration_ms'] > 100:
        print(f"  ⚠️  WARNING: DB batch commits slow (p95 >100ms)")
    
    print("\n" + "=" * 80)
    print("7. CIRCUIT BREAKER EVENTS")
    print("=" * 80)
    
    cb_events = stats['circuit_breaker_events']
    print(f"Total events: {cb_events}")
    if cb_events > 10:
        print(f"  ⚠️  WARNING: High circuit breaker activity ({cb_events} events)")
        print(f"     May indicate provider thrashing. Check provider selection logic.")
    
    print("\n" + "=" * 80)
    print("8. BOTTLENECK IDENTIFICATION")
    print("=" * 80)
    
    bottlenecks = []
    
    # Check queue blocking
    if sq['max'] >= 190:
        bottlenecks.append(("Queue Blocking", "HIGH", 
            f"Summary queue reached {sq['max']}/200. Producer blocked. Increase maxsize to 1000."))
    
    # Check chunk fan-out
    if stats['map_llm_time']['mean'] > 20000:  # 20 seconds
        bottlenecks.append(("Chunk Fan-Out", "HIGH",
            f"Map LLM time {stats['map_llm_time']['mean']/1000:.1f}s suggests many chunks. Optimize chunking."))
    
    # Check worker utilization
    if workers['mean'] < 2.5:
        bottlenecks.append(("Low Worker Utilization", "MEDIUM",
            f"Only {workers['mean']:.1f} workers active. Check adaptive scaling."))
    
    # Check GPU saturation
    for provider, provider_stats in stats['providers'].items():
        if provider == 'ollama' and provider_stats['timeouts'] / max(provider_stats['calls'], 1) > 0.15:
            bottlenecks.append(("GPU Saturation", "HIGH",
                f"Ollama timeout rate {provider_stats['timeouts']/max(provider_stats['calls'],1)*100:.1f}%. Add concurrency semaphore."))
    
    # Check DB writer
    if db['p95_duration_ms'] > 100:
        bottlenecks.append(("DB Writer Slow", "MEDIUM",
            f"DB batch commits p95={db['p95_duration_ms']:.1f}ms. Check DB performance."))
    
    if bottlenecks:
        print("\nIdentified Bottlenecks:")
        for i, (name, severity, description) in enumerate(bottlenecks, 1):
            print(f"\n{i}. [{severity}] {name}")
            print(f"   {description}")
    else:
        print("\n✓ No major bottlenecks identified!")
    
    print("\n" + "=" * 80)
    print("9. RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    if sq['max'] >= 190:
        recommendations.append("1. Increase summary_queue maxsize from 200 to 1000")
    
    if stats['map_llm_time']['mean'] > 20000:
        recommendations.append("2. Optimize chunking: increase CHUNK_SIZE to 3500, reduce OVERLAP to 275")
    
    for provider, provider_stats in stats['providers'].items():
        if provider == 'ollama' and provider_stats['timeouts'] / max(provider_stats['calls'], 1) > 0.15:
            recommendations.append("3. Add concurrency semaphore (limit=4) to prevent GPU saturation")
            break
    
    if wq['max'] >= 380:
        recommendations.append("4. Increase write_queue maxsize from 500 to 1000")
    
    if recommendations:
        print("\nRecommended Actions:")
        for rec in recommendations:
            print(f"  • {rec}")
    else:
        print("\n✓ System performing well. No immediate optimizations needed.")
    
    # Export report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pipeline_report_385_{timestamp}.json"
    metrics.export_report(filename)
    print(f"\n✓ Full report exported to: {filename}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_385_chapter_performance()

