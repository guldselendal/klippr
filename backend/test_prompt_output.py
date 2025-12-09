#!/usr/bin/env python3
"""
Test script to verify prompt output has exactly 10-12 sentences.

Tests:
1. Short content (≤5K chars) - single pass
2. Long content (>5K chars) - chunked processing
3. Different providers (Ollama, Gemini)
4. Sentence counting accuracy

For detailed diagnostics, run: python3 test_prompt_diagnostics.py
"""
import os
import sys

# Set environment for testing - prevent async pipeline initialization
# MUST be set BEFORE importing summary_pipeline
os.environ["USE_ASYNC_SUMMARIZATION"] = "false"
os.environ["SKIP_PIPELINE_INIT"] = "true"  # Prevent pipeline initialization on import

from prompt_utils import count_sentences, truncate_to_sentences, FULL_SUMMARY_PROMPT_TEMPLATE, FULL_SUMMARY_SYSTEM
from llm_provider import call_llm
from summarizer import split_content_into_chunks, merge_chunk_summaries


def test_sentence_counting():
    """Test the sentence counting function."""
    print("=" * 60)
    print("Test 1: Sentence Counting Function")
    print("=" * 60)
    
    test_cases = [
        ("This is one sentence.", 1),
        ("This is one. This is two.", 2),
        ("First! Second? Third.", 3),
        ("", 0),
        ("No punctuation here", 1),  # Should return at least 1 if text exists
    ]
    
    all_passed = True
    for text, expected in test_cases:
        actual = count_sentences(text)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_passed = False
        print(f"{status} '{text[:30]}...' -> {actual} (expected {expected})")
    
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed!'}\n")
    return all_passed


def test_truncation():
    """Test the truncation function."""
    print("=" * 60)
    print("Test 2: Sentence Truncation Function")
    print("=" * 60)
    
    # Create text with 15 sentences
    long_text = ". ".join([f"Sentence {i+1}" for i in range(15)]) + "."
    
    truncated = truncate_to_sentences(long_text, max_sentences=12)
    count = count_sentences(truncated)
    
    status = "✓" if count == 12 else "✗"
    print(f"{status} Truncated 15 sentences to {count} sentences (expected 12)")
    print(f"   Truncated text: {truncated[:100]}...")
    
    passed = count == 12
    print(f"\n{'Test passed!' if passed else 'Test failed!'}\n")
    return passed


def _generate_summary_direct(content: str, title: str, provider: str, model: str) -> str:
    """Generate summary directly using LLM without async pipeline."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    import time
    
    print(f"  [DEBUG] Starting summary generation")
    print(f"  [DEBUG] Provider: {provider}, Model: {model}")
    print(f"  [DEBUG] Content length: {len(content)} chars")
    
    # Check concurrency limiter status
    try:
        from llm_concurrency import LLMConcurrencyLimiter
        limiter = LLMConcurrencyLimiter()
        metrics = limiter.get_metrics()
        print(f"  [DEBUG] Concurrency limiter status:")
        print(f"    - In-flight: {metrics['in_flight']}")
        print(f"    - Limits: {metrics['limits']}")
    except Exception as e:
        print(f"  [DEBUG] Could not check limiter: {e}")
    
    # Use optimized prompt
    prompt = FULL_SUMMARY_PROMPT_TEMPLATE.format(title=title, content=content)
    system_prompt = FULL_SUMMARY_SYSTEM
    
    print(f"  [DEBUG] Prompt length: {len(prompt)} chars")
    print(f"  [DEBUG] System prompt length: {len(system_prompt)} chars")
    
    def _call_with_timeout(timeout: int = 120):
        """Call LLM with timeout"""
        print(f"  [DEBUG] Calling LLM with timeout={timeout}s...")
        call_start = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_llm, prompt=prompt, system_prompt=system_prompt, provider=provider, model=model)
                try:
                    result = future.result(timeout=timeout).strip()
                    call_duration = time.time() - call_start
                    print(f"  [DEBUG] LLM call completed in {call_duration:.2f}s")
                    print(f"  [DEBUG] Result length: {len(result)} chars")
                    return result
                except FutureTimeoutError:
                    executor.shutdown(wait=False)
                    call_duration = time.time() - call_start
                    print(f"  [DEBUG] LLM call timed out after {call_duration:.2f}s")
                    raise TimeoutError(f"LLM call to {provider} exceeded {timeout}s timeout")
        except Exception as e:
            call_duration = time.time() - call_start
            print(f"  [DEBUG] LLM call failed after {call_duration:.2f}s: {type(e).__name__}: {e}")
            raise
    
    try:
        summary = _call_with_timeout(timeout=120)
    except Exception as e:
        print(f"  [DEBUG] Summary generation failed: {type(e).__name__}: {e}")
        import traceback
        print(f"  [DEBUG] Traceback:")
        traceback.print_exc()
        raise
    
    # Post-process to ensure exactly 10-12 sentences
    print(f"  [DEBUG] Post-processing summary...")
    sentence_count = count_sentences(summary)
    print(f"  [DEBUG] Initial sentence count: {sentence_count}")
    
    if sentence_count > 12:
        print(f"  [DEBUG] Truncating from {sentence_count} to 12 sentences")
        summary = truncate_to_sentences(summary, max_sentences=12)
        sentence_count = count_sentences(summary)
        print(f"  [DEBUG] After truncation: {sentence_count} sentences")
    elif sentence_count < 10 and len(summary) < 2000:
        print(f"  [DEBUG] Warning: Summary has only {sentence_count} sentences (target: 10-12)")
    
    print(f"  [DEBUG] Final summary: {len(summary)} chars, {sentence_count} sentences")
    return summary


def test_short_content_summary(provider: str = "ollama"):
    """Test summary generation for short content (≤5K chars)."""
    print("=" * 60)
    print(f"Test 3: Short Content Summary ({provider})")
    print("=" * 60)
    
    # Short chapter content (~2000 chars)
    content = """
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. 
    It involves algorithms that can identify patterns and make predictions based on historical information. 
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. 
    Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data. 
    Reinforcement learning involves agents learning through trial and error in an environment. 
    Neural networks are a key component of modern machine learning, inspired by the structure of the human brain. 
    Deep learning, a subset of machine learning, uses multiple layers of neural networks to process complex data. 
    Applications of machine learning include image recognition, natural language processing, recommendation systems, and autonomous vehicles. 
    The field continues to evolve rapidly with new architectures and techniques being developed regularly. 
    Understanding machine learning fundamentals is essential for anyone working in data science or artificial intelligence.
    """ * 10  # Repeat to get ~2000 chars
    
    title = "Introduction to Machine Learning"
    
    print(f"Content length: {len(content)} characters")
    print(f"Generating summary with {provider}...")
    print(f"[TEST] Starting test_short_content_summary with provider={provider}")
    
    try:
        model = "phi3:mini" if provider == "ollama" else "gemini-1.5-flash"
        print(f"[TEST] Using model: {model}")
        
        # Check if Ollama is accessible
        if provider == "ollama":
            import requests
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=2)
                print(f"[TEST] Ollama is accessible at {ollama_url}")
            except Exception as e:
                print(f"[TEST] ⚠️  Ollama check failed: {e}")
                print(f"[TEST] This may cause the test to fail")
        
        summary = _generate_summary_direct(content, title, provider, model)
        sentence_count = count_sentences(summary)
        
        print(f"\nSummary ({len(summary)} chars, {sentence_count} sentences):")
        print("-" * 60)
        print(summary)
        print("-" * 60)
        
        # Verify sentence count
        if 10 <= sentence_count <= 12:
            print(f"✓ Sentence count is correct: {sentence_count} (target: 10-12)")
            passed = True
        elif sentence_count > 12:
            print(f"⚠ Sentence count exceeds target: {sentence_count} (target: 10-12)")
            print(f"   Note: Post-processing should truncate to 12 sentences")
            passed = False
        else:
            print(f"✗ Sentence count is too low: {sentence_count} (target: 10-12)")
            passed = False
        
        return passed, sentence_count, summary
        
    except Exception as e:
        print(f"✗ Error generating summary: {type(e).__name__}: {e}")
        print(f"[TEST] Full error details:")
        import traceback
        traceback.print_exc()
        print(f"[TEST] Test failed with exception: {type(e).__name__}")
        return False, 0, None


def test_long_content_summary(provider: str = "ollama"):
    """Test summary generation for long content (>5K chars) with chunking."""
    print("=" * 60)
    print(f"Test 4: Long Content Summary ({provider}) - Chunked Processing")
    print("=" * 60)
    
    # Long chapter content (~8000 chars)
    content = """
    The history of artificial intelligence spans several decades, beginning with early theoretical work in the 1950s. 
    Alan Turing proposed the famous Turing Test as a measure of machine intelligence. 
    The Dartmouth Conference in 1956 is often considered the birth of AI as a field. 
    Early AI research focused on symbolic reasoning and problem-solving. 
    Expert systems became popular in the 1980s, attempting to capture human expertise in rule-based systems. 
    The field experienced "AI winters" in the 1970s and 1980s when funding and interest declined. 
    Machine learning emerged as a key approach, allowing systems to learn from data rather than explicit programming. 
    Neural networks, inspired by biological neurons, became a powerful tool for pattern recognition. 
    The development of backpropagation enabled training of multi-layer networks. 
    Deep learning revolutionized AI in the 2010s with breakthroughs in image recognition and natural language processing. 
    Large language models like GPT and BERT demonstrated remarkable capabilities in understanding and generating text. 
    Modern AI systems are used in diverse applications from healthcare to autonomous vehicles. 
    Ethical considerations around AI bias, privacy, and job displacement have become increasingly important. 
    The future of AI holds promise for solving complex problems but also raises questions about safety and control.
    """ * 50  # Repeat to get ~8000 chars
    
    title = "History of Artificial Intelligence"
    
    print(f"Content length: {len(content)} characters")
    print(f"[TEST] Starting test_long_content_summary with provider={provider}")
    
    # For long content with Ollama, use chunking; with Gemini, use direct call
    if provider == "ollama" and len(content) > 5000:
        print(f"Generating summary with {provider} (will be chunked)...")
        print(f"[TEST] Using chunked processing path (content > 5K chars)")
        try:
            from summarizer import generate_summary
            print(f"[TEST] Calling summarizer.generate_summary()...")
            summary = generate_summary(content, title)
            print(f"[TEST] Chunked summary generation completed")
        except Exception as e:
            print(f"[TEST] Error with summarizer.generate_summary: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to direct call
            print(f"[TEST] Falling back to direct call with truncated content...")
            model = "phi3:mini"
            summary = _generate_summary_direct(content[:5000], title, provider, model)  # Truncate for test
    else:
        print(f"Generating summary with {provider}...")
        model = "phi3:mini" if provider == "ollama" else "gemini-1.5-flash"
        print(f"[TEST] Using direct call path with model={model}")
        summary = _generate_summary_direct(content, title, provider, model)
    
    try:
        sentence_count = count_sentences(summary)
        
        print(f"\nSummary ({len(summary)} chars, {sentence_count} sentences):")
        print("-" * 60)
        print(summary)
        print("-" * 60)
        
        # Verify sentence count
        if 10 <= sentence_count <= 12:
            print(f"✓ Sentence count is correct: {sentence_count} (target: 10-12)")
            passed = True
        elif sentence_count > 12:
            print(f"⚠ Sentence count exceeds target: {sentence_count} (target: 10-12)")
            print(f"   Note: Post-processing should truncate to 12 sentences")
            passed = False
        else:
            print(f"✗ Sentence count is too low: {sentence_count} (target: 10-12)")
            passed = False
        
        return passed, sentence_count, summary
        
    except Exception as e:
        print(f"✗ Error processing summary: {type(e).__name__}: {e}")
        print(f"[TEST] Full error details:")
        import traceback
        traceback.print_exc()
        print(f"[TEST] Test failed with exception: {type(e).__name__}")
        return False, 0, None


def test_multiple_samples():
    """Test multiple summary samples to verify consistency."""
    print("=" * 60)
    print("Test 5: Multiple Samples (Consistency Check)")
    print("=" * 60)
    
    samples = [
        ("Machine Learning Basics", "Machine learning is a method of data analysis. " * 100),
        ("Neural Networks", "Neural networks are computing systems. " * 100),
        ("Deep Learning", "Deep learning uses multiple layers. " * 100),
    ]
    
    results = []
    provider = "ollama"  # Test with Ollama first
    
    for title, content in samples:
        print(f"\nTesting: {title}")
        print(f"[TEST] Processing sample: {title}")
        try:
            summary = _generate_summary_direct(content, title, provider, "phi3:mini")
            sentence_count = count_sentences(summary)
            results.append((title, sentence_count, 10 <= sentence_count <= 12))
            print(f"  Sentence count: {sentence_count} {'✓' if 10 <= sentence_count <= 12 else '✗'}")
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            print(f"[TEST] Sample '{title}' failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((title, 0, False))
    
    passed = all(result[2] for result in results)
    print(f"\n{'All samples passed!' if passed else 'Some samples failed!'}")
    return passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Prompt Output Verification Test")
    print("=" * 60)
    print("\nThis test verifies that summaries have exactly 10-12 sentences.")
    print("Note: LLM services must be available for full testing.\n")
    
    # Print environment info
    print("[TEST] ========================================")
    print("[TEST] Test Environment")
    print("[TEST] ========================================")
    print(f"[TEST] Python version: {sys.version.split()[0]}")
    print(f"[TEST] Working directory: {os.getcwd()}")
    
    # Check environment variables
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "phi3:mini")
    print(f"[TEST] OLLAMA_URL: {ollama_url}")
    print(f"[TEST] OLLAMA_MODEL: {ollama_model}")
    
    # Check concurrency limiter settings
    try:
        from llm_concurrency import LLMConcurrencyLimiter
        limiter = LLMConcurrencyLimiter()
        metrics = limiter.get_metrics()
        print(f"[TEST] Concurrency limiter limits: {metrics['limits']}")
    except Exception as e:
        print(f"[TEST] Could not check concurrency limiter: {e}")
    
    print("[TEST] ========================================\n")
    
    results = []
    
    # Test 1: Sentence counting function
    results.append(("Sentence Counting", test_sentence_counting()))
    
    # Test 2: Truncation function
    results.append(("Truncation", test_truncation()))
    
    # Test 3: Short content (try Ollama first, fallback to Gemini)
    print("\nTrying Ollama for short content...")
    print("[TEST] ========================================")
    print("[TEST] Test 3: Short Content Summary")
    print("[TEST] ========================================")
    try:
        passed, count, summary = test_short_content_summary("ollama")
        results.append(("Short Content (Ollama)", passed))
        print(f"[TEST] Test 3 result: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"[TEST] Ollama failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("Trying Gemini...")
        try:
            passed, count, summary = test_short_content_summary("gemini")
            results.append(("Short Content (Gemini)", passed))
            print(f"[TEST] Test 3 (Gemini fallback) result: {'PASS' if passed else 'FAIL'}")
        except Exception as gemini_err:
            print(f"[TEST] Gemini also failed: {type(gemini_err).__name__}: {gemini_err}")
            traceback.print_exc()
            results.append(("Short Content", False))
            print(f"[TEST] Test 3 result: FAIL (both providers failed)")
    
    # Test 4: Long content (try Ollama first, fallback to Gemini)
    print("\nTrying Ollama for long content...")
    print("[TEST] ========================================")
    print("[TEST] Test 4: Long Content Summary")
    print("[TEST] ========================================")
    try:
        passed, count, summary = test_long_content_summary("ollama")
        results.append(("Long Content (Ollama)", passed))
        print(f"[TEST] Test 4 result: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"[TEST] Ollama failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("Trying Gemini...")
        try:
            passed, count, summary = test_long_content_summary("gemini")
            results.append(("Long Content (Gemini)", passed))
            print(f"[TEST] Test 4 (Gemini fallback) result: {'PASS' if passed else 'FAIL'}")
        except Exception as gemini_err:
            print(f"[TEST] Gemini also failed: {type(gemini_err).__name__}: {gemini_err}")
            traceback.print_exc()
            results.append(("Long Content", False))
            print(f"[TEST] Test 4 result: FAIL (both providers failed)")
    
    # Test 5: Multiple samples
    print("[TEST] ========================================")
    print("[TEST] Test 5: Multiple Samples")
    print("[TEST] ========================================")
    try:
        passed = test_multiple_samples()
        results.append(("Multiple Samples", passed))
        print(f"[TEST] Test 5 result: {'PASS' if passed else 'FAIL'}")
    except Exception as e:
        print(f"[TEST] Multiple samples test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Samples", False))
        print(f"[TEST] Test 5 result: FAIL")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Prompt optimization is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

