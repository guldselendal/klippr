#!/usr/bin/env python3
"""
Diagnostic tests for prompt output verification failures.

This script runs focused tests to identify why Ollama tests are failing.
Each test is isolated and provides specific error information.
"""
import os
import sys
import time
import requests

# Set environment to prevent async pipeline initialization
os.environ["USE_ASYNC_SUMMARIZATION"] = "false"
os.environ["SKIP_PIPELINE_INIT"] = "true"

from prompt_utils import count_sentences, truncate_to_sentences, FULL_SUMMARY_PROMPT_TEMPLATE, FULL_SUMMARY_SYSTEM
from llm_provider import call_ollama, call_llm
from llm_concurrency import LLMConcurrencyLimiter


def test_ollama_connectivity():
    """Test 1: Check if Ollama is accessible"""
    print("=" * 70)
    print("DIAGNOSTIC TEST 1: Ollama Connectivity")
    print("=" * 70)
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    print(f"[TEST] Checking Ollama at: {ollama_url}")
    
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        print(f"[TEST] ✓ Ollama is accessible")
        print(f"[TEST] Available models: {[m.get('name', 'unknown') for m in models]}")
        return True, None
    except requests.exceptions.ConnectionError as e:
        print(f"[TEST] ✗ Cannot connect to Ollama: {e}")
        print(f"[TEST] Make sure Ollama is running: ollama serve")
        return False, f"ConnectionError: {e}"
    except requests.exceptions.Timeout as e:
        print(f"[TEST] ✗ Ollama connection timeout: {e}")
        return False, f"Timeout: {e}"
    except Exception as e:
        print(f"[TEST] ✗ Unexpected error: {type(e).__name__}: {e}")
        return False, f"{type(e).__name__}: {e}"


def test_concurrency_limiter():
    """Test 2: Verify concurrency limiter is working"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 2: Concurrency Limiter")
    print("=" * 70)
    
    try:
        limiter = LLMConcurrencyLimiter()
        print(f"[TEST] ✓ Limiter initialized")
        
        metrics = limiter.get_metrics()
        print(f"[TEST] Current limits: {metrics['limits']}")
        print(f"[TEST] Current in-flight: {metrics['in_flight']}")
        
        # Test acquire/release
        print(f"[TEST] Testing acquire/release...")
        with limiter.acquire("ollama"):
            metrics_after = limiter.get_metrics()
            print(f"[TEST] After acquire - in-flight: {metrics_after['in_flight']}")
            assert metrics_after['in_flight']['ollama'] > 0, "Ollama in-flight should be > 0"
        
        metrics_final = limiter.get_metrics()
        print(f"[TEST] After release - in-flight: {metrics_final['in_flight']}")
        assert metrics_final['in_flight']['ollama'] == 0, "Ollama in-flight should be 0 after release"
        
        print(f"[TEST] ✓ Concurrency limiter working correctly")
        return True, None
    except Exception as e:
        print(f"[TEST] ✗ Concurrency limiter error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_simple_ollama_call():
    """Test 3: Make a simple Ollama call without concurrency limiter"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 3: Simple Ollama Call (No Limiter)")
    print("=" * 70)
    
    try:
        print(f"[TEST] Making direct Ollama API call...")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        
        # Use longer timeout for potentially slow models
        slow_models = ["gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "llama3:70b"]
        is_slow_model = any(slow in model.lower() for slow in slow_models)
        timeout = 300 if is_slow_model else 60
        
        print(f"[TEST] URL: {ollama_url}, Model: {model}, Timeout: {timeout}s")
        
        if is_slow_model:
            print(f"[TEST] ⚠️  Using slow model - this may take a while...")
        
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Say 'Hello, test!' and nothing else."}
                ],
                "stream": False
            },
            timeout=timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "message" not in result or "content" not in result["message"]:
            print(f"[TEST] ✗ Invalid response format: {result}")
            return False, "Invalid response format"
        
        content = result["message"]["content"].strip()
        print(f"[TEST] ✓ Ollama responded: {content[:100]}")
        return True, None
        
    except Exception as e:
        print(f"[TEST] ✗ Ollama call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_ollama_with_limiter():
    """Test 4: Make Ollama call through llm_provider (with limiter)"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 4: Ollama Call via llm_provider (With Limiter)")
    print("=" * 70)
    
    try:
        print(f"[TEST] Calling call_ollama() function...")
        start_time = time.time()
        
        result = call_ollama(
            prompt="Say 'Hello, test!' and nothing else.",
            system_prompt=None,
            model=None
        )
        
        duration = time.time() - start_time
        print(f"[TEST] ✓ Call completed in {duration:.2f}s")
        print(f"[TEST] Result: {result[:100]}")
        return True, None
        
    except Exception as e:
        print(f"[TEST] ✗ call_ollama() failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_ollama_with_call_llm():
    """Test 5: Make Ollama call via unified call_llm interface"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 5: Ollama Call via call_llm() (Unified Interface)")
    print("=" * 70)
    
    try:
        print(f"[TEST] Calling call_llm() with provider='ollama'...")
        start_time = time.time()
        
        result = call_llm(
            prompt="Say 'Hello, test!' and nothing else.",
            system_prompt=None,
            provider="ollama",
            model=None
        )
        
        duration = time.time() - start_time
        print(f"[TEST] Call completed in {duration:.2f}s")
        print(f"[TEST] Result type: {type(result)}")
        print(f"[TEST] Result length: {len(result) if result else 0} chars")
        print(f"[TEST] Result value: {repr(result)}")
        
        if not result or len(result.strip()) == 0:
            print(f"[TEST] ✗ call_llm() returned empty or whitespace-only result")
            print(f"[TEST] This indicates the response parsing is failing")
            return False, "Empty result from call_llm"
        
        print(f"[TEST] ✓ Result received: {result[:100]}")
        return True, None
        
    except Exception as e:
        print(f"[TEST] ✗ call_llm() failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_prompt_template():
    """Test 6: Verify prompt template formatting"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 6: Prompt Template Formatting")
    print("=" * 70)
    
    try:
        content = "This is a test chapter about machine learning."
        title = "Test Chapter"
        
        prompt = FULL_SUMMARY_PROMPT_TEMPLATE.format(title=title, content=content)
        print(f"[TEST] ✓ Prompt template formatted successfully")
        print(f"[TEST] Prompt length: {len(prompt)} chars")
        print(f"[TEST] Prompt preview: {prompt[:200]}...")
        
        # Check for required elements
        assert "EXACTLY 10-12" in prompt, "Prompt should mention exact sentence count"
        assert title in prompt, "Prompt should include title"
        assert content in prompt, "Prompt should include content"
        
        print(f"[TEST] ✓ Prompt contains required elements")
        return True, None
        
    except Exception as e:
        print(f"[TEST] ✗ Prompt template error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_sentence_counting():
    """Test 7: Verify sentence counting on actual LLM output"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 7: Sentence Counting on LLM Output")
    print("=" * 70)
    
    try:
        # Get a real summary from Ollama
        print(f"[TEST] Getting summary from Ollama...")
        content = "Machine learning is important. It uses algorithms. Neural networks are powerful. Deep learning is advanced."
        title = "Test"
        
        prompt = FULL_SUMMARY_PROMPT_TEMPLATE.format(title=title, content=content)
        system_prompt = FULL_SUMMARY_SYSTEM
        
        # Make direct call to see raw response
        print(f"[TEST] Making direct Ollama API call to inspect response...")
        import requests
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        # Use faster model for diagnostics, or fallback to configured model
        model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        
        # Check if model is too slow (large models)
        slow_models = ["gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "llama3:70b"]
        is_slow_model = any(slow in model.lower() for slow in slow_models)
        
        if is_slow_model:
            print(f"[TEST] ⚠️  Warning: Using potentially slow model '{model}'")
            print(f"[TEST] Consider using 'phi3:mini' for faster diagnostics")
            # Use longer timeout for slow models
            timeout = 300  # 5 minutes for large models
        else:
            timeout = 120  # 2 minutes for normal models
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        print(f"[TEST] Request details:")
        print(f"  URL: {ollama_url}/api/chat")
        print(f"  Model: {model}")
        print(f"  Timeout: {timeout}s")
        print(f"  Messages count: {len(messages)}")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  System prompt length: {len(system_prompt) if system_prompt else 0} chars")
        
        print(f"[TEST] Sending request (this may take a while for large models)...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                },
                timeout=timeout
            )
            elapsed = time.time() - start_time
            print(f"[TEST] Request completed in {elapsed:.2f}s")
        except requests.exceptions.ReadTimeout:
            elapsed = time.time() - start_time
            print(f"[TEST] ✗ Request timed out after {elapsed:.2f}s (timeout={timeout}s)")
            print(f"[TEST] Model '{model}' may be too slow. Try:")
            print(f"  - Using a faster model: export OLLAMA_MODEL=phi3:mini")
            print(f"  - Increasing timeout in test")
            raise
        
        print(f"[TEST] Response status: {response.status_code}")
        print(f"[TEST] Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"[TEST] ✗ HTTP error: {response.text[:500]}")
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
        
        result = response.json()
        print(f"[TEST] Raw response keys: {list(result.keys())}")
        print(f"[TEST] Full response structure:")
        import json
        print(json.dumps(result, indent=2)[:1000])  # First 1000 chars
        
        # Check response structure
        if "message" not in result:
            print(f"[TEST] ✗ No 'message' key in response")
            return False, "Invalid response structure: missing 'message'"
        
        if "content" not in result["message"]:
            print(f"[TEST] ✗ No 'content' key in message")
            print(f"[TEST] Message keys: {list(result['message'].keys())}")
            return False, "Invalid response structure: missing 'content'"
        
        raw_content = result["message"]["content"]
        print(f"[TEST] Raw content type: {type(raw_content)}")
        print(f"[TEST] Raw content length: {len(raw_content) if raw_content else 0}")
        print(f"[TEST] Raw content (first 500 chars): {raw_content[:500] if raw_content else '(empty)'}")
        
        if not raw_content or len(raw_content.strip()) == 0:
            print(f"[TEST] ✗ Content is empty or whitespace only")
            return False, "Empty content in response"
        
        # Now test via call_llm
        print(f"\n[TEST] Testing via call_llm() function...")
        summary = call_llm(prompt=prompt, system_prompt=system_prompt, provider="ollama")
        print(f"[TEST] Summary from call_llm: {len(summary)} chars")
        print(f"[TEST] Summary preview: {summary[:200]}...")
        
        if len(summary) == 0:
            print(f"[TEST] ✗ call_llm returned empty string")
            print(f"[TEST] This suggests call_llm is not extracting content correctly")
            return False, "call_llm returned empty string"
        
        # Count sentences
        sentence_count = count_sentences(summary)
        print(f"[TEST] Sentence count: {sentence_count}")
        
        # Check if count is reasonable
        if sentence_count == 0:
            print(f"[TEST] ⚠️  Warning: Sentence count is 0 (may indicate counting issue)")
        elif sentence_count < 5:
            print(f"[TEST] ⚠️  Warning: Only {sentence_count} sentences (may be too low)")
        else:
            print(f"[TEST] ✓ Sentence count looks reasonable")
        
        return True, None
        
    except Exception as e:
        print(f"[TEST] ✗ Sentence counting test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_concurrent_requests():
    """Test 8: Test concurrency limiter with multiple requests"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 8: Concurrent Requests (Limiter Behavior)")
    print("=" * 70)
    
    try:
        import threading
        
        limiter = LLMConcurrencyLimiter()
        results = []
        errors = []
        
        def make_request(request_id):
            try:
                print(f"[TEST] Request {request_id}: Starting...")
                start = time.time()
                with limiter.acquire("ollama"):
                    metrics = limiter.get_metrics()
                    print(f"[TEST] Request {request_id}: Acquired (in-flight: {metrics['in_flight']['ollama']})")
                    
                    # Make a quick call
                    result = call_llm(
                        prompt=f"Say 'Request {request_id}' and nothing else.",
                        provider="ollama"
                    )
                    duration = time.time() - start
                    print(f"[TEST] Request {request_id}: Completed in {duration:.2f}s")
                    results.append((request_id, True, duration))
            except Exception as e:
                print(f"[TEST] Request {request_id}: Failed - {type(e).__name__}: {e}")
                errors.append((request_id, str(e)))
                results.append((request_id, False, 0))
        
        # Launch 6 requests (should only allow 4 concurrent)
        print(f"[TEST] Launching 6 requests (limiter cap: 4)...")
        threads = []
        for i in range(6):
            t = threading.Thread(target=make_request, args=(i+1,))
            threads.append(t)
            t.start()
            time.sleep(0.1)  # Stagger starts
        
        for t in threads:
            t.join()
        
        successful = sum(1 for r in results if r[1])
        print(f"[TEST] Results: {successful}/6 requests succeeded")
        print(f"[TEST] Errors: {len(errors)}")
        
        if len(errors) > 0:
            print(f"[TEST] Error details:")
            for req_id, error in errors:
                print(f"  Request {req_id}: {error}")
        
        if successful >= 4:
            print(f"[TEST] ✓ Concurrency limiter working (at least 4 succeeded)")
            return True, None
        else:
            print(f"[TEST] ✗ Too many failures (expected at least 4 successes)")
            return False, f"Only {successful}/6 succeeded"
        
    except Exception as e:
        print(f"[TEST] ✗ Concurrent requests test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def test_full_summary_generation():
    """Test 9: Full summary generation with post-processing"""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST 9: Full Summary Generation (End-to-End)")
    print("=" * 70)
    
    try:
        content = """
        Machine learning is a subset of artificial intelligence. 
        It enables computers to learn from data. 
        There are three main types: supervised, unsupervised, and reinforcement learning.
        Neural networks are key components. 
        Deep learning uses multiple layers. 
        Applications include image recognition and NLP.
        The field evolves rapidly. 
        Understanding fundamentals is essential.
        """ * 5  # ~1000 chars
        
        title = "Introduction to Machine Learning"
        
        model = os.getenv("OLLAMA_MODEL", "phi3:mini")
        slow_models = ["gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "llama3:70b"]
        is_slow_model = any(slow in model.lower() for slow in slow_models)
        
        print(f"[TEST] Content length: {len(content)} chars")
        print(f"[TEST] Model: {model}")
        if is_slow_model:
            print(f"[TEST] ⚠️  Using slow model - this may take 2-5 minutes...")
        print(f"[TEST] Generating summary...")
        
        prompt = FULL_SUMMARY_PROMPT_TEMPLATE.format(title=title, content=content)
        system_prompt = FULL_SUMMARY_SYSTEM
        
        start = time.time()
        summary = call_llm(prompt=prompt, system_prompt=system_prompt, provider="ollama")
        duration = time.time() - start
        
        print(f"[TEST] Summary generated in {duration:.2f}s")
        print(f"[TEST] Raw summary length: {len(summary)} chars")
        
        # Post-process (same logic as summary_pipeline)
        sentence_count = count_sentences(summary)
        print(f"[TEST] Initial sentence count: {sentence_count}")
        
        if sentence_count > 12:
            print(f"[TEST] Truncating from {sentence_count} to 12...")
            summary = truncate_to_sentences(summary, max_sentences=12)
            sentence_count = count_sentences(summary)
            print(f"[TEST] After truncation: {sentence_count} sentences")
        elif sentence_count < 10:
            # Try to extend if too short
            print(f"[TEST] Summary too short ({sentence_count} sentences), attempting extension...")
            try:
                from prompt_utils import EXTEND_SUMMARY_PROMPT_TEMPLATE
                needed = 10 - sentence_count
                target_count = min(12, sentence_count + needed + 2)
                extend_prompt = EXTEND_SUMMARY_PROMPT_TEMPLATE.format(
                    current_count=sentence_count,
                    previous_summary=summary,
                    title=title,
                    content=content,
                    needed=needed,
                    target_count=target_count
                )
                print(f"[TEST] Requesting extension (need {needed} more sentences)...")
                extended_summary = call_llm(prompt=extend_prompt, system_prompt=system_prompt, provider="ollama")
                extended_count = count_sentences(extended_summary)
                print(f"[TEST] Extended summary has {extended_count} sentences")
                
                if extended_count >= 10:
                    summary = extended_summary
                    sentence_count = extended_count
                    if sentence_count > 12:
                        summary = truncate_to_sentences(summary, max_sentences=12)
                        sentence_count = count_sentences(summary)
                else:
                    print(f"[TEST] ⚠️  Extension still insufficient ({extended_count} sentences)")
            except Exception as e:
                print(f"[TEST] ⚠️  Extension failed: {e}")
        
        print(f"[TEST] Final summary: {len(summary)} chars, {sentence_count} sentences")
        print(f"[TEST] Summary preview: {summary[:300]}...")
        
        # Verify
        if 10 <= sentence_count <= 12:
            print(f"[TEST] ✓ Sentence count is correct: {sentence_count}")
            return True, None
        else:
            print(f"[TEST] ✗ Sentence count out of range: {sentence_count} (expected 10-12)")
            return False, f"Sentence count: {sentence_count}"
        
    except Exception as e:
        print(f"[TEST] ✗ Full summary generation failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 70)
    print("PROMPT OUTPUT DIAGNOSTIC TESTS")
    print("=" * 70)
    print("\nThese tests diagnose why Ollama tests are failing.\n")
    
    # Check model configuration
    model = os.getenv("OLLAMA_MODEL", "phi3:mini")
    slow_models = ["gpt-oss:20b", "llama3.2:3b", "llama3.1:8b", "llama3:70b"]
    is_slow_model = any(slow in model.lower() for slow in slow_models)
    
    if is_slow_model:
        print(f"⚠️  WARNING: Using potentially slow model '{model}'")
        print(f"   Tests may take 2-5 minutes per request.")
        print(f"   For faster diagnostics, use: export OLLAMA_MODEL=phi3:mini\n")
    
    tests = [
        ("Ollama Connectivity", test_ollama_connectivity),
        ("Concurrency Limiter", test_concurrency_limiter),
        ("Simple Ollama Call", test_simple_ollama_call),
        ("Ollama via llm_provider", test_ollama_with_limiter),
        ("Ollama via call_llm", test_ollama_with_call_llm),
        ("Prompt Template", test_prompt_template),
        ("Sentence Counting", test_sentence_counting),
        ("Concurrent Requests", test_concurrent_requests),
        ("Full Summary Generation", test_full_summary_generation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed, error = test_func()
            results.append((test_name, passed, error))
        except Exception as e:
            print(f"\n[TEST] ✗ {test_name} crashed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, f"Crash: {type(e).__name__}: {e}"))
    
    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed, error in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"       Error: {error}")
    
    total_passed = sum(1 for _, passed, _ in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    # Identify likely root cause
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    
    if not results[0][1]:  # Ollama connectivity failed
        print("🔴 ROOT CAUSE: Ollama is not accessible")
        print("   Solution: Start Ollama with 'ollama serve'")
    elif not results[1][1]:  # Limiter failed
        print("🔴 ROOT CAUSE: Concurrency limiter is broken")
        print("   Solution: Check llm_concurrency.py implementation")
    elif not results[2][1] and not results[3][1] and not results[4][1]:
        print("🔴 ROOT CAUSE: All Ollama call methods failing")
        print("   Solution: Check Ollama model availability and API compatibility")
    elif not results[5][1]:  # Prompt template failed
        print("🔴 ROOT CAUSE: Prompt template formatting issue")
        print("   Solution: Check prompt_utils.py templates")
    elif not results[6][1]:  # Sentence counting failed
        print("🟡 ROOT CAUSE: Sentence counting may be inaccurate")
        print("   Solution: Review count_sentences() function")
    elif not results[7][1]:  # Concurrent requests failed
        print("🟡 ROOT CAUSE: Concurrency limiter blocking too aggressively")
        print("   Solution: Check limiter limits and blocking behavior")
    elif not results[8][1]:  # Full generation failed
        print("🟡 ROOT CAUSE: End-to-end summary generation issue")
        print("   Solution: Check LLM response format and post-processing")
    else:
        print("🟢 All diagnostic tests passed - issue may be in test_prompt_output.py")
        print("   Solution: Review test_prompt_output.py test logic")
    
    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())

