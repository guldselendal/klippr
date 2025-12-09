# Summarizer Optimization - Code Diffs

## File 1: New `token_utils.py`

```python
"""
Token counting and token-aware chunking utilities.
Optimizes chunking for LLM context windows.
"""
import re
from typing import List

# Conservative estimate: 1 token ≈ 4 characters (for English)
TOKENS_PER_CHAR = 0.25
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Conservative estimate: 1 token ≈ 4 characters.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return int(len(text) * TOKENS_PER_CHAR)


def split_by_tokens(content: str, target_tokens: int = 1500, overlap_tokens: int = 150) -> List[str]:
    """
    Split content into chunks based on token count, not character count.
    More accurate for LLM context window management.
    
    Args:
        content: The content to split
        target_tokens: Target tokens per chunk (default: 1500, ~75% of phi3:mini context)
        overlap_tokens: Overlap in tokens (default: 150, ~10%)
    
    Returns:
        List of content chunks
    """
    if not content:
        return []
    
    # Estimate total tokens
    total_tokens = estimate_tokens(content)
    
    # If content fits in one chunk, return as-is
    if total_tokens <= target_tokens:
        return [content]
    
    # Convert token targets to character estimates
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + target_chars
        
        # Try to break at sentence boundary if possible
        if end < len(content):
            # Look for sentence endings near the chunk boundary (within 200 chars)
            for i in range(end, max(start + target_chars - 200, start), -1):
                if i < len(content) and content[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = content[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap_chars
        
        # Find nearest sentence boundary for overlap start
        if start > 0 and start < len(content):
            for i in range(start, max(start - 100, 0), -1):
                if i < len(content) and content[i] in '.!?\n':
                    start = i + 1
                    break
        
        if start >= len(content):
            break
    
    return chunks


def adaptive_chunk_size(content: str, base_tokens: int = 1500, model: str = "phi3:mini") -> int:
    """
    Adaptively determine chunk size based on content characteristics and model.
    
    Args:
        content: The content to analyze
        base_tokens: Base target tokens per chunk
        model: Model name (for context window awareness)
    
    Returns:
        Optimal chunk size in tokens
    """
    total_tokens = estimate_tokens(content)
    
    # Model-specific context windows (approximate)
    context_windows = {
        "phi3:mini": 3200,
        "llama3.2:3b": 128000,
        "gemini-1.5-flash": 1000000,
    }
    
    max_tokens = context_windows.get(model, 3200)
    safe_max = int(max_tokens * 0.75)  # Use 75% of context window
    
    # For very long content, use larger chunks to reduce merge overhead
    if total_tokens > 10000:
        return min(int(base_tokens * 1.3), safe_max)
    
    # For medium content, use base size
    if total_tokens > 5000:
        return min(base_tokens, safe_max)
    
    # For short content, use smaller chunks (but still reasonable)
    return max(int(base_tokens * 0.8), 1000)
```

## File 2: Modified `summarizer.py`

### Key Changes:
1. Import token_utils
2. Use token-aware chunking
3. Add early stopping support
4. Optimize chunk size calculation

```diff
--- a/backend/summarizer.py
+++ b/backend/summarizer.py
@@ -7,6 +7,7 @@
 from typing import Optional, Tuple, List, Dict
 import os
 import time
+import asyncio
 from concurrent.futures import ThreadPoolExecutor, as_completed
 from dotenv import load_dotenv
 from llm_provider import call_llm
@@ -14,6 +15,12 @@ from prompt_utils import (
     COMBINE_PROMPT_TEMPLATE, COMBINE_SYSTEM,
     count_sentences, truncate_to_sentences
 )
+try:
+    from token_utils import split_by_tokens, adaptive_chunk_size, estimate_tokens
+    USE_TOKEN_CHUNKING = os.getenv("SUMMARY_USE_TOKEN_CHUNKING", "true").lower() == "true"
+except ImportError:
+    USE_TOKEN_CHUNKING = False
+    split_by_tokens = None
 
 load_dotenv()
 
@@ -191,7 +198,7 @@ def generate_summary(content: str, title: str, max_length: int = 2000) -> Opti
     # Determine chunk size and overlap based on content length
     # Use chunked approach for chapters longer than 5000 characters
     CHUNK_THRESHOLD = 5000
-    CHUNK_SIZE = 2000  # Reduced from 3000 for better reliability under load
+    CHUNK_SIZE = 2000  # Legacy: character-based (fallback)
     OVERLAP = 200      # Reduced from 400 to 200 for optimization (Step 1.2)
     
     if len(content) > CHUNK_THRESHOLD:
@@ -199,7 +206,20 @@ def generate_summary(content: str, title: str, max_length: int = 2000) -> Opti
         print(f"  Using chunked parallel summarization for chapter '{title}' ({len(content)} chars)")
         
         # Split content into chunks
-        chunks = split_content_into_chunks(content, CHUNK_SIZE, OVERLAP)
+        if USE_TOKEN_CHUNKING and split_by_tokens:
+            # Use token-aware chunking (optimized)
+            target_tokens = int(os.getenv("SUMMARY_CHUNK_SIZE_TOKENS", 1500))
+            overlap_tokens = int(os.getenv("SUMMARY_OVERLAP_TOKENS", 150))
+            
+            # Adaptive chunk sizing
+            if os.getenv("SUMMARY_ADAPTIVE_CHUNKS", "true").lower() == "true":
+                target_tokens = adaptive_chunk_size(content, base_tokens=target_tokens, model="phi3:mini")
+            
+            chunks = split_by_tokens(content, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
+            print(f"  Token-aware chunking: {len(chunks)} chunks (~{target_tokens} tokens each, {overlap_tokens} overlap)")
+        else:
+            # Fallback to character-based chunking
+            chunks = split_content_into_chunks(content, CHUNK_SIZE, OVERLAP)
+            print(f"  Character-based chunking: {len(chunks)} chunks ({CHUNK_SIZE} chars each, {OVERLAP} overlap)")
+        
         print(f"  Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")
         
         # Calculate optimal worker count for chunk processing
```

## File 3: Modified `summarizer.py` - Early Stopping

```diff
--- a/backend/summarizer.py
+++ b/backend/summarizer.py
@@ -71,6 +71,7 @@ def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: st
     """
     from prompt_utils import CHUNK_PROMPT_COMPACT, CHUNK_SYSTEM_COMPACT, parse_chunk_summary_response
     from chunk_cache import get_cached_summary, cache_summary
+    from prompt_utils import STOP_SEQUENCES
     
     # Use compact prompt for optimization (Step 1.1)
     prompt_template = CHUNK_PROMPT_COMPACT
@@ -105,7 +106,12 @@ def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: st
     try:
         response = call_llm(
             prompt=prompt,
             system_prompt=CHUNK_SYSTEM_COMPACT,
             provider="ollama",
             model=model,
+            stop_sequences=STOP_SEQUENCES.get("ollama", []) if os.getenv("SUMMARY_EARLY_STOP", "true").lower() == "true" else None
         )
         
         # Parse JSON response
```

## File 4: Modified `llm_provider.py` - Async Support

```diff
--- a/backend/llm_provider.py
+++ b/backend/llm_provider.py
@@ -5,6 +5,7 @@ Supports: Ollama, OpenAI, Gemini, DeepSeek
 """
 from typing import Optional, List, Dict
 import os
+import asyncio
 from dotenv import load_dotenv
 from prompt_utils import STOP_SEQUENCES, TEMPERATURE
 from llm_concurrency import LLMConcurrencyLimiter
@@ -200,6 +201,60 @@ def call_llm(
     return result
 
 
+async def call_llm_async(
+    prompt: str,
+    system_prompt: Optional[str] = None,
+    provider: str = "ollama",
+    model: Optional[str] = None,
+    timeout: int = 300,
+    stop_sequences: Optional[List[str]] = None
+) -> str:
+    """
+    Async version of call_llm for better concurrency with asyncio.
+    
+    Args:
+        prompt: The prompt text
+        system_prompt: Optional system prompt
+        provider: LLM provider name
+        model: Model name (optional, uses default if not provided)
+        timeout: Maximum time in seconds
+        stop_sequences: Sequences that signal end of generation
    
+    Returns:
+        Generated text
+    """
+    limiter = LLMConcurrencyLimiter()
+    
+    # Acquire semaphore (non-blocking in async context)
+    await asyncio.to_thread(limiter.acquire(provider).__enter__)
+    
+    try:
+        if provider == "ollama":
+            return await call_ollama_async(prompt, system_prompt, model, timeout, stop_sequences)
+        elif provider == "openai":
+            return await call_openai_async(prompt, system_prompt, model, timeout, stop_sequences)
+        elif provider == "gemini":
+            return await call_gemini_async(prompt, system_prompt, model, timeout, stop_sequences)
+        elif provider == "deepseek":
+            return await call_deepseek_async(prompt, system_prompt, model, timeout, stop_sequences)
+        else:
+            raise ValueError(f"Unknown provider: {provider}")
+    finally:
+        await asyncio.to_thread(limiter.acquire(provider).__exit__, None, None, None)
+
+
+async def call_ollama_async(
+    prompt: str,
+    system_prompt: Optional[str] = None,
+    model: Optional[str] = None,
+    timeout: int = 300,
+    stop_sequences: Optional[List[str]] = None
+) -> str:
+    """Async Ollama call using httpx for async HTTP"""
+    import httpx
+    
+    model = model or os.getenv("OLLAMA_MODEL", "phi3:mini")
+    url = f"{os.getenv('OLLAMA_URL', 'http://localhost:11434')}/api/generate"
+    
+    payload = {
+        "model": model,
+        "prompt": prompt,
+        "system": system_prompt or "",
+        "stream": False,
+        "options": {
+            "temperature": TEMPERATURE.get("ollama", 0.2),
+            "stop": stop_sequences or STOP_SEQUENCES.get("ollama", [])
+        }
+    }
+    
+    async with httpx.AsyncClient(timeout=timeout) as client:
+        response = await client.post(url, json=payload)
+        response.raise_for_status()
+        result = response.json()
+        return result.get("response", "")
```

## File 5: Modified `summarizer.py` - Async Chunk Processing

```diff
--- a/backend/summarizer.py
+++ b/backend/summarizer.py
@@ -235,6 +235,50 @@ def generate_summary(content: str, title: str, max_length: int = 2000) -> Opti
         print(f"  Using {max_workers} parallel workers for chunk processing (CPU cores: {cpu_count}, chunks: {len(chunks)})")
         
         # Summarize chunks in parallel
+        use_async = os.getenv("SUMMARY_USE_ASYNC", "true").lower() == "true"
+        
+        if use_async:
+            # Use async I/O for better concurrency
+            chunk_summaries = await _summarize_chunks_async(chunks, title, max_workers)
+        else:
+            # Fallback to thread-based parallelism
+            chunk_summaries = _summarize_chunks_sync(chunks, title, max_workers)
+        
+        # ... rest of merge logic
+
+
+async def _summarize_chunks_async(chunks: List[str], title: str, max_workers: int) -> List[str]:
+    """Async version of chunk summarization"""
+    from llm_provider import call_llm_async
+    from prompt_utils import CHUNK_PROMPT_COMPACT, CHUNK_SYSTEM_COMPACT, parse_chunk_summary_response
+    from chunk_cache import get_cached_summary, cache_summary
+    
+    model = "phi3:mini"
+    prompt_template = CHUNK_PROMPT_COMPACT
+    
+    async def summarize_one_chunk(idx: int, chunk: str) -> tuple[int, str]:
+        """Summarize a single chunk asynchronously"""
+        # Check cache
+        cached = get_cached_summary(chunk, prompt_template, model)
+        if cached:
+            return idx, cached.get("summary", "")
+        
+        prompt = prompt_template.format(
+            idx=idx + 1,
+            total=len(chunks),
+            title=title,
+            chunk=chunk
+        )
+        
+        try:
+            response = await call_llm_async(
+                prompt=prompt,
+                system_prompt=CHUNK_SYSTEM_COMPACT,
+                provider="ollama",
+                model=model
+            )
+            parsed = parse_chunk_summary_response(response.strip())
+            cache_summary(chunk, prompt_template, model, parsed)
+            return idx, parsed.get("summary", response[:120])
+        except Exception as e:
+            print(f"Error summarizing chunk {idx + 1}: {e}")
+            return idx, chunk[:500] + "..."
+    
+    # Process chunks with controlled concurrency
+    semaphore = asyncio.Semaphore(max_workers)
+    
+    async def bounded_summarize(idx: int, chunk: str):
+        async with semaphore:
+            return await summarize_one_chunk(idx, chunk)
+    
+    tasks = [bounded_summarize(idx, chunk) for idx, chunk in enumerate(chunks)]
+    results = await asyncio.gather(*tasks)
+    
+    # Sort by index and extract summaries
+    results.sort(key=lambda x: x[0])
+    return [summary for _, summary in results]
```

## File 6: Modified `prompt_utils.py` - Early Stopping

```diff
--- a/backend/prompt_utils.py
+++ b/backend/prompt_utils.py
@@ -33,6 +33,20 @@ def truncate_to_sentences(text: str, max_sentences: int = 12) -> str:
     return truncated
 
 
+def count_sentences_streaming(text_so_far: str, target: int = 12) -> tuple[int, bool]:
+    """
+    Count sentences in streaming text and check if target reached.
+    Used for early stopping during LLM generation.
+    
+    Args:
+        text_so_far: Text generated so far
+        target: Target sentence count (default: 12)
+    
+    Returns:
+        (sentence_count, should_stop) tuple
+    """
+    sentences = count_sentences(text_so_far)
+    should_stop = sentences >= target
+    return sentences, should_stop
+
+
 # Optimized prompts with exact sentence count requirements
```

## Summary of Changes

1. **New `token_utils.py`**: Token-aware chunking utilities
2. **Modified `summarizer.py`**: 
   - Token-aware chunking (reduces chunk count)
   - Async chunk processing (better concurrency)
   - Early stopping support
3. **Modified `llm_provider.py`**: Async LLM calls with httpx
4. **Modified `prompt_utils.py`**: Early stopping utilities

## Environment Variables Added

- `SUMMARY_USE_TOKEN_CHUNKING=true` (default)
- `SUMMARY_CHUNK_SIZE_TOKENS=1500`
- `SUMMARY_OVERLAP_TOKENS=150`
- `SUMMARY_USE_ASYNC=true` (default)
- `SUMMARY_EARLY_STOP=true` (default)
- `SUMMARY_ADAPTIVE_CHUNKS=true` (default)

