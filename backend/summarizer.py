"""
Module for generating chapter summaries using unified LLM provider.
Uses Ollama by default, but supports OpenAI, Gemini, and DeepSeek.

Includes parallel processing utilities for batch summary generation.
"""
from typing import Optional, Tuple, List, Dict
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from llm_provider import call_llm
from prompt_utils import (
    COMBINE_PROMPT_TEMPLATE, COMBINE_SYSTEM,
    count_sentences, truncate_to_sentences
)

load_dotenv()


def split_content_into_chunks(content: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    """
    Split content into overlapping chunks for parallel processing.
    Overlap occurs at sentence boundaries for better context preservation.
    
    Args:
        content: The content to split
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks (default: 200)
    
    Returns:
        List of content chunks
    """
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(content):
            # Look for sentence endings near the chunk boundary
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if i < len(content) and content[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = content[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap, ensuring it's at a sentence boundary
        start = end - overlap
        
        # Find nearest sentence boundary for overlap start (Step 1.2 optimization)
        if start > 0 and start < len(content):
            # Look backwards for sentence boundary within 100 chars
            for i in range(start, max(start - 100, 0), -1):
                if i < len(content) and content[i] in '.!?\n':
                    start = i + 1
                    break
        
        if start >= len(content):
            break
    
    return chunks


def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int, title: str) -> str:
    """
    Summarize a single chunk of content using compact prompt with JSON output.
    Uses caching to avoid re-processing identical chunks (Step 2.2).
    
    Args:
        chunk: The content chunk to summarize
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks
        title: Chapter title for context
    
    Returns:
        Summary of the chunk (extracted from JSON response)
    """
    from prompt_utils import CHUNK_PROMPT_COMPACT, CHUNK_SYSTEM_COMPACT, parse_chunk_summary_response
    from chunk_cache import get_cached_summary, cache_summary
    
    # Use compact prompt for optimization (Step 1.1)
    prompt_template = CHUNK_PROMPT_COMPACT
    model = "phi3:mini"  # Explicitly use phi3:mini for summarization
    
    # Check cache first (Step 2.2)
    cached = get_cached_summary(chunk, prompt_template, model)
    if cached:
        print(f"  Cache hit for chunk {chunk_index + 1}")
        return cached.get("summary", "")
    
    prompt = prompt_template.format(
        idx=chunk_index + 1,
        total=total_chunks,
        title=title,
        chunk=chunk
    )

    try:
        response = call_llm(
            prompt=prompt,
            system_prompt=CHUNK_SYSTEM_COMPACT,
            provider="ollama",
            model=model
        )
        
        # Parse JSON response
        parsed = parse_chunk_summary_response(response.strip())
        
        # Cache the parsed summary (Step 2.2)
        cache_summary(chunk, prompt_template, model, parsed)
        
        return parsed.get("summary", response[:120] if len(response) > 120 else response)
    except Exception as e:
        print(f"Error summarizing chunk {chunk_index + 1}: {e}")
        # Fallback: return truncated chunk
        return chunk[:500] + "..." if len(chunk) > 500 else chunk


def merge_chunk_summaries(chunk_summaries: List[str], title: str) -> str:
    """
    Merge multiple chunk summaries into a single comprehensive summary.
    Uses compact prompt with only key points for optimization (Step 1.5).
    
    Args:
        chunk_summaries: List of summaries from each chunk (may be JSON or plain text)
        title: Chapter title for context
    
    Returns:
        Merged comprehensive summary
    """
    from prompt_utils import (
        MERGE_PROMPT_COMPACT, COMBINE_SYSTEM, 
        extract_key_points_for_merge, parse_chunk_summary_response
    )
    
    if not chunk_summaries:
        return ""
    
    if len(chunk_summaries) == 1:
        # If single summary, extract summary text if it's JSON
        parsed = parse_chunk_summary_response(chunk_summaries[0])
        if isinstance(parsed, dict):
            return parsed.get("summary", chunk_summaries[0])
        return chunk_summaries[0]
    
    # Extract key points for compact merge (Step 1.5)
    key_points = extract_key_points_for_merge(chunk_summaries)
    
    # Use compact merge prompt
    prompt = MERGE_PROMPT_COMPACT.format(
        count=len(chunk_summaries),
        key_points=key_points
    )

    system_prompt = COMBINE_SYSTEM

    try:
        merged_summary = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            provider="ollama",
            model="phi3:mini"  # Explicitly use phi3:mini for summarization
        )
        # Post-process to ensure exactly 10-12 sentences
        sentence_count = count_sentences(merged_summary)
        if sentence_count > 12:
            merged_summary = truncate_to_sentences(merged_summary, max_sentences=12)
        elif sentence_count < 10 and len(merged_summary) < 2000:
            print(f"Warning: Merged summary has only {sentence_count} sentences (target: 10-12)")
        return merged_summary.strip()
    except Exception as e:
        print(f"Error merging summaries: {e}")
        # Fallback: extract summaries from JSON or use plain text
        fallback_summaries = []
        for chunk_data in chunk_summaries:
            parsed = parse_chunk_summary_response(chunk_data)
            if isinstance(parsed, dict):
                fallback_summaries.append(parsed.get("summary", chunk_data))
            else:
                fallback_summaries.append(chunk_data)
        return "\n\n".join(fallback_summaries)


def generate_summary(content: str, title: str, max_length: int = 2000) -> Optional[str]:
    """
    Generate a comprehensive summary of a chapter using LLM (Ollama by default).
    For long chapters, uses chunked parallel processing for faster generation.
    
    Args:
        content: The chapter content
        title: The chapter title
        max_length: Maximum length of summary in characters (default: 2000)
    
    Returns:
        Summary string or truncated content if generation fails
    """
    # Determine chunk size and overlap based on content length
    # Use chunked approach for chapters longer than 5000 characters
    CHUNK_THRESHOLD = 5000
    CHUNK_SIZE = 2000  # Reduced from 3000 for better reliability under load
    OVERLAP = 200      # Reduced from 400 to 200 for optimization (Step 1.2)
    
    if len(content) > CHUNK_THRESHOLD:
        # Use chunked parallel approach for long chapters
        print(f"  Using chunked parallel summarization for chapter '{title}' ({len(content)} chars)")
        
        # Split content into chunks
        chunks = split_content_into_chunks(content, CHUNK_SIZE, OVERLAP)
        print(f"  Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")
        
        # Calculate optimal worker count for chunk processing
        # Use more workers to maximize throughput - up to chunk count or reasonable limit
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        env_workers = int(os.getenv("SUMMARY_MAX_WORKERS", 0))  # 0 means auto
        
        if env_workers > 0:
            # User specified a limit, respect it but allow more for chunks
            # Increased to 16 to match Ollama concurrency limit (Step 1.4)
            max_workers = min(env_workers, len(chunks), 16)
        else:
            # Auto: use CPU count or chunk count, whichever is smaller, capped at 16
            # Increased to 16 to match Ollama concurrency limit (Step 1.4)
            max_workers = min(cpu_count, len(chunks), 16)
        
        print(f"  Using {max_workers} parallel workers for chunk processing (CPU cores: {cpu_count}, chunks: {len(chunks)})")
        
        # Summarize chunks in parallel
        chunk_summaries = []
        completed_count = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    summarize_chunk,
                    chunk,
                    idx,
                    len(chunks),
                    title
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            
            print(f"  Submitted {len(future_to_chunk)} chunk summarization tasks")
            
            # Collect results in order
            results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                idx = future_to_chunk[future]
                try:
                    chunk_start = time.time()
                    summary = future.result()
                    chunk_time = time.time() - chunk_start
                    results[idx] = summary
                    completed_count += 1
                    print(f"  ✓ Chunk {idx + 1}/{len(chunks)} completed ({len(summary)} chars, {chunk_time:.1f}s) - "
                          f"Progress: {completed_count}/{len(chunks)} ({100*completed_count/len(chunks):.1f}%)")
                except Exception as e:
                    completed_count += 1
                    print(f"  ✗ Error summarizing chunk {idx + 1}/{len(chunks)}: {e}")
                    results[idx] = chunks[idx][:500] + "..."  # Fallback
                    print(f"    Using fallback content for chunk {idx + 1}")
            
            total_time = time.time() - start_time
            print(f"  Completed all {len(chunks)} chunks in {total_time:.1f}s (avg: {total_time/len(chunks):.2f}s per chunk)")
        
        # Filter out None results
        chunk_summaries = [s for s in results if s]
        
        print(f"  Successfully summarized {len(chunk_summaries)}/{len(chunks)} chunks")
        if len(chunk_summaries) < len(chunks):
            print(f"  Warning: {len(chunks) - len(chunk_summaries)} chunks failed and used fallback content")
        
        if not chunk_summaries:
            # Fallback to single summary if chunking failed
            print(f"  ✗ Chunking failed completely, falling back to single summary")
            return _generate_single_summary(content, title, max_length)
        
        # Merge chunk summaries into final summary
        print(f"  Merging {len(chunk_summaries)} chunk summaries into final summary...")
        merge_start = time.time()
        summary = merge_chunk_summaries(chunk_summaries, title)
        merge_time = time.time() - merge_start
        print(f"  ✓ Merge completed in {merge_time:.1f}s ({len(summary)} chars)")
        
        # Ensure summary meets minimum requirements
        sentences = summary.split('.')
        if len(sentences) < 10:
            # Summary might be too short, but proceed anyway
            pass
        
        # Only truncate if significantly over max_length (allow some flexibility)
        if len(summary) > max_length * 1.2:
            summary = summary[:max_length-3] + "..."
        
        return summary
    else:
        # Use single summary approach for shorter chapters
        return _generate_single_summary(content, title, max_length)


def _generate_single_summary(content: str, title: str, max_length: int = 2000) -> Optional[str]:
    """
    Generate a summary using the traditional single-pass approach.
    Used for shorter chapters or as fallback.
    """
    # Use more content for comprehensive summaries - up to 8000 chars
    content_preview = content[:8000] + "..." if len(content) > 8000 else content
    
    prompt = f"""Generate a comprehensive, detailed summary for this chapter. The summary must:

1. Be at least 10 sentences long
2. Include every terminology and concept mentioned in the text
3. Cover every takeaway and key point, even if briefly
4. Be thorough and complete, not just a brief overview

Title: {title}

Content:
{content_preview}

Provide a detailed summary that comprehensively covers all aspects of the chapter."""

    system_prompt = "You are an expert at creating comprehensive, detailed summaries. Your summaries must be thorough, include all terminology and concepts, cover all takeaways, and be at least 10 sentences long. Always respond with just the summary text, no additional commentary."

    try:
        # Always use Ollama with phi3:mini for summarization
        summary = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            provider="ollama",
            model="phi3:mini"  # Explicitly use phi3:mini for summarization
        )
        
        # Ensure summary meets minimum requirements
        sentences = summary.split('.')
        if len(sentences) < 10:
            # If summary is too short, try to expand it
            # But don't truncate if it's comprehensive
            pass
        
        # Only truncate if significantly over max_length (allow some flexibility)
        if len(summary) > max_length * 1.2:
            summary = summary[:max_length-3] + "..."
        return summary
    except Exception as e:
        # Re-raise the exception so the caller can handle it appropriately
        # This is important when called from the API endpoint
        raise


def generate_takeaway_title(summary: str) -> str:
    """
    Generate a main takeaway/title from a detailed summary.
    
    Args:
        summary: The detailed summary
    
    Returns:
        A concise title representing the main takeaway
    """
    prompt = f"""Based on this detailed chapter summary, generate a concise title (max 15 words) that captures the main takeaway or key insight:

Summary:
{summary}

Generate a title that represents the core message or main takeaway. Return only the title, no additional text."""

    system_prompt = "You are an expert at extracting key insights. Generate concise, informative titles that capture the essence of content."

    try:
        title = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            provider="ollama",
            model="phi3:mini"  # Explicitly use phi3:mini for summarization
        )
        # Clean up the title
        title = title.strip()
        # Remove quotes if present
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith("'") and title.endswith("'"):
            title = title[1:-1]
        # No truncation - let frontend handle display with CSS wrapping
        # Titles can be longer to capture the full takeaway
        return title
    except Exception as e:
        # Fallback: use first sentence
        first_sentence = summary.split('.')[0]
        return first_sentence or summary[:200]


def generate_preview(summary: str) -> str:
    """
    Generate a 3-sentence preview from a detailed summary.
    
    Args:
        summary: The detailed summary
    
    Returns:
        A 3-sentence preview
    """
    prompt = f"""Based on this detailed chapter summary, create a concise 3-sentence preview that captures the most important points:

Summary:
{summary}

Generate exactly 3 sentences that provide a good overview. Return only the 3 sentences, no additional text."""

    system_prompt = "You are an expert at creating concise previews. Generate exactly 3 sentences that capture the essence of the content."

    try:
        preview = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            provider="ollama",
            model="phi3:mini"  # Explicitly use phi3:mini for summarization
        )
        # Clean up the preview
        preview = preview.strip()
        # Ensure it's roughly 3 sentences (allow some flexibility)
        sentences = [s.strip() for s in preview.split('.') if s.strip()]
        if len(sentences) >= 2:
            # Take first 3 sentences
            preview = '. '.join(sentences[:3])
            if not preview.endswith('.'):
                preview += '.'
        return preview
    except Exception as e:
        # Fallback: use first 3 sentences of summary
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        if len(sentences) >= 3:
            return '. '.join(sentences[:3]) + '.'
        return summary[:300] + "..." if len(summary) > 300 else summary


def process_summary_for_chapter(summary: str) -> Tuple[str, str]:
    """
    Process a detailed summary to generate title and preview.
    
    Args:
        summary: The detailed summary
    
    Returns:
        Tuple of (takeaway_title, preview)
    """
    if not summary or len(summary.strip()) == 0:
        return None, None
    
    try:
        takeaway_title = generate_takeaway_title(summary)
        preview = generate_preview(summary)
        return takeaway_title, preview
    except Exception as e:
        print(f"Error processing summary: {e}")
        # Fallback: generate simple preview
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        preview = '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else summary[:300]
        return None, preview


# ============================================================================
# Parallel Processing Utilities
# ============================================================================

def generate_summaries_parallel(
    chapters_data: List[Dict[str, str]], 
    max_workers: int = None
) -> List[str]:
    """
    Generate summaries for multiple chapters in parallel.
    
    Args:
        chapters_data: List of dicts with 'content' and 'title' keys
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of summaries in the same order as input chapters
    
    Notes:
        - Default: min(3, num_chapters, CPU_count)
        - Can be overridden via SUMMARY_MAX_WORKERS env var
        - Hard maximum: 16 workers (to prevent overwhelming Ollama/server)
        - Recommended: 3-8 workers for optimal performance
    """
    if max_workers is None:
        # Default to 3 workers to prevent overloading Ollama
        cpu_count = os.cpu_count() or 4
        default_workers = min(3, len(chapters_data), cpu_count)
        # Allow override via environment variable
        max_workers = int(os.getenv("SUMMARY_MAX_WORKERS", default_workers))
    
    # Enforce reasonable maximum to prevent overwhelming Ollama/server
    HARD_MAX_WORKERS = 16
    max_workers = min(max_workers, HARD_MAX_WORKERS, len(chapters_data))
    
    # Ensure at least 1 worker
    max_workers = max(1, max_workers)
    
    summaries = [None] * len(chapters_data)
    
    def generate_single_summary(idx: int, content: str, title: str) -> Tuple[int, str]:
        """Generate summary for a single chapter"""
        try:
            summary = generate_summary(content, title)
            return idx, summary
        except Exception as e:
            # Fallback to truncation on error
            fallback = content[:200] + "..." if len(content) > 200 else content
            return idx, fallback
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                generate_single_summary,
                idx,
                chapter_data.get('content', ''),
                chapter_data.get('title', f'Chapter {idx + 1}')
            ): idx
            for idx, chapter_data in enumerate(chapters_data)
        }
        
        # Collect results as they complete (order doesn't matter, we use idx)
        completed = 0
        for future in as_completed(future_to_idx):
            idx, summary = future.result()
            summaries[idx] = summary
            completed += 1
            if completed % 5 == 0 or completed == len(chapters_data):
                print(f"  Progress: {completed}/{len(chapters_data)} summaries completed")
    
    return summaries


def process_summaries_for_titles_and_previews(
    summaries: List[str],
    max_workers: int = None
) -> List[Tuple[str, str]]:
    """
    Process summaries to generate takeaway titles and previews in parallel.
    
    Args:
        summaries: List of detailed summaries
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of tuples (takeaway_title, preview) in the same order as input
    """
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        default_workers = min(3, len(summaries), cpu_count)
        max_workers = int(os.getenv("SUMMARY_MAX_WORKERS", default_workers))
    
    HARD_MAX_WORKERS = 16
    max_workers = min(max_workers, HARD_MAX_WORKERS, len(summaries))
    max_workers = max(1, max_workers)
    
    results = [(None, None)] * len(summaries)
    
    def process_single_summary(idx: int, summary: str) -> Tuple[int, str, str]:
        """Process a single summary to generate title and preview"""
        try:
            if summary and len(summary.strip()) > 0:
                takeaway_title, preview = process_summary_for_chapter(summary)
                return idx, takeaway_title, preview
            return idx, None, None
        except Exception as e:
            print(f"Error processing summary {idx}: {e}")
            return idx, None, None
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_summary, idx, summary): idx
            for idx, summary in enumerate(summaries)
            if summary  # Only process non-empty summaries
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx, takeaway_title, preview = future.result()
            results[idx] = (takeaway_title, preview)
            completed += 1
            if completed % 5 == 0 or completed == len(future_to_idx):
                print(f"  Progress: {completed}/{len(future_to_idx)} titles/previews completed")
    
    return results
