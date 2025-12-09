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

