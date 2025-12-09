"""
Utility functions for prompt optimization and sentence counting.
"""
import re
from typing import List

# Provider-specific stop sequences to help models stop at the right point
STOP_SEQUENCES = {
    'ollama': ['\n\n\n', '---', '###'],
    'gemini': ['\n\n\n'],
    'openai': ['\n\n\n', '---'],
    'deepseek': ['\n\n\n']
}

# Temperature settings for deterministic output
TEMPERATURE = {
    'ollama': 0.2,      # Low temperature for consistency
    'gemini': 0.1,      # Very low for exact sentence count
    'openai': 0.3,
    'deepseek': 0.2
}


def count_sentences(text: str) -> int:
    """Count sentences in text by counting sentence-ending punctuation."""
    if not text:
        return 0
    # Count sentence endings: period, exclamation, question mark
    sentences = len(re.findall(r'[.!?]+(?:\s|$)', text))
    return max(1, sentences)  # At least 1 sentence if text exists


def truncate_to_sentences(text: str, max_sentences: int = 12) -> str:
    """
    Truncate text to exactly max_sentences by finding sentence boundaries.
    Returns text with exactly max_sentences sentences (or fewer if text is shorter).
    """
    if not text:
        return text
    
    # Find all sentence boundaries
    sentence_pattern = r'[.!?]+(?:\s|$)'
    matches = list(re.finditer(sentence_pattern, text))
    
    if len(matches) <= max_sentences:
        return text.strip()
    
    # Truncate at the max_sentences-th sentence boundary
    end_pos = matches[max_sentences - 1].end()
    truncated = text[:end_pos].strip()
    
    return truncated


# Optimized prompts with exact sentence count requirements
FULL_SUMMARY_PROMPT_TEMPLATE = """You are summarizing a chapter from a book. Generate a comprehensive summary that meets these EXACT requirements:

CRITICAL REQUIREMENTS (MUST FOLLOW):
1. Write EXACTLY 10-12 complete sentences (no more, no less). This is MANDATORY.
2. Count your sentences as you write. Stop at exactly 12 sentences maximum.
3. Total length: 1,800-2,000 characters (strict maximum: 2,400 characters).
4. Include EVERY key concept, terminology, and takeaway mentioned in the text.
5. Cover ALL main points, arguments, and conclusions.
6. Be thorough and complete - nothing important should be omitted.

IMPORTANT: Your response must contain between 10 and 12 sentences. Count them before finishing.

Chapter Title: {title}

Chapter Content:
{content}

Generate the summary now. Write exactly 10-12 sentences. Count them carefully."""

FULL_SUMMARY_SYSTEM = "You are an expert at creating detailed, comprehensive summaries. You ALWAYS write exactly 10-12 sentences (never fewer than 10, never more than 12). You count your sentences as you write. You stay within 2,000 characters. You never omit important concepts or terminology."

# Follow-up prompt for extending short summaries
EXTEND_SUMMARY_PROMPT_TEMPLATE = """Your previous summary had only {current_count} sentences, but you need EXACTLY 10-12 sentences.

Your previous summary:
{previous_summary}

Chapter Title: {title}
Chapter Content:
{content}

EXTEND your summary by adding {needed} more sentences (for a total of 10-12 sentences). 
- Add more details about key concepts you may have missed
- Expand on important points
- Include additional terminology or takeaways
- Ensure you reach at least 10 sentences total

Write the COMPLETE extended summary (all {target_count} sentences), not just the additions."""

COMBINE_PROMPT_TEMPLATE = """You are merging summaries from multiple sections of a chapter into a single, comprehensive summary. Requirements:

1. Write EXACTLY 10-12 complete sentences (no more, no less).
2. Total length: 1,800-2,000 characters (strict maximum: 2,400 characters).
3. Include ALL key concepts, terminology, and takeaways from ALL sections.
4. Eliminate redundancy - do not repeat the same point multiple times.
5. Ensure smooth flow and coherence between concepts from different sections.

Chapter Title: {title}

Section Summaries:
{chunk_summaries}

Generate the combined summary. Count your sentences and ensure exactly 10-12 sentences."""

COMBINE_SYSTEM = "You merge multiple summaries into one coherent, comprehensive summary. You eliminate redundancy while ensuring complete coverage. You always write exactly 10-12 sentences."

# Compact merge prompt for optimization (Step 1.5) - uses only key points
MERGE_PROMPT_COMPACT = """Merge {count} section summaries into one chapter summary (10-12 sentences, <=2000 chars).

Sections:
{key_points}

Output: Comprehensive summary covering all sections."""

def extract_key_points_for_merge(chunk_summaries: List[str]) -> str:
    """
    Extract only key_points from chunk summaries for compact merge.
    
    Args:
        chunk_summaries: List of chunk summary strings (may be JSON or plain text)
    
    Returns:
        Formatted string with key points from all sections
    """
    sections = []
    
    for i, chunk_data in enumerate(chunk_summaries):
        # Try to parse as JSON first
        parsed = parse_chunk_summary_response(chunk_data)
        
        if isinstance(parsed, dict) and "key_points" in parsed:
            key_points = parsed.get("key_points", [])
            if key_points:
                # Use key points if available
                points_text = "; ".join(key_points[:5])  # Limit to 5 points
                sections.append(f"Section {i+1}: {points_text}")
            else:
                # Fallback to summary if no key points
                summary = parsed.get("summary", chunk_data[:100])
                sections.append(f"Section {i+1}: {summary[:100]}")
        else:
            # Fallback for non-JSON summaries
            summary_text = chunk_data[:100] if isinstance(chunk_data, str) else str(chunk_data)[:100]
            sections.append(f"Section {i+1}: {summary_text}")
    
    return "\n".join(sections)

# Compact chunk prompt for optimization (Step 1.1)
CHUNK_PROMPT_COMPACT = """Summarize section {idx}/{total} of "{title}".

Output JSON:
{{"summary":"<=120 words","key_points":["<=5 bullets"],"entities":["optional"]}}

Content:
{chunk}"""

CHUNK_SYSTEM_COMPACT = "Output valid JSON only. Be concise."

# Helper function to parse chunk summary response
def parse_chunk_summary_response(response: str) -> dict:
    """
    Parse chunk summary response, extracting JSON if present.
    
    Returns:
        dict with 'summary', 'key_points', 'entities' keys
    """
    import json
    
    # Try to extract JSON from response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Ensure required fields
            if "summary" not in data:
                data["summary"] = response[:120] if len(response) > 120 else response
            if "key_points" not in data:
                data["key_points"] = []
            if "entities" not in data:
                data["entities"] = []
            return data
        except json.JSONDecodeError:
            pass
    
    # Fallback: treat entire response as summary
    return {
        "summary": response[:120] if len(response) > 120 else response,
        "key_points": [],
        "entities": []
    }

