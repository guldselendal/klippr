from typing import List, Dict, Optional
import os
import json
from dotenv import load_dotenv
from llm_provider import call_llm

# Load environment variables
load_dotenv()


def find_connections(chapters: List[Dict[str, str]], max_connections: int = 20) -> List[Dict]:
    """
    Find connections between chapters using LLM (Ollama by default).
    Returns the most relevant connections.
    """
    if len(chapters) < 2:
        return []
    
    # Prepare chapter summaries for analysis (limit content to avoid token limits)
    chapter_summaries = []
    for chapter in chapters:
        # Use title + first 800 chars for context
        content_preview = chapter['content'][:800] + "..." if len(chapter['content']) > 800 else chapter['content']
        summary = f"Title: {chapter['title']}\nContent: {content_preview}"
        chapter_summaries.append({
            'id': chapter['id'],
            'summary': summary,
            'full_chapter': chapter
        })
    
    # Analyze connections - compare each chapter with others
    # For efficiency, we'll analyze pairs and limit comparisons
    all_connections = []
    
    # Compare chapters intelligently - prioritize nearby chapters and sample others
    for i, chapter1 in enumerate(chapter_summaries):
        # Always compare with next few chapters (likely related)
        nearby_range = min(5, len(chapter_summaries) - i - 1)
        for j in range(1, nearby_range + 1):
            if i + j < len(chapter_summaries):
                chapter2 = chapter_summaries[i + j]
                connection = _analyze_connection(chapter1, chapter2)
                if connection:
                    all_connections.append(connection)
        
        # Sample connections with chapters further away (every 3rd chapter)
        for j in range(i + 6, len(chapter_summaries), 3):
            chapter2 = chapter_summaries[j]
            connection = _analyze_connection(chapter1, chapter2)
            if connection:
                all_connections.append(connection)
    
    # Sort by similarity score (highest first)
    all_connections.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Return top connections
    return all_connections[:max_connections]


def _analyze_connection(chapter1: Dict, chapter2: Dict) -> Optional[Dict]:
    """Analyze connection between two chapters using LLM (Ollama)"""
    
    prompt = f"""Analyze the connection between these two chapters from documents.

Chapter 1:
{chapter1['summary']}

Chapter 2:
{chapter2['summary']}

Please analyze if these chapters are related and how. Respond in JSON format with:
{{
    "connected": true/false,
    "similarity": 0.0-1.0 (a score indicating how related they are),
    "reason": "A brief explanation of why they are connected (or why not)"
}}

Only respond with the JSON, no additional text."""

    system_prompt = "You are an expert at analyzing textual connections and relationships between different pieces of content. Always respond with valid JSON only."
    
    try:
        # Use Ollama via unified LLM provider
        result_text = call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            provider="ollama"  # Force Ollama
        )
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        if result.get("connected", False) and result.get("similarity", 0) > 0.3:
            return {
                'chapter1': chapter1['full_chapter'],
                'chapter2': chapter2['full_chapter'],
                'similarity': float(result.get("similarity", 0)),
                'reason': result.get("reason", "Connected chapters")
            }
        
        return None
    
    except json.JSONDecodeError as e:
        result_text_safe = result_text[:200] if 'result_text' in locals() else "N/A"
        print(f"JSON decode error: {e}, response was: {result_text_safe}")
        return None
    except Exception as e:
        print(f"Error analyzing connection: {e}")
        return None