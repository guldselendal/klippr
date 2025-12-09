"""
Shared utilities for the backend application.
Consolidates common patterns to reduce duplication.
"""
import os
import re
from typing import Optional
from contextlib import contextmanager
from fastapi import HTTPException
from database import get_db_session


def clean_filename(filename: str) -> str:
    """Remove parentheses and their contents from filename."""
    name, ext = os.path.splitext(filename)
    cleaned = re.sub(r'\([^)]*\)', '', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.strip('. -')
    return cleaned + ext if cleaned else filename


def normalize_book_name(filename: str) -> str:
    """Normalize book name for comparison (remove extension, parentheses, lowercase)."""
    name_without_ext = os.path.splitext(filename)[0]
    cleaned = clean_filename(name_without_ext)
    return cleaned.lower().strip()


def validate_file_extension(filename: str, allowed_extensions: list[str] = ['.epub', '.pdf']) -> str:
    """
    Validate file extension and return normalized extension.
    
    Raises:
        HTTPException: If file extension is invalid
    """
    if not filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ext.upper() for ext in allowed_extensions)} files are supported"
        )
    return file_ext


@contextmanager
def db_session():
    """
    Context manager for database sessions.
    Ensures proper cleanup and consistent session handling.
    
    Usage:
        with db_session() as db:
            # use db here
    """
    db = next(get_db_session())
    try:
        yield db
    finally:
        db.close()


def format_chapter_response(chapter, include_content: bool = False) -> dict:
    """
    Format chapter data for API response.
    Standardizes chapter response structure across endpoints.
    """
    result = {
        'id': chapter.id,
        'title': chapter.title,
        'document_id': chapter.document_id,
        'document_title': getattr(chapter, 'document_title', None),
        'chapter_number': chapter.chapter_number,
    }
    
    if include_content:
        result['content'] = chapter.content
    
    # Add optional fields if they exist
    if hasattr(chapter, 'summary') and chapter.summary:
        result['summary'] = chapter.summary
    if hasattr(chapter, 'preview') and chapter.preview:
        result['preview'] = chapter.preview
    
    return result


def handle_parse_error(file_ext: str, error: Exception) -> HTTPException:
    """
    Create standardized HTTPException for file parsing errors.
    """
    return HTTPException(
        status_code=400,
        detail=f"Error parsing {file_ext} file: {str(error)}. The file may be corrupted or in an unsupported format."
    )

