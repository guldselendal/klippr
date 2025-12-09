#!/usr/bin/env python3
"""
Generate summaries for chapters that don't have them yet.
Run this after migration to populate summaries for existing chapters.
"""
from database import get_db_session
from models import Chapter
from summarizer import generate_summary

def generate_missing_summaries():
    """Generate summaries for all chapters that don't have one"""
    db = next(get_db_session())
    
    try:
        # Get all chapters without summaries
        chapters = db.query(Chapter).filter(
            (Chapter.summary == None) | (Chapter.summary == '')
        ).all()
        
        total = len(chapters)
        print(f"Found {total} chapters without summaries.")
        
        if total == 0:
            print("All chapters already have summaries!")
            return
        
        for idx, chapter in enumerate(chapters, 1):
            print(f"Generating summary for chapter {idx}/{total}: {chapter.title}")
            summary = generate_summary(chapter.content, chapter.title)
            chapter.summary = summary
            db.commit()
            print(f"  ✓ Summary generated ({len(summary)} chars)")
        
        print(f"\nSuccessfully generated {total} summaries!")
        
    except Exception as e:
        print(f"Error generating summaries: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    generate_missing_summaries()

