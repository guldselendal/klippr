#!/usr/bin/env python3
"""
Migration script to add performance indexes to existing databases.
Safe to run multiple times (uses IF NOT EXISTS).
"""
from database import engine
from sqlalchemy import text

def add_indexes():
    """Add indexes for better query performance"""
    print("Adding performance indexes...")
    
    with engine.connect() as conn:
        # Index on chapters.document_id (foreign key, used in joins)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chapters_document_id 
            ON chapters(document_id)
        """))
        print("  ✓ Created index: ix_chapters_document_id")
        
        # Index on chapters.chapter_number (used for ordering)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chapters_chapter_number 
            ON chapters(document_id, chapter_number)
        """))
        print("  ✓ Created index: ix_chapters_chapter_number")
        
        conn.commit()
    
    print("\nIndexes added successfully!")

if __name__ == "__main__":
    add_indexes()

