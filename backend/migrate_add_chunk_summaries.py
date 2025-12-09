#!/usr/bin/env python3
"""
Migration to add chunk_summaries table for checkpointing and resume capability.

This table stores individual chunk summaries, enabling:
- Resume processing from partial progress
- Deduplication of identical chunks
- Checkpointing for long-running chapter processing
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import engine, Base, IS_POSTGRESQL, IS_SQLITE
from models import ChunkSummary
from sqlalchemy import inspect, text


def migrate():
    """Create chunk_summaries table if it doesn't exist."""
    db_type = "postgresql" if IS_POSTGRESQL else "sqlite" if IS_SQLITE else None
    if not db_type:
        raise ValueError("Unsupported database type")
    
    # Check if table already exists
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    if "chunk_summaries" in existing_tables:
        print("✓ chunk_summaries table already exists")
        return
    
    print("Creating chunk_summaries table...")
    
    if db_type == "postgresql":
        # PostgreSQL: Use CREATE TABLE IF NOT EXISTS
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chunk_summaries (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    chapter_id VARCHAR NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_hash VARCHAR NOT NULL,
                    summary TEXT,
                    key_points TEXT,
                    entities TEXT,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chunk_summaries_chapter_id 
                ON chunk_summaries(chapter_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chunk_summaries_content_hash 
                ON chunk_summaries(content_hash)
            """))
            conn.commit()
        print("✓ chunk_summaries table created (PostgreSQL)")
    else:
        # SQLite: Use SQLAlchemy table creation
        ChunkSummary.__table__.create(bind=engine, checkfirst=True)
        print("✓ chunk_summaries table created (SQLite)")
    
    # Create indexes
    try:
        with engine.connect() as conn:
            if db_type == "postgresql":
                # Indexes already created above
                pass
            else:
                # SQLite indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS ix_chunk_summaries_chapter_id 
                    ON chunk_summaries(chapter_id)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS ix_chunk_summaries_content_hash 
                    ON chunk_summaries(content_hash)
                """))
                conn.commit()
        print("✓ Indexes created")
    except Exception as e:
        print(f"⚠️  Warning: Could not create indexes: {e}")
    
    print("✓ Migration complete: chunk_summaries table ready")


if __name__ == "__main__":
    migrate()

