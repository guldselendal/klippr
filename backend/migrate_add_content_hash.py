#!/usr/bin/env python3
"""
Migration script to add content_hash column to chapters table.
Run this once to update existing databases.

The content_hash column stores SHA256 hashes of chapter content for deduplication,
allowing the system to skip summary generation if content hasn't changed.
"""
import os
import sys
from database import DATABASE_URL, IS_POSTGRESQL, IS_SQLITE, engine
from sqlalchemy import text, inspect

def migrate():
    """Add content_hash column and index to chapters table if they don't exist"""
    
    if not IS_POSTGRESQL and not IS_SQLITE:
        print(f"Unsupported database URL format: {DATABASE_URL}")
        return
    
    print(f"Running migration for {'PostgreSQL' if IS_POSTGRESQL else 'SQLite'} database...")
    
    with engine.connect() as conn:
        try:
            # Check if content_hash column already exists
            inspector = inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('chapters')]
            
            if 'content_hash' in columns:
                print("content_hash column already exists. Checking index...")
            else:
                # Add content_hash column
                print("Adding content_hash column to chapters table...")
                if IS_POSTGRESQL:
                    conn.execute(text("""
                        ALTER TABLE chapters 
                        ADD COLUMN IF NOT EXISTS content_hash TEXT
                    """))
                else:  # SQLite
                    conn.execute(text("""
                        ALTER TABLE chapters 
                        ADD COLUMN content_hash TEXT
                    """))
                conn.commit()
                print("✓ content_hash column added successfully")
            
            # Check if index already exists
            indexes = [idx['name'] for idx in inspector.get_indexes('chapters')]
            
            if 'ix_chapters_content_hash' in indexes:
                print("ix_chapters_content_hash index already exists. Migration complete.")
            else:
                # Create index on content_hash
                print("Creating index on content_hash column...")
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS ix_chapters_content_hash 
                    ON chapters(content_hash)
                """))
                conn.commit()
                print("✓ Index created successfully")
            
            print("\nMigration completed successfully!")
            print("Note: Existing chapters will have NULL content_hash values.")
            print("Content hashes will be computed automatically for new chapters.")
            print("You can backfill existing chapters by recomputing their summaries.")
            
        except Exception as e:
            print(f"Error during migration: {e}")
            import traceback
            traceback.print_exc()
            conn.rollback()
            sys.exit(1)

if __name__ == "__main__":
    migrate()

