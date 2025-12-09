#!/usr/bin/env python3
"""
Migration script to add summary column to chapters table.
Run this once to update existing databases.
"""
import sqlite3
import os
from database import DATABASE_URL

def migrate():
    """Add summary column to chapters table if it doesn't exist"""
    # Extract database path from SQLite URL
    db_path = DATABASE_URL.replace("sqlite:///", "")
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found. Skipping migration.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if summary column already exists
        cursor.execute("PRAGMA table_info(chapters)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'summary' in columns:
            print("Summary column already exists. Migration not needed.")
            return
        
        # Add summary column
        print("Adding summary column to chapters table...")
        cursor.execute("ALTER TABLE chapters ADD COLUMN summary TEXT")
        conn.commit()
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()

