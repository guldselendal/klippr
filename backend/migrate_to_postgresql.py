#!/usr/bin/env python3
"""
Migration script to migrate data from SQLite to PostgreSQL.

Usage:
1. Ensure PostgreSQL is running and DATABASE_URL is set
2. Run: python3 migrate_to_postgresql.py

This script will:
- Export all data from SQLite
- Create tables in PostgreSQL (if not exist)
- Import all data to PostgreSQL
- Verify data integrity
"""
import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import sqlite3
from datetime import datetime

# Import models to ensure they're registered
from models import Document, Chapter, Base

def get_sqlite_connection():
    """Get SQLite connection"""
    sqlite_path = os.getenv("SQLITE_DB_PATH", "./readerz.db")
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    
    return sqlite3.connect(sqlite_path)


def get_postgresql_engine():
    """Get PostgreSQL engine"""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://klippr:klippr@localhost:5432/klippr"
    )
    
    if not database_url.startswith("postgresql://") and not database_url.startswith("postgres://"):
        raise ValueError(f"Invalid PostgreSQL URL: {database_url}")
    
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        echo=False
    )
    
    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✓ Connected to PostgreSQL: {database_url}")
    except Exception as e:
        raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
    
    return engine


def export_sqlite_data(sqlite_conn):
    """Export all data from SQLite"""
    print("\n=== Exporting data from SQLite ===")
    
    cursor = sqlite_conn.cursor()
    
    # Export documents
    cursor.execute("SELECT id, title, file_path, file_type, created_at FROM documents")
    documents = cursor.fetchall()
    print(f"  Found {len(documents)} documents")
    
    # Export chapters
    cursor.execute("""
        SELECT id, document_id, title, content, summary, preview, chapter_number, created_at 
        FROM chapters
        ORDER BY document_id, chapter_number
    """)
    chapters = cursor.fetchall()
    print(f"  Found {len(chapters)} chapters")
    
    return documents, chapters


def create_postgresql_tables(engine):
    """Create tables in PostgreSQL if they don't exist"""
    print("\n=== Creating tables in PostgreSQL ===")
    
    Base.metadata.create_all(bind=engine)
    print("  ✓ Tables created/verified")
    
    # Create indexes
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chapters_document_id 
            ON chapters(document_id)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chapters_chapter_number 
            ON chapters(document_id, chapter_number)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_documents_created_at 
            ON documents(created_at DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_chapters_created_at 
            ON chapters(created_at DESC)
        """))
        
        conn.commit()
    print("  ✓ Indexes created/verified")


def import_data(engine, documents, chapters):
    """Import data to PostgreSQL"""
    print("\n=== Importing data to PostgreSQL ===")
    
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_docs = db.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        if existing_docs > 0:
            response = input(f"  ⚠️  Found {existing_docs} existing documents in PostgreSQL. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("  Migration cancelled.")
                return
            print("  Clearing existing data...")
            db.execute(text("DELETE FROM chapters"))
            db.execute(text("DELETE FROM documents"))
            db.commit()
        
        # Import documents
        print(f"  Importing {len(documents)} documents...")
        for doc in documents:
            db.execute(text("""
                INSERT INTO documents (id, title, file_path, file_type, created_at)
                VALUES (:id, :title, :file_path, :file_type, :created_at)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    file_path = EXCLUDED.file_path,
                    file_type = EXCLUDED.file_type
            """), {
                'id': doc[0],
                'title': doc[1],
                'file_path': doc[2],
                'file_type': doc[3],
                'created_at': doc[4] if doc[4] else datetime.utcnow()
            })
            db.flush()
        
        db.commit()
        print(f"  ✓ Imported {len(documents)} documents")
        
        # Import chapters in batches
        print(f"  Importing {len(chapters)} chapters...")
        batch_size = 100
        for i in range(0, len(chapters), batch_size):
            batch = chapters[i:i + batch_size]
            
            for chapter in batch:
                db.execute(text("""
                    INSERT INTO chapters (
                        id, document_id, title, content, summary, preview, 
                        chapter_number, created_at
                    )
                    VALUES (
                        :id, :document_id, :title, :content, :summary, :preview,
                        :chapter_number, :created_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        summary = EXCLUDED.summary,
                        preview = EXCLUDED.preview,
                        chapter_number = EXCLUDED.chapter_number
                """), {
                    'id': chapter[0],
                    'document_id': chapter[1],
                    'title': chapter[2],
                    'content': chapter[3] if chapter[3] else '',
                    'summary': chapter[4] if chapter[4] else None,
                    'preview': chapter[5] if chapter[5] else None,
                    'chapter_number': chapter[6],
                    'created_at': chapter[7] if chapter[7] else datetime.utcnow()
                })
            
            db.commit()
            print(f"    Imported batch {i//batch_size + 1}/{(len(chapters)-1)//batch_size + 1}")
        
        print(f"  ✓ Imported {len(chapters)} chapters")
        
    except Exception as e:
        db.rollback()
        print(f"  ✗ Error importing data: {e}")
        raise
    finally:
        db.close()


def verify_data(engine, sqlite_conn):
    """Verify data integrity"""
    print("\n=== Verifying data integrity ===")
    
    sqlite_cursor = sqlite_conn.cursor()
    
    # Count documents
    sqlite_cursor.execute("SELECT COUNT(*) FROM documents")
    sqlite_doc_count = sqlite_cursor.fetchone()[0]
    
    with engine.connect() as conn:
        pg_doc_count = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar()
    
    print(f"  Documents: SQLite={sqlite_doc_count}, PostgreSQL={pg_doc_count}")
    
    if sqlite_doc_count != pg_doc_count:
        print(f"  ✗ Document count mismatch!")
        return False
    
    # Count chapters
    sqlite_cursor.execute("SELECT COUNT(*) FROM chapters")
    sqlite_chapter_count = sqlite_cursor.fetchone()[0]
    
    with engine.connect() as conn:
        pg_chapter_count = conn.execute(text("SELECT COUNT(*) FROM chapters")).scalar()
    
    print(f"  Chapters: SQLite={sqlite_chapter_count}, PostgreSQL={pg_chapter_count}")
    
    if sqlite_chapter_count != pg_chapter_count:
        print(f"  ✗ Chapter count mismatch!")
        return False
    
    # Sample verification: Check a few random documents
    print("  Verifying sample data...")
    sqlite_cursor.execute("SELECT id, title FROM documents LIMIT 5")
    sample_docs = sqlite_cursor.fetchall()
    
    with engine.connect() as conn:
        for doc_id, title in sample_docs:
            result = conn.execute(
                text("SELECT title FROM documents WHERE id = :id"),
                {'id': doc_id}
            ).fetchone()
            
            if not result or result[0] != title:
                print(f"  ✗ Data mismatch for document {doc_id}")
                return False
    
    print("  ✓ Data integrity verified")
    return True


def main():
    """Main migration function"""
    print("=" * 60)
    print("SQLite to PostgreSQL Migration")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("DATABASE_URL"):
        print("\n⚠️  DATABASE_URL not set. Using default:")
        print("   postgresql://klippr:klippr@localhost:5432/klippr")
        print("\n   Set DATABASE_URL environment variable to use custom connection.")
        response = input("\nContinue with default? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
    
    try:
        # Connect to databases
        sqlite_conn = get_sqlite_connection()
        pg_engine = get_postgresql_engine()
        
        # Export data
        documents, chapters = export_sqlite_data(sqlite_conn)
        
        if len(documents) == 0:
            print("\n⚠️  No data to migrate. Exiting.")
            return
        
        # Create tables
        create_postgresql_tables(pg_engine)
        
        # Import data
        import_data(pg_engine, documents, chapters)
        
        # Verify
        if verify_data(pg_engine, sqlite_conn):
            print("\n" + "=" * 60)
            print("✓ Migration completed successfully!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Set DATABASE_URL environment variable in your .env file")
            print("2. Restart your backend server")
            print("3. Test the application")
            print("\n⚠️  Keep your SQLite database as backup until you verify everything works.")
        else:
            print("\n" + "=" * 60)
            print("✗ Migration completed with errors. Please review the output above.")
            print("=" * 60)
        
        sqlite_conn.close()
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

