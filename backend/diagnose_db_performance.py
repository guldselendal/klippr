"""
Diagnostic script to measure database save performance.
Run this to identify bottlenecks in the "Saving to database" step.
"""
import time
import os
import sys
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from models import Document, Chapter
import uuid

# Database configuration (matches database.py)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://klippr:klippr@localhost:5432/klippr"
)

# Detect database type from URL
IS_POSTGRESQL = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")
IS_SQLITE = DATABASE_URL.startswith("sqlite://")

# Track SQL execution times
sql_times = []

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Record start time before SQL execution"""
    conn.info.setdefault('query_start_time', []).append(time.perf_counter())

@event.listens_for(Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Record end time after SQL execution"""
    total = time.perf_counter() - conn.info['query_start_time'].pop()
    sql_times.append({
        'statement': statement[:100],  # First 100 chars
        'time_ms': total * 1000,
        'executemany': executemany
    })

# Create engine (matching database.py)
if IS_POSTGRESQL:
    from sqlalchemy.pool import QueuePool
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
        future=True,
        connect_args={
            "connect_timeout": 10,
            "application_name": "klippr_diagnostics"
        }
    )
elif IS_SQLITE:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        future=True,
        echo=False,
        pool_pre_ping=True
    )
else:
    raise ValueError(f"Unsupported database URL format: {DATABASE_URL}")

# Set database-specific settings (matching database.py)
@event.listens_for(Engine, "connect")
def set_database_settings(dbapi_conn, connection_record):
    """Set database-specific settings for optimal performance"""
    if IS_POSTGRESQL:
        cursor = dbapi_conn.cursor()
        cursor.execute("SET timezone = 'UTC'")
        cursor.execute("SET synchronous_commit = 'on'")
        cursor.execute("SET work_mem = '16MB'")
        cursor.execute("SET max_parallel_workers_per_gather = 2")
        cursor.execute("SET statement_timeout = '300s'")
        cursor.close()
    elif IS_SQLITE:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA cache_size=-100000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA wal_autocheckpoint=2000")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        # Note: page_size can only be set on new databases
        cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

def check_database_settings():
    """Check current database settings"""
    if IS_SQLITE:
        print("\n=== SQLite PRAGMA Settings ===")
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode"))
            print(f"journal_mode: {result.scalar()}")
            
            result = conn.execute(text("PRAGMA synchronous"))
            sync_val = result.scalar()
            sync_map = {0: "OFF", 1: "NORMAL", 2: "FULL", 3: "EXTRA"}
            print(f"synchronous: {sync_map.get(sync_val, sync_val)}")
            
            result = conn.execute(text("PRAGMA foreign_keys"))
            print(f"foreign_keys: {result.scalar()}")
            
            result = conn.execute(text("PRAGMA cache_size"))
            cache_kb = result.scalar()
            print(f"cache_size: {cache_kb} KB ({abs(cache_kb) / 1024:.1f} MB if negative)")
            
            result = conn.execute(text("PRAGMA temp_store"))
            temp_map = {0: "DEFAULT", 1: "FILE", 2: "MEMORY"}
            print(f"temp_store: {temp_map.get(result.scalar(), 'UNKNOWN')}")
            
            result = conn.execute(text("PRAGMA mmap_size"))
            mmap_bytes = result.scalar()
            print(f"mmap_size: {mmap_bytes / (1024*1024):.1f} MB")
            
            result = conn.execute(text("PRAGMA page_size"))
            page_size = result.scalar()
            print(f"page_size: {page_size} bytes")
            
            result = conn.execute(text("PRAGMA busy_timeout"))
            print(f"busy_timeout: {result.scalar()} ms")
    elif IS_POSTGRESQL:
        print("\n=== PostgreSQL Settings ===")
        with engine.connect() as conn:
            result = conn.execute(text("SHOW timezone"))
            print(f"timezone: {result.scalar()}")
            
            result = conn.execute(text("SHOW synchronous_commit"))
            print(f"synchronous_commit: {result.scalar()}")
            
            result = conn.execute(text("SHOW work_mem"))
            print(f"work_mem: {result.scalar()}")
            
            result = conn.execute(text("SHOW max_parallel_workers_per_gather"))
            print(f"max_parallel_workers_per_gather: {result.scalar()}")
            
            result = conn.execute(text("SHOW statement_timeout"))
            print(f"statement_timeout: {result.scalar()}")

def simulate_bulk_insert(num_chapters=50, content_size=5000):
    """Simulate the bulk insert operation from upload_file endpoint"""
    print(f"\n=== Simulating Bulk Insert ({num_chapters} chapters) ===")
    
    # Clear previous timings
    sql_times.clear()
    
    db = SessionLocal()
    try:
        # Create a test document
        doc_id = str(uuid.uuid4())
        document = Document(
            id=doc_id,
            title="Test Document",
            file_path="/test/path",
            file_type="epub"
        )
        
        # Time document insert
        t0 = time.perf_counter()
        db.add(document)
        db.flush()
        t_flush = (time.perf_counter() - t0) * 1000
        print(f"Document flush: {t_flush:.2f} ms")
        
        # Prepare chapters data (matching upload_file structure)
        chapters_dicts = []
        for idx in range(num_chapters):
            chapters_dicts.append({
                'id': str(uuid.uuid4()),
                'document_id': doc_id,
                'title': f'Chapter {idx + 1}',
                'content': 'X' * content_size,  # Simulate chapter content
                'summary': 'X' * 2000,  # Simulate summary
                'preview': 'X' * 300,  # Simulate preview
                'chapter_number': idx + 1
            })
        
        # Time bulk insert
        t0 = time.perf_counter()
        db.bulk_insert_mappings(Chapter, chapters_dicts)
        t_bulk = (time.perf_counter() - t0) * 1000
        print(f"Bulk insert mapping: {t_bulk:.2f} ms")
        
        # Time commit
        t0 = time.perf_counter()
        db.commit()
        t_commit = (time.perf_counter() - t0) * 1000
        print(f"Commit: {t_commit:.2f} ms")
        
        total_time = t_flush + t_bulk + t_commit
        print(f"\nTotal database save time: {total_time:.2f} ms")
        print(f"  - Document flush: {t_flush:.2f} ms ({t_flush/total_time*100:.1f}%)")
        print(f"  - Bulk insert: {t_bulk:.2f} ms ({t_bulk/total_time*100:.1f}%)")
        print(f"  - Commit: {t_commit:.2f} ms ({t_commit/total_time*100:.1f}%)")
        
        # Show SQL statement timings
        if sql_times:
            print(f"\n=== SQL Statement Timings ===")
            total_sql_time = sum(st['time_ms'] for st in sql_times)
            for st in sql_times:
                pct = (st['time_ms'] / total_sql_time * 100) if total_sql_time > 0 else 0
                exec_type = "executemany" if st['executemany'] else "execute"
                print(f"{st['time_ms']:.2f} ms ({pct:.1f}%) - {exec_type}: {st['statement']}")
        
        # Clean up test data
        db.delete(document)
        db.commit()
        
        return total_time
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()

def check_database_stats():
    """Check database file size and statistics"""
    print("\n=== Database Statistics ===")
    
    if IS_SQLITE:
        db_path = DATABASE_URL.replace("sqlite:///", "")
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            print(f"Database file size: {size_mb:.2f} MB")
        else:
            print(f"Database file not found: {db_path}")
            return
    
    with engine.connect() as conn:
        # Count documents and chapters
        result = conn.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = result.scalar()
        result = conn.execute(text("SELECT COUNT(*) FROM chapters"))
        chapter_count = result.scalar()
        print(f"Documents: {doc_count}")
        print(f"Chapters: {chapter_count}")
        
        # Check for indexes
        if IS_SQLITE:
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='chapters'
            """))
            indexes = [row[0] for row in result]
            print(f"Chapter indexes: {indexes}")
        elif IS_POSTGRESQL:
            result = conn.execute(text("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'chapters'
            """))
            indexes = [row[0] for row in result]
            print(f"Chapter indexes: {indexes}")

def main():
    print("=" * 60)
    print("Database Performance Diagnostic")
    print("=" * 60)
    
    # Check database settings
    check_database_settings()
    
    # Check database stats
    check_database_stats()
    
    # Run performance tests
    print("\n" + "=" * 60)
    print("Performance Test: Small Batch (10 chapters)")
    print("=" * 60)
    time1 = simulate_bulk_insert(10, 5000)
    
    print("\n" + "=" * 60)
    print("Performance Test: Medium Batch (50 chapters)")
    print("=" * 60)
    time2 = simulate_bulk_insert(50, 5000)
    
    print("\n" + "=" * 60)
    print("Performance Test: Large Batch (100 chapters)")
    print("=" * 60)
    time3 = simulate_bulk_insert(100, 5000)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"10 chapters:  {time1:.2f} ms ({time1/10:.2f} ms/chapter)")
    print(f"50 chapters:  {time2:.2f} ms ({time2/50:.2f} ms/chapter)")
    print(f"100 chapters: {time3:.2f} ms ({time3/100:.2f} ms/chapter)")
    
    # Check if performance degrades with size
    if time3 and time2:
        scaling_factor = (time3 / 100) / (time2 / 50)
        if scaling_factor > 1.5:
            print(f"\n⚠️  WARNING: Performance degrades with batch size (scaling factor: {scaling_factor:.2f}x)")
        else:
            print(f"\n✓ Performance scales well (scaling factor: {scaling_factor:.2f}x)")

if __name__ == "__main__":
    main()

