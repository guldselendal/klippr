#!/usr/bin/env python3
"""
Optimized performance measurement script for database save operations.
Run this after optimizations to compare with baseline.
"""
import time
import uuid
from contextlib import contextmanager
from database import get_db_session, init_db
from models import Document, Chapter
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Track query counts
query_count = 0
query_times = []

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    global query_count
    query_count += 1
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = conn.info['query_start_time'].pop(-1)
    query_times.append(time.time() - total)

def create_test_data(num_chapters=100, chapter_size=5000):
    """Create test data similar to real uploads"""
    file_id = str(uuid.uuid4())
    chapters_data = []
    
    for i in range(num_chapters):
        chapters_data.append({
            'id': str(uuid.uuid4()),
            'document_id': file_id,
            'title': f'Chapter {i+1}',
            'content': 'X' * chapter_size,
            'summary': 'Summary text ' * 50,
            'preview': 'Preview text ' * 20,
            'chapter_number': i + 1
        })
    
    return file_id, chapters_data

def optimized_save_method(db, file_id, chapters_data, display_title="Test Document"):
    """Optimized save method - bulk inserts"""
    global query_count, query_times
    query_count = 0
    query_times = []
    
    # Create document
    document = Document(
        id=file_id,
        title=display_title,
        file_path=f"test_{file_id}.epub",
        file_type="epub"
    )
    db.add(document)
    db.flush()  # Flush document first so foreign key constraint is satisfied
    
    # Prepare chapters as dicts for bulk insert
    chapters_dicts = []
    for chapter_data in chapters_data:
        chapters_dicts.append({
            'id': chapter_data['id'],
            'document_id': chapter_data['document_id'],
            'title': chapter_data['title'],
            'content': chapter_data['content'],
            'summary': chapter_data['summary'],
            'preview': chapter_data['preview'],
            'chapter_number': chapter_data['chapter_number']
        })
    
    # Bulk insert (much faster than individual adds)
    if chapters_dicts:
        db.bulk_insert_mappings(Chapter, chapters_dicts)
    
    db.commit()

def measure_optimized(num_chapters=100, chapter_size=5000, iterations=3):
    """Measure optimized performance"""
    print("=" * 60)
    print(f"OPTIMIZED MEASUREMENT")
    print(f"Chapters: {num_chapters}, Chapter size: {chapter_size} chars")
    print("=" * 60)
    
    init_db()
    
    times = []
    query_counts = []
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        
        file_id, chapters_data = create_test_data(num_chapters, chapter_size)
        
        db = next(get_db_session())
        try:
            # Delete if exists
            db.query(Chapter).filter(Chapter.document_id == file_id).delete()
            db.query(Document).filter(Document.id == file_id).delete()
            db.commit()
            
            # Measure save
            start = time.time()
            optimized_save_method(db, file_id, chapters_data)
            elapsed = time.time() - start
            
            times.append(elapsed)
            query_counts.append(query_count)
            
            print(f"  Total time: {elapsed:.3f}s")
            print(f"  Queries executed: {query_count}")
            if query_times:
                print(f"  Avg query time: {sum(query_times)/len(query_times)*1000:.2f}ms")
                print(f"  Max query time: {max(query_times)*1000:.2f}ms")
            
        finally:
            db.close()
    
    avg_time = sum(times) / len(times)
    avg_queries = sum(query_counts) / len(query_counts)
    
    print("\n" + "=" * 60)
    print("OPTIMIZED SUMMARY")
    print("=" * 60)
    print(f"Average save time: {avg_time:.3f}s")
    print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")
    print(f"Average queries: {avg_queries:.0f}")
    print("=" * 60)
    
    return avg_time, avg_queries

if __name__ == "__main__":
    print("\n🚀 OPTIMIZED PERFORMANCE MEASUREMENT\n")
    
    # Test with different sizes
    print("\n📊 Small document (10 chapters)")
    opt_small_time, opt_small_queries = measure_optimized(num_chapters=10, chapter_size=2000, iterations=3)
    
    print("\n📊 Medium document (50 chapters)")
    opt_medium_time, opt_medium_queries = measure_optimized(num_chapters=50, chapter_size=5000, iterations=3)
    
    print("\n📊 Large document (100 chapters)")
    opt_large_time, opt_large_queries = measure_optimized(num_chapters=100, chapter_size=5000, iterations=3)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("Run benchmark_baseline.py first to get baseline metrics")
    print("Then compare the results above with baseline measurements")

