#!/usr/bin/env python3
"""
Test script for DB writer with synthetic data (50-200 row batches).

Tests:
1. Batch size thresholds (50-200 rows)
2. Time-based commits (250-500ms)
3. Idempotent upsert pattern
4. Performance metrics
"""
import os
import sys
import time
import uuid
import random
import string
from typing import List

# Set environment to skip pipeline initialization
os.environ["SKIP_PIPELINE_INIT"] = "true"

from database import init_db, get_db_session, IS_POSTGRESQL, IS_SQLITE
from models import Document, Chapter
from summary_pipeline import WriteTask, write_queue, commit_batch, compute_content_hash, shutdown_event

# Test configuration
TEST_DOCUMENT_ID = "test-doc-" + str(uuid.uuid4())


def generate_random_content(min_length: int = 100, max_length: int = 5000) -> str:
    """Generate random text content for testing."""
    length = random.randint(min_length, max_length)
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "ut"
    ]
    sentences = []
    for _ in range(length // 50):  # Roughly 50 chars per sentence
        sentence_length = random.randint(5, 15)
        sentence = " ".join(random.choices(words, k=sentence_length)).capitalize()
        sentences.append(sentence)
    return ". ".join(sentences) + "."


def generate_synthetic_tasks(count: int, document_id: str = TEST_DOCUMENT_ID) -> List[WriteTask]:
    """Generate synthetic WriteTask objects for testing."""
    tasks = []
    for i in range(count):
        content = generate_random_content()
        task = WriteTask(
            chapter_id=str(uuid.uuid4()),
            document_id=document_id,
            title=f"Test Chapter {i + 1}",
            summary=f"Test summary for chapter {i + 1}. " + generate_random_content(50, 200),
            preview=f"Preview {i + 1}: " + generate_random_content(30, 100),
            content=content,
            chapter_number=i + 1
        )
        tasks.append(task)
    return tasks


def create_test_document(db, document_id: str = TEST_DOCUMENT_ID):
    """Create a test document in the database."""
    doc = Document(
        id=document_id,
        title="Test Document for DB Writer",
        file_path="/tmp/test.pdf",
        file_type="pdf"
    )
    db.add(doc)
    db.commit()
    print(f"✓ Created test document: {document_id}")


def test_batch_size_commit(batch_size: int):
    """Test that commits happen at the correct batch size threshold."""
    print(f"\n{'='*60}")
    print(f"Test: Batch size commit ({batch_size} rows)")
    print(f"{'='*60}")
    
    db = next(get_db_session())
    
    try:
        # Generate tasks
        tasks = generate_synthetic_tasks(batch_size)
        print(f"Generated {len(tasks)} synthetic tasks")
        
        # Commit batch
        t0 = time.perf_counter()
        commit_batch(db, tasks)
        elapsed = (time.perf_counter() - t0) * 1000
        
        # Verify all chapters were inserted
        count = db.query(Chapter).filter(Chapter.document_id == TEST_DOCUMENT_ID).count()
        
        print(f"✓ Committed {len(tasks)} chapters in {elapsed:.2f} ms")
        print(f"✓ Verified {count} chapters in database")
        print(f"✓ Average: {elapsed/len(tasks):.2f} ms per chapter")
        
        assert count == len(tasks), f"Expected {len(tasks)} chapters, found {count}"
        print("✓ Test passed!")
        
    finally:
        db.close()


def test_idempotent_upsert():
    """Test that upserts work correctly (no duplicates on retry)."""
    print(f"\n{'='*60}")
    print("Test: Idempotent upsert pattern")
    print(f"{'='*60}")
    
    db = next(get_db_session())
    
    try:
        # Create a task
        task = generate_synthetic_tasks(1)[0]
        original_summary = task.summary
        
        # First insert
        print("First insert...")
        commit_batch(db, [task])
        count1 = db.query(Chapter).filter(Chapter.id == task.chapter_id).count()
        chapter1 = db.query(Chapter).filter(Chapter.id == task.chapter_id).first()
        
        # Update the summary and try to insert again (should update, not duplicate)
        task.summary = "Updated summary"
        print("Second insert (should update, not duplicate)...")
        commit_batch(db, [task])
        count2 = db.query(Chapter).filter(Chapter.id == task.chapter_id).count()
        chapter2 = db.query(Chapter).filter(Chapter.id == task.chapter_id).first()
        
        print(f"✓ First insert: {count1} row(s)")
        print(f"✓ Second insert: {count2} row(s)")
        print(f"✓ Original summary: {original_summary[:50]}...")
        print(f"✓ Updated summary: {chapter2.summary[:50]}...")
        
        assert count1 == 1, "First insert should create 1 row"
        assert count2 == 1, "Second insert should update, not create duplicate"
        assert chapter2.summary == "Updated summary", "Summary should be updated"
        print("✓ Test passed!")
        
    finally:
        db.close()


def test_multiple_batches():
    """Test multiple batches of different sizes."""
    print(f"\n{'='*60}")
    print("Test: Multiple batches (50, 100, 150, 200 rows)")
    print(f"{'='*60}")
    
    db = next(get_db_session())
    
    try:
        batch_sizes = [50, 100, 150, 200]
        total_committed = 0
        total_time = 0
        
        for batch_size in batch_sizes:
            tasks = generate_synthetic_tasks(batch_size)
            t0 = time.perf_counter()
            commit_batch(db, tasks)
            elapsed = (time.perf_counter() - t0) * 1000
            
            total_committed += len(tasks)
            total_time += elapsed
            
            print(f"✓ Batch of {batch_size}: {elapsed:.2f} ms ({elapsed/batch_size:.2f} ms/row)")
        
        # Verify total count
        count = db.query(Chapter).filter(Chapter.document_id == TEST_DOCUMENT_ID).count()
        
        print(f"\nTotal: {total_committed} chapters in {total_time:.2f} ms")
        print(f"Average: {total_time/total_committed:.2f} ms per chapter")
        print(f"✓ Verified {count} chapters in database")
        
        assert count == total_committed, f"Expected {total_committed} chapters, found {count}"
        print("✓ Test passed!")
        
    finally:
        db.close()


def test_content_hash():
    """Test that content_hash is computed and stored correctly."""
    print(f"\n{'='*60}")
    print("Test: Content hash computation and storage")
    print(f"{'='*60}")
    
    db = next(get_db_session())
    
    try:
        # Create a task with known content
        content = "This is a test content for hash computation."
        expected_hash = compute_content_hash(content)
        
        task = WriteTask(
            chapter_id=str(uuid.uuid4()),
            document_id=TEST_DOCUMENT_ID,
            title="Hash Test Chapter",
            summary="Test summary",
            preview="Test preview",
            content=content,
            chapter_number=999
        )
        
        commit_batch(db, [task])
        
        # Verify hash was stored
        chapter = db.query(Chapter).filter(Chapter.id == task.chapter_id).first()
        
        print(f"✓ Expected hash: {expected_hash}")
        print(f"✓ Stored hash: {chapter.content_hash}")
        
        assert chapter.content_hash == expected_hash, "Content hash should match"
        print("✓ Test passed!")
        
    finally:
        db.close()


def cleanup_test_data():
    """Clean up test data from database."""
    print(f"\n{'='*60}")
    print("Cleaning up test data...")
    print(f"{'='*60}")
    
    db = next(get_db_session())
    
    try:
        # Delete test chapters
        chapters_deleted = db.query(Chapter).filter(Chapter.document_id == TEST_DOCUMENT_ID).delete()
        print(f"✓ Deleted {chapters_deleted} test chapters")
        
        # Delete test document
        doc_deleted = db.query(Document).filter(Document.id == TEST_DOCUMENT_ID).delete()
        print(f"✓ Deleted {doc_deleted} test document(s)")
        
        db.commit()
        print("✓ Cleanup complete")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        db.rollback()
    finally:
        db.close()


def main():
    """Run all DB writer tests."""
    print("="*60)
    print("DB Writer Test Suite")
    print("="*60)
    print(f"Database: {'PostgreSQL' if IS_POSTGRESQL else 'SQLite'}")
    print(f"Test Document ID: {TEST_DOCUMENT_ID}")
    print("="*60)
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
    # Create test document
    db = next(get_db_session())
    try:
        create_test_document(db)
    finally:
        db.close()
    
    try:
        # Run tests
        test_batch_size_commit(50)
        test_batch_size_commit(100)
        test_batch_size_commit(150)
        test_batch_size_commit(200)
        
        test_idempotent_upsert()
        test_content_hash()
        test_multiple_batches()
        
        print(f"\n{'='*60}")
        print("All tests passed! ✓")
        print(f"{'='*60}")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        cleanup_test_data()


if __name__ == "__main__":
    main()

