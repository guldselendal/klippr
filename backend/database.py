from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
import os

# Database URL - supports both SQLite (legacy) and PostgreSQL
# PostgreSQL format: postgresql://user:password@localhost:5432/dbname
# SQLite format: sqlite:///./readerz.db
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://klippr:klippr@localhost:5432/klippr"
)

# Detect database type from URL
IS_POSTGRESQL = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")
IS_SQLITE = DATABASE_URL.startswith("sqlite://")

# Engine configuration optimized for PostgreSQL
if IS_POSTGRESQL:
    # PostgreSQL connection pool settings for high concurrency
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,  # Number of connections to maintain
        max_overflow=20,  # Additional connections beyond pool_size
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False,  # Set to True for SQL query logging
        future=True,  # SQLAlchemy 2.0 style
        connect_args={
            "connect_timeout": 10,  # Connection timeout in seconds
            "application_name": "klippr_backend"  # Identify connections in pg_stat_activity
        }
    )
elif IS_SQLITE:
    # SQLite configuration (legacy support)
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        future=True,
        echo=False,
        pool_pre_ping=True
    )
else:
    raise ValueError(f"Unsupported database URL format: {DATABASE_URL}")

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,  # Disable autoflush for bulk operations
    bind=engine,
    expire_on_commit=False  # Don't expire objects after commit (faster)
)

Base = declarative_base()


@event.listens_for(Engine, "connect")
def set_database_settings(dbapi_conn, connection_record):
    """Set database-specific settings on connection"""
    if IS_POSTGRESQL:
        # PostgreSQL optimizations for write performance
        cursor = dbapi_conn.cursor()
        
        # Set timezone to UTC
        cursor.execute("SET timezone = 'UTC'")
        
        # Optimize for write-heavy workloads
        cursor.execute("SET synchronous_commit = 'on'")  # Balance speed/durability
        
        # Increase work_mem for better sort/join performance (per connection)
        # Default is 4MB, increase to 16MB for better bulk insert performance
        cursor.execute("SET work_mem = '16MB'")
        
        # Enable parallel queries (PostgreSQL 9.6+)
        cursor.execute("SET max_parallel_workers_per_gather = 2")
        
        # Set statement timeout (prevent long-running queries)
        cursor.execute("SET statement_timeout = '300s'")  # 5 minutes
        
        cursor.close()
    elif IS_SQLITE:
        # SQLite PRAGMAs (legacy support)
        cursor = dbapi_conn.cursor()
        
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA cache_size=-100000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA wal_autocheckpoint=2000")
        
        # Performance optimizations: mmap_size and page_size
        # mmap_size: Use memory-mapped I/O for faster reads (256MB)
        # Note: This can be set at runtime
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        # page_size: Larger page size for better performance (8KB)
        # Note: Only takes effect on new databases; existing databases keep their page size
        cursor.execute("PRAGMA page_size=8192")
        
        is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
        if not is_production:
            cursor.execute("PRAGMA checkpoint_fullfsync=OFF")
        
        cursor.close()


def init_db():
    """Initialize database tables and indexes"""
    from models import Document, Chapter
    Base.metadata.create_all(bind=engine)
    
    # Create indexes for foreign keys and common queries
    with engine.connect() as conn:
        if IS_POSTGRESQL:
            # PostgreSQL: Use IF NOT EXISTS (PostgreSQL 9.5+)
            
            # Unique index on id for idempotent upserts (primary key already ensures uniqueness, but explicit for clarity)
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS ix_chapters_id 
                ON chapters(id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chapters_document_id 
                ON chapters(document_id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chapters_chapter_number 
                ON chapters(document_id, chapter_number)
            """))
            
            # Index for content_hash (for deduplication) - only if column exists
            try:
                # Check if content_hash column exists by querying information_schema
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'chapters' AND column_name = 'content_hash'
                """))
                if result.fetchone():
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS ix_chapters_content_hash 
                        ON chapters(content_hash)
                    """))
            except Exception as e:
                # Column doesn't exist yet, skip index creation
                pass  # Silently skip if column doesn't exist
            
            # Additional indexes for PostgreSQL performance
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_documents_created_at 
                ON documents(created_at DESC)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chapters_created_at 
                ON chapters(created_at DESC)
            """))
        else:
            # SQLite: Use IF NOT EXISTS
            
            # Unique index on id for idempotent upserts (primary key already ensures uniqueness, but explicit for clarity)
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS ix_chapters_id 
                ON chapters(id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chapters_document_id 
                ON chapters(document_id)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_chapters_chapter_number 
                ON chapters(document_id, chapter_number)
            """))
            
            # Index for content_hash (for deduplication) - only if column exists
            try:
                # Check if content_hash column exists by querying information_schema
                result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'chapters' AND column_name = 'content_hash'
                """))
                if result.fetchone():
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS ix_chapters_content_hash 
                        ON chapters(content_hash)
                    """))
            except Exception as e:
                # Column doesn't exist yet, skip index creation
                pass  # Silently skip if column doesn't exist
        
        conn.commit()


def get_db_session():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
