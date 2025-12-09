from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # 'epub' or 'pdf'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chapters = relationship("Chapter", back_populates="document", cascade="all, delete-orphan")


class Chapter(Base):
    __tablename__ = "chapters"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # Detailed comprehensive summary
    preview = Column(Text, nullable=True)  # 3-sentence preview for chapter cards
    content_hash = Column(String, nullable=True)  # SHA256 hash of content for deduplication
    chapter_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="chapters")
    chunk_summaries = relationship("ChunkSummary", back_populates="chapter", cascade="all, delete-orphan")


class ChunkSummary(Base):
    """
    Stores individual chunk summaries for checkpointing and resume capability.
    Enables resuming chapter processing from partial progress.
    """
    __tablename__ = "chunk_summaries"
    
    id = Column(String, primary_key=True)  # Format: "{chapter_id}:{chunk_index}"
    chapter_id = Column(String, ForeignKey("chapters.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)  # 0-based index of chunk within chapter
    content_hash = Column(String, nullable=False, index=True)  # SHA256 hash of chunk content
    summary = Column(Text, nullable=True)  # The summary text
    key_points = Column(Text, nullable=True)  # JSON array of key points
    entities = Column(Text, nullable=True)  # JSON array of entities
    status = Column(String, nullable=False, default="pending")  # "pending", "completed", "failed"
    error_message = Column(Text, nullable=True)  # Error message if status is "failed"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    chapter = relationship("Chapter", back_populates="chunk_summaries")

