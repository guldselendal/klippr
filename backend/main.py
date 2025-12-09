from fastapi import FastAPI, File, UploadFile, HTTPException, Body, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from database import init_db, get_db_session
from models import Document, Chapter
from parsers import parse_epub, parse_pdf
from connections import find_connections
from summarizer import generate_summary, process_summary_for_chapter, generate_summaries_parallel, process_summaries_for_titles_and_previews
from utils import (
    clean_filename, normalize_book_name, validate_file_extension,
    db_session, format_chapter_response, handle_parse_error
)

# Note: Automatic summarization on upload has been removed.
# Summaries can be generated manually via:
# - POST /api/chapters/{chapter_id}/summarize (single chapter)
# - POST /api/documents/{document_id}/summarize-all (all chapters in document)

app = FastAPI(title="ReaderZ API")

# Timing middleware for performance diagnostics
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Measure request latency for performance diagnostics"""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    # Only log slow requests (>100ms) or upload endpoints
    if dt > 100 or "/api/upload" in str(request.url):
        print(f"LATENCY {request.method} {request.url.path} {dt:.1f} ms")
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Files are stored in the 'uploads' directory relative to the backend folder
# Full path: backend/uploads/
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
print(f"Upload directory: {os.path.abspath(UPLOAD_DIR)}")


def check_duplicate(db, filename: str) -> bool:
    """Check if a book with the same normalized name already exists"""
    normalized_name = normalize_book_name(filename)
    existing_docs = db.query(Document).all()
    return any(normalize_book_name(doc.title) == normalized_name for doc in existing_docs)


@app.get("/")
async def root():
    return {"message": "ReaderZ API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for frontend to verify backend availability"""
    return {"status": "ok", "message": "Backend is running"}


@app.get("/api/pipeline/status")
async def pipeline_status():
    """Get status of LLM concurrency and metrics (async pipeline removed)"""
    try:
        from pipeline_metrics import get_all_metrics
        from llm_concurrency import LLMConcurrencyLimiter
        
        # Get comprehensive metrics
        metrics = get_all_metrics()
        
        # Get LLM concurrency metrics
        limiter = LLMConcurrencyLimiter()
        concurrency_metrics = limiter.get_metrics()
        
        return {
            "async_enabled": False,
            "message": "Automatic summarization on upload has been removed. Use manual summarize endpoints.",
            "metrics": metrics,
            "llm_concurrency": concurrency_metrics
        }
    except ImportError:
        # Fallback if metrics module not available
        return {
            "async_enabled": False,
            "message": "Automatic summarization on upload has been removed. Use manual summarize endpoints.",
            "metrics": "Metrics module not available"
        }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and parse EPUB or PDF file"""
    file_ext = validate_file_extension(file.filename)
    
    # Check for duplicates before processing
    with db_session() as db:
        if check_duplicate(db, file.filename):
            cleaned_name = clean_filename(os.path.splitext(file.filename)[0])
            raise HTTPException(
                status_code=409, 
                detail=f"Book '{cleaned_name}' already exists in your library. Duplicates are not allowed."
            )
    
    # Clean the filename (remove parentheses)
    display_title = clean_filename(file.filename)
    if not display_title or display_title == file_ext:
        raise HTTPException(status_code=400, detail="Invalid filename after cleaning")
    
    # Save file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        # Parse file based on extension
        try:
            chapters_data = parse_epub(file_path) if file_ext == '.epub' else parse_pdf(file_path)
        except Exception as parse_error:
            raise handle_parse_error(file_ext, parse_error)
        
        if not chapters_data or len(chapters_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="No chapters found in the file. The file may be empty or in an unsupported format."
            )
        
        # Save document to database
        db_save_start = time.perf_counter()
        with db_session() as db:
            document = Document(
                id=file_id,
                title=display_title,
                file_path=file_path,
                file_type=file_ext[1:]  # Remove the dot
            )
            db.add(document)
            db.flush()  # Flush document first so foreign key constraint is satisfied
            db.commit()
        t_db_total = (time.perf_counter() - db_save_start) * 1000
        print(f"DB: Document saved in {t_db_total:.2f} ms")
        
        # Save chapters without summaries (automatic summarization removed)
        print(f"Saving {len(chapters_data)} chapters without summaries...")
        
        db = next(get_db_session())
        chapters_dicts = []
        chapters = []
        
        for idx, chapter_data in enumerate(chapters_data):
            chapter_content = chapter_data.get('content', '')
            original_title = chapter_data.get('title', f'Chapter {idx + 1}')
            chapter_id = str(uuid.uuid4())
            
            chapters_dicts.append({
                'id': chapter_id,
                'document_id': file_id,
                'title': original_title,
                'content': chapter_content,
                'summary': None,  # No automatic summary
                'preview': None,  # No automatic preview
                'chapter_number': idx + 1
            })
            
            chapters.append({
                'id': chapter_id,
                'title': original_title,
                'content': chapter_content,
                'summary': None,
                'preview': None,
                'document_id': file_id,
                'document_title': display_title
            })
        
        # Bulk insert chapters
        if chapters_dicts:
            db.bulk_insert_mappings(Chapter, chapters_dicts)
        
        db.commit()
        db.close()
        
        return JSONResponse({
            "message": "File uploaded and parsed successfully.",
            "document_id": file_id,
            "chapters": chapters
        })
    
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        # Provide error message for file parsing/upload errors
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {error_detail}")


@app.post("/api/upload/batch")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and parse multiple EPUB or PDF files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    db = next(get_db_session())
    
    for file in files:
        if not file.filename:
            results.append({
                "filename": "unknown",
                "success": False,
                "error": "No filename provided"
            })
            continue
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.epub', '.pdf']:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Only EPUB and PDF files are supported"
            })
            continue
        
        # Check for duplicates
        if check_duplicate(db, file.filename):
            cleaned_name = clean_filename(os.path.splitext(file.filename)[0])
            results.append({
                "filename": file.filename,
                "success": False,
                "error": f"Book '{cleaned_name}' already exists in your library. Duplicates are not allowed."
            })
            continue
        
        # Clean the filename (remove parentheses)
        display_title = clean_filename(file.filename)
        if not display_title or display_title == file_ext:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Invalid filename after cleaning"
            })
            continue
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Parse file based on extension
            if file_ext == '.epub':
                chapters_data = parse_epub(file_path)
            else:
                chapters_data = parse_pdf(file_path)
            
            # Save to database with cleaned title
            document = Document(
                id=file_id,
                title=display_title,  # Use cleaned filename
                file_path=file_path,
                file_type=file_ext[1:]  # Remove the dot
            )
            db.add(document)
            db.flush()  # Flush document first so foreign key constraint is satisfied for bulk inserts
            
            # Save chapters without summaries (automatic summarization removed)
            print(f"Saving {len(chapters_data)} chapters without summaries...")
            
            # Prepare chapters data for bulk insert
            chapters_dicts = []
            chapters = []
            
            for idx, chapter_data in enumerate(chapters_data):
                chapter_content = chapter_data.get('content', '')
                original_title = chapter_data.get('title', f'Chapter {idx + 1}')
                chapter_id = str(uuid.uuid4())
                
                # Prepare dict for bulk insert
                chapters_dicts.append({
                    'id': chapter_id,
                    'document_id': file_id,
                    'title': original_title,
                    'content': chapter_content,
                    'summary': None,  # No automatic summary
                    'preview': None,  # No automatic preview
                    'chapter_number': idx + 1
                })
                
                # Prepare response data
                chapters.append({
                    'id': chapter_id,
                    'title': original_title,
                    'content': chapter_content,
                    'summary': None,
                    'preview': None,
                    'document_id': file_id,
                    'document_title': display_title
                })
            
            # Bulk insert chapters using bulk_insert_mappings (much faster than individual adds)
            if chapters_dicts:
                db.bulk_insert_mappings(Chapter, chapters_dicts)
            
            db.commit()
            
            results.append({
                "filename": file.filename,
                "success": True,
                "document_id": file_id,
                "chapters": chapters
            })
        
        except Exception as e:
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    db.close()
    
    # Collect all chapters from successful uploads
    all_chapters = []
    for result in results:
        if result.get("success") and result.get("chapters"):
            all_chapters.extend(result["chapters"])
    
    return JSONResponse({
        "message": f"Processed {len(files)} file(s)",
        "results": results,
        "chapters": all_chapters
    })


@app.get("/api/chapters")
async def get_chapters(include_content: bool = False):
    """Get all chapters from all documents. Set include_content=True to get full content."""
    with db_session() as db:
        chapters = db.query(Chapter).all()
        documents = {doc.id: doc.title for doc in db.query(Document).all()}
        
        result = []
        for chapter in chapters:
            chapter_data = format_chapter_response(chapter, include_content)
            chapter_data['document_title'] = documents.get(chapter.document_id, 'Unknown')
            result.append(chapter_data)
    
    return {"chapters": result}


@app.get("/api/connections")
async def get_connections():
    """Find and return connections between chapters"""
    try:
        with db_session() as db:
            chapters = db.query(Chapter).all()
            documents = {doc.id: doc.title for doc in db.query(Document).all()}
        
        if len(chapters) < 2:
            return {"connections": []}
        
        # Prepare chapter data
        chapter_data = [
            {
                'id': chapter.id,
                'title': chapter.title,
                'content': chapter.content,
                'document_id': chapter.document_id,
                'document_title': documents.get(chapter.document_id, 'Unknown')
            }
            for chapter in chapters
        ]
        
        connections = find_connections(chapter_data)
        return {"connections": connections}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding connections: {str(e)}")


@app.get("/api/documents")
async def get_documents():
    """Get all uploaded documents"""
    with db_session() as db:
        documents = db.query(Document).all()
        result = [
            {
                'id': doc.id,
                'title': doc.title,
                'file_type': doc.file_type,
                'chapter_count': db.query(Chapter).filter(Chapter.document_id == doc.id).count(),
                'uploaded_at': doc.created_at.isoformat() if doc.created_at else None
            }
            for doc in documents
        ]
    return {"documents": result}


@app.get("/api/documents/{document_id}/chapters")
async def get_document_chapters(document_id: str, include_content: bool = False):
    """Get all chapters for a specific document. Set include_content=True to get full content."""
    with db_session() as db:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        chapters = db.query(Chapter).filter(Chapter.document_id == document_id).order_by(Chapter.chapter_number).all()
        result = [
            {**format_chapter_response(chapter, include_content), 'document_title': document.title}
            for chapter in chapters
        ]
    
    return {"chapters": result}


@app.get("/api/chapters/{chapter_id}")
async def get_chapter(chapter_id: str):
    """Get a single chapter with full content"""
    with db_session() as db:
        chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")
        
        document = db.query(Document).filter(Document.id == chapter.document_id).first()
        result = format_chapter_response(chapter, include_content=True)
        result['document_title'] = document.title if document else 'Unknown'
    
    return {"chapter": result}


@app.post("/api/chapters/{chapter_id}/summarize")
async def summarize_chapter(chapter_id: str, background_tasks: BackgroundTasks = None):
    """Generate a summary for a chapter"""
    db = next(get_db_session())
    try:
        chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()
        
        if not chapter:
            db.close()
            raise HTTPException(status_code=404, detail="Chapter not found")
        
        if not chapter.content or len(chapter.content.strip()) == 0:
            db.close()
            raise HTTPException(status_code=400, detail="Chapter has no content to summarize")
        
        # Generate summary
        try:
            summary = generate_summary(chapter.content, chapter.title)
            if not summary or len(summary.strip()) == 0:
                # If summary generation failed silently, provide a fallback
                summary = chapter.content[:200] + "..." if len(chapter.content) > 200 else chapter.content
        except Exception as e:
            print(f"Error generating summary for chapter {chapter_id}: {e}")
            db.close()
            error_msg = str(e)
            if "connection" in error_msg.lower() or "ollama" in error_msg.lower():
                raise HTTPException(
                    status_code=503, 
                    detail="Cannot connect to Ollama. Please make sure Ollama is running on http://localhost:11434"
                )
            raise HTTPException(
                status_code=500, 
                detail=f"Error generating summary: {error_msg}"
            )
        
        # Generate takeaway title and preview from the summary
        takeaway_title, preview = process_summary_for_chapter(summary)
        
        # Update chapter with summary, title, and preview
        chapter.summary = summary
        if takeaway_title:
            chapter.title = takeaway_title  # Update title with main takeaway
        if preview:
            chapter.preview = preview
        db.commit()
        
        document = db.query(Document).filter(Document.id == chapter.document_id).first()
        
        result = {
            'id': chapter.id,
            'title': chapter.title,
            'content': chapter.content,
            'summary': chapter.summary or '',
            'preview': chapter.preview or '',
            'document_id': chapter.document_id,
            'document_title': document.title if document else 'Unknown',
            'chapter_number': chapter.chapter_number
        }
        
        db.close()
        return {"chapter": result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in summarize_chapter: {e}")
        import traceback
        traceback.print_exc()
        if db:
            db.close()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/api/documents/{document_id}/summarize-all")
async def summarize_all_chapters(document_id: str):
    """Generate summaries for all chapters in a document in parallel"""
    try:
        with db_session() as db:
            document = db.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            chapters = db.query(Chapter).filter(Chapter.document_id == document_id).all()
            if not chapters:
                return {"message": "No chapters found", "summarized": 0}
            
            # Prepare data for parallel processing
            chapters_data = [
                {'content': ch.content, 'title': ch.title, 'id': ch.id}
                for ch in chapters
                if ch.content and len(ch.content.strip()) > 0
            ]
            
            if not chapters_data:
                return {"message": "No chapters with content found", "summarized": 0}
        
        # Generate summaries in parallel (outside db session for long-running operation)
        max_workers = int(os.getenv("SUMMARY_MAX_WORKERS", 8))
        print(f"Generating summaries for {len(chapters_data)} chapters in parallel (using {max_workers} workers)...")
        summaries = generate_summaries_parallel(chapters_data, max_workers=max_workers)
        print(f"Successfully generated {len([s for s in summaries if s])} summaries")
        
        print(f"Generating titles and previews from summaries (using {max_workers} workers)...")
        titles_and_previews = process_summaries_for_titles_and_previews(summaries, max_workers=max_workers)
        print(f"Successfully generated titles and previews")
        
        # Update chapters with summaries, titles, and previews
        with db_session() as db:
            summarized_count = 0
            for idx, chapter_data in enumerate(chapters_data):
                chapter = db.query(Chapter).filter(Chapter.id == chapter_data['id']).first()
                if chapter and idx < len(summaries) and summaries[idx]:
                    chapter.summary = summaries[idx]
                    if idx < len(titles_and_previews):
                        takeaway_title, preview = titles_and_previews[idx]
                        if takeaway_title:
                            chapter.title = takeaway_title
                        if preview:
                            chapter.preview = preview
                    summarized_count += 1
            db.commit()
        
        return {
            "message": f"Successfully generated {summarized_count} summaries",
            "summarized": summarized_count,
            "total": len(chapters_data)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in summarize_all_chapters: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chapters"""
    with db_session() as db:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file if it exists
        if os.path.exists(document.file_path):
            try:
                os.remove(document.file_path)
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        # Delete document (cascade will delete chapters)
        db.delete(document)
        db.commit()
    
    return {"message": "Document deleted successfully"}


@app.post("/api/documents/delete-batch")
async def delete_documents_batch(request: Dict = Body(...)):
    """Delete multiple documents and all their chapters"""
    import json
    print(f"[DEBUG] Received delete-batch request: {json.dumps(request, indent=2)}")
    print(f"[DEBUG] Request type: {type(request)}")
    print(f"[DEBUG] Request keys: {request.keys() if isinstance(request, dict) else 'Not a dict'}")
    
    document_ids = request.get("document_ids", [])
    print(f"[DEBUG] Extracted document_ids: {document_ids}, count: {len(document_ids)}")
    
    if not document_ids:
        print("[DEBUG] ERROR: No document IDs provided")
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    with db_session() as db:
        deleted_count = 0
        errors = []
        deleted_titles = []
        
        total_docs = db.query(Document).count()
        print(f"[DEBUG] Total documents in database: {total_docs}")
        
        for idx, document_id in enumerate(document_ids):
            try:
                print(f"[DEBUG] Processing document {idx+1}/{len(document_ids)}: {document_id}")
                document = db.query(Document).filter(Document.id == document_id).first()
                
                if not document:
                    error_msg = f"Document {document_id} not found in database"
                    print(f"[DEBUG] {error_msg}")
                    errors.append(error_msg)
                    continue
                
                print(f"[DEBUG] Found document: {document.title}, file_path: {document.file_path}")
                
                # Delete file if it exists
                if os.path.exists(document.file_path):
                    try:
                        os.remove(document.file_path)
                        print(f"[DEBUG] Deleted file: {document.file_path}")
                    except Exception as e:
                        error_msg = f"Could not delete file for {document.title}: {str(e)}"
                        print(f"[DEBUG] ERROR: {error_msg}")
                        errors.append(error_msg)
                elif document.file_path:
                    print(f"[DEBUG] Warning: File not found at {document.file_path} for document {document.title}")
                
                # Delete document (cascade will delete chapters)
                db.delete(document)
                deleted_count += 1
                deleted_titles.append(document.title)
                print(f"[DEBUG] Marked document {document_id} for deletion")
            except Exception as e:
                error_msg = f"Error deleting document {document_id}: {str(e)}"
                print(f"[DEBUG] EXCEPTION: {error_msg}")
                print(f"[DEBUG] Exception type: {type(e).__name__}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                errors.append(error_msg)
        
        print(f"[DEBUG] Attempting to commit {deleted_count} deletions...")
        try:
            db.commit()
            print(f"[DEBUG] Successfully committed {deleted_count} deletions")
        except Exception as e:
            print(f"[DEBUG] ERROR committing: {str(e)}")
            import traceback
            print(f"[DEBUG] Commit traceback: {traceback.format_exc()}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Error committing deletions: {str(e)}")
    
    result = {
        "message": f"Deleted {deleted_count} document(s)",
        "deleted_count": deleted_count,
        "deleted_titles": deleted_titles,
        "errors": errors if errors else None
    }
    print(f"[DEBUG] Returning result: {json.dumps(result, indent=2, default=str)}")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
