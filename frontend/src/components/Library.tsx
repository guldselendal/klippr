import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, Document, Chapter, checkBackendHealth } from '../api/client';
import { AxiosError } from 'axios';
import './Library.css';

function Library() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
  const [showUpload, setShowUpload] = useState(false);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [docChapters, setDocChapters] = useState<Record<string, Chapter[]>>({});
  const [loadingChapters, setLoadingChapters] = useState<Set<string>>(new Set());
  const [summarizingChapters, setSummarizingChapters] = useState<Set<string>>(new Set());
  const [showSummary, setShowSummary] = useState<{ chapterId: string; summary: string } | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const navigate = useNavigate();

  useEffect(() => {
    const initialize = async () => {
      const isHealthy = await checkBackendHealth();
      if (!isHealthy) {
        alert('Backend server is not available. Please start the backend server on http://localhost:8000');
        setLoading(false);
        return;
      }
      loadDocuments();
    };
    initialize();
  }, []);

  const getErrorMessage = (error: unknown): string => {
    const axiosError = error as AxiosError & { userMessage?: string };
    if (axiosError.userMessage) return axiosError.userMessage;
    
    if (axiosError.response?.status === 413) return 'File is too large. Please try a smaller file.';
    if (axiosError.response?.status === 409) {
      return (axiosError.response.data as { detail?: string })?.detail || 'This file already exists in your library.';
    }
    if (axiosError.response?.status === 400) {
      return (axiosError.response.data as { detail?: string })?.detail || 'Invalid file. Only EPUB and PDF files are supported.';
    }
    if (axiosError.response?.status === 503) {
      return (axiosError.response.data as { detail?: string })?.detail || 'Service unavailable. Please check if Ollama is running.';
    }
    if (axiosError.code === 'ECONNABORTED' || axiosError.message?.includes('timeout')) {
      return 'Upload timed out. The file may be too large or processing is taking longer than expected. Please try a smaller file or check if Ollama is running.';
    }
    return axiosError.userMessage || 'An error occurred';
  };

  const loadDocuments = async () => {
    try {
      setLoading(true);
      // Always fetch fresh data (cache is invalidated after upload/delete operations)
      const docs = await api.getDocuments(false); // useCache = false to ensure fresh data
      setDocuments(docs);
    } catch (error) {
      console.error('Error loading documents:', error);
      alert(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files?.length) return;

    setUploading(true);
    setUploadStatus('Uploading file...');
    
    const statusMessages = [
      'Parsing document...',
      'Extracting chapters...',
      'Generating summaries...',
      'Generating titles and previews...',
      'Saving to database...',
    ];

    let messageIndex = 0;
    const statusInterval = setInterval(() => {
      if (messageIndex < statusMessages.length) {
        setUploadStatus(statusMessages[messageIndex++]);
      }
    }, 3000);

    try {
      if (files.length === 1) {
        setUploadStatus(`Processing ${files[0].name}...`);
        await api.uploadFile(files[0]);
      } else {
        setUploadStatus(`Processing ${files.length} files...`);
        await api.uploadFiles(Array.from(files));
      }
      
      clearInterval(statusInterval);
      setUploadStatus('✓ Complete!');
      // Force reload documents without cache to show newly uploaded file
      await loadDocuments();
      setShowUpload(false);
      // Don't show alert - the document list refresh is sufficient feedback
    } catch (error) {
      console.error('Error uploading files:', error);
      const errorMsg = getErrorMessage(error);
      setUploadStatus(`✗ Error: ${errorMsg}`);
      alert(errorMsg);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(''), 5000);
      if (event.target) event.target.value = '';
    }
  };

  const handleDelete = async (documentId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    try {
      await api.deleteDocument(documentId);
      await loadDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
      alert('Failed to delete document');
    }
  };

  const handleBatchDelete = async () => {
    if (selectedDocs.size === 0) return;
    if (!confirm(`Are you sure you want to delete ${selectedDocs.size} document(s)?`)) return;
    try {
      await api.deleteDocuments(Array.from(selectedDocs));
      setSelectedDocs(new Set());
      await loadDocuments();
    } catch (error) {
      console.error('Error deleting documents:', error);
      alert('Failed to delete documents');
    }
  };

  const toggleSelection = (docId: string) => {
    setSelectedDocs(prev => {
      const next = new Set(prev);
      next.has(docId) ? next.delete(docId) : next.add(docId);
      return next;
    });
  };

  const toggleDocumentExpansion = async (documentId: string) => {
    if (expandedDoc === documentId) {
      setExpandedDoc(null);
      return;
    }

    setExpandedDoc(documentId);

    if (!docChapters[documentId]) {
      try {
        setLoadingChapters(prev => new Set(prev).add(documentId));
        const chapters = await api.getDocumentChapters(documentId, false);
        setDocChapters(prev => ({ ...prev, [documentId]: chapters }));
      } catch (error) {
        console.error('Error loading chapters:', error);
        alert('Failed to load document chapters');
      } finally {
        setLoadingChapters(prev => {
          const next = new Set(prev);
          next.delete(documentId);
          return next;
        });
      }
    }
  };

  const handleChapterClick = (chapterId: string) => {
    navigate(`/reader/${chapterId}`);
  };

  const handleSummaryClick = (e: React.MouseEvent, chapterId: string, summary: string) => {
    e.stopPropagation();
    setShowSummary(prev => prev?.chapterId === chapterId ? null : { chapterId, summary });
  };

  const handleSummarizeClick = async (e: React.MouseEvent, chapterId: string, documentId: string) => {
    e.stopPropagation();
    
    try {
      setSummarizingChapters(prev => new Set(prev).add(chapterId));
      const updatedChapter = await api.summarizeChapter(chapterId);
      
      setDocChapters(prev => {
        const chapters = prev[documentId] || [];
        return {
          ...prev,
          [documentId]: chapters.map(ch => ch.id === chapterId ? updatedChapter : ch)
        };
      });
      
      setShowSummary({ chapterId, summary: updatedChapter.summary || '' });
    } catch (error) {
      console.error('Error generating summary:', error);
      const errorMsg = (error as AxiosError<{ detail?: string }>).response?.data?.detail || 
                      (error as Error).message || 
                      'Failed to generate summary. Please try again.';
      alert(errorMsg);
    } finally {
      setSummarizingChapters(prev => {
        const next = new Set(prev);
        next.delete(chapterId);
        return next;
      });
    }
  };

  if (loading) {
    return <div className="loading">Loading library...</div>;
  }

  return (
    <div className="library">
      <div className="library-header">
        <h2>Document Library</h2>
        <div className="library-actions">
          {selectedDocs.size > 0 && (
            <button className="btn btn-danger" onClick={handleBatchDelete}>
              Delete Selected ({selectedDocs.size})
            </button>
          )}
          <button className="btn btn-primary" onClick={() => setShowUpload(!showUpload)}>
            {showUpload ? 'Cancel' : 'Upload Document'}
          </button>
        </div>
      </div>

      {showUpload && (
        <div className="upload-section">
          <div className="upload-box">
            <input
              type="file"
              id="file-upload"
              multiple
              accept=".epub,.pdf"
              onChange={handleFileUpload}
              disabled={uploading}
              style={{ display: 'none' }}
            />
            <label htmlFor="file-upload" className="upload-label">
              {uploading ? (
                <span>Uploading...</span>
              ) : (
                <>
                  <span>📄 Click to upload EPUB or PDF files</span>
                  <span className="upload-hint">You can select multiple files</span>
                </>
              )}
            </label>
          </div>
          {uploadStatus && (
            <div className="upload-status">
              <div className="upload-status-indicator">
                <div className="upload-status-spinner"></div>
                <span>{uploadStatus}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {documents.length === 0 ? (
        <div className="empty-state">
          <p>No documents yet. Upload your first EPUB or PDF to get started!</p>
        </div>
      ) : (
        <div className="documents-list">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className={`document-card ${selectedDocs.has(doc.id) ? 'selected' : ''} ${expandedDoc === doc.id ? 'expanded' : ''}`}
            >
              <div 
                className="document-card-main"
                onClick={() => !selectedDocs.has(doc.id) && toggleDocumentExpansion(doc.id)}
              >
                <div className="document-card-header">
                  <input
                    type="checkbox"
                    checked={selectedDocs.has(doc.id)}
                    onChange={(e) => {
                      e.stopPropagation();
                      toggleSelection(doc.id);
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div className="document-card-info">
                    <h3>{doc.title}</h3>
                    <div className="document-meta-row">
                      <span className="document-type">{doc.file_type.toUpperCase()}</span>
                      <span className="document-meta">
                        {doc.chapter_count} chapter{doc.chapter_count !== 1 ? 's' : ''}
                      </span>
                      {doc.uploaded_at && (
                        <span className="document-date">
                          {new Date(doc.uploaded_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="document-card-actions">
                    <button
                      className="btn btn-small btn-danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(doc.id);
                      }}
                    >
                      Delete
                    </button>
                    <span className="expand-icon">
                      {expandedDoc === doc.id ? '▼' : '▶'}
                    </span>
                  </div>
                </div>
              </div>
              
              {expandedDoc === doc.id && (
                <div className="document-chapters">
                  {loadingChapters.has(doc.id) ? (
                    <div className="chapters-loading">Loading chapters...</div>
                  ) : docChapters[doc.id]?.length ? (
                    <div className="chapters-list">
                      {docChapters[doc.id].map((chapter) => {
                        const hasSummary = !!chapter.summary?.trim();
                        const hasPreview = !!chapter.preview?.trim();
                        const isSummarizing = summarizingChapters.has(chapter.id);
                        
                        return (
                          <div
                            key={chapter.id}
                            className="chapter-item"
                            onClick={() => handleChapterClick(chapter.id)}
                          >
                            <div className="chapter-item-number">
                              {chapter.chapter_number || '•'}
                            </div>
                            <div className="chapter-item-content">
                              <div className="chapter-item-title" title={chapter.title}>
                                {chapter.title}
                              </div>
                              {hasPreview && (
                                <div className="chapter-item-preview">{chapter.preview}</div>
                              )}
                            </div>
                            {hasSummary ? (
                              <button
                                className="btn btn-small btn-secondary chapter-summary-btn"
                                onClick={(e) => handleSummaryClick(e, chapter.id, chapter.summary || '')}
                                title="View detailed summary"
                              >
                                📄 Summary
                              </button>
                            ) : (
                              <button
                                className="btn btn-small btn-primary chapter-summary-btn"
                                onClick={(e) => handleSummarizeClick(e, chapter.id, doc.id)}
                                disabled={isSummarizing}
                                title="Generate summary"
                              >
                                {isSummarizing ? '⏳ Summarizing...' : '✨ Summarize'}
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="chapters-empty">No chapters found</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {showSummary && (
        <div className="summary-modal-overlay" onClick={() => setShowSummary(null)}>
          <div className="summary-modal" onClick={(e) => e.stopPropagation()}>
            <div className="summary-modal-header">
              <h3>Chapter Summary</h3>
              <button className="btn btn-small btn-secondary" onClick={() => setShowSummary(null)}>
                ✕
              </button>
            </div>
            <div className="summary-modal-content">
              {showSummary.summary || 'No summary available'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Library;
