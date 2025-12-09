import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api, Chapter } from '../api/client';
import './Reader.css';

function Reader() {
  const { chapterId } = useParams<{ chapterId: string }>();
  const navigate = useNavigate();
  const [currentChapter, setCurrentChapter] = useState<Chapter | null>(null);
  const [allChapters, setAllChapters] = useState<Chapter[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(-1);
  const [loading, setLoading] = useState(true);
  const [showChaptersList, setShowChaptersList] = useState(false);

  useEffect(() => {
    if (chapterId) loadChapter(chapterId);
  }, [chapterId]);

  const loadChapter = async (id: string) => {
    try {
      setLoading(true);
      const chapters = await api.getAllChapters(false);
      setAllChapters(chapters);
      
      const index = chapters.findIndex(ch => ch.id === id);
      if (index === -1) {
        alert('Chapter not found');
        navigate('/');
        return;
      }
      
      setCurrentIndex(index);
      const fullChapter = await api.getChapter(id);
      setCurrentChapter(fullChapter);
    } catch (error) {
      console.error('Error loading chapter:', error);
      alert('Failed to load chapter');
      navigate('/');
    } finally {
      setLoading(false);
    }
  };

  const goToChapter = (index: number) => {
    if (index >= 0 && index < allChapters.length) {
      navigate(`/reader/${allChapters[index].id}`);
    }
  };

  if (loading) {
    return <div className="loading">Loading chapter...</div>;
  }

  if (!currentChapter) {
    return <div className="error">Chapter not found</div>;
  }

  return (
    <div className="reader">
      <div className="reader-header">
        <button className="btn btn-secondary" onClick={() => navigate('/')}>
          ← Back to Library
        </button>
        <div className="reader-nav">
          <button
            className="btn btn-secondary"
            onClick={() => goToChapter(currentIndex - 1)}
            disabled={currentIndex === 0}
          >
            ← Previous
          </button>
          <span className="chapter-counter">
            {currentIndex + 1} / {allChapters.length}
          </span>
          <button
            className="btn btn-secondary"
            onClick={() => goToChapter(currentIndex + 1)}
            disabled={currentIndex === allChapters.length - 1}
          >
            Next →
          </button>
        </div>
        <button
          className="btn btn-secondary"
          onClick={() => setShowChaptersList(!showChaptersList)}
        >
          {showChaptersList ? 'Hide' : 'Show'} Chapters
        </button>
      </div>

      {showChaptersList && (
        <div className="chapters-sidebar">
          <h3>All Chapters</h3>
          <div className="chapters-list">
            {allChapters.map((chapter, idx) => (
              <div
                key={chapter.id}
                className={`chapter-item ${idx === currentIndex ? 'active' : ''}`}
                onClick={() => goToChapter(idx)}
              >
                <div className="chapter-item-title">{chapter.title}</div>
                {chapter.summary && (
                  <div className="chapter-item-summary">{chapter.summary}</div>
                )}
                <div className="chapter-item-doc">{chapter.document_title}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="reader-content">
        <div className="chapter-header">
          <h2>{currentChapter.title}</h2>
          <p className="chapter-source">From: {currentChapter.document_title}</p>
        </div>
        <div className="chapter-content">
          {currentChapter.content ? (
            currentChapter.content.split('\n').map((paragraph, idx) => (
              <p key={idx}>{paragraph}</p>
            ))
          ) : (
            <p>Loading content...</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default Reader;
