import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, Connection } from '../api/client';
import './Connections.css';

function Connections() {
  const [connections, setConnections] = useState<Connection[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadConnections();
  }, []);

  const loadConnections = async () => {
    try {
      setLoading(true);
      const conns = await api.getConnections();
      setConnections(conns);
    } catch (error: any) {
      console.error('Error loading connections:', error);
      const errorMsg = error.response?.data?.detail || 'Failed to load connections';
      if (errorMsg.includes('API key')) {
        alert('API key not configured. Connections feature requires OpenAI API key.');
      } else {
        alert(errorMsg);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleChapterClick = (chapterId: string) => {
    navigate(`/reader/${chapterId}`);
  };

  if (loading) {
    return <div className="loading">Finding connections...</div>;
  }

  return (
    <div className="connections">
      <div className="connections-header">
        <h2>Chapter Connections</h2>
        <button className="btn btn-secondary" onClick={loadConnections}>
          Refresh
        </button>
      </div>

      {connections.length === 0 ? (
        <div className="empty-state">
          <p>No connections found. Upload more documents to discover relationships between chapters.</p>
        </div>
      ) : (
        <div className="connections-list">
          {connections.map((conn, idx) => (
            <div key={idx} className="connection-card">
              <div className="connection-header">
                <span className="similarity-badge">
                  {(conn.similarity * 100).toFixed(0)}% similar
                </span>
              </div>
              <div className="connection-chapters">
                <div
                  className="connection-chapter"
                  onClick={() => handleChapterClick(conn.chapter1.id)}
                >
                  <h4>{conn.chapter1.title}</h4>
                  <p className="chapter-doc">{conn.chapter1.document_title}</p>
                </div>
                <div className="connection-arrow">↔</div>
                <div
                  className="connection-chapter"
                  onClick={() => handleChapterClick(conn.chapter2.id)}
                >
                  <h4>{conn.chapter2.title}</h4>
                  <p className="chapter-doc">{conn.chapter2.document_title}</p>
                </div>
              </div>
              <div className="connection-reason">
                <strong>Connection:</strong> {conn.reason}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Connections;

