import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Library from './components/Library';
import Reader from './components/Reader';
import Connections from './components/Connections';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/reader/:chapterId" element={<Reader />} />
          <Route path="/connections" element={<Connections />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
