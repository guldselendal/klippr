import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <h1>Klippr</h1>
        </Link>
        <div className="navbar-links">
          <Link 
            to="/" 
            className={location.pathname === '/' ? 'active' : ''}
          >
            Library
          </Link>
          <Link 
            to="/connections" 
            className={location.pathname === '/connections' ? 'active' : ''}
          >
            Connections
          </Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;

