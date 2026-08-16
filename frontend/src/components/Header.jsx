import { Bell, Moon, Search, Sun } from "lucide-react";

function Header({ dark, setDark }) {
  return (
    <header className="header">
      <div className="search">
        <Search size={18} />
        <input placeholder="Search assessments..." />
      </div>

      <div className="header-actions">
        <button className="header-button">
          <Bell size={19} />
          <span className="notification"></span>
        </button>

        <button className="header-button" onClick={() => setDark(!dark)}>
          {dark ? <Sun size={19} /> : <Moon size={19} />}
        </button>

        <div className="user">
          <div className="user-avatar">AM</div>
          <div>
            <strong>Ashutosh</strong>
            <span>Administrator</span>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
