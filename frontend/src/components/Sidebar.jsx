import {
  Activity,
  BarChart3,
  ClipboardPlus,
  LayoutDashboard,
  BrainCircuit,
  Settings,
  Dices,
} from "lucide-react";

function Sidebar({ activePage, setActivePage }) {
  const items = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "Assessment", icon: ClipboardPlus },
    { name: "Prediction", icon: BrainCircuit },
    { name: "Synthetic Lab", icon: Dices },
    { name: "Analytics", icon: BarChart3 },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">
          <Activity size={22} />
        </div>

        <div>
          <h2>DiaSense</h2>
          <span>Risk Intelligence</span>
        </div>
      </div>

      <nav>
        {items.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.name}
              className={
                activePage === item.name ? "nav-item active" : "nav-item"
              }
              onClick={() => setActivePage(item.name)}
            >
              <Icon size={19} />
              <span>{item.name}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <button className="nav-item">
          <Settings size={19} />
          <span>Settings</span>
        </button>

        <div className="sidebar-status">
          <span className="status-dot"></span>
          Model API Online
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
