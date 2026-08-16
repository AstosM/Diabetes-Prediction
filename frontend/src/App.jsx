import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Dashboard from "./components/Dashboard";
import Assessment from "./components/Assessment";
import Analytics from "./components/Analytics";
import Prediction from "./components/Prediction";
import SyntheticLab from "./components/SyntheticLab";
import "./App.css";

function App() {
  const [dark, setDark] = useState(true);
  const [activePage, setActivePage] = useState("Dashboard");

  return (
    <div className={dark ? "app dark" : "app light"}>
      <Sidebar activePage={activePage} setActivePage={setActivePage} />

      <div className="main">
        <Header dark={dark} setDark={setDark} />

        {activePage === "Dashboard" && (
          <Dashboard setActivePage={setActivePage} />
        )}

        {activePage === "Assessment" && <Assessment />}

        {activePage === "Prediction" && <Prediction />}

        {activePage === "Synthetic Lab" && <SyntheticLab />}

        {activePage === "Analytics" && <Analytics />}
      </div>
    </div>
  );
}

export default App;
