import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  Brain,
  ClipboardCheck,
  Users,
} from "lucide-react";

function Dashboard({ setActivePage }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [assessments, setAssessments] = useState([]);

  useEffect(() => {
    Promise.all([
      fetch("http://127.0.0.1:8000/model-info").then((res) => res.json()),
      fetch("http://127.0.0.1:8000/analytics").then((res) => res.json()),
    ])
      .then(([modelData, analyticsData]) => {
        setModelInfo(modelData);
        setAnalytics(analyticsData);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []);

  return (
    <div className="dashboard-content">
      <div className="page-heading">
        <div>
          <p className="eyebrow">OVERVIEW</p>

          <h1>Good morning, Ashutosh</h1>

          <p>Monitor diabetes risk assessments and model performance.</p>
        </div>

        <button
          className="primary-button"
          onClick={() => setActivePage("Prediction")}
        >
          <ClipboardCheck size={18} />
          New Assessment
        </button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-top">
            <span>Total Patients</span>

            <div className="stat-icon">
              <Users size={19} />
            </div>
          </div>

          <h2>{analytics ? analytics.total_records : "--"}</h2>

          <div className="stat-change">
            <ArrowUpRight size={15} />
            Dataset records
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-top">
            <span>Diabetes Rate</span>

            <div className="stat-icon">
              <Activity size={19} />
            </div>
          </div>

          <h2>{analytics ? `${analytics.diabetes_rate}%` : "--"}</h2>

          <div className="stat-change">
            <Activity size={15} />
            Dataset prevalence
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-top">
            <span>Model Accuracy</span>

            <div className="stat-icon">
              <Brain size={19} />
            </div>
          </div>

          <h2>{modelInfo ? `${modelInfo.accuracy}%` : "--"}</h2>

          <div className="stat-change">
            <ArrowUpRight size={15} />

            {modelInfo ? modelInfo.model : "Loading model"}
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="panel welcome-panel">
          <div className="panel-header">
            <div>
              <span className="panel-label">AI RISK ENGINE</span>

              <h2>Diabetes Risk Assessment</h2>
            </div>

            <div className="model-status">
              <span></span>
              Model Active
            </div>
          </div>

          <p>
            Analyze patient health indicators using the trained Linear Support
            Vector Machine model.
          </p>

          <button
            className="primary-button"
            onClick={() => setActivePage("Prediction")}
          >
            Start Assessment
            <ArrowUpRight size={17} />
          </button>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <span className="panel-label">MODEL</span>

              <h2>Performance</h2>
            </div>
          </div>

          <div className="performance">
            <div className="performance-ring">
              <strong>{modelInfo ? `${modelInfo.accuracy}%` : "--"}</strong>

              <span>Accuracy</span>
            </div>

            <div className="performance-details">
              <div>
                <span>Algorithm</span>

                <strong>{modelInfo ? modelInfo.model : "--"}</strong>
              </div>

              <div>
                <span>Dataset</span>

                <strong>PIMA Diabetes</strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="panel recent-panel">
        <div className="panel-header">
          <div>
            <span className="panel-label">DATASET</span>

            <h2>Dataset Overview</h2>
          </div>

          <button
            className="text-button"
            onClick={() => setActivePage("Analytics")}
          >
            View analytics
          </button>
        </div>

        <div className="assessment-table">
          <div className="table-head">
            <span>Metric</span>
            <span>Value</span>
            <span>Status</span>
            <span>Source</span>
            <span>Type</span>
          </div>

          <div className="table-row">
            <strong>Total Records</strong>

            <span>{analytics ? analytics.total_records : "--"}</span>

            <b className="result-low">Available</b>

            <span>PIMA Dataset</span>

            <span>Clinical</span>
          </div>

          <div className="table-row">
            <strong>Diabetic Cases</strong>

            <span>{analytics ? analytics.diabetic : "--"}</span>

            <b className="result-high">Positive</b>

            <span>PIMA Dataset</span>

            <span>Outcome</span>
          </div>

          <div className="table-row">
            <strong>Non-Diabetic Cases</strong>

            <span>{analytics ? analytics.non_diabetic : "--"}</span>

            <b className="result-low">Negative</b>

            <span>PIMA Dataset</span>

            <span>Outcome</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
