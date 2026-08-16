import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  Database,
  Gauge,
  Users,
} from "lucide-react";

function Analytics() {
  const [analytics, setAnalytics] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch("https://diabetes-prediction-m6o4.onrender.com/analytics").then((res) => res.json()),
      fetch("https://diabetes-prediction-m6o4.onrender.com/model-info").then((res) => res.json()),
    ])
      .then(([analyticsData, modelData]) => {
        setAnalytics(analyticsData);
        setModelInfo(modelData);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []);

  const total = analytics?.total_records || 0;
  const diabetic = analytics?.diabetic || 0;
  const nonDiabetic = analytics?.non_diabetic || 0;

  const diabeticWidth = total ? `${(diabetic / total) * 100}%` : "0%";

  const nonDiabeticWidth = total ? `${(nonDiabetic / total) * 100}%` : "0%";

  return (
    <div className="analytics-page">
      <div className="analytics-heading">
        <div>
          <span className="section-kicker">MODEL INTELLIGENCE</span>
          <h1>Analytics & Insights</h1>
          <p>
            Explore dataset composition and machine-learning model performance.
          </p>
        </div>

        <div className="dataset-chip">
          <Database size={17} />
          PIMA Diabetes Dataset
        </div>
      </div>

      <div className="analytics-metrics">
        <div className="analytics-metric">
          <div className="metric-icon">
            <Database size={20} />
          </div>
          <span>Dataset Size</span>
          <strong>{analytics ? total : "--"}</strong>
          <small>Clinical records</small>
        </div>

        <div className="analytics-metric">
          <div className="metric-icon">
            <Users size={20} />
          </div>
          <span>Diabetic Cases</span>
          <strong>{analytics ? diabetic : "--"}</strong>
          <small>Positive outcomes</small>
        </div>

        <div className="analytics-metric">
          <div className="metric-icon">
            <Activity size={20} />
          </div>
          <span>Prevalence</span>
          <strong>{analytics ? `${analytics.diabetes_rate}%` : "--"}</strong>
          <small>Dataset diabetes rate</small>
        </div>

        <div className="analytics-metric">
          <div className="metric-icon">
            <Gauge size={20} />
          </div>
          <span>Model Accuracy</span>
          <strong>{modelInfo ? `${modelInfo.accuracy}%` : "--"}</strong>
          <small>Holdout test accuracy</small>
        </div>
      </div>

      <div className="analytics-main-grid">
        <div className="analytics-panel distribution-panel">
          <div className="analytics-panel-heading">
            <div>
              <span>OUTCOME DISTRIBUTION</span>
              <h2>Dataset composition</h2>
            </div>

            <BarChart3 size={21} />
          </div>

          <div className="distribution-total">
            <strong>{analytics ? total : "--"}</strong>
            <span>Total records</span>
          </div>

          <div className="distribution-bars">
            <div className="distribution-item">
              <div className="distribution-label">
                <span>
                  <i className="legend positive"></i>
                  Diabetic
                </span>

                <strong>{diabetic}</strong>
              </div>

              <div className="bar-track">
                <div
                  className="bar-fill diabetic-bar"
                  style={{ width: diabeticWidth }}
                ></div>
              </div>
            </div>

            <div className="distribution-item">
              <div className="distribution-label">
                <span>
                  <i className="legend negative"></i>
                  Non-Diabetic
                </span>

                <strong>{nonDiabetic}</strong>
              </div>

              <div className="bar-track">
                <div
                  className="bar-fill healthy-bar"
                  style={{ width: nonDiabeticWidth }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        <div className="analytics-panel model-panel">
          <div className="analytics-panel-heading">
            <div>
              <span>MODEL PROFILE</span>
              <h2>Prediction engine</h2>
            </div>

            <Brain size={21} />
          </div>

          <div className="model-score">
            <div className="score-circle">
              <strong>{modelInfo ? `${modelInfo.accuracy}%` : "--"}</strong>
              <span>accuracy</span>
            </div>
          </div>

          <div className="model-properties">
            <div>
              <span>Algorithm</span>
              <strong>{modelInfo ? modelInfo.model : "--"}</strong>
            </div>

            <div>
              <span>Kernel</span>
              <strong>Linear</strong>
            </div>

            <div>
              <span>Dataset</span>
              <strong>PIMA</strong>
            </div>

            <div>
              <span>Task</span>
              <strong>Binary Classification</strong>
            </div>
          </div>
        </div>
      </div>

      <div className="analytics-panel methodology-panel">
        <div className="analytics-panel-heading">
          <div>
            <span>ML PIPELINE</span>
            <h2>How the prediction engine works</h2>
          </div>
        </div>

        <div className="pipeline">
          <div className="pipeline-step">
            <div className="pipeline-number">01</div>
            <h3>Patient Input</h3>
            <p>
              Eight clinical indicators are collected through the assessment
              interface.
            </p>
          </div>

          <div className="pipeline-line"></div>

          <div className="pipeline-step">
            <div className="pipeline-number">02</div>
            <h3>Standardization</h3>
            <p>
              Patient features are transformed using the trained StandardScaler.
            </p>
          </div>

          <div className="pipeline-line"></div>

          <div className="pipeline-step">
            <div className="pipeline-number">03</div>
            <h3>SVM Inference</h3>
            <p>The Linear SVM evaluates the standardized feature vector.</p>
          </div>

          <div className="pipeline-line"></div>

          <div className="pipeline-step">
            <div className="pipeline-number">04</div>
            <h3>Risk Result</h3>
            <p>
              The API returns the predicted outcome and model information to the
              React interface.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Analytics;
