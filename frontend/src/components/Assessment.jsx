import { useState } from "react";
import {
  Search,
  Filter,
  Eye,
  CalendarDays,
  UserRound,
  Activity,
  X,
} from "lucide-react";

function Assessment() {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");
  const [selectedAssessment, setSelectedAssessment] = useState(null);

  const assessments = [
    {
      id: "DS-001",
      patient: "Patient #001",
      age: 33,
      glucose: 103,
      bmi: 43.3,
      bloodPressure: 30,
      pregnancies: 1,
      insulin: 83,
      result: "Low Risk",
      date: "Today, 10:42 AM",
    },
    {
      id: "DS-002",
      patient: "Patient #002",
      age: 48,
      glucose: 142,
      bmi: 31.2,
      bloodPressure: 72,
      pregnancies: 4,
      insulin: 180,
      result: "High Risk",
      date: "Today, 09:18 AM",
    },
    {
      id: "DS-003",
      patient: "Patient #003",
      age: 41,
      glucose: 118,
      bmi: 28.4,
      bloodPressure: 68,
      pregnancies: 2,
      insulin: 94,
      result: "Low Risk",
      date: "Yesterday, 04:32 PM",
    },
    {
      id: "DS-004",
      patient: "Patient #004",
      age: 56,
      glucose: 167,
      bmi: 35.8,
      bloodPressure: 82,
      pregnancies: 6,
      insulin: 250,
      result: "High Risk",
      date: "Yesterday, 01:15 PM",
    },
    {
      id: "DS-005",
      patient: "Patient #005",
      age: 29,
      glucose: 96,
      bmi: 24.7,
      bloodPressure: 64,
      pregnancies: 1,
      insulin: 70,
      result: "Low Risk",
      date: "Aug 14, 11:06 AM",
    },
  ];

  const filteredAssessments = assessments.filter((item) => {
    const matchesSearch =
      item.patient.toLowerCase().includes(search.toLowerCase()) ||
      item.id.toLowerCase().includes(search.toLowerCase()) ||
      item.result.toLowerCase().includes(search.toLowerCase());

    const matchesFilter = filter === "All" || item.result === filter;

    return matchesSearch && matchesFilter;
  });

  return (
    <div className="page-content">
      <div className="page-heading">
        <div>
          <p className="eyebrow">ASSESSMENTS</p>
          <h1>Patient Assessments</h1>
          <p>
            Review previous diabetes risk assessments and prediction results.
          </p>
        </div>
      </div>

      <div className="assessment-summary">
        <div className="summary-card">
          <div className="summary-icon">
            <Activity size={21} />
          </div>

          <div>
            <span>Total assessments</span>
            <strong>{assessments.length}</strong>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">
            <UserRound size={21} />
          </div>

          <div>
            <span>High risk</span>
            <strong>
              {assessments.filter((item) => item.result === "High Risk").length}
            </strong>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">
            <CalendarDays size={21} />
          </div>

          <div>
            <span>Latest assessment</span>
            <strong>Today</strong>
          </div>
        </div>
      </div>

      <div className="assessment-toolbar">
        <div className="assessment-search">
          <Search size={19} />

          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search patient, assessment ID or result..."
          />
        </div>

        <div className="assessment-filter">
          <Filter size={17} />

          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="All">All results</option>
            <option value="High Risk">High Risk</option>
            <option value="Low Risk">Low Risk</option>
          </select>
        </div>
      </div>

      <div className="assessment-list">
        <div className="assessment-list-header">
          <span>Assessment</span>
          <span>Patient</span>
          <span>Age</span>
          <span>Glucose</span>
          <span>BMI</span>
          <span>Result</span>
          <span>Date</span>
          <span></span>
        </div>

        {filteredAssessments.map((item) => (
          <div className="assessment-list-row" key={item.id}>
            <strong>{item.id}</strong>

            <span>{item.patient}</span>

            <span>{item.age}</span>

            <span>{item.glucose} mg/dL</span>

            <span>{item.bmi}</span>

            <b
              className={
                item.result === "High Risk" ? "result-high" : "result-low"
              }
            >
              {item.result}
            </b>

            <span>{item.date}</span>

            <button
              className="view-assessment"
              onClick={() => setSelectedAssessment(item)}
              title="View assessment"
            >
              <Eye size={17} />
            </button>
          </div>
        ))}

        {filteredAssessments.length === 0 && (
          <div className="empty-assessments">
            <Search size={30} />
            <h3>No assessments found</h3>
            <p>Try another patient name, ID, or result.</p>
          </div>
        )}
      </div>

      {selectedAssessment && (
        <div
          className="assessment-modal-overlay"
          onClick={() => setSelectedAssessment(null)}
        >
          <div
            className="assessment-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="assessment-modal-header">
              <div>
                <span className="eyebrow">ASSESSMENT DETAILS</span>
                <h2>{selectedAssessment.id}</h2>
              </div>

              <button
                className="modal-close"
                onClick={() => setSelectedAssessment(null)}
              >
                <X size={20} />
              </button>
            </div>

            <div
              className={
                selectedAssessment.result === "High Risk"
                  ? "modal-result high"
                  : "modal-result low"
              }
            >
              <span>Prediction result</span>
              <strong>{selectedAssessment.result}</strong>
            </div>

            <div className="modal-grid">
              <div>
                <span>Patient</span>
                <strong>{selectedAssessment.patient}</strong>
              </div>

              <div>
                <span>Age</span>
                <strong>{selectedAssessment.age} years</strong>
              </div>

              <div>
                <span>Glucose</span>
                <strong>{selectedAssessment.glucose} mg/dL</strong>
              </div>

              <div>
                <span>BMI</span>
                <strong>{selectedAssessment.bmi}</strong>
              </div>

              <div>
                <span>Blood Pressure</span>
                <strong>{selectedAssessment.bloodPressure} mmHg</strong>
              </div>

              <div>
                <span>Pregnancies</span>
                <strong>{selectedAssessment.pregnancies}</strong>
              </div>

              <div>
                <span>Insulin</span>
                <strong>{selectedAssessment.insulin}</strong>
              </div>

              <div>
                <span>Assessment time</span>
                <strong>{selectedAssessment.date}</strong>
              </div>
            </div>

            <div className="modal-footer">
              <span>Model</span>
              <strong>Linear SVM</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Assessment;
