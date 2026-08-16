import { useState } from "react";
import {
  Activity,
  Dices,
  UserRound,
  HeartPulse,
  ShieldAlert,
  ShieldCheck,
  RefreshCw,
} from "lucide-react";

function SyntheticLab() {
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(false);

  const generatePatient = (riskLevel) => {
    setLoading(true);

    setTimeout(() => {
      const highRisk = riskLevel === "high";

      const generated = {
        age: highRisk
          ? Math.floor(Math.random() * 20) + 46
          : Math.floor(Math.random() * 25) + 20,

        gender: ["Female", "Male", "Other"][Math.floor(Math.random() * 3)],

        pregnancies: highRisk
          ? Math.floor(Math.random() * 6) + 4
          : Math.floor(Math.random() * 3),

        glucose: highRisk
          ? Math.floor(Math.random() * 40) + 140
          : Math.floor(Math.random() * 35) + 75,

        blood_pressure: highRisk
          ? Math.floor(Math.random() * 25) + 75
          : Math.floor(Math.random() * 20) + 60,

        skin_thickness: Math.floor(Math.random() * 40) + 10,

        insulin: highRisk
          ? Math.floor(Math.random() * 120) + 150
          : Math.floor(Math.random() * 120) + 50,

        bmi: highRisk
          ? Number((Math.random() * 10 + 30).toFixed(1))
          : Number((Math.random() * 7 + 19).toFixed(1)),

        diabetes_pedigree: highRisk
          ? Number((Math.random() * 0.8 + 0.7).toFixed(3))
          : Number((Math.random() * 0.5 + 0.1).toFixed(3)),

        physical_activity: highRisk
          ? "Sedentary"
          : ["Moderate", "Active"][Math.floor(Math.random() * 2)],

        smoking: highRisk ? Math.random() > 0.5 : false,

        alcohol: highRisk
          ? ["Occasional", "Regular"][Math.floor(Math.random() * 2)]
          : "Never",

        family_history: highRisk ? Math.random() > 0.25 : Math.random() > 0.75,
      };

      setPatient(generated);
      setLoading(false);
    }, 500);
  };

  const runPrediction = async () => {
    if (!patient) return;

    setLoading(true);

    try {
      const response = await fetch("https://diabetes-prediction-m6o4.onrender.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pregnancies: patient.pregnancies,
          glucose: patient.glucose,
          blood_pressure: patient.blood_pressure,
          skin_thickness: patient.skin_thickness,
          insulin: patient.insulin,
          bmi: patient.bmi,
          diabetes_pedigree: patient.diabetes_pedigree,
          age: patient.age,
        }),
      });

      const data = await response.json();

      setPatient({
        ...patient,
        prediction: data,
      });
    } catch {
      setPatient({
        ...patient,
        prediction: {
          error: true,
        },
      });
    }

    setLoading(false);
  };

  const riskFactors = patient
    ? [
        patient.glucose > 126 && "Elevated glucose",
        patient.bmi > 30 && "High BMI",
        patient.age > 45 && "Age-related risk",
        patient.pregnancies > 3 && "Multiple pregnancies",
        patient.family_history && "Family history",
        patient.physical_activity === "Sedentary" && "Low physical activity",
      ].filter(Boolean)
    : [];

  return (
    <div className="synthetic-page">
      <div className="page-heading">
        <div>
          <p className="eyebrow">SIMULATION LAB</p>
          <h1>Synthetic Patient Laboratory</h1>
          <p>
            Generate realistic patient profiles and test the diabetes risk
            engine.
          </p>
        </div>

        <div className="prediction-model">
          <Activity size={18} />
          Simulation Mode
        </div>
      </div>

      <div className="synthetic-actions">
        <button
          className="synthetic-action"
          onClick={() => generatePatient("normal")}
        >
          <div>
            <ShieldCheck size={24} />
          </div>

          <span>
            <strong>Generate Low-Risk Patient</strong>
            <small>Generate a realistic lower-risk profile</small>
          </span>
        </button>

        <button
          className="synthetic-action high"
          onClick={() => generatePatient("high")}
        >
          <div>
            <ShieldAlert size={24} />
          </div>

          <span>
            <strong>Generate High-Risk Patient</strong>
            <small>Simulate multiple diabetes risk factors</small>
          </span>
        </button>

        <button
          className="synthetic-action"
          onClick={() =>
            generatePatient(Math.random() > 0.5 ? "high" : "normal")
          }
        >
          <div>
            <Dices size={24} />
          </div>

          <span>
            <strong>Random Patient</strong>
            <small>Generate a randomized clinical profile</small>
          </span>
        </button>
      </div>

      {!patient && (
        <div className="synthetic-empty">
          <UserRound size={38} />
          <h2>No synthetic patient generated</h2>
          <p>
            Choose a simulation type above to create a patient profile for model
            testing.
          </p>
        </div>
      )}

      {patient && (
        <div className="synthetic-layout">
          <div className="synthetic-profile">
            <div className="synthetic-profile-header">
              <div className="profile-large">
                {patient.gender === "Female" ? "F" : "M"}
              </div>

              <div>
                <span>SYNTHETIC PATIENT</span>
                <h2>Generated Profile</h2>
              </div>

              <button
                className="refresh-button"
                onClick={() =>
                  generatePatient(patient.glucose > 126 ? "high" : "normal")
                }
              >
                <RefreshCw size={18} />
              </button>
            </div>

            <div className="synthetic-section">
              <h3>Demographics</h3>

              <div className="synthetic-grid">
                <div>
                  <span>Age</span>
                  <strong>{patient.age} years</strong>
                </div>

                <div>
                  <span>Gender</span>
                  <strong>{patient.gender}</strong>
                </div>

                <div>
                  <span>Pregnancies</span>
                  <strong>{patient.pregnancies}</strong>
                </div>

                <div>
                  <span>Family History</span>
                  <strong>{patient.family_history ? "Yes" : "No"}</strong>
                </div>
              </div>
            </div>

            <div className="synthetic-section">
              <h3>Clinical measurements</h3>

              <div className="synthetic-grid">
                <div>
                  <span>Glucose</span>
                  <strong>{patient.glucose} mg/dL</strong>
                </div>

                <div>
                  <span>Blood Pressure</span>
                  <strong>{patient.blood_pressure} mmHg</strong>
                </div>

                <div>
                  <span>BMI</span>
                  <strong>{patient.bmi} kg/m²</strong>
                </div>

                <div>
                  <span>Insulin</span>
                  <strong>{patient.insulin} μU/mL</strong>
                </div>

                <div>
                  <span>Skin Thickness</span>
                  <strong>{patient.skin_thickness} mm</strong>
                </div>

                <div>
                  <span>Diabetes Pedigree</span>
                  <strong>{patient.diabetes_pedigree}</strong>
                </div>
              </div>
            </div>

            <div className="synthetic-section">
              <h3>Lifestyle factors</h3>

              <div className="synthetic-grid">
                <div>
                  <span>Physical Activity</span>
                  <strong>{patient.physical_activity}</strong>
                </div>

                <div>
                  <span>Smoking</span>
                  <strong>{patient.smoking ? "Yes" : "No"}</strong>
                </div>

                <div>
                  <span>Alcohol</span>
                  <strong>{patient.alcohol}</strong>
                </div>
              </div>
            </div>

            <button
              className="primary-button synthetic-predict"
              onClick={runPrediction}
              disabled={loading}
            >
              {loading ? "Running model..." : "Run SVM Prediction"}
            </button>
          </div>

          <div className="synthetic-analysis">
            <div className="panel-header">
              <div>
                <span className="panel-label">RISK ANALYSIS</span>
                <h2>Risk Factor Breakdown</h2>
              </div>
            </div>

            <div className="risk-factors">
              {riskFactors.length > 0 ? (
                riskFactors.map((factor) => (
                  <div className="risk-factor" key={factor}>
                    <HeartPulse size={17} />
                    <span>{factor}</span>
                  </div>
                ))
              ) : (
                <div className="no-risk">
                  No major threshold-based risk factors detected.
                </div>
              )}
            </div>

            {patient.prediction && (
              <div
                className={
                  patient.prediction.prediction === 1
                    ? "synthetic-result high"
                    : "synthetic-result low"
                }
              >
                {patient.prediction.prediction === 1 ? (
                  <ShieldAlert size={30} />
                ) : (
                  <ShieldCheck size={30} />
                )}

                <span>Linear SVM prediction</span>

                <strong>
                  {patient.prediction.error
                    ? "Prediction unavailable"
                    : patient.prediction.result}
                </strong>

                {!patient.prediction.error && (
                  <small>Model accuracy: {patient.prediction.accuracy}%</small>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default SyntheticLab;
