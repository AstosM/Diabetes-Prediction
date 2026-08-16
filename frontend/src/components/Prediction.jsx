import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";

function Prediction() {
  const [form, setForm] = useState({
    pregnancies: "",
    glucose: "",
    blood_pressure: "",
    skin_thickness: "",
    insulin: "",
    bmi: "",
    diabetes_pedigree: "",
    age: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          pregnancies: Number(form.pregnancies),
          glucose: Number(form.glucose),
          blood_pressure: Number(form.blood_pressure),
          skin_thickness: Number(form.skin_thickness),
          insulin: Number(form.insulin),
          bmi: Number(form.bmi),
          diabetes_pedigree: Number(form.diabetes_pedigree),
          age: Number(form.age),
        }),
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({
        error: "Unable to connect to the prediction service.",
      });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm({
      pregnancies: "",
      glucose: "",
      blood_pressure: "",
      skin_thickness: "",
      insulin: "",
      bmi: "",
      diabetes_pedigree: "",
      age: "",
    });

    setResult(null);
  };

  return (
    <div className="prediction-page">
      <div className="page-heading">
        <div>
          <p className="eyebrow">AI ASSESSMENT</p>
          <h1>Diabetes Risk Prediction</h1>
          <p>
            Enter patient health indicators to generate an SVM-based risk
            assessment.
          </p>
        </div>

        <div className="prediction-model">
          <ShieldCheck size={18} />
          <span>Linear SVM</span>
        </div>
      </div>

      <div className="prediction-layout">
        <form className="prediction-form" onSubmit={handleSubmit}>
          <div className="form-section">
            <div className="form-section-heading">
              <div className="section-number">01</div>

              <div>
                <h2>Patient information</h2>
                <p>Basic demographic and pregnancy information.</p>
              </div>
            </div>

            <div className="form-grid">
              <div className="field">
                <label>Pregnancies</label>
                <input
                  type="number"
                  name="pregnancies"
                  value={form.pregnancies}
                  onChange={handleChange}
                  min="0"
                  max="20"
                  placeholder="e.g. 2"
                  required
                />
              </div>

              <div className="field">
                <label>Age</label>
                <div className="input-unit">
                  <input
                    type="number"
                    name="age"
                    value={form.age}
                    onChange={handleChange}
                    min="1"
                    max="120"
                    placeholder="e.g. 34"
                    required
                  />
                  <span>years</span>
                </div>
              </div>
            </div>
          </div>

          <div className="form-section">
            <div className="form-section-heading">
              <div className="section-number">02</div>

              <div>
                <h2>Clinical measurements</h2>
                <p>Enter the patient's measured health indicators.</p>
              </div>
            </div>

            <div className="form-grid">
              <div className="field">
                <label>Glucose</label>
                <div className="input-unit">
                  <input
                    type="number"
                    name="glucose"
                    value={form.glucose}
                    onChange={handleChange}
                    min="0"
                    max="300"
                    placeholder="e.g. 120"
                    required
                  />
                  <span>mg/dL</span>
                </div>
              </div>

              <div className="field">
                <label>Blood Pressure</label>
                <div className="input-unit">
                  <input
                    type="number"
                    name="blood_pressure"
                    value={form.blood_pressure}
                    onChange={handleChange}
                    min="0"
                    max="200"
                    placeholder="e.g. 72"
                    required
                  />
                  <span>mmHg</span>
                </div>
              </div>

              <div className="field">
                <label>Skin Thickness</label>
                <div className="input-unit">
                  <input
                    type="number"
                    name="skin_thickness"
                    value={form.skin_thickness}
                    onChange={handleChange}
                    min="0"
                    max="100"
                    placeholder="e.g. 25"
                    required
                  />
                  <span>mm</span>
                </div>
              </div>

              <div className="field">
                <label>Insulin</label>
                <div className="input-unit">
                  <input
                    type="number"
                    name="insulin"
                    value={form.insulin}
                    onChange={handleChange}
                    min="0"
                    max="900"
                    placeholder="e.g. 100"
                    required
                  />
                  <span>μU/mL</span>
                </div>
              </div>

              <div className="field">
                <label>BMI</label>
                <div className="input-unit">
                  <input
                    type="number"
                    step="0.1"
                    name="bmi"
                    value={form.bmi}
                    onChange={handleChange}
                    min="0"
                    max="70"
                    placeholder="e.g. 27.5"
                    required
                  />
                  <span>kg/m²</span>
                </div>
              </div>

              <div className="field">
                <label>Diabetes Pedigree Function</label>
                <input
                  type="number"
                  step="0.001"
                  name="diabetes_pedigree"
                  value={form.diabetes_pedigree}
                  onChange={handleChange}
                  min="0"
                  max="5"
                  placeholder="e.g. 0.245"
                  required
                />
              </div>
            </div>
          </div>

          <div className="form-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={resetForm}
            >
              <RotateCcw size={17} />
              Reset
            </button>

            <button
              type="submit"
              className="primary-button prediction-submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <Activity size={18} />
                  Analyzing...
                </>
              ) : (
                <>
                  Run Prediction
                  <ArrowRight size={18} />
                </>
              )}
            </button>
          </div>
        </form>

        <div className="prediction-result">
          {!result && (
            <div className="result-empty">
              <div className="result-icon">
                <Activity size={30} />
              </div>

              <h2>Awaiting assessment</h2>

              <p>
                Complete the patient information and clinical measurements. The
                trained model will analyze the values and return a risk
                classification.
              </p>

              <div className="result-note">
                <ShieldCheck size={17} />
                <span>Powered by a trained Linear SVM model</span>
              </div>
            </div>
          )}

          {result?.error && (
            <div className="result-error">
              <AlertTriangle size={32} />
              <h2>Connection unavailable</h2>
              <p>{result.error}</p>
              <span>Make sure the FastAPI server is running.</span>
            </div>
          )}

          {result && !result.error && (
            <div
              className={
                result.prediction === 1
                  ? "result-content high"
                  : "result-content low"
              }
            >
              <div className="result-top">
                {result.prediction === 1 ? (
                  <AlertTriangle size={34} />
                ) : (
                  <CheckCircle2 size={34} />
                )}

                <span>Prediction complete</span>
              </div>

              <h2>{result.result}</h2>

              <p className="result-description">
                {result.prediction === 1
                  ? "The model has identified a higher likelihood of diabetes based on the submitted health indicators."
                  : "The model has identified a lower likelihood of diabetes based on the submitted health indicators."}
              </p>

              <div className="result-details">
                <div>
                  <span>Model</span>
                  <strong>{result.model}</strong>
                </div>

                <div>
                  <span>Accuracy</span>
                  <strong>{result.accuracy}%</strong>
                </div>
              </div>

              <div className="result-disclaimer">
                This prediction is generated by a machine learning model and
                should not be considered a medical diagnosis.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Prediction;
