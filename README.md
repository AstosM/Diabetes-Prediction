Run locally
pip install -r requirements.txt
streamlit run app.py

# 🩺 Clinical Diabetes Diagnostics & Predictive Intelligence Application

An end-to-end medical diagnostics application leveraging machine learning to predict diabetic patient outcomes based on historical health indicators. Powered by a mathematically optimal Support Vector Machine (SVM) pipeline, this architecture translates raw patient data inputs into real-time health-risk classifications via an interactive Streamlit data interface.

---

## 🚀 Key Architectural Highlights
* **Mathematical Core:** Utilizes a **Support Vector Machine (SVM)** classifier with a linear kernel execution layer to maximize hyperplane margins between high-risk categorical health distributions.
* **Feature Engineering Pipeline:** Implements z-score normalization via `StandardScaler` to prevent high-magnitude metrics (e.g., Insulin, Glucose) from biasing the classification boundaries.
* **Interactive Healthcare UI:** Features a dynamic **Streamlit** multi-tab control framework allowing medical consultants to review clinical distributions, explore feature correlations, and compute live diagnostics.

---

## 📊 Analytical Architecture & Interface Flow
The diagnostics application maps workflows fluidly from raw data validation straight to interactive local model inferences:

```text
  [DATA SOURCE]            [PROCESSING PIPELINE]            [PRODUCTION INTERFACE]
┌──────────────┐         ┌─────────────────────────┐       ┌───────────────────────────┐
│  PIMA India  │ ──────➔ │ StandardScaler Fit/Trans │ ────➔ │ Streamlit Frontend Engine │
│ Diabetes CSV │         │ SVM Hyperplane Training │       │   (🏠 Home / 📊 EDA Tabs)  │
└──────────────┘         └─────────────────────────┘       └─────────────┬─────────────┘
                                                                         │
                                                                         ▼
                                                            [🤖 Real-Time Inferences]
                                                            - Diagnostic Inputs
                                                            - Local Array Transforms
