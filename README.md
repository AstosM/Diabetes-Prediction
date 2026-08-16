# Diabetes Risk Prediction Portal — Backend

A FastAPI-based backend for the **Diabetes Risk Prediction Portal**.
The application uses a **Linear Support Vector Machine (SVM)** machine-learning model to predict diabetes risk from patient health parameters.

## 🚀 Features

- FastAPI REST API
- Diabetes risk prediction using Linear SVM
- StandardScaler feature normalization
- Dataset analytics
- Model accuracy information
- Health-check endpoint
- Automatic API documentation with Swagger
- Input validation using Pydantic
- CORS support for frontend integration

## 🛠️ Technology Stack

- **Python**
- **FastAPI**
- **Uvicorn**
- **Pandas**
- **Scikit-learn**
- **Pydantic**
- **Support Vector Machine (SVM)**

## 📁 Project Structure

```text
backend/
│
├── main.py
├── model.py
├── diabetes.csv
├── requirements.txt
└── README.md
```

### `main.py`

Contains the FastAPI application and API endpoints.

### `model.py`

Loads the diabetes dataset, preprocesses the data, trains the Linear SVM model, and calculates model accuracy.

### `diabetes.csv`

Dataset used for training and evaluating the machine-learning model.

### `requirements.txt`

Contains the Python dependencies required to run the backend.

## 🧠 Machine Learning Pipeline

```text
Diabetes Dataset
       ↓
Data Loading
       ↓
Feature / Target Separation
       ↓
Train-Test Split
       ↓
StandardScaler
       ↓
Linear SVM
       ↓
Model Training
       ↓
Accuracy Evaluation
       ↓
Prediction API
```

## 📊 Input Features

The prediction model uses the following eight features:

| Feature           | Description                  |
| ----------------- | ---------------------------- |
| Pregnancies       | Number of pregnancies        |
| Glucose           | Plasma glucose concentration |
| Blood Pressure    | Diastolic blood pressure     |
| Skin Thickness    | Triceps skin fold thickness  |
| Insulin           | Insulin level                |
| BMI               | Body Mass Index              |
| Diabetes Pedigree | Diabetes pedigree function   |
| Age               | Patient age                  |

## 🔌 API Endpoints

### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Diabetes Risk Portal API is running"
}
```

### GET `/health`

Returns the health status of the backend.

```json
{
  "status": "healthy"
}
```

### GET `/model-info`

Returns information about the machine-learning model.

Example:

```json
{
  "model": "Support Vector Machine",
  "kernel": "linear",
  "accuracy": 77.92
}
```

### GET `/analytics`

Returns dataset statistics including:

- Total records
- Diabetic records
- Non-diabetic records
- Diabetes rate
- Available features
- Average glucose
- Average BMI
- Average age
- Average blood pressure

### POST `/predict`

Accepts patient information and returns the predicted diabetes risk.

Example request:

```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 79,
  "bmi": 25.5,
  "diabetes_pedigree": 0.35,
  "age": 35
}
```

Example response:

```json
{
  "prediction": 0,
  "result": "Low Risk",
  "model": "Linear SVM",
  "accuracy": 77.92
}
```

A prediction of `1` indicates **High Risk**, while `0` indicates **Low Risk**.

> **Note:** This project is intended for educational and demonstration purposes. Predictions should not be considered a medical diagnosis.

## ⚙️ Installation

Clone the repository and navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```powershell
.\venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Backend

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## 📚 API Documentation

FastAPI automatically provides interactive Swagger documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

Alternative ReDoc documentation:

```text
http://127.0.0.1:8000/redoc
```

## 🌐 Frontend Integration

The backend supports CORS for the React frontend during local development.

Local frontend:

```text
http://localhost:5173
```

When deploying the application, update the `allow_origins` configuration in `main.py` with the deployed frontend URL.

## ☁️ Deployment

The backend can be deployed on platforms such as **Render** or other services supporting Python/FastAPI applications.

### Render Configuration

**Root Directory**

```text
backend
```

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

After deployment, verify:

```text
https://YOUR-BACKEND-URL/health
```

and:

```text
https://YOUR-BACKEND-URL/docs
```

## 🔐 Input Validation

The API validates incoming patient data using Pydantic.

Examples of validation limits include:

- Pregnancies: `0–20`
- Glucose: `0–300`
- Blood Pressure: `0–200`
- Skin Thickness: `0–100`
- Insulin: `0–900`
- BMI: `0–70`
- Diabetes Pedigree: `0–5`
- Age: `1–120`

Invalid input is automatically rejected by the API.

## 📌 Project Purpose

The purpose of this project is to demonstrate the development of a complete machine-learning-powered web application that combines:

```text
Machine Learning
       +
FastAPI Backend
       +
React Frontend
       =
Diabetes Risk Prediction Portal
```

The system provides a simple interface for entering patient parameters and receiving a machine-learning-based diabetes risk prediction.

## 👨‍💻 Author

Developed as a BTech project demonstrating machine learning, REST API development, and frontend-backend integration.
