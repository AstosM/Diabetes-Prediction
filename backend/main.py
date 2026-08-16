from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import model, scaler, accuracy, df

app = FastAPI(title="Diabetes Risk Portal API")

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        # Add your deployed frontend URL here after deployment
        # Example:
        # "https://your-frontend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST MODEL

class PatientData(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=0, le=300)
    blood_pressure: float = Field(ge=0, le=200)
    skin_thickness: float = Field(ge=0, le=100)
    insulin: float = Field(ge=0, le=900)
    bmi: float = Field(ge=0, le=70)
    diabetes_pedigree: float = Field(ge=0, le=5)
    age: int = Field(ge=1, le=120)

# ROOT

@app.get("/")
def root():
    return {
        "message": "Diabetes Risk Portal API is running"
    }

# HEALTH CHECK

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# MODEL INFORMATION

@app.get("/model-info")
def model_info():
    return {
        "model": "Support Vector Machine",
        "kernel": "linear",
        "accuracy": round(accuracy * 100, 2)
    }

# ANALYTICS

@app.get("/analytics")
def analytics():
    total_records = len(df)

    diabetic = int(df["Outcome"].sum())
    non_diabetic = total_records - diabetic

    return {
        "total_records": total_records,
        "diabetic": diabetic,
        "non_diabetic": non_diabetic,
        "diabetes_rate": round(
            (diabetic / total_records) * 100,
            2
        ),
        "features": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree",
            "Age"
        ],
        "averages": {
            "glucose": round(
                float(df["Glucose"].mean()),
                2
            ),
            "bmi": round(
                float(df["BMI"].mean()),
                2
            ),
            "age": round(
                float(df["Age"].mean()),
                2
            ),
            "blood_pressure": round(
                float(df["BloodPressure"].mean()),
                2
            )
        }
    }

# PREDICTION

@app.post("/predict")
def predict(data: PatientData):

    input_data = [[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age
    ]]

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        result = "High Risk"
    else:
        result = "Low Risk"

    return {
        "prediction": int(prediction),
        "result": result,
        "model": "Linear SVM",
        "accuracy": round(accuracy * 100, 2)
    }
