"""
MediPredict AI – FastAPI Backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import os
from typing import Literal

app = FastAPI(
    title="MediPredict AI",
    description="Clinical-grade disease risk prediction API powered by Gradient Boosting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(prefix: str):
    files = [f"{prefix}_model.pkl", f"{prefix}_scaler.pkl", f"{prefix}_cols.pkl"]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} not found. Run train_models.py first.")
    return (
        joblib.load(f"{prefix}_model.pkl"),
        joblib.load(f"{prefix}_scaler.pkl"),
        joblib.load(f"{prefix}_cols.pkl"),
    )

def run_prediction(prefix: str, input_dict: dict) -> dict:
    model, scaler, cols = load_model(prefix)
    arr = np.array([input_dict.get(c, 0) for c in cols]).reshape(1, -1)
    arr_sc = scaler.transform(arr)
    pred = int(model.predict(arr_sc)[0])
    prob = model.predict_proba(arr_sc)[0].tolist()
    return {
        "prediction": pred,
        "risk_probability": round(prob[1] * 100, 2),
        "safe_probability": round(prob[0] * 100, 2),
    }

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class HeartInput(BaseModel):
    age:      int   = Field(..., ge=29, le=77,    example=55,   description="Patient age in years")
    sex:      int   = Field(..., ge=0,  le=1,     example=1,    description="0 = Female, 1 = Male")
    cp:       int   = Field(..., ge=0,  le=3,     example=2,    description="Chest pain type (0-3)")
    trestbps: int   = Field(..., ge=90, le=200,   example=130,  description="Resting blood pressure (mm Hg)")
    chol:     int   = Field(..., ge=120,le=564,   example=246,  description="Serum cholesterol (mg/dL)")
    fbs:      int   = Field(..., ge=0,  le=1,     example=0,    description="Fasting blood sugar > 120 mg/dL")
    restecg:  int   = Field(..., ge=0,  le=2,     example=0,    description="Resting ECG result (0-2)")
    thalach:  int   = Field(..., ge=71, le=202,   example=150,  description="Maximum heart rate achieved")
    exang:    int   = Field(..., ge=0,  le=1,     example=0,    description="Exercise-induced angina (0/1)")
    oldpeak:  float = Field(..., ge=0.0,le=6.2,  example=1.0,  description="ST depression induced by exercise")
    slope:    int   = Field(..., ge=0,  le=2,     example=1,    description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,  le=3,     example=0,    description="Number of major vessels (0-3)")
    thal:     int   = Field(..., ge=1,  le=3,     example=2,    description="Thalassemia type (1=Normal, 2=Fixed, 3=Reversible)")

class DiabetesInput(BaseModel):
    Pregnancies:              int   = Field(..., ge=0,   le=17,  example=3)
    Glucose:                  int   = Field(..., ge=44,  le=199, example=120,  description="Plasma glucose (mg/dL)")
    BloodPressure:            int   = Field(..., ge=24,  le=122, example=72,   description="Diastolic blood pressure (mm Hg)")
    SkinThickness:            int   = Field(..., ge=7,   le=99,  example=27,   description="Triceps skin fold thickness (mm)")
    Insulin:                  int   = Field(..., ge=14,  le=846, example=100,  description="2-Hour serum insulin (μU/mL)")
    BMI:                      float = Field(..., ge=18.0,le=67.0,example=31.0)
    DiabetesPedigreeFunction: float = Field(..., ge=0.07,le=2.42,example=0.45)
    Age:                      int   = Field(..., ge=21,  le=81,  example=33)

class LiverInput(BaseModel):
    Age:                        int   = Field(..., ge=4,   le=90,   example=45)
    Gender:                     int   = Field(..., ge=0,   le=1,    example=1,   description="0 = Female, 1 = Male")
    Total_Bilirubin:            float = Field(..., ge=0.4, le=75.0, example=1.5, description="mg/dL")
    Direct_Bilirubin:           float = Field(..., ge=0.1, le=19.7, example=0.5, description="mg/dL")
    Alkaline_Phosphotase:       int   = Field(..., ge=63,  le=2110, example=200, description="IU/L")
    Alamine_Aminotransferase:   int   = Field(..., ge=7,   le=2000, example=35,  description="IU/L")
    Aspartate_Aminotransferase: int   = Field(..., ge=10,  le=4929, example=40,  description="IU/L")
    Total_Protiens:             float = Field(..., ge=2.7, le=9.6,  example=6.8, description="g/dL")
    Albumin:                    float = Field(..., ge=0.9, le=5.5,  example=3.5, description="g/dL")
    Albumin_and_Globulin_Ratio: float = Field(..., ge=0.3, le=2.8,  example=0.95)

class LungInput(BaseModel):
    GENDER:               int = Field(..., ge=0, le=1, example=1, description="0 = Female, 1 = Male")
    AGE:                  int = Field(..., ge=20, le=90, example=55)
    SMOKING:              int = Field(..., ge=0, le=1, example=1)
    YELLOW_FINGERS:       int = Field(..., ge=0, le=1, example=0)
    ANXIETY:              int = Field(..., ge=0, le=1, example=0)
    PEER_PRESSURE:        int = Field(..., ge=0, le=1, example=0)
    CHRONIC_DISEASE:      int = Field(..., ge=0, le=1, example=1)
    FATIGUE:              int = Field(..., ge=0, le=1, example=1)
    ALLERGY:              int = Field(..., ge=0, le=1, example=0)
    WHEEZING:             int = Field(..., ge=0, le=1, example=1)
    ALCOHOL_CONSUMING:    int = Field(..., ge=0, le=1, example=0)
    COUGHING:             int = Field(..., ge=0, le=1, example=1)
    SHORTNESS_OF_BREATH:  int = Field(..., ge=0, le=1, example=1)
    SWALLOWING_DIFFICULTY:int = Field(..., ge=0, le=1, example=0)
    CHEST_PAIN:           int = Field(..., ge=0, le=1, example=1)

# ── Response Schema ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    prediction:       int
    risk_probability: float
    safe_probability: float
    label:            str
    disease:          str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "MediPredict AI is running 🚀"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

@app.post("/predict/heart", response_model=PredictionResponse, tags=["Predictions"])
def predict_heart(data: HeartInput):
    try:
        result = run_prediction("heart", data.model_dump())
        result["label"] = "Heart Disease Detected" if result["prediction"] == 1 else "No Heart Disease Detected"
        result["disease"] = "Heart Disease"
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict/diabetes", response_model=PredictionResponse, tags=["Predictions"])
def predict_diabetes(data: DiabetesInput):
    try:
        result = run_prediction("diabetes", data.model_dump())
        result["label"] = "Diabetes Detected" if result["prediction"] == 1 else "No Diabetes Detected"
        result["disease"] = "Diabetes"
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict/liver", response_model=PredictionResponse, tags=["Predictions"])
def predict_liver(data: LiverInput):
    try:
        result = run_prediction("liver", data.model_dump())
        result["label"] = "Liver Disease Detected" if result["prediction"] == 1 else "No Liver Disease Detected"
        result["disease"] = "Liver Disease"
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict/lung", response_model=PredictionResponse, tags=["Predictions"])
def predict_lung(data: LungInput):
    try:
        result = run_prediction("lung", data.model_dump())
        result["label"] = "High Lung Cancer Risk" if result["prediction"] == 1 else "Low Lung Cancer Risk"
        result["disease"] = "Lung Cancer"
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
