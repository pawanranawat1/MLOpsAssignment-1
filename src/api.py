"""FastAPI model serving application."""

import logging
from typing import Optional
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
Instrumentator().instrument(app).expose(app)

model = None


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=0, le=2)
    ca: Optional[float] = Field(default=None, ge=0, le=3)
    thal: Optional[float] = Field(default=None, ge=0, le=7)


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        logger.warning("Model file does not exist: %s", MODEL_PATH)
        return
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs"}


@app.post("/predict")
def predict(patient: Patient):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    data = patient.model_dump()

    # Convert the input dictionary to a DataFrame so that
    # the sklearn pipeline receives a 2D tabular input.
    input_df = pd.DataFrame([data])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    logger.info(
        "Prediction request processed: prediction=%s confidence=%.4f",
        prediction,
        probability,
    )

    return {
        "prediction": prediction,
        "prediction_label": (
            "heart disease risk" if prediction == 1 else "no heart disease risk"
        ),
        "confidence": round(probability if prediction == 1 else 1 - probability, 4),
        "disease_probability": round(probability, 4),
    }
