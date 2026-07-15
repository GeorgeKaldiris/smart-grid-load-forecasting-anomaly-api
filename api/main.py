from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_load


app = FastAPI(
    title="Smart Grid Load Forecasting & Anomaly Detection API",
    description="API for predicting smart grid power consumption and detecting anomalies.",
    version="1.0.0",)


class PredictionRequest(BaseModel):
    solar_power_kw: float
    wind_power_kw: float
    temperature_c: float
    humidity_percent: float
    electricity_price_usd_kwh: float

    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)

    lag_1: float
    lag_4: float
    lag_96: float

    rolling_mean_4: float
    rolling_mean_24: float
    rolling_mean_96: float

    actual_power_kw: Optional[float] = None


@app.get("/")
def root():
    return {"message": "Smart Grid Load Forecasting & Anomaly Detection API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = request.dict()

    try:
        prediction = predict_load(input_data)
        return prediction

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))