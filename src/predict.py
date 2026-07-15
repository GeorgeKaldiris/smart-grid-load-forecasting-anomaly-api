from pathlib import Path
import json
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "load_forecasting_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"


INPUT_FEATURE_MAPPING = {
    "solar_power_kw": "Solar Power (kW)",
    "wind_power_kw": "Wind Power (kW)",
    "temperature_c": "Temperature (°C)",
    "humidity_percent": "Humidity (%)",
    "electricity_price_usd_kwh": "Electricity Price (USD/kWh)",
    "hour": "hour",
    "day_of_week": "day_of_week",
    "month": "month",
    "is_weekend": "is_weekend",
    "lag_1": "lag_1",
    "lag_4": "lag_4",
    "lag_96": "lag_96",
    "rolling_mean_4": "rolling_mean_4",
    "rolling_mean_24": "rolling_mean_24",
    "rolling_mean_96": "rolling_mean_96",
}


def load_artifacts():
    """
    Load the trained model and the model configuration from disk.
    """
    model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, config


def prepare_model_input(input_data: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """
    Validate the input data, map API-friendly feature names to model feature names,
    and return a DataFrame with the correct feature order.
    """
    missing_features = []

    for api_feature in INPUT_FEATURE_MAPPING:
        if api_feature not in input_data:
            missing_features.append(api_feature)

    if missing_features:
        raise ValueError(f"Missing input features: {missing_features}")

    model_input = {}

    for api_feature, model_feature in INPUT_FEATURE_MAPPING.items():
        model_input[model_feature] = input_data[api_feature]

    input_df = pd.DataFrame([model_input])

    input_df = input_df[feature_columns]

    return input_df


def predict_load(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Predict power consumption and optionally calculate anomaly information
    when actual_power_kw is provided.
    """
    model, config = load_artifacts()

    feature_columns = config["feature_columns"]
    anomaly_threshold = config["anomaly_threshold"]

    input_df = prepare_model_input(input_data, feature_columns)

    predicted_power = float(model.predict(input_df)[0])

    result = {
        "predicted_power_kw": predicted_power,
        "model_type": config["model_type"],
        "anomaly_threshold": anomaly_threshold,
    }

    if "actual_power_kw" in input_data and input_data["actual_power_kw"] is not None:
        actual_power = float(input_data["actual_power_kw"])
        residual = actual_power - predicted_power
        abs_residual = abs(residual)

        result.update({
            "actual_power_kw": actual_power,
            "residual": residual,
            "abs_residual": abs_residual,
            "is_anomaly": abs_residual > anomaly_threshold,
        })

    return result


if __name__ == "__main__":
    sample_input = {
        "solar_power_kw": 3.5,
        "wind_power_kw": 2.1,
        "temperature_c": 27.0,
        "humidity_percent": 60.0,
        "electricity_price_usd_kwh": 0.15,
        "hour": 14,
        "day_of_week": 2,
        "month": 6,
        "is_weekend": 0,
        "lag_1": 6.2,
        "lag_4": 6.0,
        "lag_96": 6.5,
        "rolling_mean_4": 6.1,
        "rolling_mean_24": 6.3,
        "rolling_mean_96": 6.4,
        "actual_power_kw": 7.0,
    }

    prediction = predict_load(sample_input)
    print(prediction)