from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "smart_grid_dataset.csv"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "load_forecasting_model.joblib"
CONFIG_PATH = MODELS_DIR / "model_config.json"

TIMESTAMP_COL = "Timestamp"
TARGET = "Power Consumption (kW)"

SAFE_FEATURES = [
    "Solar Power (kW)",
    "Wind Power (kW)",
    "Temperature (°C)",
    "Humidity (%)",
    "Electricity Price (USD/kWh)",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_4",
    "lag_96",
    "rolling_mean_4",
    "rolling_mean_24",
    "rolling_mean_96",
]


def load_data() -> pd.DataFrame:
    """
    Load the dataset, convert the timestamp column to datetime,
    and sort the data chronologically.
    """
    df = pd.read_csv(DATA_PATH)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    return df


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based, lag and rolling mean features for forecasting.
    """
    df_model = dataframe.copy()

    df_model["hour"] = df_model[TIMESTAMP_COL].dt.hour
    df_model["day_of_week"] = df_model[TIMESTAMP_COL].dt.dayofweek
    df_model["month"] = df_model[TIMESTAMP_COL].dt.month
    df_model["is_weekend"] = df_model["day_of_week"].isin([5, 6]).astype(int)

    df_model["lag_1"] = df_model[TARGET].shift(1)
    df_model["lag_4"] = df_model[TARGET].shift(4)
    df_model["lag_96"] = df_model[TARGET].shift(96)

    df_model["rolling_mean_4"] = df_model[TARGET].shift(1).rolling(window=4).mean()
    df_model["rolling_mean_24"] = df_model[TARGET].shift(1).rolling(window=24).mean()
    df_model["rolling_mean_96"] = df_model[TARGET].shift(1).rolling(window=96).mean()

    df_model = df_model.dropna().reset_index(drop=True)

    return df_model


def split_data(df_model: pd.DataFrame):
    """
    Split the dataset chronologically into train and test sets.
    """
    X = df_model[SAFE_FEATURES]
    y = df_model[TARGET]

    split_index = int(len(df_model) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test, split_index


def evaluate_regression(y_true, predictions) -> tuple[float, float]:
    """
    Calculate MAE and RMSE for regression evaluation.
    """
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    return mae, rmse


def train_model():
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data()
    df_model = create_features(df)

    X_train, X_test, y_train, y_test, split_index = split_data(df_model)

    baseline_predictions = X_test["lag_1"]
    baseline_mae, baseline_rmse = evaluate_regression(y_test, baseline_predictions)

    model = HistGradientBoostingRegressor(
        max_iter=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    model_mae, model_rmse = evaluate_regression(y_test, predictions)

    mae_improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
    rmse_improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100

    train_predictions = model.predict(X_train)
    train_residuals = y_train.values - train_predictions
    train_abs_residuals = np.abs(train_residuals)

    anomaly_threshold = float(np.percentile(train_abs_residuals, 95))

    test_results = df_model.iloc[split_index:].copy()
    test_results["actual_power"] = y_test.values
    test_results["predicted_power"] = predictions
    test_results["residual"] = test_results["actual_power"] - test_results["predicted_power"]
    test_results["abs_residual"] = test_results["residual"].abs()

    test_results["is_anomaly"] = (
        test_results["abs_residual"] > anomaly_threshold
    ).astype(int)

    anomaly_rate = test_results["is_anomaly"].mean() * 100

    joblib.dump(model, MODEL_PATH)

    model_config = {
        "model_type": "HistGradientBoostingRegressor",
        "target_column": TARGET,
        "timestamp_column": TIMESTAMP_COL,
        "feature_columns": SAFE_FEATURES,
        "train_test_split": "chronological_80_20",
        "anomaly_threshold": float(anomaly_threshold),
        "metrics": {
            "baseline_mae": float(baseline_mae),
            "baseline_rmse": float(baseline_rmse),
            "model_mae": float(model_mae),
            "model_rmse": float(model_rmse),
            "mae_improvement_percent": float(mae_improvement),
            "rmse_improvement_percent": float(rmse_improvement),
            "test_anomaly_rate_percent": float(anomaly_rate)
        }
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=4)

    print("Training completed successfully.")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Model MAE: {model_mae:.4f}")
    print(f"Model RMSE: {model_rmse:.4f}")
    print(f"MAE improvement: {mae_improvement:.2f}%")
    print(f"RMSE improvement: {rmse_improvement:.2f}%")
    print(f"Anomaly threshold: {anomaly_threshold:.4f}")
    print(f"Test anomaly rate: {anomaly_rate:.2f}%")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved config to: {CONFIG_PATH}")


if __name__ == "__main__":
    train_model()