# Smart Grid Load Forecasting & Anomaly Detection API

An end-to-end machine learning project for smart grid power consumption forecasting and residual-based anomaly detection.

The project includes exploratory analysis, feature engineering, model training, saved ML artifacts, a FastAPI inference API, automated tests, Docker containerization and GitHub Actions CI.

---

## Project Overview

The goal of this project is to predict smart grid power consumption using historical and contextual features and to detect potential anomalies by comparing actual power consumption against the model prediction.

The API can be used in two modes:

1. **Forecasting only**  
   Returns the predicted power consumption.

2. **Forecasting + anomaly detection**  
   If `actual_power_kw` is provided, the API calculates the residual and flags whether the point is anomalous.

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- Docker
- pytest
- GitHub Actions CI

---

## Project Structure

```text
.
├── api/
│   └── main.py
├── data/
│   └── .gitkeep
├── models/
│   ├── load_forecasting_model.joblib
│   └── model_config.json
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_predict.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains smart grid measurements such as:

- Timestamp
- Power Consumption
- Solar Power
- Wind Power
- Temperature
- Humidity
- Electricity Price
- Grid-related measurements
- Operational indicators such as overload and transformer fault

The main forecasting target is:

```text
Power Consumption (kW)
```

The raw dataset is not included in this repository.

To retrain the model, place the dataset at:

```text
data/smart_grid_dataset.csv
```

---

## Feature Engineering

The model uses safe forecasting features, including:

- Renewable generation features
- Weather-related features
- Electricity price
- Time-based features
- Lag features
- Rolling mean features

Examples:

```text
lag_1  -> previous 15-minute value
lag_4  -> value from approximately 1 hour before
lag_96 -> value from approximately 1 day before
```

Rolling mean features:

```text
rolling_mean_4  -> average of the previous 4 values
rolling_mean_24 -> average of the previous 24 values
rolling_mean_96 -> average of the previous 96 values
```

A `shift(1)` is applied before rolling calculations to avoid using the current target value in feature creation.

---

## Leakage Prevention

During experimentation, some same-timestamp electrical variables showed extremely high correlation with the target, especially `Current (A)`.

These features were excluded from the final safe feature set to avoid proxy leakage and unrealistic performance.

The final model uses only features that are more appropriate for a forecasting setup.

---

## Model

The final model is:

```text
HistGradientBoostingRegressor
```

It was selected after comparing it against a naive lag baseline and a Random Forest model using the same safe feature set.

The model is trained using a chronological 80/20 split, which is more appropriate for time-series forecasting than a random split.

---

## Evaluation Results

Final model performance:

```text
Naive Baseline MAE: 3.4724
Naive Baseline RMSE: 4.2507

Final Model MAE: 2.5952
Final Model RMSE: 2.9977

MAE improvement: 25.26%
RMSE improvement: 29.48%
```

---

## Anomaly Detection

Anomaly detection is based on forecast residuals.

The residual is calculated as:

```text
residual = actual_power_kw - predicted_power_kw
```

The absolute residual is compared against a threshold calculated from the training residual distribution.

```text
is_anomaly = abs_residual > anomaly_threshold
```

This allows the API to flag cases where the actual power consumption differs significantly from the expected value.

---

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python src/train.py
```

Run the prediction script:

```bash
python src/predict.py
```

Run the FastAPI app:

```bash
python -m uvicorn api.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
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
  "actual_power_kw": 7.0
}
```

Example response:

```json
{
  "predicted_power_kw": 6.3469298489421355,
  "model_type": "HistGradientBoostingRegressor",
  "anomaly_threshold": 4.881138557279104,
  "actual_power_kw": 7.0,
  "residual": 0.6530701510578645,
  "abs_residual": 0.6530701510578645,
  "is_anomaly": false
}
```

---

## Run Tests

```bash
python -m pytest
```

The tests cover:

- Prediction with `actual_power_kw`
- Prediction without `actual_power_kw`
- Missing required feature validation

---

## Run with Docker

Build the Docker image:

```bash
docker build -t smart-grid-api .
```

Run the container:

```bash
docker run -p 8000:8000 smart-grid-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

If port `8000` is already in use:

```bash
docker run -p 8001:8000 smart-grid-api
```

Then open:

```text
http://127.0.0.1:8001/docs
```

---

## CI/CD

This project includes a GitHub Actions CI workflow.

On every push or pull request to `main`, the workflow:

1. Checks out the repository
2. Sets up Python 3.11
3. Installs dependencies
4. Runs the pytest test suite

---

## MLOps Note

During Docker testing, the saved scikit-learn model initially failed to load due to a dependency version mismatch between the local training environment and the Docker environment.

To fix this, the package versions were pinned in `requirements.txt`.

This is important in ML projects because serialized model artifacts should be loaded with compatible library versions.

---

## What This Project Demonstrates

This project demonstrates:

- Exploratory data analysis
- Time-series feature engineering
- Regression model training
- Leakage-aware feature selection
- Model evaluation against a baseline
- Residual-based anomaly detection
- Model serialization with joblib
- FastAPI model serving
- Dockerized inference API
- Automated testing with pytest
- GitHub Actions CI
