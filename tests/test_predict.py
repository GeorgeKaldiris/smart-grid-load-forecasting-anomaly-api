import pytest

from src.predict import predict_load


def get_sample_input(include_actual_power: bool = True):
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
    }

    if include_actual_power:
        sample_input["actual_power_kw"] = 7.0

    return sample_input


def test_predict_load_returns_prediction_with_actual_power():
    sample_input = get_sample_input(include_actual_power=True)

    result = predict_load(sample_input)

    assert "predicted_power_kw" in result
    assert "model_type" in result
    assert "anomaly_threshold" in result
    assert "actual_power_kw" in result
    assert "residual" in result
    assert "abs_residual" in result
    assert "is_anomaly" in result

    assert isinstance(result["predicted_power_kw"], float)
    assert isinstance(result["actual_power_kw"], float)
    assert isinstance(result["residual"], float)
    assert isinstance(result["abs_residual"], float)
    assert isinstance(result["is_anomaly"], bool)


def test_predict_load_returns_prediction_without_actual_power():
    sample_input = get_sample_input(include_actual_power=False)

    result = predict_load(sample_input)

    assert "predicted_power_kw" in result
    assert "model_type" in result
    assert "anomaly_threshold" in result

    assert "actual_power_kw" not in result
    assert "residual" not in result
    assert "abs_residual" not in result
    assert "is_anomaly" not in result

    assert isinstance(result["predicted_power_kw"], float)


def test_predict_load_missing_feature_raises_error():
    sample_input = {
        "solar_power_kw": 3.5,
        "wind_power_kw": 2.1,
    }

    with pytest.raises(ValueError) as error:
        predict_load(sample_input)

    assert "Missing input features" in str(error.value)