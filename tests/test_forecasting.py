"""Pruebas del Sprint 03 con datos sintéticos reproducibles."""
from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest

from tempus.demo import synthetic_series
from tempus.forecasting import (
    ForecastConfig, METHODS, error_metrics, forecast_one_step,
    walk_forward_evaluation,
)


def test_baseline_uses_last_known_price():
    history = synthetic_series()["Close"].iloc[:90]
    assert forecast_one_step(history, "Ingenuo") == pytest.approx(history.iloc[-1])


@pytest.mark.parametrize("method", METHODS)
def test_each_method_predicts_finite_price(method):
    price = forecast_one_step(synthetic_series()["Close"].iloc[:100], method)
    assert np.isfinite(price) and price > 0


def test_config_rejects_invalid_values():
    for kwargs in ({"test_fraction": 0.9}, {"step": 0}, {"min_train": 20}):
        with pytest.raises(ValueError):
            ForecastConfig(**kwargs)
    with pytest.raises(FrozenInstanceError):
        ForecastConfig().step = 3


def test_metrics_known_values():
    idx = pd.date_range("2025-01-01", periods=3)
    result = error_metrics(pd.Series([10., 20., 30.], index=idx),
                           pd.Series([11., 18., 30.], index=idx))
    assert result["n"] == 3
    assert result["MAE"] == pytest.approx(1.0)
    assert result["RMSE"] == pytest.approx((5 / 3)**0.5)


def test_metrics_reject_incompatible_index():
    with pytest.raises(ValueError):
        error_metrics(pd.Series([1, 2]), pd.Series([1, 2], index=[3, 4]))


def test_walk_forward_origin_never_uses_test_target(monkeypatch):
    frame = synthetic_series()
    origins = []

    def spy(history, method):
        origins.append((history.index.max(), len(history), method))
        return float(history.iloc[-1])

    monkeypatch.setattr("tempus.forecasting.forecast_one_step", spy)
    pred, scores, meta = walk_forward_evaluation(
        frame, ForecastConfig(test_fraction=0.2, step=5, min_train=70),
        methods=("Ingenuo",),
    )
    assert len(pred) >= 5
    assert all(origin < target for (origin, _, _), target in
               zip(origins, pred.index))
    assert scores.iloc[0]["n"] == len(pred)
    assert meta["horizonte_sesiones"] == 1
    assert pred.iloc[0]["entrenamiento_n"] == int(len(frame) * 0.8)


def test_missing_predictions_are_not_masked(monkeypatch):
    frame = synthetic_series()
    def intermittent(history, method):
        if method == "ARIMA(1,1,1)" and len(history) % 2 == 0:
            raise ValueError("simulated failure")
        return float(history.iloc[-1])
    monkeypatch.setattr("tempus.forecasting.forecast_one_step", intermittent)
    pred, scores, meta = walk_forward_evaluation(
        frame, ForecastConfig(test_fraction=0.2, step=1, min_train=70))
    assert len(meta["fallos"]) > 0
    assert int(scores["n"].nunique()) == 1
    assert len(pred) > meta["evaluaciones_comunes"]


def test_walk_forward_does_not_mutate_input(monkeypatch):
    frame = synthetic_series()
    original = frame.copy(deep=True)
    monkeypatch.setattr("tempus.forecasting.forecast_one_step",
                        lambda history, method: float(history.iloc[-1]))
    walk_forward_evaluation(frame, ForecastConfig(step=4, min_train=70))
    pd.testing.assert_frame_equal(frame, original)
