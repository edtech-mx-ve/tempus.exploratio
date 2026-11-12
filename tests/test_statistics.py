"""Sprint 02: pruebas de estadística y comparación con series sintéticas."""
import numpy as np
import pandas as pd
import pytest

from tempus.demo import synthetic_series
from tempus.statistics import (
    descriptive_statistics, stationarity_diagnostics, dependence_profile,
    align_returns, correlation_matrix, normalized_price_panel, drawdown_series,
)


def series_with_shift(multiplier: float = 1.0) -> pd.DataFrame:
    """Crea una serie OHLCV válida con variación y escala controladas."""
    frame = synthetic_series().copy()
    for key in ("Open", "High", "Low", "Close"):
        frame[key] *= multiplier
    return frame


def test_descriptive_statistics():
    result = descriptive_statistics(synthetic_series())
    assert result["observaciones"] == 130
    assert result["rendimientos_validos"] == 129
    assert np.isfinite(result["volatilidad_anual_pct"])


def test_dependence_profile():
    data = synthetic_series()["Close"].pct_change().dropna()
    result = dependence_profile(data, max_lags=20)
    assert len(result) == 21
    assert result.iloc[0]["ACF"] == pytest.approx(1)
    assert result.iloc[0]["PACF"] == pytest.approx(1)


def test_dependence_limits():
    with pytest.raises(ValueError):
        dependence_profile(synthetic_series()["Close"], max_lags=-3)
    with pytest.raises(ValueError):
        dependence_profile(pd.Series([2] * 40))


def test_stationarity_diagnostics():
    rng = np.random.default_rng(25)
    values = pd.Series(rng.normal(size=500))
    result = stationarity_diagnostics(values)
    assert 0 <= result["adf_p"] <= 1
    assert 0 <= result["kpss_p"] <= 1
    assert result["n"] == 500


def test_stationarity_rejects_bad_input():
    with pytest.raises(ValueError):
        stationarity_diagnostics(pd.Series([1] * 200))
    with pytest.raises(ValueError):
        stationarity_diagnostics(pd.Series(range(8)))


def test_comparison_misaligned_calendars():
    a = series_with_shift()
    b = series_with_shift(2.0).iloc[5:]
    aligned = align_returns({"JNJ": a, "MSFT": b})
    assert len(aligned) == len(b) - 1
    corr = correlation_matrix({"JNJ": a, "MSFT": b})
    assert corr.loc["JNJ", "MSFT"] == pytest.approx(1)
    panel = normalized_price_panel({"JNJ": a, "MSFT": b})
    assert (panel.iloc[0] == 100).all()
    assert panel.index.min() == b.index.min()


def test_drawdown_never_positive():
    dd = drawdown_series(synthetic_series())
    assert (dd <= 0).all()
    assert dd.iloc[0] == pytest.approx(0)


def test_statistics_do_not_change_input():
    data = synthetic_series()
    original = data.copy(deep=True)
    descriptive_statistics(data)
    dependence_profile(data["Close"])
    drawdown_series(data)
    pd.testing.assert_frame_equal(original, data)
