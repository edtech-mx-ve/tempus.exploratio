"""Pruebas: Sprint 05, extrapolación sin fuga y calendario bursátil."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempus.future import FutureConfig, forecast_future, future_sessions, HORIZONS
from tempus.future_report import create_future_appendix


@pytest.fixture()
def prices() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=160, name="Date")
    close = 100 + np.arange(160) * 0.12 + np.sin(np.arange(160) / 11)
    return pd.DataFrame({
        "Open": close - 0.15, "High": close + 0.4,
        "Low": close - 0.4, "Close": close,
        "Volume": np.full(160, 10000),
    }, index=dates)


def test_config_validations() -> None:
    for h in HORIZONS:
        assert FutureConfig(horizon=h).horizon == h
    with pytest.raises(ValueError):
        FutureConfig(horizon=3)
    with pytest.raises(ValueError):
        FutureConfig(horizon=True)
    with pytest.raises(ValueError):
        FutureConfig(min_train=20)


def test_market_sessions_skip_holidays() -> None:
    # Thanksgiving 2026 falls on Thursday 26 November.
    days = future_sessions(pd.Timestamp("2026-11-25"), 5)
    expected = pd.DatetimeIndex(["2026-11-27","2026-11-30","2026-12-01",
                                 "2026-12-02","2026-12-03"], name="Date")
    pd.testing.assert_index_equal(days, expected)


def test_forecast_future_all_models_and_intervals(prices: pd.DataFrame) -> None:
    before = prices.copy(deep=True)
    table, metadata = forecast_future(prices, FutureConfig(horizon=5))
    assert len(table) == 15
    assert table["Date"].nunique() == 5
    assert table["Date"].min() > prices.index[-1]
    assert metadata["ultimo_dato_observado"] == str(prices.index[-1].date())
    assert metadata["observaciones_entrenamiento"] == 160
    assert (table["pronostico"] > 0).all()
    assert np.isfinite(table["pronostico"]).all()
    for name in ("Ingenuo", "Suavizamiento exponencial"):
        vals = table[table["modelo"] == name]
        assert vals["limite_inferior"].isna().all()
        assert vals["limite_superior"].isna().all()
    arima = table[table["modelo"] == "ARIMA(1,1,1)"]
    assert np.isfinite(arima["limite_inferior"]).all()
    assert (arima["limite_inferior"] < arima["pronostico"]).all()
    assert (arima["limite_superior"] > arima["pronostico"]).all()
    pd.testing.assert_frame_equal(before, prices)


def test_naive_constant_and_one_session(prices: pd.DataFrame) -> None:
    table, _ = forecast_future(prices, FutureConfig(horizon=20),
                               methods=("Ingenuo",))
    assert len(table) == 20
    assert (table["pronostico"] == prices["Close"].iloc[-1]).all()
    assert table["Date"].is_unique


def test_invalid_input(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        forecast_future(prices.iloc[:35])
    with pytest.raises(ValueError):
        forecast_future(prices, methods=("otro",))
    with pytest.raises(ValueError):
        forecast_future(prices, methods=("Ingenuo", "Ingenuo"))
    changed = prices.copy()
    changed.loc[changed.index[-1], "Close"] = np.nan
    with pytest.raises(ValueError):
        forecast_future(changed)


def test_future_appendix_requires_five(tmp_path) -> None:
    with pytest.raises(ValueError):
        create_future_appendix({}, tmp_path / "no.pdf")


def test_future_cli_existing_output(tmp_path, prices: pd.DataFrame) -> None:
    """La ejecución por lotes jamás reemplaza un directorio preexistente."""
    from future_cli import generate_future_bundle
    src = tmp_path / "original"
    src.mkdir()
    out = tmp_path / "results"
    out.mkdir()
    (out / "keep.txt").write_text("no modificar", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_future_bundle("evidence", src, out)
    assert (out / "keep.txt").read_text(encoding="utf-8") == "no modificar"


def test_future_cli_rejects_unexpected_source(tmp_path) -> None:
    from future_cli import generate_future_bundle
    with pytest.raises(ValueError):
        generate_future_bundle("http", tmp_path, tmp_path / "salida")
