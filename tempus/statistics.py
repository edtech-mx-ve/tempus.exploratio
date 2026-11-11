"""Herramientas estadísticas puras para TEMPUS EXPLORATIO, Sprint 02.

Las pruebas ADF/KPSS son diagnósticos exploratorios, no certifican
estacionariedad por sí mismas. Todos los cálculos preservan la cronología.
"""
from __future__ import annotations

import logging
import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from tempus.core import enrich_prices, normalize_prices

LOG = logging.getLogger(__name__)


def descriptive_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Resume rendimientos diarios y calidad de la observación temporal."""
    clean = enrich_prices(frame)
    returns = clean["return_1d"].dropna()
    if len(returns) < 3:
        raise ValueError("Se requieren al menos cuatro precios para estadística descriptiva")
    return {
        "observaciones": int(len(clean)),
        "rendimientos_validos": int(len(returns)),
        "media_diaria_pct": float(returns.mean() * 100),
        "mediana_diaria_pct": float(returns.median() * 100),
        "desviacion_diaria_pct": float(returns.std(ddof=1) * 100),
        "volatilidad_anual_pct": float(returns.std(ddof=1) * np.sqrt(252) * 100),
        "asimetria": float(returns.skew()),
        "curtosis_exceso": float(returns.kurt()),
        "minimo_diario_pct": float(returns.min() * 100),
        "maximo_diario_pct": float(returns.max() * 100),
        "rendimiento_acumulado_pct": float((clean["Close"].iloc[-1] / clean["Close"].iloc[0] - 1) * 100),
    }


def stationarity_diagnostics(
    values: pd.Series, significance: float = 0.05
) -> dict[str, Any]:
    """Ejecuta ADF y KPSS sobre una serie finita y no constante.

    ADF: H0 = raíz unitaria. KPSS: H0 = estacionariedad alrededor de una constante.
    Los resultados de ambas pruebas deben interpretarse conjuntamente.
    """
    if not 0 < significance < 1:
        raise ValueError("significance debe estar entre 0 y 1")
    if not isinstance(values, pd.Series):
        raise TypeError("Se espera una Serie de pandas")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 30:
        raise ValueError("Se requieren al menos 30 observaciones válidas")
    array = numeric.to_numpy(dtype=float)
    if not np.isfinite(array).all() or np.isclose(array.std(), 0):
        raise ValueError("Serie no finita o aproximadamente constante")
    try:
        adf_args: dict[str, Any] = {"regression": "c", "autolag": "AIC"}
        # Statsmodels 0.15 incorpora result_object y cambia su valor por defecto
        # en una versión futura. En 0.14 esta opción aún no existe.
        if "result_object" in inspect.signature(adfuller).parameters:
            adf_args["result_object"] = False
        adf_stat, adf_p, adf_lags, adf_obs, *_ = adfuller(array, **adf_args)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            kpss_stat, kpss_p, kpss_lags, _ = kpss(array, regression="c", nlags="auto")
        notices = [str(w.message) for w in recorded]
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        LOG.warning("No se pudo completar ADF/KPSS: %s", type(exc).__name__)
        raise ValueError("No fue posible estimar las pruebas de estacionariedad") from exc

    if adf_p < significance and kpss_p >= significance:
        interpretation = "Evidencia compatible con estacionariedad en nivel"
    elif adf_p >= significance and kpss_p < significance:
        interpretation = "Evidencia compatible con no estacionariedad en nivel"
    else:
        interpretation = "Diagnósticos inconclusos o contradictorios"
    return {
        "n": int(len(numeric)),
        "alpha": significance,
        "adf_estadistico": float(adf_stat),
        "adf_p": float(adf_p),
        "adf_rezagos": int(adf_lags),
        "adf_observaciones": int(adf_obs),
        "kpss_estadistico": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_rezagos": int(kpss_lags),
        "interpretacion": interpretation,
        "advertencias": notices,
    }


def dependence_profile(values: pd.Series, max_lags: int = 25) -> pd.DataFrame:
    """Calcula ACF y PACF sobre serie finita con rezagos acotados."""
    if not isinstance(values, pd.Series):
        raise TypeError("Se espera una Serie de pandas")
    if not isinstance(max_lags, int) or isinstance(max_lags, bool) or max_lags < 1:
        raise ValueError("max_lags debe ser un entero positivo")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    array = numeric.to_numpy(dtype=float)
    if len(array) < 12 or not np.isfinite(array).all() or np.isclose(array.std(), 0):
        raise ValueError("Se requieren al menos doce datos finitos no constantes")
    safe_lags = min(max_lags, len(array) // 2 - 1)
    if safe_lags < 1:
        raise ValueError("Serie demasiado corta")
    return pd.DataFrame({
        "rezago": np.arange(safe_lags + 1, dtype=int),
        "ACF": acf(array, nlags=safe_lags, fft=True, missing="raise"),
        "PACF": pacf(array, nlags=safe_lags, method="ywm"),
    })


def align_returns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Alinea rendimientos por fecha sin interpolar sesiones bursátiles ausentes."""
    if not isinstance(frames, dict) or len(frames) < 2:
        raise ValueError("Se requieren al menos dos series")
    collected: dict[str, pd.Series] = {}
    for ticker, frame in frames.items():
        if not isinstance(ticker, str) or not ticker or ticker in collected:
            raise ValueError("Ticker inválido")
        enriched = enrich_prices(frame)
        collected[ticker] = enriched["return_1d"].rename(ticker)
    aligned = pd.concat(collected, axis=1, join="inner").dropna(how="any").sort_index()
    if len(aligned) < 30:
        raise ValueError("Menos de treinta rendimientos comunes")
    return aligned


def correlation_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Correlación de Pearson entre rendimientos contemporáneos comunes."""
    return align_returns(frames).corr(method="pearson")


def normalized_price_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Normaliza a 100 desde la primera fecha común a todos los activos."""
    if not isinstance(frames, dict) or len(frames) < 2:
        raise ValueError("Se requieren al menos dos series")
    closes = pd.concat(
        {ticker: normalize_prices(frame)["Close"] for ticker, frame in frames.items()},
        axis=1, join="inner",
    ).dropna(how="any").sort_index()
    if len(closes) < 2:
        raise ValueError("No hay suficientes fechas comunes")
    return closes.div(closes.iloc[0]).mul(100)


def drawdown_series(frame: pd.DataFrame) -> pd.Series:
    """Calcula caída relativa desde máximo histórico observado hasta cada fecha."""
    close = normalize_prices(frame)["Close"]
    return (close / close.cummax() - 1).rename("drawdown")
