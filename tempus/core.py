"""Validaciones y métricas: transformaciones sin efectos secundarios."""
from __future__ import annotations
from datetime import date
import numpy as np
import pandas as pd

REQUIRED = ("Open", "High", "Low", "Close", "Volume")

def validate_period(start: date, end: date) -> None:
    """Comprueba límites inclusivo/exclusivo para descarga."""
    if not isinstance(start, date) or not isinstance(end, date):
        raise TypeError("start y end deben ser fechas")
    if start >= end:
        raise ValueError("La fecha inicial debe ser anterior a la final")
    if end > date.today() + pd.Timedelta(days=1):
        raise ValueError("No se permiten periodos futuros")

def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida OHLCV, ordena fechas y entrega una copia limpia sin rellenar huecos."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("No se recibieron cotizaciones")
    if not all(col in frame.columns for col in REQUIRED):
        raise ValueError("Faltan columnas OHLCV requeridas")
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("El índice debe contener fechas")
    if result.index.hasnans or result.index.duplicated().any():
        raise ValueError("Fechas ausentes o duplicadas")
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()
    for col in REQUIRED:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if result[list(REQUIRED)].isna().any().any():
        raise ValueError("Existen datos OHLCV ausentes o no numéricos")
    numeric = result[["Open", "High", "Low", "Close"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Existen precios no finitos")
    if (numeric <= 0).any().any():
        raise ValueError("Todos los precios deben ser positivos")
    if (result["Volume"] < 0).any() or not np.isfinite(result["Volume"].to_numpy(dtype=float)).all():
        raise ValueError("Volumen inválido")
    # Yahoo puede devolver discrepancias sub-centavo por el ajuste y redondeo
    # de OHLC. Aceptamos solo errores relativos <= 1e-6; nunca corregimos precios.
    # Una discrepancia material sigue produciendo un error audit-able.
    max_esperado = result[["Open", "Close", "Low"]].max(axis=1)
    tol_high = np.maximum(1e-7, max_esperado.abs() * 1e-6)
    high_error = max_esperado - result["High"]
    high_invalido = high_error > tol_high
    if high_invalido.any():
        fecha = high_invalido[high_invalido].index[0]
        diferencia = high_error.loc[fecha]
        raise ValueError(
            f"High inconsistente el {fecha.date()}: diferencia USD {diferencia:.8f}"
        )

    min_esperado = result[["Open", "Close", "High"]].min(axis=1)
    tol_low = np.maximum(1e-7, min_esperado.abs() * 1e-6)
    low_error = result["Low"] - min_esperado
    low_invalido = low_error > tol_low
    if low_invalido.any():
        fecha = low_invalido[low_invalido].index[0]
        diferencia = low_error.loc[fecha]
        raise ValueError(
            f"Low inconsistente el {fecha.date()}: diferencia USD {diferencia:.8f}"
        )
    result.index.name = "Date"
    return result

def enrich_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores causales; cada fila usa solo el presente y el pasado."""
    result = normalize_prices(frame)
    close = result["Close"]
    ret = close.pct_change(fill_method=None)
    result["return_1d"] = ret
    result["return_lag_1"] = ret.shift(1)
    result["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    result["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    result["volatility_20"] = ret.rolling(20, min_periods=20).std()
    result["normalized_100"] = 100 * close / close.iloc[0]
    return result

def summarize(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Calcula un resumen descriptivo sobre datos validados."""
    data = enrich_prices(frame)
    ret = data["return_1d"].dropna()
    return {
        "observaciones": len(data),
        "inicio": data.index.min().date().isoformat(),
        "fin": data.index.max().date().isoformat(),
        "cierre_final": float(data["Close"].iloc[-1]),
        "rendimiento_acumulado_pct": float((data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100),
        "volatilidad_anual_pct": float(ret.std() * np.sqrt(252) * 100) if len(ret) >= 2 else float("nan"),
    }
