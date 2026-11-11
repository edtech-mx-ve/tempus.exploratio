"""TEMPUS EXPLORATIO · Sprint 03: pronósticos evaluados cronológicamente.

Métodos: persistencia (naive), suavizamiento exponencial simple, ARIMA.
La validación walk-forward reentrena utilizando exclusivamente datos conocidos
en cada origen. Las fechas sin negociación no se interpolan.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from tempus.core import normalize_prices

LOG = logging.getLogger(__name__)
METHODS = ("Ingenuo", "Suavizamiento exponencial", "ARIMA(1,1,1)")


@dataclass(frozen=True)
class ForecastConfig:
    """Parámetros reproducibles de evaluación de una sesión bursátil."""

    test_fraction: float = 0.20
    step: int = 10
    min_train: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.test_fraction, (int, float)) or not 0.05 <= self.test_fraction <= 0.40:
            raise ValueError("test_fraction debe estar entre 0.05 y 0.40")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or not 1 <= self.step <= 30:
            raise ValueError("step debe ser un entero entre 1 y 30")
        if isinstance(self.min_train, bool) or not isinstance(self.min_train, int) or self.min_train < 40:
            raise ValueError("min_train debe ser al menos 40")


def forecast_one_step(history: pd.Series, method: str) -> float:
    """Pronostica el siguiente cierre usando solo el histórico recibido.

    La estimación no utiliza la fila que se intenta predecir.
    """
    if method not in METHODS:
        raise ValueError(f"Método desconocido: {method}")
    if not isinstance(history, pd.Series):
        raise TypeError("history debe ser una Series")
    if len(history) < 40:
        raise ValueError("Se requieren al menos 40 cierres")
    values = pd.to_numeric(history, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Los precios deben ser positivos y finitos")
    if method == "Ingenuo":
        return float(values[-1])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if method == "Suavizamiento exponencial":
            model = SimpleExpSmoothing(values, initialization_method="estimated")
            fitted = model.fit(optimized=True)
        else:
            fitted = ARIMA(values, order=(1, 1, 1), enforce_stationarity=False,
                           enforce_invertibility=False).fit()
        result = float(np.asarray(fitted.forecast(steps=1), dtype=float)[0])
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"Pronóstico no válido del método {method}")
    return result


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """MAE, RMSE y sMAPE en una escala común y sobre pares válidos."""
    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("Se esperan dos Series")
    if not actual.index.equals(predicted.index) or len(actual) == 0:
        raise ValueError("Las fechas de observado y predicho deben coincidir y no estar vacías")
    a = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Se encontraron valores no finitos")
    err = a - p
    denom = np.abs(a) + np.abs(p)
    smape = np.divide(2 * np.abs(err), denom, out=np.zeros_like(denom), where=denom > 0)
    return {
        "n": int(len(a)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "sMAPE_pct": float(100 * np.mean(smape)),
    }


def walk_forward_evaluation(
    frame: pd.DataFrame,
    config: ForecastConfig | None = None,
    methods: tuple[str, ...] = METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evalúa un paso adelante en fechas fijas del segmento final.

    Cada origen se reentrena hasta t-1 para pronosticar t. El parámetro step
    controla cuántas fechas se omiten entre orígenes (no el horizonte).
    Si algún ajuste falla se registra y ese método no recibe una métrica
    artificial: solo se comparan fechas completas comunes a todos.
    """
    cfg = config if config is not None else ForecastConfig()
    if not isinstance(cfg, ForecastConfig):
        raise TypeError("config debe ser ForecastConfig")
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("La selección de métodos es inválida")
    clean = normalize_prices(frame)
    close = clean["Close"]
    cut = int(np.floor(len(close) * (1 - cfg.test_fraction)))
    if cut < cfg.min_train or len(close) - cut < 5:
        raise ValueError("Serie insuficiente para entrenar y evaluar")
    observations: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    for idx in range(cut, len(close), cfg.step):
        history = close.iloc[:idx]
        record: dict[str, object] = {
            "Date": close.index[idx],
            "observado": float(close.iloc[idx]),
            "ultimo_conocido": float(history.iloc[-1]),
            "entrenamiento_n": len(history),
        }
        for method in methods:
            try:
                record[method] = forecast_one_step(history, method)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                LOG.warning("Falló %s en %s: %s", method, close.index[idx].date(), type(exc).__name__)
                record[method] = np.nan
                failed.append({"fecha": str(close.index[idx].date()), "modelo": method,
                               "tipo_error": type(exc).__name__})
        observations.append(record)

    predictions = pd.DataFrame.from_records(observations).set_index("Date")
    predictions.index.name = "Date"
    valid = predictions.dropna(subset=list(methods))
    if valid.empty:
        raise RuntimeError("Ninguna fecha contiene pronósticos completos comparables")
    scores = []
    for method in methods:
        item = {"modelo": method}
        item.update(error_metrics(valid["observado"], valid[method]))
        scores.append(item)
    score_frame = pd.DataFrame(scores).sort_values(["MAE", "RMSE"], kind="stable").reset_index(drop=True)
    metadata: dict[str, object] = {
        "config": asdict(cfg),
        "horizonte_sesiones": 1,
        "periodo_entrenamiento_inicio": str(close.index[0].date()),
        "periodo_entrenamiento_fin_inicial": str(close.index[cut - 1].date()),
        "periodo_prueba_inicio": str(close.index[cut].date()),
        "periodo_prueba_fin": str(close.index[-1].date()),
        "evaluaciones_programadas": len(predictions),
        "evaluaciones_comunes": len(valid),
        "fallos": failed,
        "nota": "Evaluación histórica, no garantiza resultados futuros. "
                "Cada origen pronostica la próxima sesión; solo se evalúa uno de cada step orígenes.",
    }
    return predictions, score_frame, metadata
