"""Sprint 05: pronósticos posteriores al último cierre, sin reentrenamiento con futuros.

Calendario: sesiones NYSE/XNYS, sin inventar cierres de fin de semana o festivos.
Los intervalos ARIMA son condicionales al modelo (no garantía de cobertura).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import warnings

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from tempus.core import normalize_prices
from tempus.forecasting import METHODS

LOG = logging.getLogger(__name__)
HORIZONS = (1, 5, 10, 20)
CONFIDENCE = 0.95


@dataclass(frozen=True)
class FutureConfig:
    """Configuración restringida para reproducir el pronóstico."""

    horizon: int = 5
    confidence: float = CONFIDENCE
    min_train: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or self.horizon not in HORIZONS:
            raise ValueError(f"horizon debe pertenecer a {HORIZONS}")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool) or not 0.8 <= self.confidence <= 0.99:
            raise ValueError("confidence debe estar entre 0.80 y 0.99")
        if isinstance(self.min_train, bool) or not isinstance(self.min_train, int) or self.min_train < 40:
            raise ValueError("min_train debe ser un entero >= 40")


def future_sessions(last_observed: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Devuelve las siguientes sesiones NYSE posteriores a la última fecha.

    Un cierre histórico puede provenir de un snapshot desactualizado; el primer
    día pronosticado se toma después de él, no después de la fecha actual.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError(f"horizon debe pertenecer a {HORIZONS}")
    if not isinstance(last_observed, pd.Timestamp) or pd.isna(last_observed):
        raise TypeError("last_observed debe ser una fecha pandas válida")
    if last_observed.tzinfo is not None:
        last_observed = last_observed.tz_localize(None)
    last = last_observed.normalize()
    # Horizonte máximo 20 sesiones; 100 días calendario cubren feriados razonables.
    end = last + pd.Timedelta(days=100)
    try:
        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(last + pd.Timedelta(days=1), end)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise RuntimeError("No se pudo consultar el calendario oficial XNYS") from exc
    if len(sessions) < horizon:
        raise RuntimeError("El calendario XNYS no contiene suficientes sesiones futuras")
    return pd.DatetimeIndex(sessions[:horizon], name="Date").tz_localize(None)


def _model_forecast(values: np.ndarray, method: str, steps: int, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta una única vez cada modelo al histórico completo."""
    if method == "Ingenuo":
        points = np.full(steps, float(values[-1]), dtype=float)
        missing = np.full(steps, np.nan)
        return points, missing, missing.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if method == "Suavizamiento exponencial":
            fitted = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
            points = np.asarray(fitted.forecast(steps), dtype=float)
            missing = np.full(steps, np.nan)
            return points, missing, missing.copy()
        if method == "ARIMA(1,1,1)":
            fitted = ARIMA(values, order=(1, 1, 1), enforce_stationarity=False,
                           enforce_invertibility=False).fit()
            pred = fitted.get_forecast(steps=steps)
            points = np.asarray(pred.predicted_mean, dtype=float)
            bounds = np.asarray(pred.conf_int(alpha=1.0-confidence), dtype=float)
            if bounds.shape != (steps, 2):
                raise RuntimeError("Intervalos ARIMA inesperados")
            return points, bounds[:, 0], bounds[:, 1]
    raise ValueError(f"Método desconocido: {method}")


def forecast_future(
    frame: pd.DataFrame,
    config: FutureConfig | None = None,
    methods: tuple[str, ...] = METHODS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pronostica sesiones reales posteriores al último dato observado.

    Retorna tabla larga de tres modelos por sesión, intervalos condicionales
    únicamente para ARIMA, y metadatos de trazabilidad. No consulta precios nuevos.
    """
    cfg = FutureConfig() if config is None else config
    if not isinstance(cfg, FutureConfig):
        raise TypeError("config debe ser FutureConfig")
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Selección de métodos inválida")

    clean = normalize_prices(frame)
    if len(clean) < cfg.min_train:
        raise ValueError(f"Se requieren {cfg.min_train} cierres para el pronóstico futuro")
    close = clean["Close"]
    values = close.to_numpy(dtype=float)
    session_dates = future_sessions(pd.Timestamp(close.index[-1]), cfg.horizon)
    records: list[dict[str, object]] = []
    for method in methods:
        try:
            point, low, high = _model_forecast(values, method, cfg.horizon, cfg.confidence)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            LOG.exception("Ajuste futuro falló en modelo %s", method)
            raise RuntimeError(f"No se pudo estimar {method}: {type(exc).__name__}") from exc
        if not np.isfinite(point).all() or (point <= 0).any():
            raise ValueError(f"Predicciones no válidas en {method}")
        if method == "ARIMA(1,1,1)" and (not np.isfinite(low).all() or not np.isfinite(high).all()):
            raise ValueError("Intervalo ARIMA no finito")
        for offset, day in enumerate(session_dates):
            records.append({
                "Date": day,
                "sesion": offset + 1,
                "modelo": method,
                "pronostico": float(point[offset]),
                "limite_inferior": float(low[offset]) if np.isfinite(low[offset]) else np.nan,
                "limite_superior": float(high[offset]) if np.isfinite(high[offset]) else np.nan,
                "intervalo_tipo": f"ARIMA condicional {cfg.confidence:.0%}" if method == "ARIMA(1,1,1)" else "No estimado",
            })
    table = pd.DataFrame.from_records(records)
    table = table.sort_values(["Date", "modelo"], kind="stable").reset_index(drop=True)
    metadata: dict[str, object] = {
        "metodo": "Ajuste con histórico completo; extrapolación posterior al último cierre",
        "config": asdict(cfg),
        "modelos": list(methods),
        "observaciones_entrenamiento": len(clean),
        "ultimo_dato_observado": close.index[-1].date().isoformat(),
        "primer_pronostico": session_dates[0].date().isoformat(),
        "ultimo_pronostico": session_dates[-1].date().isoformat(),
        "ultimo_cierre": float(values[-1]),
        "calendario": "XNYS (New York Stock Exchange)",
        "datos_externos_adicionales": False,
        "intervalos": "Solo ARIMA: intervalo condicional al modelo; no incorpora riesgo de cambios estructurales.",
        "advertencia": "Pronóstico experimental, no recomendación financiera. "
            "No garantiza precisión ni incorpora eventos, noticias o ajustes futuros. "
            "Las fechas se calculan desde el último dato del archivo, que podría estar desactualizado.",
    }
    return table, metadata
