"""Consolidación verificable de resultados TEMPUS EXPLORATIO.

La lectura de archivos y la escritura del reporte están separadas de las
transformaciones puras. No ejecuta descargas ni ajustes de modelos.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from tempus.assets import ASSETS
from tempus.forecasting import METHODS, error_metrics

LOG = logging.getLogger(__name__)
N_EXPECTED = 26
COLUMNS = ("Date", "observado", "ultimo_conocido", "entrenamiento_n", *METHODS)
METRICS = ("MAE", "RMSE", "sMAPE_pct")


@dataclass(frozen=True)
class Evidence:
    """Resultado inmutable de validación; los DataFrame se usan solo como lectura."""
    metrics: pd.DataFrame
    forecasts: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, dict[str, object]]
    source_hashes: Mapping[str, str]
    panel: pd.DataFrame
    correlations: pd.DataFrame
    indicators: Mapping[str, pd.DataFrame]


def safe_input_path(directory: Path, filename: str) -> Path:
    """Construye una ruta sin aceptar nombres arbitrarios ni enlaces externos."""
    if Path(filename).name != filename or filename.startswith("."):
        raise ValueError("Nombre de evidencia inválido")
    base = Path(directory).resolve(strict=True)
    target = base / filename
    if not target.is_file() or target.is_symlink() or target.resolve().parent != base:
        raise FileNotFoundError(f"Archivo regular no disponible: {filename}")
    if target.stat().st_size > 20_000_000:
        raise ValueError(f"Archivo excesivo: {filename}")
    return target


def validate_forecasts(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Valida fechas, valores y correspondencia de predicción ingenua."""
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError(f"{ticker}: faltan columnas de pronóstico")
    result = frame[list(COLUMNS)].copy()
    result["Date"] = pd.to_datetime(result["Date"], format="%Y-%m-%d", errors="raise")
    if result["Date"].isna().any() or result["Date"].duplicated().any():
        raise ValueError(f"{ticker}: fechas inválidas o duplicadas")
    if not result["Date"].is_monotonic_increasing or len(result) < 5:
        raise ValueError(f"{ticker}: cronología insuficiente")
    numeric = list(COLUMNS[1:])
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="raise")
    values = result[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{ticker}: valores ausentes/no finitos")
    if (result[list(METHODS) + ["observado", "ultimo_conocido"]] <= 0).any().any():
        raise ValueError(f"{ticker}: los precios deben ser positivos")
    if (result["entrenamiento_n"] < 100).any() or not result["entrenamiento_n"].is_monotonic_increasing:
        raise ValueError(f"{ticker}: entrenamiento inválido")
    if not np.allclose(result["Ingenuo"], result["ultimo_conocido"], atol=1e-7, rtol=0):
        raise ValueError(f"{ticker}: la línea base ingenua no coincide con último cierre")
    return result.set_index("Date")


def validate_scores(frame: pd.DataFrame, forecasts: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Recalcula métricas a partir de predicciones, sin confiar en la tabla recibida."""
    expected = {"modelo", "n", *METRICS}
    if not expected.issubset(frame.columns) or len(frame) != len(METHODS):
        raise ValueError(f"{ticker}: tabla de métricas incompleta")
    if set(frame["modelo"]) != set(METHODS) or frame["modelo"].duplicated().any():
        raise ValueError(f"{ticker}: nombres de modelos inesperados")
    original = frame.copy()
    for method in METHODS:
        row = original.loc[original["modelo"] == method].iloc[0]
        recalculated = error_metrics(forecasts["observado"], forecasts[method])
        if int(row["n"]) != recalculated["n"]:
            raise ValueError(f"{ticker}: n no corresponde con las predicciones")
        for metric in METRICS:
            observed = float(row[metric])
            if not np.isfinite(observed) or not np.isclose(
                observed, float(recalculated[metric]), rtol=1e-8, atol=1e-8
            ):
                raise ValueError(f"{ticker}: {metric} inconsistente para {method}")
    return original


def validate_metadata(meta: dict[str, object], forecasts: pd.DataFrame, ticker: str) -> None:
    """Comprueba cobertura de fechas y comparabilidad de la configuración."""
    cfg = meta.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"{ticker}: configuración faltante")
    if cfg.get("test_fraction") != 0.2 or cfg.get("step") != 10 or cfg.get("min_train") != 100:
        raise ValueError(f"{ticker}: configuración distinta a la evaluación de referencia")
    if meta.get("horizonte_sesiones") != 1:
        raise ValueError(f"{ticker}: horizonte incompatible")
    if meta.get("evaluaciones_comunes") != len(forecasts):
        raise ValueError(f"{ticker}: cantidad de fechas incompatible")
    if meta.get("evaluaciones_programadas") != len(forecasts):
        raise ValueError(f"{ticker}: faltan evaluaciones")
    if meta.get("fallos") != []:
        raise ValueError(f"{ticker}: la evaluación registra fallos")
    if meta.get("periodo_prueba_inicio") != str(forecasts.index[0].date()):
        raise ValueError(f"{ticker}: inicio de prueba inesperado")
    if meta.get("periodo_prueba_fin") < str(forecasts.index[-1].date()):
        raise ValueError(f"{ticker}: cierre de prueba anterior al último pronóstico")


def load_evidence(directory: Path, expected_tickers: tuple[str, ...] = tuple(ASSETS)) -> Evidence:
    """Carga los CSV/JSON exportados y comprueba su integridad matemática."""
    if not expected_tickers or len(set(expected_tickers)) != len(expected_tickers):
        raise ValueError("Selección de activos vacía o duplicada")
    if any(t not in ASSETS for t in expected_tickers):
        raise ValueError("Activo desconocido")
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"Directorio inexistente: {base}")
    forecasts: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    tables: list[pd.DataFrame] = []
    indicators: dict[str, pd.DataFrame] = {}
    for ticker in expected_tickers:
        files = {
            kind: safe_input_path(base, f"tempus_{ticker}_{kind}.{ext}")
            for kind, ext in [("metricas", "csv"), ("pronosticos", "csv"), ("evaluacion", "json")]
        }
        for source in files.values():
            hashes[source.name] = sha256(source.read_bytes()).hexdigest()
        preds = validate_forecasts(pd.read_csv(files["pronosticos"]), ticker)
        scores = validate_scores(pd.read_csv(files["metricas"]), preds, ticker)
        with files["evaluacion"].open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not isinstance(meta, dict):
            raise ValueError(f"{ticker}: JSON de evaluación inválido")
        validate_metadata(meta, preds, ticker)
        forecasts[ticker] = preds
        metadata[ticker] = meta
        scores.insert(0, "ticker", ticker)
        tables.append(scores)
        indicator_filename = f"tempus_{ticker}_indicadores.csv"
        if (base / indicator_filename).is_file():
            indicator_path = safe_input_path(base, indicator_filename)
            hashes[indicator_filename] = sha256(indicator_path.read_bytes()).hexdigest()
            ind = pd.read_csv(indicator_path)
            if not {"Date", "Close", "return_1d"}.issubset(ind.columns):
                raise ValueError(f"{ticker}: indicadores incompletos")
            indicators[ticker] = ind.copy()

    first_dates = next(iter(forecasts.values())).index
    if any(not first_dates.equals(x.index) for x in forecasts.values()):
        raise ValueError("Las empresas no fueron evaluadas en las mismas fechas")
    if any(x["entrenamiento_n"].tolist() != next(iter(forecasts.values()))["entrenamiento_n"].tolist()
           for x in forecasts.values()):
        raise ValueError("Las ventanas de entrenamiento no coinciden")
    panel_path = safe_input_path(base, "tempus_series_base100.csv")
    corr_path = safe_input_path(base, "tempus_correlaciones.csv")
    for source in (panel_path, corr_path):
        hashes[source.name] = sha256(source.read_bytes()).hexdigest()
    panel = pd.read_csv(panel_path, parse_dates=["Date"]).set_index("Date")
    if list(panel.columns) != list(expected_tickers) or len(panel) < 100:
        raise ValueError("Panel de series incompleto")
    if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Panel sin orden cronológico")
    if not np.isfinite(panel.to_numpy(dtype=float)).all() or (panel <= 0).any().any():
        raise ValueError("Panel de precios con valores inválidos")
    if not np.allclose(panel.iloc[0], 100.0, atol=1e-6):
        raise ValueError("Las series deben iniciar con índice 100")
    correlations = pd.read_csv(corr_path, index_col=0)
    if list(correlations.columns) != list(expected_tickers) or list(correlations.index) != list(expected_tickers):
        raise ValueError("Matriz de correlaciones no coincide con activos")
    array = correlations.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.allclose(array, array.T, atol=1e-10):
        raise ValueError("Matriz de correlaciones inválida")
    if not np.allclose(np.diag(array), 1, atol=1e-10) or (abs(array) > 1.0000001).any():
        raise ValueError("Coeficientes de correlación fuera de rango")
    returns = panel.pct_change().dropna()
    verified = returns.corr().to_numpy()
    if not np.allclose(array, verified, atol=1e-7, rtol=0):
        raise ValueError("Correlaciones incoherentes con el panel")
    LOG.info("Evidencia comprobada: %d activos, %d predicciones por activo",
             len(expected_tickers), len(first_dates))
    return Evidence(
        metrics=pd.concat(tables, ignore_index=True), forecasts=forecasts,
        metadata=metadata, source_hashes=hashes, panel=panel,
        correlations=correlations, indicators=indicators
    )


def best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """Selecciona por MAE sin mezclar monedas ni escalas entre activos."""
    if metrics.empty:
        raise ValueError("Se necesita una tabla de métricas no vacía")
    data = metrics.sort_values(["ticker", "MAE", "RMSE", "modelo"], kind="stable")
    return data.groupby("ticker", sort=False).head(1).reset_index(drop=True)


def metadata_for_export(evidence: Evidence) -> dict[str, object]:
    """Contrato JSON desacoplado del reporte, listo para el módulo de IA."""
    return {
        "schema_version": "1.0.0",
        "producto": "TEMPUS DECISIO",
        "modulo_origen": "TEMPUS EXPLORATIO",
        "horizonte_sesiones": 1,
        "test_fraction": 0.20,
        "evaluation_step": 10,
        "metrics": list(METRICS),
        "tickers": list(evidence.forecasts),
        "periodo": {
            "primera_fecha": str(evidence.panel.index.min().date()),
            "ultima_fecha": str(evidence.panel.index.max().date()),
        },
        "source_sha256": dict(evidence.source_hashes),
        "limitaciones": [
            "Las 26 evaluaciones por activo no cubren cada sesión bursátil.",
            "Sin prueba de superioridad estadística entre modelos.",
            "No se incluyen recomendaciones de inversión.",
            "Los modelos de IA deben construir sus variables con información conocida hasta cada origen."
        ],
    }


def export_contract(evidence: Evidence, directory: Path, overwrite: bool = False) -> list[Path]:
    """Escribe contrato y datos validados; jamás borra datos externos."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, bytes] = {}
    outputs["tempus_exploratio_metadata.json"] = json.dumps(
        metadata_for_export(evidence), ensure_ascii=False, indent=2).encode("utf-8")
    outputs["tempus_exploratio_metricas.csv"] = evidence.metrics.to_csv(index=False).encode("utf-8")
    outputs["tempus_exploratio_base100.csv"] = evidence.panel.to_csv().encode("utf-8")
    for ticker, df in evidence.forecasts.items():
        outputs[f"tempus_exploratio_{ticker}_pronosticos.csv"] = df.to_csv().encode("utf-8")
    for filename in outputs:
        dest = target / filename
        if dest.exists() and not overwrite:
            raise FileExistsError(f"No se reemplaza archivo existente: {dest}")
    paths = []
    for filename, payload in outputs.items():
        dest = target / filename
        if not overwrite:
            with dest.open("xb") as handle:
                handle.write(payload)
        else:
            dest.write_bytes(payload)
        paths.append(dest)
    LOG.info("Contrato exportado: %d archivos en %s", len(paths), target)
    return paths
