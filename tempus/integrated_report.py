"""Informe integral del Módulo I: evaluación histórica y cinco pronósticos futuros.

Consume exclusivamente evidencia exportada; nunca descarga ni reentrena modelos.
Valida coherencia con el histórico para evitar mezclar snapshots diferentes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

from tempus.assets import ASSETS
from tempus.forecasting import METHODS
from tempus.future import future_sessions
from tempus.future_report import create_future_appendix
from tempus.comparative_report import create_comparative_appendix
from tempus.report_pdf import create_academic_pdf
from tempus.reporting import Evidence

LOG = logging.getLogger(__name__)
REQUIRED_TICKERS = tuple(ASSETS)
REQUIRED_HORIZONS = (1, 5, 10, 20)


def load_future_exports(
    directory: Path,
    evidence: Evidence,
) -> dict[str, tuple[pd.DataFrame, dict[str, object]]]:
    """Valida CSV/JSON de los cinco activos contra la evidencia histórica."""
    base = Path(directory).resolve(strict=True)
    if not base.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {base}")
    if set(evidence.indicators) != set(REQUIRED_TICKERS):
        raise ValueError("Faltan CSV de indicadores históricos para validar pronósticos")

    results: dict[str, tuple[pd.DataFrame, dict[str, object]]] = {}
    for ticker in REQUIRED_TICKERS:
        csv_path = base / f"tempus_{ticker}_futuro.csv"
        json_path = base / f"tempus_{ticker}_futuro.json"
        for path in (csv_path, json_path):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"Falta archivo o no es regular: {path}")

        raw = pd.read_csv(csv_path)
        required = {
            "Date", "sesion", "modelo", "pronostico",
            "limite_inferior", "limite_superior", "intervalo_tipo",
        }
        if not required.issubset(raw.columns):
            raise ValueError(f"{ticker}: CSV futuro sin columnas requeridas")

        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{ticker}: JSON de trazabilidad incorrecto") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"{ticker}: metadatos no son un objeto")

        history = evidence.indicators[ticker].copy()
        history["Date"] = pd.to_datetime(history["Date"], errors="raise")
        history = history.sort_values("Date")
        if history["Date"].isna().any() or not history["Date"].is_unique:
            raise ValueError(f"{ticker}: fechas del histórico inválidas")
        last_date = pd.Timestamp(history["Date"].iloc[-1]).normalize()
        last_close = float(history["Close"].iloc[-1])
        if not np.isfinite(last_close) or last_close <= 0:
            raise ValueError(f"{ticker}: último precio del histórico inválido")

        if meta.get("ultimo_dato_observado") != last_date.date().isoformat():
            raise ValueError(f"{ticker}: el pronóstico corresponde a otro snapshot (fecha)")
        if not np.isclose(float(meta.get("ultimo_cierre", np.nan)), last_close, rtol=0, atol=0.02):
            raise ValueError(f"{ticker}: el pronóstico corresponde a otro snapshot (precio)")
        if meta.get("calendario") != "XNYS (New York Stock Exchange)":
            raise ValueError(f"{ticker}: calendario bursátil no reconocido")
        if int(meta.get("observaciones_entrenamiento", -1)) != len(history):
            raise ValueError(f"{ticker}: longitud de entrenamiento inconsistente")
        if meta.get("config", {}).get("confidence") != 0.95:
            raise ValueError(f"{ticker}: nivel de confianza inesperado")

        raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
        if raw["Date"].isna().any() or raw.duplicated(["Date", "modelo"]).any():
            raise ValueError(f"{ticker}: fechas o predicciones duplicadas")
        steps = int(raw["sesion"].max())
        if steps not in REQUIRED_HORIZONS or steps != int(meta["config"]["horizon"]):
            raise ValueError(f"{ticker}: horizonte incompatible con CSV/JSON")
        if steps < 20:
            raise ValueError(
                f"{ticker}: se requieren 20 sesiones para incluir horizontes 1,5,10,20")
        if len(raw) != steps * len(METHODS):
            raise ValueError(f"{ticker}: cantidad de pronósticos inconsistente")
        expected_days = future_sessions(last_date, steps)
        for model in METHODS:
            rows = raw[raw["modelo"] == model].sort_values("sesion")
            if len(rows) != steps or rows["sesion"].tolist() != list(range(1, steps + 1)):
                raise ValueError(f"{ticker}: faltan sesiones del modelo {model}")
            if not pd.DatetimeIndex(rows["Date"]).equals(pd.DatetimeIndex(expected_days)):
                raise ValueError(f"{ticker}: fechas bursátiles inválidas para {model}")
            points = pd.to_numeric(rows["pronostico"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(points).all() or (points <= 0).any():
                raise ValueError(f"{ticker}: pronósticos no finitos o negativos")
            lower = pd.to_numeric(rows["limite_inferior"], errors="coerce").to_numpy(float)
            upper = pd.to_numeric(rows["limite_superior"], errors="coerce").to_numpy(float)
            if model == "ARIMA(1,1,1)":
                if not (np.isfinite(lower).all() and np.isfinite(upper).all()
                        and (lower < points).all() and (points < upper).all()):
                    raise ValueError(f"{ticker}: intervalos ARIMA no válidos")
            elif np.isfinite(lower).any() or np.isfinite(upper).any():
                raise ValueError(f"{ticker}: intervalos no esperados para {model}")
        if set(raw["modelo"]) != set(METHODS):
            raise ValueError(f"{ticker}: conjunto de modelos incompleto")
        if meta.get("primer_pronostico") != expected_days[0].date().isoformat():
            raise ValueError(f"{ticker}: primera fecha futura inconsistente")
        if meta.get("ultimo_pronostico") != expected_days[-1].date().isoformat():
            raise ValueError(f"{ticker}: última fecha futura inconsistente")
        results[ticker] = (raw.sort_values(["Date", "modelo"]).reset_index(drop=True), meta)
        LOG.info("Evidencia futura verificada: %s, %s observaciones", ticker, len(raw))
    return results


def create_integrated_pdf(
    evidence: Evidence,
    futures: Mapping[str, tuple[pd.DataFrame, dict[str, object]]],
    destination: Path,
    overwrite: bool = False,
) -> Path:
    """Construye un PDF único: informe histórico + cinco gráficas de futuro.

    No sobrescribe. Genera páginas intermedias en una carpeta temporal.
    """
    if set(futures) != set(REQUIRED_TICKERS):
        raise ValueError("Se requieren los cinco activos para el informe integral")
    target = Path(destination).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Ya existe el informe: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="tempus_reporte_") as folder:
        scratch = Path(folder)
        historical = create_academic_pdf(
            evidence, scratch / "historico.pdf")
        comparative = create_comparative_appendix(
            evidence, scratch / "comparativo.pdf")
        appendix = create_future_appendix(
            futures, scratch / "pronosticos.pdf",
            historical=evidence.indicators)
        writer = PdfWriter()
        writer.append(str(historical))
        writer.append(str(comparative))
        writer.append(str(appendix))
        writer.add_metadata({
            "/Title": "TEMPUS EXPLORATIO - Informe integral de diez años",
            "/Subject": "Practica 1 - Historico, evaluacion y pronosticos futuros de cinco empresas",
            "/Author": "TEMPUS DECISIO",
        })
        temporary_pdf = target.with_name(target.name + ".partial")
        try:
            with temporary_pdf.open("xb") as output:
                writer.write(output)
            # Comprobación de integridad antes de publicar el PDF.
            verify = PdfReader(str(temporary_pdf))
            if len(verify.pages) < 10:
                raise RuntimeError("Informe integral incompleto")
            temporary_pdf.replace(target)
        except Exception:
            temporary_pdf.unlink(missing_ok=True)
            raise
        finally:
            writer.close()
    LOG.info("Informe integral creado: %s (%s bytes)", target, target.stat().st_size)
    return target
