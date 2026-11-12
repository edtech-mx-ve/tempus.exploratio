"""Selección y validación de periodos históricos de 5 y 10 años.

Las fechas de fin son inclusivas para el usuario; yfinance recibe fin exclusivo.
Este módulo no hace peticiones web ni modifica archivos.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import pandas as pd

from tempus.io import list_snapshots, load_snapshot

CUTOFF = date(2026, 10, 30)
PERIODS = (5, 10)


def period_start(years: int, end: date = CUTOFF) -> date:
    """Obtiene la fecha de inicio al retroceder un número de años naturales."""
    if type(years) is not int or years not in PERIODS:
        raise ValueError("El periodo debe ser de 5 o 10 años")
    if not isinstance(end, date):
        raise TypeError("La fecha final debe ser date")
    return (pd.Timestamp(end) - pd.DateOffset(years=years)).date()


def covers_period(index: pd.DatetimeIndex, start: date, end: date,
                  tolerance_days: int = 7) -> bool:
    """Verifica cobertura aproximada de los extremos, respetando festivos bursátiles."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return False
    if not isinstance(start, date) or not isinstance(end, date) or start >= end:
        return False
    if type(tolerance_days) is not int or not 0 <= tolerance_days <= 10:
        raise ValueError("Tolerancia de cobertura inválida")
    return (index.min().date() <= start + pd.Timedelta(days=tolerance_days)
            and index.max().date() >= end - pd.Timedelta(days=tolerance_days))


def load_covered_snapshot(ticker: str, directory: Path,
                          start: date, end: date) -> tuple[pd.DataFrame, Path] | None:
    """Selecciona un snapshot íntegro que cubra el periodo solicitado.

    Ignora versiones cuya metadata no cubre la ventana, pero valida hash al cargar.
    Un fallo de integridad no se silencia: el usuario debe conocerlo.
    """
    for path in list_snapshots(ticker, directory):
        metadata_path = path.with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            first = date.fromisoformat(metadata["first_date"])
            last = date.fromisoformat(metadata["last_date"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Metadatos de snapshot no válidos: {path.name}") from exc
        if first <= start + pd.Timedelta(days=7) and last >= end - pd.Timedelta(days=7):
            frame = load_snapshot(path, directory, ticker)
            if covers_period(frame.index, start, end):
                return frame.loc[(frame.index.date >= start) & (frame.index.date <= end)].copy(), path
    return None
