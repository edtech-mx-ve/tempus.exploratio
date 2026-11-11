"""Lectura segura y reproducible de cinco series públicas incluidas con el repositorio.

No realiza escrituras, descargas ni utiliza secretos. Las exportaciones corresponden
al corte histórico del 30 de octubre de 2026 y pueden estar desactualizadas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tempus.assets import ASSETS
from tempus.core import normalize_prices

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "evidence" / "sprint03" / "public_manifest.json"
COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")


def load_public_series(ticker: str, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Carga una exportación pública verificada por SHA-256, sin mutar el archivo.

    Args:
        ticker: Uno de los cinco activos declarados por el proyecto.
        root: Directorio del repositorio; útil para pruebas y validación aislada.
    """
    if ticker not in ASSETS:
        raise ValueError("Ticker público no permitido")

    repo = Path(root).resolve()
    manifest_path = repo / "evidence" / "sprint03" / "public_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = payload["series"][ticker]
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("No se encontró el manifiesto público válido") from exc

    expected_path = repo / "evidence" / "sprint03" / f"tempus_{ticker}_indicadores.csv"
    file_path = (repo / str(entry.get("path", ""))).resolve()
    if file_path != expected_path.resolve():
        raise ValueError("Ruta de exportación pública inválida")
    try:
        raw_bytes = file_path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"No existe el histórico público de {ticker}") from exc

    if hashlib.sha256(raw_bytes).hexdigest() != entry.get("sha256"):
        raise ValueError("La exportación pública no coincide con su hash")

    try:
        raw = pd.read_csv(file_path, parse_dates=["Date"])
        if raw["Date"].isna().any():
            raise ValueError("Fechas inválidas en histórico público")
        raw = raw.set_index("Date")
        prices = normalize_prices(raw[list(COLUMNS)])
    except (KeyError, ValueError, TypeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Histórico público de {ticker} inválido") from exc

    if len(prices) != entry.get("rows"):
        raise ValueError("Número de filas público distinto del manifiesto")
    if prices.index[-1].date().isoformat() != entry.get("last_date"):
        raise ValueError("Fecha de corte distinta del manifiesto")

    return prices
