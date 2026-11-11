"""Adquisición externa, persistencia y metadatos auditables."""
from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import json
import logging
import os
import re
import uuid
import pandas as pd
from tempus.assets import ASSETS
from tempus.core import normalize_prices, validate_period

LOG = logging.getLogger(__name__)
VERSION_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")

def download_ticker(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Obtiene datos del proveedor; reporta fallos sin confundirlos con datos vacíos."""
    validate_period(start, end)
    if ticker not in ASSETS:
        raise ValueError(f"Ticker no admitido: {ticker}")
    try:
        import yfinance as yf
        downloaded = yf.Ticker(ticker).history(
            start=start.isoformat(), end=end.isoformat(), interval="1d",
            auto_adjust=True, actions=True, raise_errors=True, timeout=15
        )
    except (ConnectionError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
        LOG.error("Descarga fallida para %s: %s", ticker, type(exc).__name__)
        raise RuntimeError(f"No fue posible descargar {ticker}") from exc
    return normalize_prices(downloaded)

def save_snapshot(frame: pd.DataFrame, ticker: str, directory: Path) -> Path:
    """Crea snapshot inmutable (Parquet + JSON); evita reemplazos accidentales."""
    if ticker not in ASSETS:
        raise ValueError("Ticker no admitido")
    data = normalize_prices(frame)
    target = Path(directory).resolve() / ticker
    target.mkdir(parents=True, exist_ok=True)
    # UUID es solo identificador de versión; no se utiliza para reproducir experimentos.
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    final = target / f"{version}.parquet"
    pending = target / f".{version}.tmp.parquet"
    metadata = target / f"{version}.json"
    try:
        data.to_parquet(pending)
        digest = hashlib.sha256(pending.read_bytes()).hexdigest()
        payload = {
            "ticker": ticker, "asset": ASSETS[ticker].metadata(),
            "version": version, "source": "Yahoo Finance / yfinance o importación verificada",
            "adjustment": "OHLC ajustado si la fuente fue yfinance con auto_adjust=True",
            "records": len(data), "first_date": str(data.index.min().date()),
            "last_date": str(data.index.max().date()),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": digest
        }
        with metadata.open("x", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, ensure_ascii=False)
        with final.open("xb") as output:
            output.write(pending.read_bytes())
    except (OSError, ValueError, ImportError) as exc:
        final.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        raise RuntimeError("No fue posible persistir la versión") from exc
    finally:
        pending.unlink(missing_ok=True)
    LOG.info("Snapshot guardado: %s (%d filas)", ticker, len(data))
    return final

def list_snapshots(ticker: str, directory: Path) -> list[Path]:
    """Devuelve únicamente snapshots de un ticker admitido, en orden descendente."""
    if ticker not in ASSETS:
        raise ValueError("Ticker no admitido")
    folder = Path(directory).resolve() / ticker
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.glob("*.parquet") if VERSION_RE.fullmatch(p.stem)), reverse=True)

def load_snapshot(path: Path, directory: Path, ticker: str) -> pd.DataFrame:
    """Carga un snapshot del directorio permitido, comprobando su hash."""
    if ticker not in ASSETS:
        raise ValueError("Ticker no admitido")
    base = (Path(directory).resolve() / ticker).resolve()
    resolved = Path(path).resolve()
    if resolved.parent != base or not VERSION_RE.fullmatch(resolved.stem):
        raise ValueError("Ruta de snapshot no permitida")
    metadata = resolved.with_suffix(".json")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    if info.get("ticker") != ticker:
        raise ValueError("Metadatos de ticker incorrectos")
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != info.get("sha256"):
        raise ValueError("La suma de verificación no coincide")
    return normalize_prices(pd.read_parquet(resolved))
