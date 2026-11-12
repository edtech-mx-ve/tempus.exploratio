"""CLI de cierre predictivo: evidencia o snapshots -> CSV, JSON, PDF.

Ejemplo: python future_cli.py --source snapshots --output reports/futuro
No sobreescribe un directorio de salida existente.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

import pandas as pd

from tempus.assets import ASSETS
from tempus.core import normalize_prices
from tempus.future import FutureConfig, forecast_future
from tempus.future_report import create_future_appendix
from tempus.io import list_snapshots, load_snapshot
from pypdf import PdfWriter

LOG = logging.getLogger("tempus.future_cli")


def load_source(ticker: str, source: str, directory: Path) -> tuple[pd.DataFrame, Path]:
    """Carga datos validados y devuelve el archivo de procedencia."""
    if ticker not in ASSETS:
        raise ValueError("Ticker no admitido")
    if source == "snapshots":
        versions = list_snapshots(ticker, directory)
        if not versions:
            raise FileNotFoundError(f"Sin snapshot para {ticker}: {directory}")
        return load_snapshot(versions[0], directory, ticker), versions[0]
    if source == "evidence":
        file = directory / f"tempus_{ticker}_indicadores.csv"
        if not file.is_file():
            raise FileNotFoundError(f"Falta evidencia de {ticker}: {file}")
        raw = pd.read_csv(file, index_col="Date", parse_dates=["Date"])
        return normalize_prices(raw), file
    raise ValueError("Fuente no admitida")


def generate_future_bundle(
    source: str, directory: Path, output: Path, base_report: Path | None = None,
) -> list[Path]:
    """Procesa las cinco series y crea resultados reproducibles sin sobrescribir."""
    if source not in {"snapshots", "evidence"}:
        raise ValueError("source debe ser snapshots o evidence")
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio de origen: {directory}")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"Ya existe el directorio de salida: {destination}")
    # Validar todas las entradas antes de generar los archivos.
    frames = {ticker: load_source(ticker, source, directory) for ticker in ASSETS}
    outputs: dict[str, tuple[pd.DataFrame, dict[str, object]]] = {}
    source_provenance = {}
    for ticker, (frame, path) in frames.items():
        pred, metadata = forecast_future(frame, FutureConfig(horizon=20))
        outputs[ticker] = (pred, metadata)
        source_provenance[ticker] = {
            "archivo_relativo": str(path.resolve().relative_to(directory)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observaciones": len(frame),
            "ultima_fecha": frame.index.max().date().isoformat(),
        }
        LOG.info("%s: %s filas de pronóstico", ticker, len(pred))
    destination.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []
    for ticker, (pred, metadata) in outputs.items():
        csv_file = destination / f"tempus_{ticker}_futuro.csv"
        json_file = destination / f"tempus_{ticker}_futuro.json"
        pred.to_csv(csv_file, index=False, float_format="%.10g")
        json_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        created.extend([csv_file, json_file])
    report = create_future_appendix(outputs, destination / "TEMPUS_EXPLORATIO_Anexo_Pronostico_Futuro.pdf")
    created.append(report)
    if base_report is not None:
        base_pdf = Path(base_report).resolve()
        if not base_pdf.is_file():
            raise FileNotFoundError(f"Informe base no disponible: {base_pdf}")
        merged = destination / "TEMPUS_EXPLORATIO_Informe_Final_Sprint_05.pdf"
        writer = PdfWriter()
        writer.append(str(base_pdf))
        writer.append(str(report))
        writer.add_metadata({
            "/Title": "TEMPUS EXPLORATIO - Informe académico final - Sprint 05",
            "/Subject": "Práctica 1: análisis de cinco series web y pronósticos futuros",
        })
        with merged.open("xb") as stream:
            writer.write(stream)
        created.append(merged)
    manifest = {
        "source": source,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "proveniencia_entradas": source_provenance,
        "resultados": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in created},
        "advertencia": "Precios futuros estimados, no observaciones ni garantía de rendimientos.",
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    created.append(manifest_path)
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pronósticos futuros de TEMPUS EXPLORATIO")
    parser.add_argument("--source", choices=["snapshots", "evidence"], default="snapshots")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-report", type=Path, default=None,
                        help="PDF de Sprint 04 para construir el informe consolidado")
    args = parser.parse_args(argv)
    source_dir = args.input or (Path("data/snapshots") if args.source == "snapshots"
                                else Path("evidence/sprint03"))
    output = args.output or (Path("reports") / (
        "futuro_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    try:
        base_report = args.base_report
        if base_report is None:
            candidate = Path("reports/TEMPUS_EXPLORATIO_Informe_Academico.pdf")
            if candidate.is_file():
                base_report = candidate
        files = generate_future_bundle(args.source, source_dir, output, base_report)
    except (FileNotFoundError, FileExistsError, ValueError, TypeError,
            RuntimeError, OSError, ImportError, KeyError) as exc:
        LOG.error("No se pudo generar paquete futuro: %s", exc)
        return 1
    for file in files:
        print(f"GENERADO {file}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
