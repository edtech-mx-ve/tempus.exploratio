"""Crea una nueva ejecución de informe con análisis comparativo, sin reentrenar.

Uso (desde raíz del proyecto):
python rebuild_comparative_report.py --run-dir reports/integral_10y_20261112T042339296752Z
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil

from tempus.integrated_report import create_integrated_pdf, load_future_exports
from tempus.reporting import load_evidence

LOG = logging.getLogger("tempus.rebuild")
PDF_NAME = "TEMPUS_EXPLORATIO_Informe_Integral_10_Anos.pdf"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(run_dir: Path) -> Path:
    original = Path(run_dir).resolve(strict=True)
    if not original.is_dir():
        raise NotADirectoryError(f"Carpeta inválida: {original}")
    manifest = original / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != "10y.1" or len(data.get("tickers", [])) != 5:
        raise ValueError("Manifiesto original no válido o incompleto")
    for relative, expected in data.get("sha256_archivos", {}).items():
        path = original / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(original):
            raise FileNotFoundError(f"Evidencia insegura o ausente: {relative}")
        if _sha(path) != expected:
            raise ValueError(f"Firma SHA-256 incorrecta: {relative}")
    evidence = load_evidence(original / "evidence")
    if len(evidence.indicators) != 5 or any(len(f) < 2000 for f in evidence.indicators.values()):
        raise ValueError("Se requieren cinco series completas de diez años")
    load_future_exports(original / "future", evidence)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = original.parent / f"integral_10y_{stamp}"
    staged = original.parent / f".preparando_comparativo_{stamp}"
    if final.exists() or staged.exists():
        raise FileExistsError("Ya existe una salida con este identificador")
    staged.mkdir()
    try:
        for name in ("evidence", "future", "contrato"):
            source = original / name
            if source.is_dir():
                shutil.copytree(source, staged / name, symlinks=False)
        ev = load_evidence(staged / "evidence")
        futures = load_future_exports(staged / "future", ev)
        pdf = create_integrated_pdf(ev, futures, staged / PDF_NAME)
        data["pdf"] = pdf.name
        data["seccion_comparativa"] = (
            "Análisis comparativo de las cinco acciones antes de pronósticos futuros"
        )
        data["sha256_archivos"] = {
            str(file.relative_to(staged)).replace("\\\\", "/"): _sha(file)
            for file in sorted(staged.rglob("*")) if file.is_file() and file.name != "manifest.json"
        }
        (staged / "manifest.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.rename(final)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    LOG.info("Informe verificado creado: %s", final / PDF_NAME)
    print(f"FINALIZADO: {final}")
    print(f"PDF: {final / PDF_NAME}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.run_dir)
    except (ValueError, FileNotFoundError, FileExistsError, OSError,
            RuntimeError, KeyError, json.JSONDecodeError) as exc:
        LOG.error("No se generó el informe: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
