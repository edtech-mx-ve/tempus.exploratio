"""Consolidación automatizada local del estudio de 10 años.

Lee únicamente snapshots Parquet con hash validado: no accede a la web.
Genera métricas históricas, pronósticos futuros, gráficos PDF, contrato y manifiesto.
Uso: python run_10y_pipeline.py --snapshots data/snapshots --output-root reports
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from tempus.assets import ASSETS
from tempus.core import enrich_prices
from tempus.forecasting import ForecastConfig, METHODS, walk_forward_evaluation
from tempus.future import FutureConfig, forecast_future
from tempus.history_period import CUTOFF, covers_period, load_covered_snapshot, period_start
from tempus.integrated_report import create_integrated_pdf, load_future_exports
from tempus.reporting import export_contract, load_evidence
from tempus.statistics import normalized_price_panel

LOG = logging.getLogger("tempus.pipeline_10y")
EXPECTED_COUNT = 2514
EXPECTED_TICKERS = tuple(ASSETS)
METRIC_COLS = ("MAE", "RMSE", "sMAPE_pct")


def load_verified_history(directory: Path, until: date) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    """Carga cinco snapshots de Yahoo Finance con SHA-256 y ventana común comprobada."""
    start = period_start(10, until)
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, str]] = {}
    for ticker in EXPECTED_TICKERS:
        result = load_covered_snapshot(ticker, directory, start, until)
        if result is None:
            raise FileNotFoundError(f"{ticker}: no existe snapshot local íntegro de diez años")
        frame, filepath = result
        if not covers_period(frame.index, start, until, tolerance_days=4):
            raise ValueError(f"{ticker}: cobertura histórica incompleta")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{ticker}: índice con duplicados o sin ordenar")
        if len(frame) < 2400:
            raise ValueError(f"{ticker}: solo {len(frame)} sesiones, insuficientes para diez años")
        if frame.index.max().date() != until and (until - frame.index.max().date()).days > 3:
            raise ValueError(f"{ticker}: la última fecha no corresponde al periodo solicitado")
        meta_path = filepath.with_suffix(".json")
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        if "Yahoo Finance" not in str(info.get("source", "")):
            raise ValueError(f"{ticker}: procedencia del snapshot no identificada como Yahoo Finance")
        frames[ticker] = frame
        sources[ticker] = {
            "snapshot": str(filepath.resolve()),
            "sha256": str(info["sha256"]),
            "inicio": frame.index.min().date().isoformat(),
            "fin": frame.index.max().date().isoformat(),
            "filas": len(frame),
        }
        LOG.info("Validada %s: %s observaciones (%s a %s)", ticker, len(frame),
                 sources[ticker]["inicio"], sources[ticker]["fin"])
    if len({tuple(frame.index) for frame in frames.values()}) != 1:
        raise ValueError("Las cinco series no comparten exactamente las fechas de negociación")
    if len({len(frame) for frame in frames.values()}) != 1:
        raise ValueError("Las cinco empresas no tienen el mismo número de filas")
    return frames, sources


def _write_csv(frame: pd.DataFrame, destination: Path, index: bool = False) -> None:
    if destination.exists():
        raise FileExistsError(f"Archivo existente: {destination}")
    frame.to_csv(destination, index=index, date_format="%Y-%m-%d", encoding="utf-8")


def _write_json(payload: dict, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Archivo existente: {destination}")
    with destination.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def generate_evidence(
    frames: dict[str, pd.DataFrame],
    evidence_dir: Path,
    future_dir: Path,
    *,
    test_fraction: float = 0.20,
    step: int = 10,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Evalúa sin fuga temporal y produce los contratos CSV/JSON existentes."""
    evidence_dir.mkdir(parents=True, exist_ok=False)
    future_dir.mkdir(parents=True, exist_ok=False)
    config = ForecastConfig(test_fraction=test_fraction, step=step, min_train=100)
    scores_all: list[pd.DataFrame] = []
    meta_all: dict[str, dict] = {}
    for ticker in EXPECTED_TICKERS:
        frame = frames[ticker]
        LOG.info("%s: evaluando walk-forward, puede llevar varios minutos...", ticker)
        predictions, scores, metadata = walk_forward_evaluation(frame, config)
        if metadata["fallos"] or metadata["evaluaciones_programadas"] != metadata["evaluaciones_comunes"]:
            raise RuntimeError(f"{ticker}: fallos de modelos; no se generará el informe")
        if not np.isfinite(scores[list(METRIC_COLS)].to_numpy(dtype=float)).all():
            raise ValueError(f"{ticker}: métricas inválidas")
        if len(predictions) < 5:
            raise ValueError(f"{ticker}: menos de 5 orígenes comparables")
        preds = predictions.reset_index()
        preds["Date"] = pd.to_datetime(preds["Date"]).dt.strftime("%Y-%m-%d")
        enriched = enrich_prices(frame).reset_index()
        enriched["Date"] = pd.to_datetime(enriched["Date"]).dt.strftime("%Y-%m-%d")
        _write_csv(preds, evidence_dir / f"tempus_{ticker}_pronosticos.csv")
        _write_csv(scores, evidence_dir / f"tempus_{ticker}_metricas.csv")
        _write_json(metadata, evidence_dir / f"tempus_{ticker}_evaluacion.json")
        _write_csv(enriched, evidence_dir / f"tempus_{ticker}_indicadores.csv")

        LOG.info("%s: pronóstico a 20 sesiones desde el último dato observado...", ticker)
        future, future_meta = forecast_future(frame, FutureConfig(horizon=20, confidence=0.95))
        future["Date"] = pd.to_datetime(future["Date"]).dt.strftime("%Y-%m-%d")
        _write_csv(future, future_dir / f"tempus_{ticker}_futuro.csv")
        _write_json(future_meta, future_dir / f"tempus_{ticker}_futuro.json")
        score_copy = scores.assign(ticker=ticker)
        scores_all.append(score_copy)
        meta_all[ticker] = metadata
        LOG.info("%s: %s orígenes × 3 modelos; 20 sesiones × 3 modelos futuros.",
                 ticker, len(predictions))

    panel = normalized_price_panel(frames)
    _write_csv(panel.reset_index(), evidence_dir / "tempus_series_base100.csv")
    _write_csv(panel.pct_change().dropna().corr(), evidence_dir / "tempus_correlaciones.csv", index=True)
    return pd.concat(scores_all, ignore_index=True), meta_all


def run(*, snapshots: Path, output_root: Path, until: date = CUTOFF, step: int = 10) -> Path:
    """Ejecución transaccional: solo una carpeta nueva, manifiesto final al completar."""
    if not isinstance(until, date) or until > date.today():
        raise ValueError("Fecha de corte inválida o futura")
    if step != 10:
        raise ValueError("El contrato de evaluación requiere step=10; no alterarlo sin validar el contrato")
    if not Path(snapshots).is_dir():
        raise FileNotFoundError(f"No existe el directorio de snapshots: {snapshots}")
    frames, sources = load_verified_history(Path(snapshots), until)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / f"integral_10y_{stamp}"
    folder.mkdir(exist_ok=False)
    evidence_dir = folder / "evidence"
    future_dir = folder / "future"
    LOG.info("Salida nueva: %s", folder)

    scores, metas = generate_evidence(frames, evidence_dir, future_dir, step=step)
    evidence = load_evidence(evidence_dir)
    futures = load_future_exports(future_dir, evidence)
    pdf = create_integrated_pdf(evidence, futures, folder / "TEMPUS_EXPLORATIO_Informe_Integral_10_Anos.pdf")
    contract_dir = folder / "contrato"
    contract = export_contract(evidence, contract_dir)
    hashes = {}
    for file in sorted(folder.rglob("*")):
        if file.is_file() and file.name != "manifest.json":
            hashes[str(file.relative_to(folder)).replace("\\", "/")] = sha256(file.read_bytes()).hexdigest()
    summary = {
        "version": "10y.1",
        "fecha_corte": until.isoformat(),
        "periodo": [period_start(10, until).isoformat(), until.isoformat()],
        "frecuencia": "diaria bursatil",
        "tickers": list(EXPECTED_TICKERS),
        "fuente": "snapshots locales verificados procedentes de Yahoo Finance / yfinance",
        "sources": sources,
        "observaciones_por_empresa": {ticker: len(frame) for ticker, frame in frames.items()},
        "registros_totales": sum(map(len, frames.values())),
        "config_walk_forward": {"test_fraction": 0.20, "step": step, "horizon": 1},
        "evaluaciones_por_empresa": {ticker: meta["evaluaciones_comunes"] for ticker, meta in metas.items()},
        "evaluaciones_totales": int(sum(meta["evaluaciones_comunes"] for meta in metas.values()) * len(METHODS)),
        "pronosticos_futuros": {"sesiones": 20, "metodos": list(METHODS),
                                 "observaciones_por_empresa": 20 * len(METHODS)},
        "metricas": list(METRIC_COLS),
        "mejor_por_mae": {
            ticker: str(scores.loc[scores.ticker.eq(ticker)].sort_values("MAE").iloc[0]["modelo"])
            for ticker in EXPECTED_TICKERS
        },
        "pdf": pdf.name,
        "contrato_archivos": len(contract),
        "sha256_archivos": hashes,
        "advertencia": "Pronosticos posteriores a la ultima observacion del snapshot; pueden ser fechas ya pasadas respecto de hoy. Uso academico.",
    }
    _write_json(summary, folder / "manifest.json")
    LOG.info("INFORME VERIFICADO: %s", pdf)
    print(f"\nFINALIZADO: {folder}\nPDF: {pdf}")
    print(f"Total: {summary['registros_totales']} registros; {summary['evaluaciones_totales']} predicciones retrospectivas")
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", default="data/snapshots")
    parser.add_argument("--output-root", default="reports")
    parser.add_argument("--until", default=CUTOFF.isoformat())
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(snapshots=Path(args.snapshots), output_root=Path(args.output_root),
            until=date.fromisoformat(args.until))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        LOG.exception("No se completó la consolidación: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
