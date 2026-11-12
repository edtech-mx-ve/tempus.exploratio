"""Auditoría local reproducible de cálculos y evidencias de TEMPUS EXPLORATIO.

No descarga datos, no recalcula ARIMA y no sustituye las pruebas de interacción UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tempus.assets import ASSETS
from tempus.core import enrich_prices
from tempus.history_period import CUTOFF, period_start
from tempus.statistics import (
    descriptive_statistics, stationarity_diagnostics, dependence_profile,
    normalized_price_panel, correlation_matrix, drawdown_series,
)

LOG = logging.getLogger(__name__)
TICKERS = tuple(ASSETS)
EXPECTED_MODELS = ("Ingenuo", "Suavizamiento exponencial", "ARIMA(1,1,1)")


@dataclass(frozen=True)
class AuditResult:
    """Resultado sin estado global y listo para representación o exportación."""
    checks: pd.DataFrame
    summary: pd.DataFrame
    prices: pd.DataFrame
    returns: pd.DataFrame
    volatility: pd.DataFrame
    volume: pd.DataFrame
    base100: pd.DataFrame
    correlation: pd.DataFrame
    metrics: pd.DataFrame
    future: pd.DataFrame
    report_path: Path | None
    report_folder: Path | None

    @property
    def passed(self) -> bool:
        return bool(len(self.checks)) and bool(self.checks["Estado"].eq("OK").all())


def _check(rows: list[dict[str, str]], component: str, item: str, passed: bool,
           detail: str) -> None:
    rows.append({"Componente": component, "Elemento": item,
                 "Estado": "OK" if passed else "ERROR", "Detalle": detail})


def _safe_csv(path: Path, base: Path) -> pd.DataFrame:
    resolved = path.resolve()
    if not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        raise FileNotFoundError(f"Archivo de evidencia no encontrado: {path.name}")
    return pd.read_csv(resolved)


def _report_latest(reports_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    found = sorted(reports_dir.glob("integral_10y_*/manifest.json"), reverse=True)
    for manifest_path in found:
        try:
            info = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (info.get("version") == "10y.1"
                    and set(info.get("tickers", [])) == set(TICKERS)):
                return manifest_path, info
        except (OSError, ValueError, TypeError):
            continue
    return None, None


def _verify_report(report_dir: Path, data: dict[str, Any],
                   sources: dict[str, dict[str, str]],
                   rows: list[dict[str, str]]) -> tuple[Path | None, pd.DataFrame, pd.DataFrame]:
    """Valida TODOS los hashes del manifiesto, no solamente el PDF."""
    hashes = data.get("sha256_archivos")
    if not isinstance(hashes, dict) or not hashes:
        _check(rows, "Informe", "Manifiesto SHA-256", False, "Faltan hashes")
        return None, pd.DataFrame(), pd.DataFrame()
    invalid: list[str] = []
    for rel, digest in hashes.items():
        p = (report_dir / rel.replace(chr(92), "/")).resolve()
        if (not p.is_relative_to(report_dir.resolve())
                or not p.is_file() or sha256(p.read_bytes()).hexdigest() != digest):
            invalid.append(str(rel))
    _check(rows, "Informe", "Archivos y SHA-256", not invalid,
           f"{len(hashes) - len(invalid)}/{len(hashes)} íntegros" +
           (f"; fallos: {', '.join(invalid[:3])}" if invalid else ""))
    matching = [
        t for t in TICKERS
        if data.get("sources", {}).get(t, {}).get("sha256") == sources.get(t, {}).get("sha256")
    ]
    _check(rows, "Informe", "Snapshots de origen", len(matching) == len(TICKERS),
           f"{len(matching)}/5 hashes de origen coincidentes")
    counts = data.get("observaciones_por_empresa", {})
    total = int(sum(int(counts.get(t, 0)) for t in TICKERS))
    expected = sum(int(sources[t]["filas"]) for t in TICKERS)
    _check(rows, "Informe", "Recuento histórico",
           total == expected and int(data.get("registros_totales", -1)) == expected,
           f"{expected:,} registros comprobados")
    pdf_name = data.get("pdf", "")
    pdf = (report_dir / pdf_name).resolve() if isinstance(pdf_name, str) else report_dir
    if not isinstance(pdf_name, str) or Path(pdf_name).name != pdf_name or not pdf.is_file():
        pdf = None
    _check(rows, "Informe", "PDF", pdf is not None and not invalid,
           pdf.name if pdf else "No disponible o con integridad inválida")
    evidence = report_dir / "evidence"
    future_dir = report_dir / "future"
    metrics: list[pd.DataFrame] = []
    futures: list[pd.DataFrame] = []
    expected_models = {"Ingenuo", "Suavizamiento exponencial", "ARIMA(1,1,1)"}
    for ticker in TICKERS:
        try:
            score = _safe_csv(evidence / f"tempus_{ticker}_metricas.csv", report_dir)
            pred = _safe_csv(evidence / f"tempus_{ticker}_pronosticos.csv", report_dir)
            future = _safe_csv(future_dir / f"tempus_{ticker}_futuro.csv", report_dir)
            if not {"MAE", "RMSE", "sMAPE_pct"}.issubset(score.columns):
                raise ValueError("Faltan métricas")
            if score[["MAE", "RMSE", "sMAPE_pct"]].isna().any().any():
                raise ValueError("Métricas vacías")
            for column in ["MAE", "RMSE", "sMAPE_pct"]:
                if not np.isfinite(pd.to_numeric(score[column], errors="coerce")).all():
                    raise ValueError(f"{column} no finito")
            origins = int(data.get("evaluaciones_por_empresa", {}).get(ticker, 0))
            if len(pred) != origins or origins <= 0:
                raise ValueError(f"Orígenes incoherentes: {len(pred)} / {origins}")
            if len(future) != 60:
                raise ValueError(f"Se esperaban 60 estimaciones futuras; hay {len(future)}")
            if "modelo" not in future.columns or set(future["modelo"]) != expected_models:
                raise ValueError("Modelos futuros incorrectos")
            score["Empresa"] = ticker
            future["Empresa"] = ticker
            metrics.append(score)
            futures.append(future)
            _check(rows, "Evaluación y futuro", ticker, True,
                   f"{len(pred)} orígenes × 3 modelos; {len(future)} estimaciones")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _check(rows, "Evaluación y futuro", ticker, False, str(exc))
    if invalid:
        # Los datos podrían renderizarse para diagnóstico, pero no se declaran validados.
        pdf = None
    return pdf, (pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()), (
        pd.concat(futures, ignore_index=True) if futures else pd.DataFrame())


def run_integral_audit(
    snapshots_dir: Path, reports_dir: Path,
    *, until=CUTOFF,
) -> AuditResult:
    """Ejecuta cálculos deterministas, diagnósticos y auditoría del informe local.

    Nunca reentrena ni reescribe archivos. Lanza excepción ante ausencia de snapshots;
    otros errores quedan marcados en la tabla de comprobaciones.
    """
    from tempus.history_period import load_covered_snapshot

    rows: list[dict[str, str]] = []
    sources: dict[str, dict[str, str]] = {}
    frames: dict[str, pd.DataFrame] = {}
    summary: list[dict[str, object]] = []
    prices: list[pd.DataFrame] = []
    returns: list[pd.DataFrame] = []
    vols: list[pd.DataFrame] = []
    volumes: list[pd.DataFrame] = []
    start = period_start(10, until)
    for ticker in TICKERS:
        try:
            loaded = load_covered_snapshot(ticker, Path(snapshots_dir), start, until)
            if loaded is None:
                raise FileNotFoundError("Snapshot íntegro de diez años no encontrado")
            frame, path = loaded
            if (len(frame) < 2400 or frame.index.has_duplicates
                    or not frame.index.is_monotonic_increasing):
                raise ValueError("Cobertura, fechas u orden incorrectos")
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if "Yahoo Finance" not in str(meta.get("source", "")):
                raise ValueError("La procedencia no es Yahoo Finance")
            frames[ticker] = frame
            sources[ticker] = {"sha256": str(meta["sha256"]), "filas": str(len(frame))}
            stats = descriptive_statistics(frame)
            enrich = enrich_prices(frame)
            station = stationarity_diagnostics(enrich["return_1d"].dropna())
            depend = dependence_profile(enrich["return_1d"].dropna(), max_lags=20)
            if not np.isfinite(depend[["ACF", "PACF"]].to_numpy()).all():
                raise ValueError("ACF/PACF contiene valores no finitos")
            _check(rows, "Históricos", ticker, True, f"{len(frame):,} sesiones; SHA-256 verificado")
            _check(rows, "Estadística", ticker, True,
                   f"ADF p={station['adf_p']:.3f}; KPSS p={station['kpss_p']:.3f}; ACF/PACF: 20 rezagos")
            summary.append({
                "Empresa": ticker, "Sesiones": len(frame),
                "Precio final (USD/acción)": round(float(frame["Close"].iloc[-1]), 2),
                "Rend. acumulado (%)": round(float(stats["rendimiento_acumulado_pct"]), 2),
                "Vol. anualizada (%)": round(float(stats["volatilidad_anual_pct"]), 2),
                "ADF p": round(float(station["adf_p"]), 4),
                "KPSS p": round(float(station["kpss_p"]), 4),
            })
            display = enrich.reset_index()
            sample = pd.DataFrame({"Fecha": display["Date"], "Empresa": ticker,
                                   "Precio (USD/acción)": display["Close"]})
            prices.append(sample)
            returns.append(pd.DataFrame({
                "Fecha": display["Date"], "Empresa": ticker,
                "Rendimiento (%)": display["return_1d"] * 100}))
            volatility = display["return_1d"].rolling(window=20, min_periods=20).std() * np.sqrt(252) * 100
            vols.append(pd.DataFrame({"Fecha": display["Date"], "Empresa": ticker,
                                      "Volatilidad anualizada (%)": volatility}))
            volumes.append(pd.DataFrame({"Fecha": display["Date"], "Empresa": ticker,
                                         "Acciones negociadas": display["Volume"]}))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError,
                np.linalg.LinAlgError) as exc:
            LOG.exception("Auditoría del activo %s: %s", ticker, exc)
            _check(rows, "Históricos/estadística", ticker, False, str(exc))

    base = pd.DataFrame()
    corr = pd.DataFrame()
    if len(frames) == 5:
        try:
            base = normalized_price_panel(frames)
            corr = correlation_matrix(frames)
            if len({tuple(frame.index) for frame in frames.values()}) != 1:
                raise ValueError("Fechas distintas entre series")
            if corr.isna().any().any() or base.isna().any().any():
                raise ValueError("Comparación con valores faltantes")
            _check(rows, "Multiserie", "Comparación", True,
                   f"{len(base):,} fechas alineadas; índice base 100 y correlación")
        except (ValueError, TypeError, KeyError) as exc:
            _check(rows, "Multiserie", "Comparación", False, str(exc))
    else:
        _check(rows, "Multiserie", "Comparación", False, "Se requieren cinco históricos válidos")

    pdf: Path | None = None
    metrics = pd.DataFrame()
    future = pd.DataFrame()
    manifest_path, manifest = _report_latest(Path(reports_dir))
    if manifest is None or manifest_path is None:
        _check(rows, "Informe", "Consolidación", False, "No hay manifiesto 10y.1")
    elif len(frames) != 5:
        _check(rows, "Informe", "Consolidación", False, "Faltan históricos para cotejar los hashes")
    else:
        pdf, metrics, future = _verify_report(manifest_path.parent, manifest, sources, rows)
    def concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return AuditResult(
        checks=pd.DataFrame(rows), summary=pd.DataFrame(summary),
        prices=concat(prices), returns=concat(returns), volatility=concat(vols),
        volume=concat(volumes), base100=base, correlation=corr,
        metrics=metrics, future=future, report_path=pdf,
        report_folder=manifest_path.parent if manifest_path else None,
    )
