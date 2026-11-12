"""Regresión v0.5.1: integración de la evidencia histórica y los cinco pronósticos."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader

from tempus.integrated_report import load_future_exports, create_integrated_pdf
from tempus.reporting import load_evidence

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "evidence" / "sprint03"
FUTURE = ROOT / "reports" / "sprint05_futuro"


@pytest.fixture(scope="module")
def evidence():
    return load_evidence(HISTORY)


def test_five_futures_and_horizons(evidence):
    futures = load_future_exports(FUTURE, evidence)
    assert len(futures) == 5
    for ticker, (frame, meta) in futures.items():
        assert len(frame) == 60
        assert set(frame["sesion"]) == set(range(1, 21))
        assert frame["Date"].min() > pd.Timestamp(meta["ultimo_dato_observado"])
        assert frame["modelo"].nunique() == 3


def test_integrated_report_contains_five_assets(evidence, tmp_path):
    pdf = create_integrated_pdf(
        evidence, load_future_exports(FUTURE, evidence), tmp_path / "final.pdf")
    assert pdf.stat().st_size > 120_000
    report = PdfReader(str(pdf))
    assert len(report.pages) >= 12
    text = "\n".join(page.extract_text() for page in report.pages)
    assert "Evaluación" in text or "evaluación" in text
    assert "Limitaciones" in text
    for ticker in ("JNJ", "MSFT", "JPM", "WMT", "XOM"):
        assert f"{ticker}: estimaciones" in text
    with pytest.raises(FileExistsError):
        create_integrated_pdf(evidence, load_future_exports(FUTURE, evidence), pdf)


def _clone_futures(tmp_path):
    src = tmp_path / "fuentes"
    src.mkdir()
    for ticker in ("JNJ", "MSFT", "JPM", "WMT", "XOM"):
        for ext in ("csv", "json"):
            shutil.copy2(
                FUTURE / f"tempus_{ticker}_futuro.{ext}",
                src / f"tempus_{ticker}_futuro.{ext}",
            )
    return src


def test_reject_mismatched_last_close(evidence, tmp_path):
    folder = _clone_futures(tmp_path)
    file = folder / "tempus_JNJ_futuro.json"
    meta = json.loads(file.read_text(encoding="utf-8"))
    meta["ultimo_cierre"] = float(meta["ultimo_cierre"]) + 20
    file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="otro snapshot"):
        load_future_exports(folder, evidence)


def test_reject_mismatched_last_date(evidence, tmp_path):
    folder = _clone_futures(tmp_path)
    file = folder / "tempus_JPM_futuro.json"
    meta = json.loads(file.read_text(encoding="utf-8"))
    meta["ultimo_dato_observado"] = "2023-01-01"
    file.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="otro snapshot"):
        load_future_exports(folder, evidence)


def test_reject_malformed_calendar_dates(evidence, tmp_path):
    folder = _clone_futures(tmp_path)
    file = folder / "tempus_MSFT_futuro.csv"
    frame = pd.read_csv(file)
    frame.loc[0, "Date"] = "2026-11-07"  # sábado, NYSE cerrado
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="fechas bursátiles"):
        load_future_exports(folder, evidence)


def test_reject_missing_or_short_horizon(evidence, tmp_path):
    folder = _clone_futures(tmp_path)
    file = folder / "tempus_WMT_futuro.csv"
    frame = pd.read_csv(file)
    frame = frame.loc[frame["sesion"] <= 5]
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="horizonte incompatible"):
        load_future_exports(folder, evidence)


def test_reject_non_arima_intervals(evidence, tmp_path):
    folder = _clone_futures(tmp_path)
    file = folder / "tempus_XOM_futuro.csv"
    frame = pd.read_csv(file)
    frame.loc[frame["modelo"] == "Ingenuo", "limite_inferior"] = 150
    frame.to_csv(file, index=False)
    with pytest.raises(ValueError, match="intervalos no esperados"):
        load_future_exports(folder, evidence)
