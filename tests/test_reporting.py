"""Pruebas de validación y reporte Sprint 04 sin depender de internet."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from tempus.reporting import (
    load_evidence, validate_forecasts, validate_scores, best_models,
    export_contract, metadata_for_export, safe_input_path
)
from tempus.report_pdf import create_academic_pdf

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "evidence" / "sprint03"


@pytest.fixture(scope="module")
def evidence():
    """Reusa exportaciones reales verificadas entre pruebas."""
    return load_evidence(SOURCE)


def test_five_assets_and_common_dates(evidence):
    assert len(evidence.forecasts) == 5
    assert len(evidence.metrics) == 15
    assert all(len(df) == 26 for df in evidence.forecasts.values())
    reference = evidence.forecasts["JNJ"].index
    assert all(df.index.equals(reference) for df in evidence.forecasts.values())


def test_metrics_recomputed(evidence):
    for ticker, pred in evidence.forecasts.items():
        for method in ("Ingenuo", "Suavizamiento exponencial", "ARIMA(1,1,1)"):
            reported = evidence.metrics.query("ticker == @ticker and modelo == @method").iloc[0]
            assert np.isclose(np.abs(pred["observado"] - pred[method]).mean(),
                              reported["MAE"], atol=1e-9)


def test_best_mae_and_contract(evidence):
    winners = dict(zip(best_models(evidence.metrics)["ticker"],
                       best_models(evidence.metrics)["modelo"]))
    assert winners["JNJ"] == "ARIMA(1,1,1)"
    assert winners["WMT"] == "ARIMA(1,1,1)"
    assert winners["JPM"] == "Ingenuo"
    contract = metadata_for_export(evidence)
    assert contract["schema_version"] == "1.0.0"
    assert len(contract["source_sha256"]) >= 17


def test_reject_bad_forecast_baseline(evidence):
    test = evidence.forecasts["JNJ"].reset_index().copy()
    test.loc[0, "Ingenuo"] = 0.001
    with pytest.raises(ValueError, match="línea base"):
        validate_forecasts(test, "JNJ")


def test_reject_duplicate_dates(evidence):
    test = evidence.forecasts["JNJ"].reset_index().copy()
    test.loc[1, "Date"] = test.loc[0, "Date"]
    with pytest.raises(ValueError, match="fechas"):
        validate_forecasts(test, "JNJ")


def test_reject_tampered_score(evidence):
    scores = evidence.metrics.query("ticker == 'JNJ'").drop(columns="ticker").copy()
    scores.loc[scores["modelo"] == "Ingenuo", "MAE"] += 1.0
    with pytest.raises(ValueError, match="inconsistente"):
        validate_scores(scores, evidence.forecasts["JNJ"], "JNJ")


def test_reject_tampered_correlation(tmp_path):
    for f in SOURCE.glob("*"):
        shutil.copy2(f, tmp_path / f.name)
    file = tmp_path / "tempus_correlaciones.csv"
    matrix = pd.read_csv(file, index_col=0)
    matrix.loc["JNJ", "MSFT"] = 0.9
    matrix.loc["MSFT", "JNJ"] = 0.9
    matrix.to_csv(file)
    with pytest.raises(ValueError, match="Correlaciones incoherentes"):
        load_evidence(tmp_path)


def test_reject_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evidence(tmp_path)


def test_safe_file_reject_path_traversal():
    with pytest.raises(ValueError):
        safe_input_path(SOURCE, "../secret.txt")


def test_contract_no_overwrite(evidence, tmp_path):
    files = export_contract(evidence, tmp_path)
    assert len(files) == 8
    assert all(f.is_file() for f in files)
    with pytest.raises(FileExistsError):
        export_contract(evidence, tmp_path)
    data = json.loads((tmp_path / "tempus_exploratio_metadata.json").read_text(encoding="utf-8"))
    assert data["horizonte_sesiones"] == 1


def test_pdf_generation(evidence, tmp_path):
    pdf = create_academic_pdf(evidence, tmp_path / "informe.pdf")
    assert pdf.is_file() and pdf.stat().st_size > 20_000
    with pytest.raises(FileExistsError):
        create_academic_pdf(evidence, pdf)
