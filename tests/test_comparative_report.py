"""Pruebas autónomas del análisis comparativo; no requieren datos externos."""
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
import pytest

from tempus.assets import ASSETS
from tempus.comparative_report import comparative_rows, create_comparative_appendix
from tempus.reporting import Evidence


@pytest.fixture
def evidence():
    tickers = tuple(ASSETS)
    dates = pd.bdate_range("2016-11-01", periods=90)
    indicators = {}
    metrics = []
    for j, ticker in enumerate(tickers):
        close = np.linspace(100 + j * 20, 125 + j * 23, len(dates))
        indicators[ticker] = pd.DataFrame({"Date": dates, "Close": close})
        for i, model in enumerate(("Ingenuo", "Suavizamiento exponencial", "ARIMA(1,1,1)")):
            metrics.append({"ticker": ticker, "modelo": model,
                            "MAE": 1.0 + i, "RMSE": 1.5 + i,
                            "sMAPE_pct": 2.0 + i})
    return Evidence(metrics=pd.DataFrame(metrics), forecasts={},
                    metadata={}, source_hashes={}, panel=pd.DataFrame(),
                    correlations=pd.DataFrame(), indicators=indicators)


def test_comparative_metrics(evidence):
    rows = comparative_rows(evidence)
    assert [r["ticker"] for r in rows] == list(ASSETS)
    assert all(r["n"] == 90 and r["volatilidad_pct"] >= 0 for r in rows)
    assert all(r["mejor_modelo"] == "Ingenuo" for r in rows)


def test_report_contains_five_companies(evidence, tmp_path):
    path = create_comparative_appendix(evidence, tmp_path / "analisis.pdf")
    text = "\\n".join(page.extract_text() for page in PdfReader(str(path)).pages)
    assert "ANÁLISIS COMPARATIVO" in text
    assert all(ticker in text for ticker in ASSETS)
    assert "Volatilidad" in text


def test_no_overwrite(evidence, tmp_path):
    path = create_comparative_appendix(evidence, tmp_path / "analisis.pdf")
    with pytest.raises(FileExistsError):
        create_comparative_appendix(evidence, path)


def test_rejects_missing_company(evidence):
    evidence.indicators.pop("XOM")
    with pytest.raises(ValueError, match="cinco series"):
        comparative_rows(evidence)
