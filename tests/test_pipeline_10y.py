"""Pruebas de protección y coherencia del cierre histórico de diez años."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import run_10y_pipeline as pipeline


def test_corte_futuro_se_rechaza(tmp_path: Path):
    (tmp_path / "snapshots").mkdir()
    with pytest.raises(ValueError, match="futura"):
        pipeline.run(snapshots=tmp_path / "snapshots", output_root=tmp_path / "out",
                     until=date(2099, 1, 1))


def test_directorio_inexistente_se_rechaza(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="snapshots"):
        pipeline.run(snapshots=tmp_path / "ausente",
                     output_root=tmp_path / "out")


def test_no_reemplaza_exportacion_csv(tmp_path: Path):
    target = tmp_path / "ejemplo.csv"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        pipeline._write_csv(pd.DataFrame({"a": [2]}), target)
    assert target.read_text(encoding="utf-8") == "original"


def test_no_reemplaza_exportacion_json(tmp_path: Path):
    target = tmp_path / "metadatos.json"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        pipeline._write_json({"a": 2}, target)
    assert target.read_text(encoding="utf-8") == "original"


def test_exporta_fechas_iso(tmp_path: Path):
    out = tmp_path / "preds.csv"
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-10-30"]),
                          "observado": [200.4]})
    pipeline._write_csv(frame, out)
    result = pd.read_csv(out)
    assert result.loc[0, "Date"] == "2026-10-30"


def test_no_acepta_snapshots_sinteticos_como_reales(tmp_path: Path, monkeypatch):
    from tempfile import TemporaryDirectory
    dates = pd.bdate_range(start="2016-10-31", end="2026-10-30", name="Date")
    close = np.linspace(100, 200, len(dates))
    frame = pd.DataFrame({"Open": close, "High": close * 1.01,
                          "Low": close * 0.99, "Close": close,
                          "Volume": np.full(len(dates), 1000000)}, index=dates)
    directory = tmp_path / "snapshots"
    directory.mkdir()
    parquet = directory / "sample.parquet"
    # No lee Parquet en esta prueba: se prueba solo la procedencia declarada.
    meta = parquet.with_suffix(".json")
    meta.write_text('{"source":"datos sintéticos","sha256":"demo"}', encoding="utf-8")
    monkeypatch.setattr(pipeline, "load_covered_snapshot",
                        lambda *args, **kwargs: (frame, parquet))
    with pytest.raises(ValueError, match="Yahoo Finance"):
        pipeline.load_verified_history(directory, date(2026, 10, 30))


def test_no_publica_informe_incompleto(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pipeline, "load_verified_history",
                        lambda *_: ({x: pd.DataFrame() for x in pipeline.EXPECTED_TICKERS}, {}))
    def fail(*args, **kwargs):
        raise RuntimeError("ajuste interrumpido")
    monkeypatch.setattr(pipeline, "generate_evidence", fail)
    src = tmp_path / "snapshots"
    src.mkdir()
    with pytest.raises(RuntimeError, match="ajuste interrumpido"):
        pipeline.run(snapshots=src, output_root=tmp_path / "reports")
    assert not list((tmp_path / "reports").rglob("manifest.json"))
