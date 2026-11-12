"""Pruebas de ventana temporal y selección de snapshots."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
import pytest
from tempus.history_period import CUTOFF, covers_period, period_start, load_covered_snapshot

def test_period_dates():
    assert period_start(10) == date(2016,10,30)
    assert period_start(5) == date(2021,10,30)

@pytest.mark.parametrize("bad", [0,4,11, "10", 10.0, True])
def test_period_rejects_invalid(bad):
    with pytest.raises(ValueError):
        period_start(bad)

def test_coverage_and_missing():
    index = pd.bdate_range("2016-10-31", "2026-10-30")
    assert covers_period(index, date(2016,10,30), CUTOFF)
    assert not covers_period(index, date(2015,10,30), CUTOFF)
    assert not covers_period(pd.DatetimeIndex([]), date(2016,10,30), CUTOFF)
    assert not covers_period(index, CUTOFF, CUTOFF)

def test_load_when_no_snapshots(tmp_path):
    assert load_covered_snapshot("JNJ", tmp_path, date(2016,10,30), CUTOFF) is None

def test_source_uses_local_ten_year_snapshots_only():
    app = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "years = 10" in app
    assert "load_covered_snapshot(ticker, STORE, first, last)" in app
    assert "run_10y_pipeline.py" in app
    assert "sha256_archivos" in app
    assert "Fecha (sesiones bursátiles diarias)" in app


def test_cli_download_all_mock(tmp_path, monkeypatch):
    from datetime import date
    import refresh_10y
    import pandas as pd
    seen = []
    def fake_download(symbol, start, end):
        seen.append((symbol, start, end))
        return pd.DataFrame(
            {"Close": [10.0] * 3},
            index=pd.DatetimeIndex(["2016-10-31", "2022-01-03", "2026-10-30"], name="Date")
        )
    def fake_save(frame, symbol, folder):
        return tmp_path / symbol / "fake.parquet"
    monkeypatch.setattr(refresh_10y, "download_ticker", fake_download)
    monkeypatch.setattr(refresh_10y, "save_snapshot", fake_save)
    assert refresh_10y.run(date(2026,10,30), tmp_path) == 0
    assert len(seen) == 5
    assert all(start == date(2016,10,30) and end == date(2026,10,31)
               for _, start, end in seen)

def test_cli_rejects_future(monkeypatch):
    import refresh_10y
    monkeypatch.setattr(refresh_10y, "run", lambda *args: pytest.fail("Should not run"))
    assert refresh_10y.main(["--until", "2099-01-01"]) == 2
