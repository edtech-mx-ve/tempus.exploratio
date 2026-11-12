from pathlib import Path
from datetime import date
import ast

def test_single_ten_year_period():
    src = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "years = 10" in src
    assert "first = period_start(years, CUTOFF)" in src
    assert 'st.radio("Periodo histórico"' not in src
    assert '"5 años"' not in src
    assert 'if years != 5' not in src

def test_local_snapshots_first():
    src = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert '["Snapshot local", "Yahoo Finance"], index=0' in src
    assert 'load_covered_snapshot(ticker, STORE, first, last)' in src
    assert 'not covers_period(frame.index, first, last)' in src

def test_old_report_not_misrepresented():
    src = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert "run_10y_pipeline.py" in src
    assert "integral_10y_*/manifest.json" in src
    assert "sha256_archivos" in src
    assert "load_evidence(source_path)" not in src
