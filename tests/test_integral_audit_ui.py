"""Pruebas del panel de auditoría, sin datos externos ni navegación web."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tempus.integral_audit import _report_latest, run_integral_audit


def test_panel_incluye_las_seis_graficas_y_leyendas():
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text(encoding="utf-8")
    ui = (root / "tempus" / "integral_audit_ui.py").read_text(encoding="utf-8")
    assert '"Prueba integral"' in app
    assert 'render_integral_audit(STORE, Path("reports"))' in app
    for label in (
        "Precio (USD/acción)", "Rendimiento (%)",
        "Volatilidad anualizada (%)", "Acciones negociadas",
        "Índice (base 100)", "Fecha (sesiones bursátiles diarias)"
    ):
        assert label in ui
    assert "Ejecutar prueba integral" in ui


def test_auditoria_sin_snapshots_no_declara_exito(tmp_path):
    result = run_integral_audit(tmp_path / "snapshots", tmp_path / "reports")
    assert not result.passed
    assert result.summary.empty
    assert result.report_path is None
    assert "ERROR" in set(result.checks["Estado"])


def test_manifiesto_malformado_se_descarta(tmp_path):
    manifest = tmp_path / "integral_10y_test" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"version":"incorrecta","tickers":["JNJ"]}', encoding="utf-8")
    found, data = _report_latest(tmp_path)
    assert found is None and data is None


def test_no_se_reentrenan_modelos_en_panel():
    from tempus import integral_audit as audit
    import inspect
    code = inspect.getsource(audit)
    assert "walk_forward_evaluation(" not in code
    assert "forecast_future(" not in code
