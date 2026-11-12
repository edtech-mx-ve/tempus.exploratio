"""Verificaciones de regresión de la navegación y terminología v0.5.2."""
from pathlib import Path


def test_pestanas_permanecen_visibles_al_desplazarse():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'key="tempus_workspace"' in source
    assert 'key="tempus_navigation"' in source
    assert 'st.radio(' in source
    assert 'position: sticky !important;' in source
    assert 'z-index: 999;' in source
    assert "Informe final" in source
    assert "Pronóstico futuro" in source


def test_metodos_estadisticos_explicitos_en_interfaz_y_reporte():
    base = Path(__file__).resolve().parents[1]
    app = (base / "app.py").read_text(encoding="utf-8")
    report = (base / "tempus" / "report_pdf.py").read_text(encoding="utf-8")
    assert "pronóstico ingenuo" in app
    assert "suavizamiento exponencial simple" in app
    assert "(SES) y ARIMA(1,1,1)" in report
    assert "ARIMA(1,1,1)" in app
    assert "MAE, RMSE y sMAPE" in app
    assert "Métodos estadísticos utilizados:" in report
    assert "(SES) y ARIMA(1,1,1)" in report
