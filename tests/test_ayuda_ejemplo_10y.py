from pathlib import Path

def test_ayuda_diez_anios_y_ejemplo():
    text = (Path(__file__).resolve().parents[1] / "tempus" / "info_pages.py").read_text(encoding="utf-8")
    assert 'elif page == "Ayuda":' in text
    assert "Ejemplo práctico · Prueba integral con JNJ" in text
    assert "2016-10-31 a 2026-10-30" in text
    assert "2.514 registros" in text
    assert "Ejecutar prueba integral" in text
    assert "no simula automáticamente clics" in text
    assert "selecciona 5 o 10 años" not in text.lower()

def test_fundamentos_y_unidades():
    text = (Path(__file__).resolve().parents[1] / "tempus" / "info_pages.py").read_text(encoding="utf-8")
    for term in ("MA20 y MA50", "sMAPE", "ADF/KPSS", "ACF/PACF", "USD/acción", "volatilidad anualizada"):
        assert term.lower() in text.lower()
