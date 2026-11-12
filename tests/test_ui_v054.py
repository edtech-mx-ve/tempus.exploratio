"""Regresión para el panel informativo y correcciones de TEMPUS v0.5.4."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
INFO = ROOT / "tempus" / "sidebar_info.py"


def test_seis_opciones_informativas():
    contents = INFO.read_text(encoding="utf-8")
    for label in ("Acerca de", "Dataset", "Fundamentos", "Trivia", "Ayuda", "Institucional"):
        assert f'"{label}"' in contents
    assert "render_sidebar_information()" in APP.read_text(encoding="utf-8")
    assert "st.expander(" not in contents


def test_terminologia_estadistica_y_contexto():
    contents = (ROOT / "tempus" / "info_pages.py").read_text(encoding="utf-8")
    for term in ("Ingenuo", "Suavizamiento Exponencial Simple",
                 "ARIMA(1,1,1)", "MAE", "RMSE", "sMAPE", "Yahoo Finance",
                 "JNJ", "MSFT", "JPM", "WMT", "XOM"):
        assert term in contents


def test_informe_final_ultimo_y_rutas_conservadas():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "NAV_LABELS"
                for t in node.targets)
    )
    labels = [node.value for node in assignment.value.elts]
    assert len(labels) == 10
    assert labels[-3:] == ["Pronóstico futuro", "Informe final", "Prueba integral"]
    text = APP.read_text(encoding="utf-8")
    assert text.count("if selected_tab == NAV_LABELS[") == 9
    assert 'if selected_tab == NAV_LABELS[7]:' in text
    assert 'if selected_tab == NAV_LABELS[8]:' in text


def test_no_condicional_de_visualizacion_automatica():
    text = APP.read_text(encoding="utf-8")
    assert 'st.warning(st.session_state["origin"]) if ' not in text
    assert 'if "SIMULACIÓN" in st.session_state["origin"]:' in text
    assert '    st.success(st.session_state["origin"])' in text
