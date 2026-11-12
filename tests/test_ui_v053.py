"""Pruebas de regresión de la navegación persistente y limpieza del diagnóstico."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_navegacion_nueve_secciones_y_un_panel_por_seleccion():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    labels = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "NAV_LABELS"
                          for target in node.targets))
    assert len(labels.value.elts) == 10
    assert source.count("if selected_tab == NAV_LABELS[") == 9
    assert "st.tabs(" not in source
    assert "st.radio(" in source


def test_no_muestra_inspeccion_de_streamlit():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"help", "dir", "vars"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            assert not (isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "st"
                        and node.func.attr == "help")
        if isinstance(node.func, ast.Name) and node.func.id in banned:
            assert not any(isinstance(arg, ast.Name) and arg.id == "st"
                           for arg in node.args)


def test_anclaje_sticky_atado_al_workspace():
    source = APP.read_text(encoding="utf-8")
    assert ".st-key-tempus_workspace" in source
    assert ":has(.st-key-tempus_navigation)" in source
    assert "position: sticky !important;" in source
    assert "overflow-x: auto;" in source
