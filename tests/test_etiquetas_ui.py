"""Pruebas estáticas de presentación: no alteran modelos ni snapshots."""
from pathlib import Path
import ast

SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def test_sintaxis_de_app():
    ast.parse(SOURCE.read_text(encoding="utf-8"))


def test_unidades_explicitas():
    text = SOURCE.read_text(encoding="utf-8")
    for token in (
        "Precio (USD/acción)",
        "Fecha (sesiones bursátiles diarias)",
        "Rendimiento diario (%)",
        "Caída desde máximo (%)",
        "Índice (base 100)",
        "Rezago (sesiones bursátiles)",
        "MA20 y MA50 corresponden",
        "sesiones bursátiles",
    ):
        assert token in text, token


def test_no_modifica_valores_estadisticos():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'df["return_1d"] * 100' in text
    assert 'df["drawdown"] * 100' in text
    assert 'forecast_future(' in text
    assert 'walk_forward_evaluation(' in text


def test_contraste_menu_tema_claro():
    text = SOURCE.read_text(encoding="utf-8")
    assert "background: #F3F6FB;" in text
    assert "color: #172033 !important;" in text
