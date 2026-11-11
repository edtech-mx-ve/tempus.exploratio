"""Navegación informativa lateral de TEMPUS EXPLORATIO.

El lateral contiene únicamente enlaces de navegación. Las páginas completas
se dibujan en el espacio principal y pueden abrirse sin cargar cotizaciones.
"""
from __future__ import annotations

import streamlit as st

INFORMATION_PAGES = (
    "Análisis financiero",
    "Acerca de",
    "Dataset",
    "Fundamentos",
    "Trivia",
    "Ayuda",
    "Institucional",
)


def render_sidebar_information() -> str:
    """Devuelve la vista seleccionada, sin ejecutar análisis ni crear desplegables."""
    st.divider()
    st.caption("INFORMACIÓN DEL PROYECTO")
    return st.radio(
        "Explorar información",
        INFORMATION_PAGES,
        key="tempus_information_page",
        label_visibility="collapsed",
    )
