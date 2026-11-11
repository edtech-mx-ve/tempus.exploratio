"""Páginas informativas centrales de TEMPUS EXPLORATIO.

No ejecutan pronósticos ni descargan datos. Su único efecto es renderizar
contenido estático sustentado en las características de la aplicación.
"""
from __future__ import annotations

import streamlit as st

from tempus.assets import ASSETS
from tempus.sidebar_info import INFORMATION_PAGES


def _intro(title: str, subtitle: str) -> None:
    st.header(title)
    st.caption(subtitle)


def render_information_page(page: str, ticker: str) -> None:
    """Renderiza una página completa y accesible en el panel principal.

    Args:
        page: Una de las seis páginas informativas.
        ticker: Símbolo seleccionado en el control lateral.

    Raises:
        ValueError: Si el nombre de la página o el símbolo no se admite.
    """
    if page not in INFORMATION_PAGES[1:]:
        raise ValueError(f"Página informativa no reconocida: {page}")
    if ticker not in ASSETS:
        raise ValueError(f"Activo no admitido: {ticker}")

    if page == "Acerca de":
        _intro("Acerca de TEMPUS EXPLORATIO",
               "TEMPUS DECISIO · Módulo I · Análisis estadístico de series financieras")
        st.markdown("""
**TEMPUS EXPLORATIO** es una aplicación web académica desarrollada en
Python para adquirir, explorar, comparar y pronosticar cinco series temporales
financieras obtenidas de la web.

Su propósito es comprender la relación entre **histórico, patrones temporales,
evaluación cronológica y estimaciones posteriores al último dato**.
""")
        a, b, c = st.columns(3)
        a.metric("Series financieras", "5")
        b.metric("Métodos de pronóstico", "3")
        c.metric("Horizontes disponibles", "1 · 5 · 10 · 20")
        st.subheader("Recorrido de análisis")
        st.markdown(
            "**Adquisición y snapshots → Calidad → Exploración y estadística → "
            "Comparación de series → Evaluación walk-forward → "
            "Pronósticos futuros → Informe final.**"
        )
        st.markdown("""
**Métodos implementados:** Ingenuo, Suavizamiento Exponencial Simple (SES)
y ARIMA(1,1,1). Su desempeño se compara mediante MAE, RMSE y sMAPE.

Los pronósticos son experimentales y no constituyen recomendaciones de inversión.
""")

    elif page == "Dataset":
        _intro("Dataset · Cinco series financieras",
               "Origen web, estructura, cobertura y trazabilidad de los datos")
        st.markdown(
            "Las cotizaciones se adquieren de **Yahoo Finance**, mediante "
            "`yfinance`, y se conservan como **snapshots locales versionados**."
        )
        st.dataframe(
            [
                {"Símbolo": "JNJ", "Empresa": "Johnson & Johnson", "Sector": "Salud"},
                {"Símbolo": "MSFT", "Empresa": "Microsoft", "Sector": "Tecnología"},
                {"Símbolo": "JPM", "Empresa": "JPMorgan Chase", "Sector": "Banca"},
                {"Símbolo": "WMT", "Empresa": "Walmart", "Sector": "Comercio minorista"},
                {"Símbolo": "XOM", "Empresa": "ExxonMobil", "Sector": "Energía"},
            ],
            hide_index=True,
            width="stretch",
        )
        st.info(f"Activo seleccionado: {ticker} · {ASSETS[ticker].company}")
        st.subheader("Variables y tratamiento")
        st.markdown("""
La base temporal utiliza **fecha, apertura, máximo, mínimo, cierre y volumen**
(OHLCV). A partir del cierre se calculan indicadores, rendimientos y
medidas descriptivas y de riesgo. La cobertura y el número de registros
dependen del snapshot que se haya cargado.
""")
        st.subheader("¿Qué relación tiene con Big Data?")
        st.markdown("""
Los mercados generan información financiera de forma continua.
En esta práctica se estudian **cinco series históricas provenientes de la web**;
el tamaño de los snapshots locales no demuestra procesamiento distribuido
de grandes volúmenes ni adquisición en tiempo real.
""")
        st.markdown(
            "**Fuentes y herramientas:** [Yahoo Finance](https://finance.yahoo.com/) · "
            "[yfinance](https://github.com/ranaroussi/yfinance)"
        )

    elif page == "Fundamentos":
        _intro("Fundamentos teóricos",
               "Conceptos necesarios para interpretar las estadísticas y los pronósticos")
        st.subheader("1. Serie temporal y calidad de datos")
        st.markdown(
            "Una serie temporal reúne observaciones ordenadas cronológicamente. "
            "La limpieza comprueba fechas, duplicados, valores faltantes y rangos válidos."
        )
        st.subheader("2. Exploración y comparación")
        st.markdown(
            "Se estudian rendimientos, medias móviles, volatilidad anualizada, "
            "correlaciones, drawdown y diagnósticos de estacionariedad y dependencia."
        )
        st.subheader("3. Métodos estadísticos utilizados")
        st.markdown("""
- **Ingenuo (Naive):** utiliza el último cierre conocido como estimación.
- **Suavizamiento Exponencial Simple (SES):** da mayor peso a observaciones recientes.
- **ARIMA(1,1,1):** modela relaciones temporales mediante autorregresión,
  diferenciación y media móvil.
""")
        st.subheader("4. Evaluación temporal")
        st.markdown(
            "La evaluación **walk-forward** respeta el orden temporal. "
            "Las métricas **MAE**, **RMSE** y **sMAPE** comparan las estimaciones "
            "con las observaciones posteriores disponibles."
        )
        st.subheader("5. Extrapolación e incertidumbre")
        st.markdown(
            "Los horizontes disponibles son **1, 5, 10 y 20 sesiones NYSE**. "
            "Los intervalos condicionales del **95 %** se estiman para ARIMA; "
            "los otros métodos se presentan sin intervalos calculados. "
            "Los intervalos no garantizan cobertura ante cambios estructurales."
        )

    elif page == "Trivia":
        _intro("Trivia · TEMPUS DECISIO",
               "Nombre, concepto y curiosidades del análisis financiero temporal")
        st.subheader("¿Qué significa TEMPUS DECISIO?")
        st.markdown(
            "**Tempus** alude al tiempo; **Decisio**, a la decisión. "
            "La idea central es analizar evidencia temporal para apoyar decisiones."
        )
        st.subheader("¿Qué significa TEMPUS EXPLORATIO?")
        st.markdown(
            "Es la etapa dedicada a explorar los datos, estudiar patrones, "
            "comparar métodos estadísticos y documentar estimaciones."
        )
        st.subheader("¿Por qué cinco empresas?")
        st.markdown(
            "Permiten contrastar series de salud, tecnología, banca, "
            "comercio minorista y energía bajo un mismo procedimiento."
        )
        st.subheader("¿Por qué los pronósticos utilizan sesiones bursátiles?")
        st.markdown(
            "La Bolsa de Nueva York no opera todos los días. "
            "El calendario **XNYS** evita asignar sesiones a fines de semana "
            "y feriados bursátiles."
        )

    elif page == "Ayuda":
        _intro("Ayuda · Recorrido rápido",
               "Cómo utilizar TEMPUS EXPLORATIO de principio a fin")
        st.markdown("""
1. En **Control de datos**, elige empresa, fuente y fechas.
2. Pulsa **Cargar / adquirir datos**.
3. Revisa **Precio e indicadores**, **Rendimientos**, **Calidad** y
   **Estadística temporal**.
4. Examina **Comparación multiserie** y **Pronósticos estadísticos**.
5. En **Pronóstico futuro**, elige horizonte y calcula las estimaciones.
6. Finaliza en **Informe final** para consolidar los resultados y descargar el PDF.
""")
        st.subheader("Problemas frecuentes")
        st.markdown("""
- **No hay snapshots:** adquiere los datos desde Yahoo Finance o emplea
  la demostración sintética, que se identifica explícitamente.
- **No aparece el resultado esperado:** comprueba que el botón de cálculo
  correspondiente se haya ejecutado.
- **No se ven cambios de interfaz:** recarga el navegador con `Ctrl + F5`.
- **Pronóstico con fecha anterior a hoy:** actualiza las cotizaciones y
  vuelve a calcular; «futuro» significa posterior al último dato del archivo.
""")
        st.caption(
            "Ejecución local habitual: python -m streamlit run app.py · "
            "http://localhost:8501"
        )

    elif page == "Institucional":
        _intro("Institucional", "Identificación académica y técnica del proyecto")
        st.markdown("**PROYECTO ACADÉMICO**")
        st.subheader("Instituto Internacional de Aguascalientes")
        st.markdown("**Maestría en Inteligencia Artificial para la Transformación Digital**")
        st.markdown("""
**Asignatura:** Tópicos de IA para la Transformación Digital

**Aplicación:** TEMPUS DECISIO — TEMPUS EXPLORATIO

**Módulo:** I — Exploración de series financieras y pronóstico estadístico

**Alumno:** Antonio Nicolás Toro González

**Tutor:** Dr. Francisco Javier Luna Rosas
""")
        st.subheader("Descripción")
        st.markdown(
            "Aplicación web didáctica en Python que analiza cinco series históricas "
            "obtenidas de la web, aplica validaciones y estadística temporal, "
            "evalúa tres métodos de pronóstico y genera un informe integral."
        )
        st.subheader("Identificación técnica")
        st.dataframe(
            [
                {"Componente": "Lenguaje", "Tecnología": "Python"},
                {"Componente": "Interfaz", "Tecnología": "Streamlit y Plotly"},
                {"Componente": "Datos", "Tecnología": "Yahoo Finance, yfinance, Parquet"},
                {"Componente": "Análisis", "Tecnología": "pandas, NumPy, statsmodels"},
                {"Componente": "Métodos", "Tecnología": "Ingenuo, SES, ARIMA(1,1,1)"},
                {"Componente": "Evaluación", "Tecnología": "Walk-forward, MAE, RMSE, sMAPE"},
                {"Componente": "Resultados", "Tecnología": "CSV, JSON e informe PDF"},
            ],
            hide_index=True,
            width="stretch",
        )
