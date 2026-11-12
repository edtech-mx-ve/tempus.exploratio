"""TEMPUS EXPLORATIO · Módulo I · Interfaz operativa v0.5.5."""
from datetime import date
from pathlib import Path
from PIL import Image
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from tempus.assets import ASSETS
from tempus.sidebar_info import render_sidebar_information
from tempus.info_pages import render_information_page
from tempus.core import enrich_prices, summarize, normalize_prices
from tempus.demo import synthetic_series
from tempus.io import download_ticker, load_snapshot, list_snapshots, save_snapshot
from tempus.public_data import load_public_series
from tempus.forecasting import ForecastConfig, METHODS, walk_forward_evaluation
from tempus.future import FutureConfig, HORIZONS, forecast_future
from tempus.reporting import load_evidence, best_models
from tempus.statistics import (
    descriptive_statistics, stationarity_diagnostics, dependence_profile,
    align_returns, correlation_matrix, normalized_price_panel, drawdown_series,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
STORE = Path("data/snapshots")
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "tempus-logo.png"
if not LOGO_PATH.is_file():
    raise FileNotFoundError(f"Logotipo no encontrado: {LOGO_PATH}")
LOGO = Image.open(LOGO_PATH)

st.set_page_config(page_title="TEMPUS EXPLORATIO", page_icon=LOGO, layout="wide")
st.markdown("""
<style>
:root {--tempus-blue:#1767E8; --tempus-pink:#EF3299}
h1,h2,h3 {letter-spacing:-.025em}
div.stButton > button[kind="primary"] {background:#1767E8;color:white;border:0}
[data-testid="stMetric"] {border:1px solid #d9e5f7;border-radius:12px;padding:12px}
/* Navegación independiente: el contenedor que aloja el selector permanece
   dentro del bloque grande del workspace, lo que permite sticky hasta su final.
   No fijar el propio radio: su caja padre es demasiado corta. */
.st-key-tempus_workspace div[data-testid="stVerticalBlock"]
  > div[data-testid="stElementContainer"]:has(.st-key-tempus_navigation) {
    position: sticky !important;
    top: 3.5rem;
    z-index: 999;
    background: var(--background-color, #0e1117);
    border-bottom: 1px solid rgba(140, 145, 152, .45);
    padding: .35rem 0 .55rem;
    box-shadow: 0 4px 12px rgba(0,0,0,.18);
}
/* Fallback: diferentes versiones pueden variar sus envoltorios. */
.st-key-tempus_workspace .st-key-tempus_navigation {
    background: var(--background-color, #0e1117);
}
.st-key-tempus_navigation [data-testid="stRadio"] > div[role="radiogroup"] {
    flex-wrap: nowrap;
    overflow-x: auto;
    scrollbar-width: thin;
    padding-bottom: .25rem;
    gap: .15rem;
}
.st-key-tempus_navigation [data-testid="stRadio"] label {
    white-space: nowrap;
    padding: .35rem .55rem;
    border-radius: .3rem;
    cursor: pointer;
}
.st-key-tempus_navigation [data-testid="stRadio"] label:has(input:checked) {
    border-bottom: 2px solid #ef3299;
    color: #ef3299;
}
@media(max-width: 640px) {
    .st-key-tempus_workspace div[data-testid="stVerticalBlock"]
      > div[data-testid="stElementContainer"]:has(.st-key-tempus_navigation) {
        top: 3rem;
    }
}
</style>""", unsafe_allow_html=True)
col_logo, col_presentacion = st.columns([1, 8])
with col_logo:
    st.image(str(LOGO_PATH), width=108)
with col_presentacion:
    st.title("TEMPUS EXPLORATIO")
    st.caption("TEMPUS DECISIO · Módulo I · Exploración estadística y adquisición versionada")
st.info("v0.5.6 · Edición pública preparada para GitHub y Streamlit Community Cloud.")

with st.sidebar:
    st.header("Control de datos")
    ticker = st.selectbox("Activo financiero", list(ASSETS), format_func=lambda k: f"{k} · {ASSETS[k].company}")
    source = st.radio("Origen", ["Snapshot local", "Histórico incluido", "Yahoo Finance", "Demostración sintética"], index=0)
    first = st.date_input("Desde", value=date(2021, 11, 1))
    last = st.date_input("Hasta (inclusive)", value=date(2026, 10, 31))
    get_data = st.button("Cargar / adquirir datos", type="primary", width="stretch")
    selected_info_page = render_sidebar_information()

if get_data:
    try:
        if source == "Demostración sintética":
            frame = synthetic_series()
            origin = "SIMULACIÓN — sin datos reales"
        elif source == "Histórico incluido":
            frame = load_public_series(ticker)
            origin = "HISTÓRICO PÚBLICO INCLUIDO — corte 2026-10-30 (puede estar desactualizado)"
        elif source == "Yahoo Finance":
            if first >= last:
                raise ValueError("El inicio debe ser anterior al final")
            # yfinance interpreta el límite end como exclusivo
            frame = download_ticker(ticker, first, last + pd.Timedelta(days=1))
            saved = save_snapshot(frame, ticker, STORE)
            origin = f"YAHOO FINANCE — versión {saved.stem}"
        else:
            versions = list_snapshots(ticker, STORE)
            if versions:
                frame = load_snapshot(versions[0], STORE, ticker)
                origin = f"SNAPSHOT LOCAL — versión {versions[0].stem}"
            else:
                frame = load_public_series(ticker)
                origin = "HISTÓRICO PÚBLICO INCLUIDO — corte 2026-10-30 (sin snapshots locales)"
        st.session_state["frame"] = frame
        st.session_state["ticker"] = ticker
        st.session_state["origin"] = origin
    except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
        logging.exception("Error al cargar")
        st.error(f"No se pudo cargar la serie: {exc}")

# Las páginas informativas se muestran sin requerir datos cargados.
if selected_info_page != "Análisis financiero":
    render_information_page(selected_info_page, ticker)
    st.stop()

if "frame" not in st.session_state:
    st.markdown("### Inicia un análisis")
    st.write("Selecciona un origen y pulsa **Cargar / adquirir datos**. La opción **Histórico incluido** permite explorar las cinco series reales archivadas sin acceso a Yahoo Finance.")
    st.stop()

frame = normalize_prices(st.session_state["frame"])
ticker_used = st.session_state["ticker"]
if "SIMULACIÓN" in st.session_state["origin"]:
    st.warning(st.session_state["origin"])
else:
    st.success(st.session_state["origin"])
data = enrich_prices(frame)
summary = summarize(frame)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observaciones", f"{summary['observaciones']:,}")
col2.metric("Cierre final", f"{summary['cierre_final']:,.2f}")
col3.metric("Rentabilidad histórica", f"{summary['rendimiento_acumulado_pct']:.2f}%")
col4.metric("Volatilidad anualizada", f"{summary['volatilidad_anual_pct']:.2f}%")
NAV_LABELS = (
    "Precio e indicadores", "Rendimientos", "Datos y exportación",
    "Calidad", "Estadística temporal", "Comparación multiserie",
    "Pronósticos estadísticos", "Pronóstico futuro", "Informe final",
)

with st.container(key="tempus_workspace"):
    with st.container(key="tempus_navigation"):
        selected_tab = st.radio(
            "Sección de TEMPUS EXPLORATIO",
            NAV_LABELS,
            horizontal=True,
            label_visibility="collapsed",
            key="tempus_selected_section",
        )
    if selected_tab == NAV_LABELS[0]:
        display = data.reset_index()
        fig = px.line(display, x="Date", y=["Close", "sma_20", "sma_50"],
                      labels={"value":"Precio (USD)", "Date":"Fecha", "variable":"Serie"},
                      title=f"{ticker_used} · Cierre y medias móviles")
        st.plotly_chart(fig, width="stretch")
    if selected_tab == NAV_LABELS[1]:
        fig = px.histogram(data.reset_index(), x="return_1d", nbins=45,
                           title="Distribución de rendimientos diarios")
        st.plotly_chart(fig, width="stretch")
    if selected_tab == NAV_LABELS[2]:
        st.dataframe(data.tail(100), width="stretch")
        st.download_button("Descargar CSV enriquecido",
                           data=data.to_csv().encode("utf-8"),
                           file_name=f"tempus_{ticker_used}_indicadores.csv",
                           mime="text/csv")
    if selected_tab == NAV_LABELS[3]:
        st.json({"origen": st.session_state["origin"],
                 "activo": ASSETS[ticker_used].metadata(),
                 "periodo": [summary["inicio"], summary["fin"]],
                 "registros": len(frame),
                 "duplicados": int(frame.index.duplicated().sum()),
                 "faltantes_OHLCV": int(frame[["Open","High","Low","Close","Volume"]].isna().sum().sum())})

    if selected_tab == NAV_LABELS[4]:
        st.subheader("Laboratorio estadístico · " + ticker_used)
        st.caption("Pruebas exploratorias; no constituyen evidencia suficiente para pronosticar.")
        try:
            stats = descriptive_statistics(frame)
            st.dataframe(pd.DataFrame([stats]).T.rename(columns={0: "Valor"}), width="stretch")
            st.plotly_chart(
                px.line(drawdown_series(frame).reset_index(), x="Date", y="drawdown",
                        title="Caída desde máximo histórico (drawdown)",
                        labels={"drawdown": "Caída relativa", "Date": "Fecha"}),
                width="stretch",
            )
            target = st.radio("Variable para dependencia y estacionariedad",
                              ["Rendimientos diarios", "Cierre ajustado"], horizontal=True)
            selected = data["return_1d"] if target == "Rendimientos diarios" else data["Close"]
            alpha = st.select_slider("Nivel de significación", options=[0.01, 0.05, 0.10], value=0.05)
            max_lags = st.slider("Rezagos máximos", min_value=5, max_value=45, value=25)
            diag = stationarity_diagnostics(selected, significance=alpha)
            st.write("**ADF:** H₀ = presencia de raíz unitaria. **KPSS:** H₀ = estacionariedad en nivel.")
            m1, m2 = st.columns(2)
            m1.metric("ADF · valor p", f"{diag['adf_p']:.4f}")
            m2.metric("KPSS · valor p", f"{diag['kpss_p']:.4f}")
            st.write("**Interpretación:** " + diag["interpretacion"])
            if diag["advertencias"]:
                st.caption("Advertencias estadísticas: " + " | ".join(diag["advertencias"]))
            dependence = dependence_profile(selected, max_lags=max_lags)
            st.plotly_chart(
                px.bar(dependence.melt(id_vars="rezago", value_vars=["ACF", "PACF"],
                                        var_name="funcion", value_name="coeficiente"),
                       x="rezago", y="coeficiente", color="funcion", barmode="group",
                       title="ACF y PACF por rezago"),
                width="stretch",
            )
            st.download_button(
                "Descargar diagnóstico estadístico (CSV)",
                data=pd.DataFrame([{k: v for k, v in diag.items() if k != "advertencias"}])
                    .to_csv(index=False).encode("utf-8"),
                file_name=f"tempus_{ticker_used}_estacionariedad.csv",
                mime="text/csv",
            )
        except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
            st.warning(f"Análisis estadístico no disponible: {exc}")

    if selected_tab == NAV_LABELS[5]:
        st.subheader("Comparación de cinco series · snapshots locales")
        st.caption("Se emplean únicamente snapshots locales, sin nuevas descargas ni datos simulados.")
        frames: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for symbol in ASSETS:
            try:
                found = list_snapshots(symbol, STORE)
                if not found:
                    missing.append(symbol)
                    continue
                frames[symbol] = load_snapshot(found[0], STORE, symbol)
            except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
                logging.warning("Snapshot de %s no disponible: %s", symbol, type(exc).__name__)
                missing.append(symbol)
        if missing:
            st.warning("Faltan snapshots locales válidos: " + ", ".join(missing))
        if len(frames) >= 2:
            try:
                common_returns = align_returns(frames)
                st.caption(f"Observaciones alineadas de rendimientos: {len(common_returns):,}")
                panel = normalized_price_panel(frames)
                st.plotly_chart(
                    px.line(panel.reset_index().melt(id_vars="Date", var_name="ticker",
                                                      value_name="base100"),
                            x="Date", y="base100", color="ticker",
                            title="Índice base 100 desde la primera sesión común",
                            labels={"base100": "Índice (base 100)", "Date": "Fecha"}),
                    width="stretch",
                )
                correlation = correlation_matrix(frames)
                st.plotly_chart(
                    px.imshow(correlation, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                              text_auto=".2f", title="Correlación de Pearson de rendimientos diarios"),
                    width="stretch",
                )
                st.download_button(
                    "Descargar matriz de correlación (CSV)",
                    data=correlation.to_csv().encode("utf-8"),
                    file_name="tempus_correlaciones.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Descargar series normalizadas (CSV)",
                    data=panel.to_csv().encode("utf-8"),
                    file_name="tempus_series_base100.csv",
                    mime="text/csv",
                )
            except (ValueError, TypeError, RuntimeError) as exc:
                st.warning(f"No se pudo comparar las series: {exc}")
        else:
            st.info("Descarga y guarda por lo menos dos activos desde Yahoo Finance.")

    if selected_tab == NAV_LABELS[6]:
        st.subheader("Pronósticos y evaluación temporal · " + ticker_used)
        st.caption("Comparación a una sesión bursátil. Datos de la prueba nunca entran "
                   "en el entrenamiento del pronóstico que los evalúa.")
        c1, c2 = st.columns(2)
        with c1:
            test_fraction = st.select_slider(
                "Proporción de prueba cronológica",
                options=[0.10, 0.15, 0.20, 0.25, 0.30], value=0.20)
        with c2:
            step = st.select_slider(
                "Frecuencia de orígenes evaluados (sesiones)",
                options=[1, 5, 10, 20], value=10,
                help="El horizonte siempre es una sesión. Un paso de 10 evalúa "
                     "una fecha de cada diez para reducir costo computacional.")
        st.write("**Modelos:** pronóstico ingenuo, suavizamiento exponencial simple "
                 "y ARIMA(1,1,1), con parámetros fijados previamente.")
        st.caption("Los parámetros no se optimizan sobre el conjunto de prueba. "
                   "Este módulo compara modelos de referencia, no produce señales de inversión.")
        if st.button("Ejecutar evaluación walk-forward", type="primary", key="run_forecast"):
            try:
                with st.spinner("Estimando pronósticos históricos..."):
                    preds, scores, metadata = walk_forward_evaluation(
                        frame, ForecastConfig(test_fraction=test_fraction,
                                              step=step, min_train=100))
                st.session_state["forecast_result"] = (
                    ticker_used, st.session_state["origin"], test_fraction,
                    step, preds, scores, metadata)
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
                logging.exception("Error durante evaluación temporal")
                st.error(f"No fue posible evaluar la serie: {exc}")
        result = st.session_state.get("forecast_result")
        if result is not None:
            used, used_origin, used_fraction, used_step, preds, scores, metadata = result
            if (used == ticker_used and used_origin == st.session_state["origin"]
                    and used_fraction == test_fraction and used_step == step):
                st.success(f"Comparación completada: {metadata['evaluaciones_comunes']} "
                           f"fechas válidas comunes.")
                st.dataframe(scores, width="stretch", hide_index=True)
                show = preds[["observado", *METHODS]].dropna()
                plot_data = show.reset_index().melt(
                    id_vars="Date", var_name="serie", value_name="precio")
                st.plotly_chart(px.line(
                    plot_data, x="Date", y="precio", color="serie",
                    title="Pronóstico a una sesión y cierre observado"),
                    width="stretch")
                st.write("**Trazabilidad de la evaluación**")
                st.json({k: v for k, v in metadata.items() if k != "fallos"})
                if metadata["fallos"]:
                    st.warning(f"{len(metadata['fallos'])} ajustes fallidos. "
                               "Métricas calculadas solo en fechas comunes con "
                               "todos los modelos disponibles.")
                st.download_button("Descargar pronósticos (CSV)",
                                   data=preds.to_csv().encode("utf-8"),
                                   file_name=f"tempus_{ticker_used}_pronosticos.csv",
                                   mime="text/csv")
                st.download_button("Descargar métricas (CSV)",
                                   data=scores.to_csv(index=False).encode("utf-8"),
                                   file_name=f"tempus_{ticker_used}_metricas.csv",
                                   mime="text/csv")
                import json
                st.download_button("Descargar metadatos (JSON)",
                                   data=json.dumps(metadata, ensure_ascii=False, indent=2)
                                       .encode("utf-8"),
                                   file_name=f"tempus_{ticker_used}_evaluacion.json",
                                   mime="application/json")
            else:
                st.info("Hay resultados de otra configuración. Ejecuta la evaluación actual.")


    if selected_tab == NAV_LABELS[8]:
        st.subheader("Consolidación final · TEMPUS EXPLORATIO")
        st.caption("Métodos estadísticos utilizados: pronóstico ingenuo (último cierre), "
                   "suavizamiento exponencial simple (SES) y ARIMA(1,1,1). "
                   "Evaluación cronológica walk-forward con MAE, RMSE y sMAPE.")
        st.caption("Informe integral: evaluación histórica y pronósticos posteriores "
                   "al último cierre. Se utilizan exportaciones verificadas, sin descargar "
                   "cotizaciones ni volver a entrenar los modelos.")
        source_path = Path("evidence/sprint03")
        st.code(str(source_path.resolve()), language=None)
        if not source_path.is_dir():
            st.warning("No existe evidence/sprint03. Copia los CSV y JSON originales "
                       "antes de consolidar.")
        else:
            try:
                evidence = load_evidence(source_path)
                st.success("Evidencia verificada: cinco empresas, tres modelos y fechas comunes.")
                st.dataframe(evidence.metrics, width="stretch", hide_index=True)
                st.write("**Mejor modelo por MAE dentro de cada empresa**")
                st.dataframe(best_models(evidence.metrics), width="stretch", hide_index=True)
                future_dir = Path("reports/sprint05_futuro")
                st.write("**Pronósticos futuros para las cinco empresas**")
                st.caption(f"Directorio de resultados: {future_dir.resolve()}")
                from tempus.integrated_report import (
                    load_future_exports, create_integrated_pdf,
                )
                futures_ready = None
                try:
                    futures_ready = load_future_exports(future_dir, evidence)
                    summary_rows = []
                    for asset, (forecast_table, meta_future) in futures_ready.items():
                        for horizon in (1, 5, 10, 20):
                            subset = forecast_table.loc[
                                (forecast_table["sesion"] == horizon)
                                & (forecast_table["modelo"] == "ARIMA(1,1,1)")
                            ]
                            point = subset.iloc[0]
                            summary_rows.append({
                                "Empresa": asset,
                                "Último dato": meta_future["ultimo_dato_observado"],
                                "Horizonte": horizon,
                                "Fecha estimada": pd.Timestamp(point["Date"]).date(),
                                "ARIMA USD": round(float(point["pronostico"]), 4),
                            })
                    st.dataframe(pd.DataFrame(summary_rows), width="stretch",
                                 hide_index=True)
                    st.info("Las fechas estimadas son posteriores al último dato "
                            "del snapshot, pero podrían ser anteriores a hoy.")
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                    st.warning("No se pudo verificar el conjunto completo de pronósticos "
                               f"futuros: {exc}. Ejecuta future_cli.py para obtener "
                               "los horizontes de 1, 5, 10 y 20 sesiones.")
                if st.button("Generar informe final integral PDF y contrato",
                             type="primary", disabled=futures_ready is None):
                    from tempus.reporting import export_contract
                    from datetime import datetime, timezone
                    try:
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        destination = Path("reports") / f"integral_{stamp}"
                        pdf_file = destination / "TEMPUS_EXPLORATIO_Informe_Final_Integral.pdf"
                        # La generación de PDF verifica coherencia antes de publicar.
                        created_pdf = create_integrated_pdf(
                            evidence, futures_ready, pdf_file)
                        contract_files = export_contract(evidence, destination)
                        st.session_state["final_pdf"] = created_pdf.read_bytes()
                        st.session_state["final_pdf_filename"] = created_pdf.name
                        st.success("Informe integral generado: análisis histórico, "
                                   "evaluación y cinco gráficas de pronóstico futuro. "
                                   f"Contrato: {len(contract_files)} archivos. "
                                   f"Carpeta: {destination}")
                    except (OSError, ValueError, TypeError, RuntimeError,
                            ImportError, FileExistsError, KeyError) as exc:
                        logging.exception("Fallo al generar informe integral")
                        st.error(f"No se pudo generar el informe integral: {exc}")
                if "final_pdf" in st.session_state:
                    st.download_button("Descargar informe final integral PDF",
                        data=st.session_state["final_pdf"],
                        file_name=st.session_state["final_pdf_filename"],
                        mime="application/pdf")
            except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
                logging.exception("La evidencia no pasa validación")
                st.error(f"Datos no válidos para consolidación: {exc}")


    if selected_tab == NAV_LABELS[7]:
        st.subheader("Pronóstico futuro · " + ticker_used)
        st.caption("Se ajustan modelos con todo el histórico seleccionado y se estiman "
                   "las próximas sesiones NYSE después de la última observación disponible.")
        last_day = frame.index.max().date()
        st.info(f"Última fecha del conjunto cargado: {last_day}. "
                "Si el snapshot es antiguo, las 'fechas futuras' pueden estar ya en el pasado "
                "respecto al día actual. Actualiza desde Yahoo Finance para un pronóstico vigente.")
        horizon_future = st.select_slider("Horizonte futuro (sesiones bursátiles)",
                                          options=list(HORIZONS), value=5,
                                          help="Se generan todos los pasos 1..horizonte, no solo el último.")
        if st.button("Calcular pronóstico futuro", type="primary", key="run_future"):
            try:
                with st.spinner("Ajustando modelos con la serie histórica completa..."):
                    future_table, future_meta = forecast_future(
                        frame, FutureConfig(horizon=horizon_future))
                st.session_state["future_result"] = (
                    ticker_used, st.session_state["origin"], len(frame), str(last_day),
                    horizon_future, future_table, future_meta
                )
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
                logging.exception("No fue posible generar pronósticos futuros")
                st.error(f"Pronóstico futuro no disponible: {exc}")
        future_result = st.session_state.get("future_result")
        if future_result is not None:
            used, used_origin, used_n, used_end, used_horizon, estimates, future_info = future_result
            if (used == ticker_used and used_origin == st.session_state["origin"]
                    and used_n == len(frame) and used_end == str(last_day)
                    and used_horizon == horizon_future):
                st.success(f"{used_horizon} sesiones generadas; "
                           f"{len(estimates)} filas para {len(METHODS)} modelos.")
                st.dataframe(estimates, width="stretch", hide_index=True)
                history = frame[["Close"]].tail(60).rename(columns={"Close": "observado"}).reset_index()
                future_lines = estimates.rename(columns={"pronostico": "precio", "modelo": "serie"})
                previous = history.rename(columns={"observado": "precio"})
                previous["serie"] = "Histórico observado"
                lines = pd.concat([previous[["Date", "serie", "precio"]],
                                   future_lines[["Date", "serie", "precio"]]], ignore_index=True)
                chart = px.line(lines, x="Date", y="precio", color="serie", markers=True,
                                title="Últimos 60 cierres y estimaciones posteriores",
                                labels={"Date": "Fecha", "precio": "USD", "serie": "Modelo"})
                arima_bounds = estimates[estimates["modelo"] == "ARIMA(1,1,1)"].sort_values("Date")
                if not arima_bounds.empty:
                    chart.add_scatter(x=arima_bounds["Date"], y=arima_bounds["limite_superior"],
                                      mode="lines", line=dict(width=0), showlegend=False)
                    chart.add_scatter(x=arima_bounds["Date"], y=arima_bounds["limite_inferior"],
                                      mode="lines", line=dict(width=0), fill="tonexty",
                                      fillcolor="rgba(81,110,224,0.13)",
                                      name="ARIMA · intervalo condicional 95%")
                st.plotly_chart(chart, width="stretch")
                st.warning("Los intervalos del 95% se estiman solamente para ARIMA y dependen "
                           "de sus supuestos. Los modelos ingenuo y SES no tienen intervalo "
                           "estimado. El pronóstico no constituye asesoramiento de inversión.")
                st.json(future_info)
                import json
                st.download_button("Descargar pronóstico futuro (CSV)",
                                   data=estimates.to_csv(index=False).encode("utf-8"),
                                   file_name=f"tempus_{ticker_used}_futuro.csv",
                                   mime="text/csv")
                st.download_button("Descargar trazabilidad futura (JSON)",
                                   data=json.dumps(future_info, ensure_ascii=False, indent=2).encode("utf-8"),
                                   file_name=f"tempus_{ticker_used}_futuro.json",
                                   mime="application/json")
            else:
                st.info("Ejecuta el pronóstico con la selección actual.")
