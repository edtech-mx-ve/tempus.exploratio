"""TEMPUS EXPLORATIO · Módulo I · Interfaz operativa v0.6.2."""
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
    background: #F3F6FB;
    border-bottom: 1px solid rgba(140, 145, 152, .45);
    padding: .35rem 0 .55rem;
    box-shadow: 0 4px 12px rgba(0,0,0,.18);
}
/* Fallback: diferentes versiones pueden variar sus envoltorios. */
.st-key-tempus_workspace .st-key-tempus_navigation {
    background: #F3F6FB;
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
    color: #172033 !important;
}
.st-key-tempus_navigation [data-testid="stRadio"] label p {
    color: #172033 !important;
}
.st-key-tempus_navigation [data-testid="stRadio"] label:has(input:checked) {
    border-bottom: 2px solid #ef3299;
    color: #ef3299;
}
.st-key-tempus_navigation [data-testid="stRadio"] label:has(input:checked) p {
    color: #b60062 !important;
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
st.info("v0.6.2 · Histórico real de 10 años · Frecuencia diaria bursátil · Pronósticos y evaluación.")

from tempus.history_period import CUTOFF, period_start, covers_period, load_covered_snapshot

# Único periodo admitido en la edición local: 10 años de sesiones bursátiles.
years = 10
first = period_start(years, CUTOFF)
last = CUTOFF

with st.sidebar:
    st.header("Control de datos")
    ticker = st.selectbox("Activo financiero", list(ASSETS),
                          format_func=lambda k: f"{k} · {ASSETS[k].company}")
    st.caption(f"**Periodo histórico: 10 años** · {first:%d/%m/%Y} – {last:%d/%m/%Y}")
    source = st.radio("Origen", ["Snapshot local", "Yahoo Finance"], index=0,
                      help="Snapshot local: datos de diez años ya descargados. "
                           "Yahoo Finance: adquiere una nueva versión real.")
    get_data = st.button("Cargar / adquirir datos", type="primary", width="stretch")
    download_all = st.button("Descargar las 5 empresas (10 años)",
                             help="Descarga y versiona las cinco series reales desde Yahoo Finance.")
    st.caption("Frecuencia: diaria bursátil · Precios: USD/acción · "
               "10 años ≈ 2.500 sesiones por empresa.")
    selected_info_page = render_sidebar_information()

if get_data or download_all:
    try:
        if download_all:
            errors = []
            completed = []
            with st.spinner("Descargando y guardando diez años de las cinco series..."):
                for symbol in ASSETS:
                    try:
                        data_new = download_ticker(symbol, first, last + pd.Timedelta(days=1))
                        if not covers_period(data_new.index, first, last):
                            raise ValueError("La descarga no cubre los diez años solicitados.")
                        saved_new = save_snapshot(data_new, symbol, STORE)
                        completed.append(symbol)
                        if symbol == ticker:
                            st.session_state["frame"] = data_new
                            st.session_state["ticker"] = ticker
                            st.session_state["origin"] = (
                                f"YAHOO FINANCE · 10 años · {saved_new.stem}"
                            )
                    except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
                        logging.exception("Error de descarga de %s", symbol)
                        errors.append(f"{symbol}: {exc}")
            if completed:
                st.success("Guardadas versiones nuevas: " + ", ".join(completed))
            if errors:
                st.error("Descargas pendientes: " + " | ".join(errors))
        else:
            if source == "Yahoo Finance":
                frame = download_ticker(ticker, first, last + pd.Timedelta(days=1))
                if not covers_period(frame.index, first, last):
                    raise ValueError("Los datos descargados no cubren los diez años solicitados.")
                saved = save_snapshot(frame, ticker, STORE)
                origin = f"YAHOO FINANCE · 10 años — versión {saved.stem}"
            else:
                chosen = load_covered_snapshot(ticker, STORE, first, last)
                if chosen is None:
                    raise FileNotFoundError(
                        "No existe un snapshot local válido de diez años. "
                        "Pulsa «Descargar las 5 empresas (10 años)»."
                    )
                frame, selected_snapshot = chosen
                origin = f"SNAPSHOT LOCAL · 10 años — {selected_snapshot.stem}"
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
    st.write("Selecciona **Snapshot local** y pulsa **Cargar / adquirir datos** para visualizar los diez años descargados.")
    st.stop()

frame = normalize_prices(st.session_state["frame"])
if st.session_state["ticker"] != ticker or not covers_period(frame.index, first, last):
    st.warning("La serie cargada no coincide con el activo o el periodo seleccionado. "
               "Pulsa «Cargar / adquirir datos» para actualizarla.")
    st.stop()
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
    "Prueba integral",
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
                      labels={"value":"Precio (USD/acción)", "Date":"Fecha (sesiones bursátiles diarias)", "variable":"Serie"},
                      title=f"{ticker_used} · Cierre y medias móviles (frecuencia diaria bursátil)")
        st.caption("MA20 y MA50 corresponden a 20 y 50 sesiones bursátiles, respectivamente.")
        st.plotly_chart(fig, width="stretch")
    if selected_tab == NAV_LABELS[1]:
        rendimientos_pct = data.reset_index().assign(rendimiento_pct=lambda df: df["return_1d"] * 100)
        fig = px.histogram(rendimientos_pct, x="rendimiento_pct", nbins=45,
                           title="Distribución de rendimientos por sesión bursátil",
                           labels={"rendimiento_pct":"Rendimiento diario (%)", "count":"Número de sesiones bursátiles"})
        fig.update_layout(yaxis_title="Número de sesiones bursátiles")
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
            drawdown_pct = drawdown_series(frame).reset_index().assign(
                drawdown_pct=lambda df: df["drawdown"] * 100)
            st.plotly_chart(
                px.line(drawdown_pct, x="Date", y="drawdown_pct",
                        title="Caída desde máximo histórico (drawdown)",
                        labels={"drawdown_pct": "Caída desde máximo (%)",
                                "Date": "Fecha (sesiones bursátiles diarias)"}),
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
                       title="ACF y PACF por rezago",
                       labels={"rezago": "Rezago (sesiones bursátiles)",
                               "coeficiente": "Correlación (sin unidad)", "funcion": "Función"}),
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
        st.subheader("Comparación de cinco series · snapshots del periodo seleccionado")
        st.caption("Se emplean snapshots locales íntegros con cobertura del periodo; no se descargan datos aquí.")
        frames: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for symbol in ASSETS:
            try:
                covered = load_covered_snapshot(symbol, STORE, first, last)
                if covered is None:
                    missing.append(symbol)
                    continue
                frames[symbol] = covered[0]
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
                            labels={"base100": "Índice (base 100)", "Date": "Fecha (sesiones bursátiles diarias)"}),
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
                    title="Pronóstico a 1 sesión bursátil y cierre observado", labels={"Date":"Fecha (sesiones bursátiles diarias)", "precio":"Precio (USD/acción)"}),
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
        st.subheader("Consolidación final · diez años")
        st.caption("Cinco series diarias bursátiles; modelos Ingenuo, "
                   "Suavizamiento Exponencial Simple y ARIMA(1,1,1). "
                   "Evaluación walk-forward: MAE, RMSE y sMAPE.")
        st.info("Análisis integral de 10 años (2016–2026): las métricas y los "
                "pronósticos se calcularon automáticamente utilizando 12.570 "
                "registros históricos reales de cinco empresas, sin necesidad "
                "de ejecutar cada análisis manualmente.")
        st.code(
            r".\\.venv\\Scripts\\python.exe run_10y_pipeline.py "
            r"--snapshots data/snapshots --output-root reports",
            language="powershell",
        )
        import json
        from hashlib import sha256
        generated = sorted(Path("reports").glob("integral_10y_*/manifest.json"),
                           reverse=True)
        if not generated:
            st.warning("Todavía no existe un informe consolidado de diez años. "
                       "Ejecuta el comando anterior en PowerShell.")
        else:
            try:
                manifest_path = generated[0]
                summary_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if summary_data.get("version") != "10y.1" or set(summary_data.get("tickers", [])) != set(ASSETS):
                    raise ValueError("Manifiesto de evaluación no válido")
                pdf_name = summary_data["pdf"]
                if Path(pdf_name).name != pdf_name:
                    raise ValueError("Nombre de PDF inseguro")
                pdf_path = manifest_path.parent / pdf_name
                digest = sha256(pdf_path.read_bytes()).hexdigest()
                if digest != summary_data["sha256_archivos"].get(pdf_name):
                    raise ValueError("El PDF no coincide con su manifiesto SHA-256")
                current_snapshot = load_covered_snapshot(ticker_used, STORE, first, last)
                if current_snapshot is None:
                    raise ValueError("No hay snapshot de diez años para verificar el informe")
                _, current_path = current_snapshot
                loaded_digest = json.loads(
                    current_path.with_suffix(".json").read_text(encoding="utf-8")
                )["sha256"]
                if loaded_digest != summary_data["sources"][ticker_used]["sha256"]:
                    raise ValueError("El informe fue generado desde otro snapshot. "
                                     "Reejecuta la consolidación.")
                st.success("Informe integral de diez años generado y verificado.")
                st.write(f"**Periodo:** {summary_data['periodo'][0]} a {summary_data['periodo'][1]}")
                st.write(f"**Registros:** {summary_data['registros_totales']:,} · "
                         f"**Predicciones retrospectivas:** {summary_data['evaluaciones_totales']:,}")
                st.dataframe(pd.DataFrame([
                    {"Empresa": t, "Observaciones": summary_data["observaciones_por_empresa"][t],
                     "Evaluaciones": summary_data["evaluaciones_por_empresa"][t],
                     "Menor MAE": summary_data["mejor_por_mae"][t]}
                    for t in ASSETS
                ]), hide_index=True, width="stretch")
                st.download_button(
                    "Descargar informe integral PDF · 10 años",
                    data=pdf_path.read_bytes(), file_name=pdf_path.name,
                    mime="application/pdf",
                )
                st.caption(f"Evidencias y trazabilidad: {manifest_path.parent}")
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                st.error(f"No se pudo validar el informe guardado: {exc}")

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
                                title=f"Últimos 60 cierres y pronóstico a {used_horizon} sesiones bursátiles",
                                labels={"Date": "Fecha (sesiones bursátiles diarias)", "precio": "Precio (USD/acción)", "serie": "Modelo"})
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

    if selected_tab == "Prueba integral":
        from tempus.integral_audit_ui import render_integral_audit
        render_integral_audit(STORE, Path("reports"))
