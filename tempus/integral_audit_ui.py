"""Sección compacta para ejecutar y presentar una auditoría de diez años."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from tempus.integral_audit import AuditResult, run_integral_audit


def _draw_chart(result: AuditResult, view: str) -> None:
    """Etiquetas explícitas, leyenda por empresa, fechas y unidad."""
    x_title = "Fecha (sesiones bursátiles diarias)"
    settings = {
        "Precio histórico": (result.prices, "Precio (USD/acción)",
                             "Cierres históricos · 10 años"),
        "Rendimiento diario": (result.returns, "Rendimiento (%)",
                              "Rendimiento por sesión bursátil"),
        "Volatilidad": (result.volatility, "Volatilidad anualizada (%)",
                        "Volatilidad móvil de 20 sesiones, anualizada (√252)"),
        "Volumen": (result.volume, "Acciones negociadas",
                    "Volumen diario de negociación"),
    }
    if view == "Índice base 100":
        if result.base100.empty:
            st.warning("Comparación normalizada no disponible.")
            return
        melted = result.base100.reset_index().melt(
            id_vars="Date", var_name="Empresa", value_name="Índice (base 100)"
        )
        chart = px.line(
            melted, x="Date", y="Índice (base 100)", color="Empresa",
            title="Comparación normalizada · base 100 en la primera sesión común",
            labels={"Date": x_title, "Empresa": "Empresa"},
        )
    elif view == "Correlación":
        if result.correlation.empty:
            st.warning("Matriz de correlación no disponible.")
            return
        chart = px.imshow(result.correlation, zmin=-1, zmax=1,
                          color_continuous_scale="RdBu_r", text_auto=".2f",
                          labels={"color": "Coeficiente r"},
                          title="Correlación de Pearson de rendimientos diarios")
        chart.update_xaxes(title="Empresa")
        chart.update_yaxes(title="Empresa")
    else:
        frame, y_label, title = settings[view]
        if frame.empty:
            st.warning("No hay datos para esta gráfica.")
            return
        chart = px.line(
            frame, x="Fecha", y=y_label, color="Empresa", title=title,
            labels={"Fecha": x_title, "Empresa": "Empresa"},
        )
    chart.update_layout(
        legend_title_text="Empresa", hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=25),
    )
    st.plotly_chart(chart, width="stretch")


def render_integral_audit(
    snapshots_dir: Path = Path("data/snapshots"),
    reports_dir: Path = Path("reports"),
) -> None:
    st.subheader("Prueba integral · 10 años")
    st.caption(
        "Comprueba snapshots, indicadores, ADF/KPSS, ACF/PACF, "
        "comparación de las cinco empresas, evaluaciones, pronósticos y hashes del PDF. "
        "No descarga datos ni reentrena modelos."
    )
    if st.button("Ejecutar prueba integral", type="primary", key="tempus_integral_audit_run"):
        with st.spinner("Verificando los datos y las evidencias de diez años..."):
            try:
                st.session_state["tempus_integral_audit_result"] = run_integral_audit(
                    Path(snapshots_dir), Path(reports_dir)
                )
            except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                st.session_state.pop("tempus_integral_audit_result", None)
                st.error(f"La prueba integral no pudo completarse: {exc}")

    result: AuditResult | None = st.session_state.get("tempus_integral_audit_result")
    if result is None:
        st.info("Pulsa «Ejecutar prueba integral» para revisar los resultados guardados.")
        return

    passed = int(result.checks["Estado"].eq("OK").sum())
    total = len(result.checks)
    a, b, c = st.columns(3)
    a.metric("Comprobaciones aprobadas", f"{passed}/{total}")
    b.metric("Registros", f"{int(result.summary['Sesiones'].sum()):,}" if not result.summary.empty else "0")
    c.metric("Empresas validadas", f"{len(result.summary)}/5")
    if result.passed:
        st.success("Cálculos y archivos del informe: comprobaciones superadas.")
    else:
        st.error("Existen comprobaciones fallidas. Revisa la tabla antes de usar el informe.")

    st.dataframe(result.checks, width="stretch", hide_index=True)
    st.download_button(
        "Descargar resultado de la prueba (CSV)",
        data=result.checks.to_csv(index=False).encode("utf-8-sig"),
        file_name="tempus_auditoria_integral_10y.csv", mime="text/csv",
        key="tempus_integral_audit_csv",
    )

    if result.summary.empty:
        return

    st.markdown("**Resumen de las cinco empresas**")
    st.dataframe(result.summary, width="stretch", hide_index=True)
    choice = st.selectbox(
        "Gráfica y unidad", [
            "Precio histórico", "Rendimiento diario", "Volatilidad",
            "Volumen", "Índice base 100", "Correlación",
        ], key="tempus_integral_audit_view"
    )
    _draw_chart(result, choice)
    st.caption(
        "Frecuencia: sesiones bursátiles diarias · MA20/MA50: 20 y 50 sesiones · "
        "Las gráficas muestran datos observados, excepto donde se indiquen pronósticos."
    )

    if not result.metrics.empty:
        st.markdown("**Evaluación retrospectiva (walk-forward)**")
        view_metrics = result.metrics.copy()
        columns = [c for c in ("Empresa", "modelo", "MAE", "RMSE", "sMAPE_pct")
                   if c in view_metrics.columns]
        st.dataframe(view_metrics[columns].round(4), width="stretch", hide_index=True)
        st.caption("MAE y RMSE en USD/acción · sMAPE en % · Horizonte: 1 sesión bursátil.")

    if not result.future.empty and {"Date", "pronostico", "modelo", "Empresa"}.issubset(result.future.columns):
        st.markdown("**Pronósticos ya generados · 20 sesiones bursátiles**")
        future_asset = st.selectbox(
            "Empresa del pronóstico", list(sorted(result.future["Empresa"].unique())),
            key="tempus_integral_future_asset"
        )
        view = result.future.loc[result.future["Empresa"].eq(future_asset)].copy()
        view["Date"] = pd.to_datetime(view["Date"], errors="coerce")
        plot = px.line(
            view, x="Date", y="pronostico", color="modelo", markers=True,
            labels={
                "Date": "Fecha (sesiones bursátiles diarias)",
                "pronostico": "Precio previsto (USD/acción)",
                "modelo": "Modelo",
            }, title=f"{future_asset} · Pronóstico a 20 sesiones bursátiles",
        )
        st.plotly_chart(plot, width="stretch")
        st.caption("Pronósticos calculados desde 30/10/2026; no equivalen necesariamente a precios actuales.")

    if result.report_path is not None and result.passed:
        st.download_button(
            "Descargar informe integral PDF verificado · 10 años",
            data=result.report_path.read_bytes(),
            file_name=result.report_path.name,
            mime="application/pdf",
            key="tempus_integral_audit_pdf",
        )

    st.caption(
        "Alcance: verificación programática de datos y cálculos. "
        "No sustituye la prueba de clics, navegación, descargas o presentación "
        "en distintos navegadores."
    )
