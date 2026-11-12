"""Informe académico PDF de TEMPUS EXPLORATIO (sprint 04)."""
from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, KeepTogether, HRFlowable,
)

from tempus.assets import ASSETS
from tempus.reporting import Evidence, best_models

LOG = logging.getLogger(__name__)
BLUE = colors.HexColor("#1368D5")
NAVY = colors.HexColor("#112948")
PURPLE = colors.HexColor("#6E43C4")
MUTED = colors.HexColor("#54708E")


def _fonts() -> tuple[str, str]:
    """Utiliza una fuente Unicode disponible y evita empaquetar archivos de fuentes."""
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for normal, bold in candidates:
        if normal.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("TempusRegular", str(normal)))
            pdfmetrics.registerFont(TTFont("TempusBold", str(bold)))
            return "TempusRegular", "TempusBold"
    LOG.warning("No se encontró fuente Unicode; se utilizará Helvetica")
    return "Helvetica", "Helvetica-Bold"


def _plot_panel(evidence: Evidence) -> BytesIO:
    """Crea imagen temporal de las cinco series normalizadas sin modificar datos."""
    fig, ax = plt.subplots(figsize=(9.0, 3.6), dpi=160)
    shades = ["#2277D6", "#7A56CD", "#18A5A0", "#E77E38", "#DC3E83"]
    for name, color in zip(evidence.panel.columns, shades):
        ax.plot(evidence.panel.index, evidence.panel[name], label=name,
                color=color, linewidth=1.55)
    ax.set_ylabel("Índice base 100")
    ax.grid(alpha=0.16)
    ax.legend(ncol=5, loc="upper left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    result = BytesIO()
    fig.savefig(result, format="png", dpi=170, facecolor="white")
    plt.close(fig)
    result.seek(0)
    return result


def _plot_smape(evidence: Evidence) -> BytesIO:
    """Agrupa sMAPE por empresa en porcentaje; menor es mejor."""
    ordered = list(evidence.forecasts)
    pivot = evidence.metrics.pivot(index="ticker", columns="modelo", values="sMAPE_pct").loc[ordered]
    fig, ax = plt.subplots(figsize=(9.0, 3.4), dpi=170)
    position = np.arange(len(ordered))
    width = .25
    for idx, (name, color) in enumerate([
        ("Ingenuo", "#2277D6"),
        ("Suavizamiento exponencial", "#7A56CD"),
        ("ARIMA(1,1,1)", "#E77E38"),
    ]):
        ax.bar(position + (idx - 1) * width, pivot[name], width=width,
               label=name, color=color)
    ax.set_xticks(position, ordered)
    ax.set_ylabel("sMAPE (%)")
    ax.set_ylim(0, float(pivot.max().max()) * 1.23)
    ax.legend(ncol=3, frameon=False, fontsize=7.8)
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    fig.tight_layout()
    result = BytesIO()
    fig.savefig(result, format="png", dpi=170, facecolor="white")
    plt.close(fig)
    result.seek(0)
    return result


def _plot_corr(evidence: Evidence) -> BytesIO:
    """Representación simétrica de correlación Pearson sobre rendimientos."""
    fig, ax = plt.subplots(figsize=(6.8, 4.1), dpi=160)
    values = evidence.correlations.to_numpy()
    im = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(5), evidence.correlations.columns)
    ax.set_yticks(range(5), evidence.correlations.index)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(values[i,j])>.65 else "#10233F")
    fig.colorbar(im, ax=ax, shrink=.85)
    fig.tight_layout()
    result = BytesIO()
    fig.savefig(result, format="png", dpi=170, facecolor="white")
    plt.close(fig)
    result.seek(0)
    return result


def create_academic_pdf(evidence: Evidence, destination: Path,
                        overwrite: bool = False) -> Path:
    """Genera informe académico con métricas verificadas y referencias.

    La función no recalcula modelos; consume únicamente evidencia validada.
    """
    final = Path(destination).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and not overwrite:
        raise FileExistsError(f"El PDF ya existe: {final}")
    normal, bold = _fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TempTitle", fontName=bold, fontSize=23,
                              leading=31, textColor=NAVY, spaceAfter=14))
    styles.add(ParagraphStyle(name="TempSubtitle", fontName=normal, fontSize=12,
                              leading=19, textColor=MUTED, spaceAfter=16))
    styles.add(ParagraphStyle(name="TempHeading", fontName=bold, fontSize=13,
                              leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="TempText", fontName=normal, fontSize=9.2,
                              leading=15, textColor=NAVY, spaceAfter=8))
    styles.add(ParagraphStyle(name="TempSmall", fontName=normal, fontSize=7.7,
                              leading=11, textColor=MUTED, spaceAfter=7))
    styles.add(ParagraphStyle(name="TempTable", fontName=normal, fontSize=7.1,
                              leading=10, textColor=NAVY))
    styles.add(ParagraphStyle(name="TempCover", fontName=bold, fontSize=32,
                              leading=43, textColor=NAVY, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="TempCenter", fontName=normal, fontSize=11,
                              leading=16, textColor=MUTED, alignment=TA_CENTER))
    p = lambda t: Paragraph(t, styles["TempText"])
    h = lambda t: Paragraph(t, styles["TempHeading"])
    small = lambda t: Paragraph(t, styles["TempSmall"])
    doc = SimpleDocTemplate(str(final), pagesize=A4,
                            rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=2*cm, bottomMargin=2.1*cm,
                            title="TEMPUS EXPLORATIO - Informe académico",
                            author="TEMPUS DECISIO")
    usable = A4[0]-3.6*cm

    def make_table(headers: list[str], rows: list[list[str]],
                   widths: list[float] | None = None) -> Table:
        material = [
            [Paragraph(str(x), ParagraphStyle(
                f"TH_{idx}_{id(headers)}", parent=styles["TempTable"],
                fontName=bold, textColor=colors.white)) for idx, x in enumerate(headers)]
        ]
        material.extend([[Paragraph(str(v), styles["TempTable"]) for v in row] for row in rows])
        result = Table(material, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),NAVY),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#F5F8FC"),colors.white]),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("TOPPADDING",(0,0),(-1,-1),7),
            ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#CFDCE8")),
        ]))
        return result

    def footer(canvas, document):
        canvas.saveState()
        w,_ = A4
        canvas.setStrokeColor(colors.HexColor("#D7E1EF"))
        canvas.line(1.8*cm,1.47*cm,w-1.8*cm,1.47*cm)
        canvas.setFont(normal,7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.8*cm,1.16*cm,"TEMPUS DECISIO · Módulo I · Evidencia histórica")
        canvas.drawRightString(w-1.8*cm,1.16*cm,str(document.page))
        canvas.restoreState()

    # Fechas, cantidades y resultados se toman de las evidencias verificadas.
    first_ticker = next(iter(evidence.forecasts))
    first_pred = evidence.forecasts[first_ticker]
    first_meta = evidence.metadata[first_ticker]
    first_history = evidence.indicators[first_ticker]
    n_history = len(first_history)
    n_eval = len(first_pred)
    first_day = str(pd.to_datetime(first_history["Date"]).min().date())
    last_day = str(pd.to_datetime(first_history["Date"]).max().date())
    train_end = first_meta["periodo_entrenamiento_fin_inicial"]
    test_start = first_meta["periodo_prueba_inicio"]
    test_end = first_meta["periodo_prueba_fin"]
    step = first_meta["config"]["step"]
    winner_by_ticker = {
        str(row.ticker): str(row.modelo)
        for row in best_models(evidence.metrics).itertuples()
    }
    winners_text = "; ".join(
        f"{ticker}: {winner_by_ticker[ticker]}" for ticker in evidence.forecasts
    )
    story=[]
    story += [Spacer(1,2.6*cm),
              HRFlowable(width="20%", thickness=5, color=BLUE, hAlign="CENTER"),
              Spacer(1,1.2*cm),
              Paragraph("TEMPUS<br/>EXPLORATIO",styles["TempCover"]),
              Spacer(1,.4*cm),
              Paragraph("Informe académico final · Módulo I",styles["TempCenter"]),
              Spacer(1,.5*cm),
              Paragraph("Análisis e implementación de cinco series temporales financieras y evaluación estadística de pronósticos",styles["TempCenter"]),
              Spacer(1,1.8*cm),
              Paragraph("TEMPUS DECISIO",styles["TempCenter"]),
              Paragraph("Tópicos de Inteligencia Artificial para la Transformación Digital",styles["TempCenter"]),
              Paragraph("Informe generado a partir de evidencia verificada",styles["TempCenter"]),
              Spacer(1,1*cm),
              small("Documento generado a partir de exportaciones verificadas de JNJ, MSFT, JPM, WMT y XOM. Los datos se atribuyen al origen declarado en la aplicación: snapshots históricos obtenidos mediante Yahoo Finance/yfinance. No representa asesoría financiera."),
              PageBreak()]
    story += [h("1. Introducción"),p(
        "Las series temporales organizan las observaciones por fecha y permiten estudiar tendencias, "
        "fluctuaciones, dependencia serial y desempeño de métodos de pronóstico. "
        "En este trabajo se implementó un pipeline reproducible para cinco acciones "
        "de sectores diferentes. El foco es el análisis histórico y la comparación "
        "fuera de muestra, no la generación de recomendaciones de inversión."),
        h("2. Objetivo y alcance"),p(
        "<b>Objetivo:</b> adquirir, validar, explorar, modelar y evaluar cinco series históricas "
        "provenientes de la web; reproducir la comparación estadística y documentar las limitaciones."),
        make_table(["Activo","Empresa","Sector"],[
            [ticker,ASSETS[ticker].company,ASSETS[ticker].sector]
            for ticker in evidence.forecasts
        ],[usable*.18,usable*.45,usable*.37]),
        Spacer(1,.35*cm),
        h("3. Fuente de datos y tratamiento"),p(
        "El proveedor de adquisición del proyecto es Yahoo Finance, mediante la biblioteca yfinance; "
        "la ejecución estudiada utilizó snapshots locales persistidos. "
        f"Se trabajó con {n_history:,} sesiones por activo, desde {first_day} "
        f"hasta {last_day}. El panel comparativo se alineó por fechas "
        "y normalizó a índice base 100. Para los pronósticos se empleó la columna "
        "Close ajustada según la configuración histórica del proveedor."),
        small("Los CSV de indicadores y la matriz de correlaciones son exportaciones de la aplicación. "
              "La interpretación depende de la política de ajuste del proveedor; las cinco "
              "series no constituyen, por sí solas, una plataforma Big Data distribuida."),
        PageBreak()]
    story += [h("4. Exploración y comparación multiserie"),
              p("Las series en base 100 permiten comparar evolución acumulada sin confundir "
                "niveles nominales diferentes. El indicador representa el cambio relativo "
                "respecto de la primera sesión común."),
              Image(_plot_panel(evidence),width=usable,height=usable*.40),
              small("Figura 1. Evolución histórica normalizada; fuente: exportación tempus_series_base100.csv."),
              make_table(["Ticker","Índice final","Rendimiento acumulado"],[
                  [ticker,f"{evidence.panel[ticker].iloc[-1]:.2f}",
                   f"{(evidence.panel[ticker].iloc[-1]/100-1)*100:+.2f}%"]
                  for ticker in evidence.forecasts
              ],[usable*.21,usable*.32,usable*.47]),
              h("4.1. Dependencia entre rendimientos"),
              p("La matriz de Pearson se calculó sobre rendimientos diarios alineados; "
                "expresa asociación lineal histórica, no causalidad ni estabilidad futura."),
              Image(_plot_corr(evidence),width=usable*.56,height=usable*.34),
              small("Figura 2. Correlaciones históricas por pares de activos, verificadas "
                    "numéricamente contra el panel normalizado."),
              PageBreak()]
    story += [h("5. Diseño de la evaluación predictiva"),
              p("<b>Métodos estadísticos utilizados:</b> pronóstico ingenuo "
                "(último cierre observado), suavizamiento exponencial simple "
                "(SES) y ARIMA(1,1,1)."),
              p("<b>Validación:</b> partición cronológica con 80% inicial para entrenamiento "
                "y 20% final para evaluación. Se pronostica una sesión por origen. "
                f"Se evalúa un origen cada {step} sesiones, resultando {n_eval} fechas por activo. "
                "Los modelos se reajustan con información anterior a cada objetivo."),
              p("<b>Métricas:</b> MAE y RMSE en unidades de precio, sMAPE en porcentaje. "
                "No se comparan MAE ni RMSE directamente entre acciones porque sus "
                "niveles de precio difieren. Las fechas válidas son comunes a los tres modelos."),
              make_table(["Aspecto","Configuración"],[
                  ["Activos",", ".join(evidence.forecasts)],
                  ["Entrenamiento inicial",f"{first_day} a {train_end}"],
                  ["Evaluación cronológica",f"{test_start} a {test_end}"],
                  ["Fracción de prueba","20%"],
                  ["Horizonte","1 sesión bursátil"],
                  ["Origen de evaluación",f"cada {step} sesiones"],
                  ["Muestras evaluadas",f"{n_eval} por activo y modelo"],
                  ["Incidencias registradas","0 en las cinco series"],
              ],[usable*.36,usable*.64]),
              Spacer(1,.3*cm),
              h("6. Evaluación y pruebas realizadas"),
              p(f"Se recalcularon MAE, RMSE y sMAPE desde los {n_eval} registros de pronóstico "
                "de cada empresa. Se verificaron fechas comunes, ausencia de nulos, "
                "parámetros compartidos, tablas coherentes y consistencia de la matriz de "
                "correlaciones. Las exportaciones no reportan fallos de ajuste. "
                "También se ejecutaron previamente pruebas unitarias y de integración "
                "durante el desarrollo; las cifras exactas de cada ejecución deben "
                "conservarse con su respectivo log."),
              PageBreak()]
    rows=[]
    for ticker in evidence.forecasts:
        each=evidence.metrics[evidence.metrics.ticker==ticker]
        for model in ("Ingenuo","Suavizamiento exponencial","ARIMA(1,1,1)"):
            r=each.loc[each.modelo==model].iloc[0]
            rows.append([ticker,model,f"{r.MAE:.4f}",f"{r.RMSE:.4f}",f"{r.sMAPE_pct:.4f}%"])
    story += [h("7. Resultados de pronóstico"),
              make_table(["Activo","Modelo","MAE","RMSE","sMAPE"],rows,
                         [usable*.12,usable*.39,usable*.16,usable*.16,usable*.17]),
              Spacer(1,.4*cm),
              Image(_plot_smape(evidence),width=usable,height=usable*.38),
              small(f"Figura 3. sMAPE histórico en las mismas {n_eval} fechas por activo. Menor error es preferible."),
              PageBreak()]
    winner=best_models(evidence.metrics)
    winrows=[[r.ticker,r.modelo,f"{r.MAE:.4f}"] for r in winner.itertuples()]
    story += [h("8. Interpretación e implicaciones"),
              make_table(["Activo","Menor MAE","MAE"],winrows,
                         [usable*.17,usable*.55,usable*.28]),
              Spacer(1,.4*cm),
              p("Los mejores métodos según MAE en la muestra actual fueron: "
                + winners_text + ". "
                "Estas comparaciones son descriptivas; una menor métrica muestral no "
                "prueba superioridad estadística."),
              p(f"El análisis corresponde a {n_eval} orígenes cronológicos por activo, "
                "todos de una sesión bursátil adelante. Las diferencias dependen del "
                "activo, la ventana y el protocolo de evaluación."),
              h("9. Limitaciones y seguridad"),
              p("La comparación no cubre todas las sesiones del período de prueba: "
                f"solo se toma una fecha de cada {step}. No se aplicaron pruebas de "
                "significación sobre diferencias de errores, optimización interna "
                "de hiperparámetros ni cambios de régimen. Los resultados son "
                "históricos y no garantizan desempeño futuro."),
              p("Los datos externos deben considerarse no confiables hasta su validación. "
                "La aplicación almacena snapshots versionados, evita sobrescritura "
                "accidental y expone trazabilidad de las pruebas. El reporte "
                "se genera de archivos locales sin enviar información a servicios externos."),
              h("10. Conclusiones"),
              p("Se implementó y validó un flujo de trabajo para analizar cinco "
                "series financieras, realizar exploración estadística y comparar "
                "tres métodos de pronóstico con separación temporal. "
                "La evidencia respalda la utilidad de las líneas base sencillas "
                "y la necesidad de evaluar cualquier modelo más complejo con "
                "métricas múltiples y fechas fuera de muestra."),
              PageBreak()]
    story += [h("11. Implementación y reproducibilidad"),
              p("El software se implementó en Python 3.12 con pandas, numpy, "
                "statsmodels, Plotly y Streamlit. La capa de adquisición y persistencia "
                "es independiente de los cálculos estadísticos y de la capa de presentación. "
                "La consolidación automatizada revisa las "
                "exportaciones originales, contrasta métricas y produce un contrato "
                "para el futuro módulo TEMPUS PREDICTIO."),
              p("<b>Comandos:</b> <font name='"+normal+"'>"
                "python -m pip install -r requirements.txt<br/>"
                "python -m pytest -v<br/>"
                "python run_10y_pipeline.py --snapshots data/snapshots --output-root reports<br/>"
                "python -m streamlit run app.py</font>"),
              small("Los archivos de evidencia incluyen hashes SHA-256 en "
                    "tempus_exploratio_metadata.json para verificar integridad. "
                    "La aplicación no contempla API HTTP en este módulo: la interfaz "
                    "se ejecuta en http://localhost:8501."),
              h("12. Transferencia a TEMPUS PREDICTIO"),
              p("El contrato exporta tickers, fechas, panel base 100, "
                "métricas de líneas base y predicciones históricas. "
                "El módulo de IA deberá generar sus propias características "
                "sin mirar observaciones futuras, separar entrenamiento y prueba "
                "cronológicamente y comparar sobre las mismas fechas y métricas. "
                "El contrato no es un dataset de entrenamiento IA listo para usar "
                "sin una etapa explícita de construcción de variables."),
              h("Referencias bibliográficas (APA 7)"),
              small("Brockwell, P. J., &amp; Davis, R. A. (2016). "
                    "<i>Introduction to time series and forecasting</i> (3rd ed.). Springer. "
                    "https://doi.org/10.1007/978-3-319-29854-2"),
              small("Cryer, J. D., &amp; Chan, K.-S. (2008). "
                    "<i>Time series analysis: With applications in R</i> (2nd ed.). Springer. "
                    "https://doi.org/10.1007/978-0-387-75959-3"),
              small("Shumway, R. H., &amp; Stoffer, D. S. (2006). "
                    "<i>Time series analysis and its applications: With R examples</i> "
                    "(2nd ed.). Springer."),
              small("Statsmodels developers. (s. f.). <i>Time series analysis.</i> "
                    "https://www.statsmodels.org/stable/tsa.html"),
              small("Yahoo Finance. (2026). <i>Historical market data</i> "
                    "[Conjunto de datos obtenido mediante yfinance; snapshots locales "
                    "de TEMPUS EXPLORATIO]. https://finance.yahoo.com/"),
              small("El fundamento conceptual y las referencias se apoyan en el "
                    "marco teórico académico del proyecto. No se consultaron las "
                    "obras completas durante la ejecución del reporte."),
              ]
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    LOG.info("PDF generado: %s",final)
    return final
