"""Anexo académico de cinco pronósticos futuros con gráficas y trazabilidad."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)

from tempus.future import HORIZONS
from tempus.forecasting import METHODS

TICKERS = ("JNJ", "MSFT", "JPM", "WMT", "XOM")
NAVY = colors.HexColor("#152A4B")
CYAN = "#159DC9"
MAGENTA = "#DE477E"
VIOLET = "#7156D4"


def _chart_future(
    ticker: str,
    frame: pd.DataFrame,
    metadata: dict[str, object],
    historical: pd.DataFrame | None,
) -> BytesIO:
    """Gráfica legible: últimas 60 observaciones, modelos e IC condicional."""
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
    if historical is not None:
        times = pd.to_datetime(historical["Date"], errors="raise")
        closing = pd.to_numeric(historical["Close"], errors="raise")
        history = pd.DataFrame({"Date": times, "Close": closing}).tail(60)
        ax.plot(history["Date"], history["Close"], label="Histórico observado",
                color="#176FA8", linewidth=1.7, marker=".", markersize=2.5)
    else:
        last = pd.Timestamp(metadata["ultimo_dato_observado"])
        ax.scatter([last], [float(metadata["ultimo_cierre"])],
                   color="#176FA8", s=20, label="Último cierre observado")
    models = [
        ("Ingenuo", "#E68439"),
        ("Suavizamiento exponencial", MAGENTA),
        ("ARIMA(1,1,1)", VIOLET),
    ]
    for name, hue in models:
        sub = frame.loc[frame["modelo"] == name].sort_values("sesion")
        x = pd.to_datetime(sub["Date"])
        y = sub["pronostico"].astype(float)
        ax.plot(x, y, label=name, color=hue, linewidth=1.7,
                marker=".", markersize=2.8)
        if name == "ARIMA(1,1,1)":
            low = sub["limite_inferior"].astype(float)
            high = sub["limite_superior"].astype(float)
            ax.fill_between(x, low, high, alpha=0.14, color=VIOLET,
                            label="ARIMA: intervalo condicional 95%")
    ax.axvline(pd.Timestamp(metadata["ultimo_dato_observado"]),
               linestyle="--", linewidth=0.9, color="#64748B")
    ax.set_ylabel("Precio (USD)", fontsize=9)
    ax.set_title(f"{ticker}: histórico y pronósticos posteriores",
                 fontsize=10.5, loc="left", color="#152A4B")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper left", fontsize=7.4, ncol=2, frameon=False)
    ax.tick_params(axis="x", rotation=18, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=145, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def create_future_appendix(
    results: Mapping[str, tuple[pd.DataFrame, dict[str, object]]],
    destination: Path,
    historical: Mapping[str, pd.DataFrame] | None = None,
) -> Path:
    """Produce anexo de 5 series; no sobrescribe un archivo existente.

    El parámetro `historical` añade 60 precios observados a cada figura.
    Los valores futuros siguen marcados como estimados, no realizados.
    """
    if set(results) != set(TICKERS):
        raise ValueError("Se requieren exactamente las cinco series para el anexo")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Ya existe el informe: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TempTitle", parent=styles["Title"], fontSize=19, leading=24,
        textColor=NAVY, spaceAfter=10))
    styles.add(ParagraphStyle(
        name="TempBody", parent=styles["BodyText"], fontSize=9.3,
        leading=14.5, spaceAfter=8))
    styles.add(ParagraphStyle(
        name="TempCaption", parent=styles["BodyText"], fontSize=7.5,
        leading=10, textColor=colors.HexColor("#52657C"), spaceAfter=8))
    story = [
        Paragraph("TEMPUS EXPLORATIO | PRONÓSTICOS FUTUROS", styles["TempTitle"]),
        Paragraph("Anexo integrado a la Práctica 1", styles["Heading2"]),
        Paragraph(
            "Las cifras siguientes son estimaciones posteriores a la última observación "
            "de cada archivo, <b>no datos observados</b>. Se presentan por separado "
            "de las métricas de evaluación histórica walk-forward.", styles["TempBody"]),
        Paragraph(
            "<b>Método:</b> ajuste de los modelos ingenuo, suavizamiento exponencial simple "
            "y ARIMA(1,1,1) con el histórico completo. Se pronostican sesiones de la bolsa "
            "de Nueva York (XNYS); se muestran cortes de 1, 5, 10 y 20 sesiones. "
            "El intervalo ARIMA al 95% es condicional a sus supuestos.",
            styles["TempBody"]),
        Paragraph(
            "<b>Alcance Big Data:</b> se utilizan muestras financieras obtenidas mediante "
            "una fuente web; el volumen procesado no equivale por sí mismo a Big Data "
            "ni supone procesamiento distribuido.", styles["TempBody"]),
    ]
    for index, ticker in enumerate(TICKERS):
        frame, meta = results[ticker]
        story += [
            Paragraph(f"{index + 1}. {ticker}: estimaciones posteriores al cierre",
                      styles["Heading2"]),
            Paragraph(
                f"Último dato observado: {meta['ultimo_dato_observado']} | "
                f"Último cierre: {float(meta['ultimo_cierre']):.2f} USD | "
                f"Observaciones utilizadas: {int(meta['observaciones_entrenamiento']):,}.",
                styles["TempBody"]),
        ]
        plot = _chart_future(
            ticker, frame, meta,
            historical.get(ticker) if historical is not None else None,
        )
        story.append(Image(plot, width=16.7 * cm, height=6.15 * cm))
        story.append(Paragraph(
            f"Figura {index + 1}. Últimos cierres conocidos y estimaciones por modelo; "
            "la banda violeta muestra únicamente el intervalo condicional de ARIMA.",
            styles["TempCaption"]))
        rows = [["Modelo", "Sesión", "Fecha", "Estimación (USD)", "IC 95% ARIMA (USD)"]]
        limited = frame[frame["sesion"].isin(HORIZONS)].copy()
        method_order = {name: i for i, name in enumerate(METHODS)}
        limited["_sort"] = limited["modelo"].map(method_order)
        limited = limited.sort_values(["sesion", "_sort"])
        for _, record in limited.iterrows():
            lower, upper = record["limite_inferior"], record["limite_superior"]
            interval = (
                f"{lower:.2f} a {upper:.2f}"
                if pd.notna(lower) and pd.notna(upper) else "No estimado"
            )
            rows.append([
                str(record["modelo"]), str(int(record["sesion"])),
                pd.Timestamp(record["Date"]).date().isoformat(),
                f"{float(record['pronostico']):.2f}", interval,
            ])
        table = Table(
            rows, colWidths=[3.75*cm, 1.55*cm, 2.65*cm, 3.25*cm, 5.0*cm],
            repeatRows=1, hAlign="LEFT",
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), NAVY),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 7.25),
            ("LEADING", (0,0), (-1,-1), 9),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.white, colors.HexColor("#F0F5FC")]),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.extend([
            table, Spacer(1, 8),
            Paragraph(
                "Interpretación: la comparación de niveles estimados no permite afirmar "
                "que se observarán esos precios. Una pendiente casi horizontal es "
                "coherente con los modelos de referencia a corto plazo. "
                "La incertidumbre crece con el horizonte; acontecimientos inesperados "
                "pueden situar el precio fuera del intervalo.",
                styles["TempCaption"]),
        ])
        if index < len(TICKERS) - 1:
            story.append(PageBreak())
    story.extend([
        Paragraph("Limitaciones y recomendaciones de interpretación", styles["Heading2"]),
        Paragraph(
            "Los resultados históricos de MAE, RMSE y sMAPE se calcularon sobre 26 "
            "orígenes seleccionados cada diez sesiones. Estos resultados no prueban "
            "que las estimaciones futuras serán correctas. No son señales de compra "
            "o venta ni recomendaciones financieras. Los modelos no incorporan "
            "eventos futuros ni cambios de régimen.",
            styles["TempBody"]),
        Paragraph(
            "Las fechas de predicción son posteriores al último cierre del archivo "
            "de entrada; pueden ser anteriores al día en que se lea el informe. "
            "Para estimaciones vigentes se debe actualizar el histórico y "
            "volver a ejecutar el generador, conservando procedencia y versiones.",
            styles["TempBody"]),
    ])
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#65758D"))
        canvas.drawString(1.6*cm, 1.05*cm,
                          "TEMPUS DECISIO · Módulo I · Pronósticos experimentales")
        canvas.drawRightString(A4[0] - 1.6*cm, 1.05*cm, str(doc.page))
        canvas.restoreState()
    try:
        doc = SimpleDocTemplate(
            str(target), pagesize=A4, topMargin=1.35*cm,
            bottomMargin=1.65*cm, leftMargin=1.55*cm,
            rightMargin=1.55*cm,
            title="TEMPUS EXPLORATIO - Anexo de pronósticos de cinco empresas",
            author="TEMPUS DECISIO",
        )
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target
