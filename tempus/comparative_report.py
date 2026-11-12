"""Apéndice comparativo descriptivo previo a pronósticos futuros.

No descarga datos ni ajusta modelos; emplea evidencia histórica verificada.
"""
from __future__ import annotations

import logging
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)

from tempus.assets import ASSETS
from tempus.reporting import Evidence

LOG = logging.getLogger(__name__)
TICKERS = tuple(ASSETS)
NAVY = colors.HexColor("#142440")
ACCENT = colors.HexColor("#1767E8")


def comparative_rows(evidence: Evidence) -> list[dict[str, object]]:
    """Resume indicadores históricos y error de evaluación para cinco activos."""
    if set(evidence.indicators) != set(TICKERS):
        raise ValueError("Se requieren las cinco series históricas")
    if not {"ticker", "modelo", "MAE", "RMSE", "sMAPE_pct"}.issubset(evidence.metrics.columns):
        raise ValueError("Faltan métricas de evaluación verificadas")
    results = []
    for ticker in TICKERS:
        frame = evidence.indicators[ticker]
        if not {"Date", "Close"}.issubset(frame.columns):
            raise ValueError(f"{ticker}: faltan fecha o cierre")
        dates = pd.to_datetime(frame["Date"], errors="raise")
        close = pd.to_numeric(frame["Close"], errors="raise").to_numpy(dtype=float)
        if len(close) < 30 or not np.isfinite(close).all() or (close <= 0).any():
            raise ValueError(f"{ticker}: precios no válidos o cobertura insuficiente")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{ticker}: fechas inválidas")
        returns = close[1:] / close[:-1] - 1.0
        cum_return = (close[-1] / close[0] - 1) * 100
        annual_vol = float(np.std(returns, ddof=1) * np.sqrt(252) * 100)
        drawdown = float(np.min(close / np.maximum.accumulate(close) - 1) * 100)
        scores = evidence.metrics.loc[evidence.metrics["ticker"] == ticker].copy()
        if len(scores) != 3 or scores["modelo"].duplicated().any():
            raise ValueError(f"{ticker}: se requieren tres modelos evaluados")
        for col in ("MAE", "RMSE", "sMAPE_pct"):
            scores[col] = pd.to_numeric(scores[col], errors="raise")
            if not np.isfinite(scores[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{ticker}: métrica inválida")
        best = scores.sort_values(["sMAPE_pct", "modelo"], kind="stable").iloc[0]
        results.append({
            "ticker": ticker,
            "inicio": dates.iloc[0].date().isoformat(),
            "fin": dates.iloc[-1].date().isoformat(),
            "n": len(close),
            "retorno_pct": float(cum_return),
            "volatilidad_pct": annual_vol,
            "drawdown_pct": drawdown,
            "mejor_modelo": str(best["modelo"]),
            "smape_pct": float(best["sMAPE_pct"]),
            "mae": float(best["MAE"]),
            "rmse": float(best["RMSE"]),
        })
    return results


def create_comparative_appendix(evidence: Evidence, destination: Path) -> Path:
    """Genera páginas comparativas que preceden el bloque de pronósticos futuros."""
    rows = comparative_rows(evidence)
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError(f"El apéndice ya existe: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TempusComparisonTitle", parent=styles["Heading1"], fontSize=15,
        leading=20, textColor=NAVY, spaceAfter=13,
    ))
    styles.add(ParagraphStyle(
        name="TempusComparisonText", parent=styles["BodyText"], fontSize=9,
        leading=14, textColor=NAVY, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="TempusComparisonSmall", parent=styles["BodyText"], fontSize=7.5,
        leading=11, textColor=NAVY,
    ))
    width = A4[0] - 3.6 * cm
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.75*cm, bottomMargin=1.75*cm,
        title="Análisis comparativo de las cinco acciones",
    )
    p = lambda msg: Paragraph(msg, styles["TempusComparisonText"])
    story = [
        Paragraph("ANÁLISIS COMPARATIVO DE LAS CINCO ACCIONES",
                  styles["TempusComparisonTitle"]),
        p("Resumen descriptivo calculado con los cierres históricos y las métricas "
          "walk-forward verificadas. Las cinco series corresponden a sesiones "
          "bursátiles diarias."),
        p(f"<b>Periodo observado:</b> {rows[0]['inicio']} a {rows[0]['fin']}. "
          f"<b>Registros:</b> {sum(r['n'] for r in rows):,} en total."),
    ]
    header = ["Acción", "Variación<br/>del cierre (%)", "Volatilidad<br/>anual (%)",
              "Caída máxima<br/>(%)", "Mejor sMAPE<br/>(%)"]
    table_rows = [[Paragraph(x, styles["TempusComparisonSmall"]) for x in header]]
    for r in rows:
        table_rows.append([
            r["ticker"],
            f"{r['retorno_pct']:+.2f}",
            f"{r['volatilidad_pct']:.2f}",
            f"{r['drawdown_pct']:.2f}",
            f"{r['smape_pct']:.3f}",
        ])
    table = Table(table_rows, colWidths=[width*.15,width*.23,width*.23,width*.20,width*.19],
                  repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F1F5FA")]),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),8),
        ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#CBD5E1")),
    ]))
    story += [table,Spacer(1,.35*cm),p(
        "<b>Lectura:</b> la variación compara el último cierre con el primero; "
        "la volatilidad es la desviación estándar de rendimientos diarios multiplicada "
        "por raíz de 252; la caída máxima mide el descenso desde un máximo previo. "
        "Son indicadores históricos, no probabilidades de pérdida."
    )]
    for r in rows:
        direction = "subió" if r["retorno_pct"] > 0 else ("bajó" if r["retorno_pct"] < 0 else "no cambió")
        company = escape(ASSETS[r["ticker"]].company)
        statement = (
            f"<b>{r['ticker']} · {company}.</b> "
            f"El cierre {direction} {abs(r['retorno_pct']):.2f}% en el periodo. "
            f"Volatilidad anualizada: {r['volatilidad_pct']:.2f}%; "
            f"drawdown máximo: {r['drawdown_pct']:.2f}%. "
            f"En la evaluación retrospectiva, el menor sMAPE "
            f"({r['smape_pct']:.3f}%) correspondió a "
            f"{escape(r['mejor_modelo'])} "
            f"(MAE: {r['mae']:.4f} USD/acción; "
            f"RMSE: {r['rmse']:.4f} USD/acción)."
        )
        story.append(p(statement))
    higher_return = max(rows, key=lambda r:r["retorno_pct"])
    higher_vol = max(rows, key=lambda r:r["volatilidad_pct"])
    story += [
        p(f"<b>Comparación general:</b> {higher_return['ticker']} presentó la mayor "
          f"variación acumulada del cierre ({higher_return['retorno_pct']:+.2f}%) "
          f"y {higher_vol['ticker']} la volatilidad anualizada más elevada "
          f"({higher_vol['volatilidad_pct']:.2f}%). "
          "La precisión se compara mediante sMAPE, no por MAE nominal entre acciones."),
        p("<b>Limitaciones:</b> las variaciones se calculan a partir del campo Close "
          "del snapshot y no equivalen necesariamente a rentabilidad total con dividendos. "
          "El sMAPE procede de una prueba retrospectiva de una sesión y no demuestra "
          "superioridad estadística ni garantiza el futuro. "
          "Esta comparación no constituye asesoramiento de inversión."),
    ]
    doc.build(story)
    LOG.info("Análisis comparativo creado: %s", out)
    return out
