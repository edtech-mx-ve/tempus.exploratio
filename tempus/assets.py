"""Catálogo de entidades y cinco subclases financieras."""
from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class FinancialTimeSeries:
    """Identidad de una serie; los cálculos son funciones compartidas."""
    ticker: ClassVar[str] = ""
    company: ClassVar[str] = ""
    sector: ClassVar[str] = ""

    @classmethod
    def metadata(cls) -> dict[str, str]:
        """Devuelve la identidad inmutable para trazabilidad."""
        if not cls.ticker or not cls.company or not cls.sector:
            raise ValueError("La subclase debe definir ticker, company y sector")
        return {"ticker": cls.ticker, "company": cls.company, "sector": cls.sector}

class HealthcareSeries(FinancialTimeSeries):
    ticker, company, sector = "JNJ", "Johnson & Johnson", "Salud"

class TechnologySeries(FinancialTimeSeries):
    ticker, company, sector = "MSFT", "Microsoft", "Tecnología"

class BankingSeries(FinancialTimeSeries):
    ticker, company, sector = "JPM", "JPMorgan Chase", "Financiero"

class RetailSeries(FinancialTimeSeries):
    ticker, company, sector = "WMT", "Walmart", "Comercio minorista"

class EnergySeries(FinancialTimeSeries):
    ticker, company, sector = "XOM", "ExxonMobil", "Energía"

ASSETS: dict[str, type[FinancialTimeSeries]] = {
    cls.ticker: cls for cls in (
        HealthcareSeries, TechnologySeries, BankingSeries,
        RetailSeries, EnergySeries
    )
}
