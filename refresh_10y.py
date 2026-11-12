"""Descarga reproducible de cinco históricos diarios de 10 años para TEMPUS.

Uso: python refresh_10y.py [--until YYYY-MM-DD] [--output data/snapshots]
El periodo termina inclusivamente en --until; no sobreescribe snapshots existentes.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import logging
from pathlib import Path
import sys

from tempus.assets import ASSETS
from tempus.history_period import CUTOFF, covers_period, period_start
from tempus.io import download_ticker, save_snapshot

LOG = logging.getLogger("tempus.refresh_10y")


def run(until: date, output: Path, tickers: tuple[str, ...] | None = None) -> int:
    """Descarga los cinco activos; informa de fallos individuales sin ocultarlos."""
    start = period_start(10, until)
    LOG.info("Periodo solicitado: %s → %s (inclusive), sesiones diarias", start, until)
    successes: list[str] = []
    failures: list[str] = []
    for symbol in (tickers if tickers is not None else tuple(ASSETS)):
        try:
            frame = download_ticker(symbol, start, until + timedelta(days=1))
            if not covers_period(frame.index, start, until):
                raise ValueError("La respuesta de Yahoo Finance no cubre los diez años")
            saved = save_snapshot(frame, symbol, output)
            successes.append(symbol)
            LOG.info("%s: %s sesiones; %s → %s; snapshot %s",
                     symbol, len(frame), frame.index.min().date(),
                     frame.index.max().date(), saved)
        except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
            failures.append(symbol)
            LOG.error("%s: %s", symbol, exc)
    print(f"Descargadas: {len(successes)}/5 · {', '.join(successes)}")
    if failures:
        print(f"Pendientes: {', '.join(failures)}", file=sys.stderr)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", default=CUTOFF.isoformat(),
                        help="Última fecha incluida YYYY-MM-DD; por defecto 2026-10-30")
    parser.add_argument("--output", default="data/snapshots",
                        help="Directorio para versiones Parquet + JSON")
    parser.add_argument("--tickers", nargs="+", choices=tuple(ASSETS), default=None,
                        help="Descarga solo estas empresas (ej.: --tickers JPM XOM)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        until = date.fromisoformat(args.until)
        if until > date.today():
            raise ValueError("La fecha final no puede estar en el futuro")
        return run(until, Path(args.output), tuple(args.tickers) if args.tickers else None)
    except (ValueError, OSError) as exc:
        LOG.error("Parámetro inválido: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
