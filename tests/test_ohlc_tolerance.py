from datetime import date

import pandas as pd
import pytest

from tempus.core import normalize_prices


def cotizaciones(high=100.0, low=99.0, close=99.5):
    frame = pd.DataFrame({
        "Open": [99.7], "High": [high], "Low": [low],
        "Close": [close], "Volume": [1000],
    }, index=pd.DatetimeIndex(["2026-10-30"], name="Date"))
    return frame


def test_high_redondeo_subcentavo_admitido():
    # Una diferencia ínfima atribuible a precisión numérica.
    data = cotizaciones(high=99.99999, close=100.0)
    assert normalize_prices(data)["High"].iloc[0] == pytest.approx(99.99999)


def test_high_realmente_inconsistente_rechazado():
    with pytest.raises(ValueError, match=r"High inconsistente el 2026-10-30"):
        normalize_prices(cotizaciones(high=99.9, close=100.0))


def test_low_redondeo_subcentavo_admitido():
    data = cotizaciones(low=99.00001, close=99.0)
    assert normalize_prices(data)["Low"].iloc[0] == pytest.approx(99.00001)


def test_low_realmente_inconsistente_rechazado():
    with pytest.raises(ValueError, match=r"Low inconsistente el 2026-10-30"):
        normalize_prices(cotizaciones(low=99.1, close=99.0))


def test_no_muta_entrada():
    data = cotizaciones(high=99.99999, close=100.0)
    antes = data.copy(deep=True)
    normalize_prices(data)
    pd.testing.assert_frame_equal(data, antes)
