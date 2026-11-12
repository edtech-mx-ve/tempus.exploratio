"""Pruebas offline sin dependencia de Yahoo Finance."""
import numpy as np
import pandas as pd
import pytest
from tempus.assets import ASSETS
from tempus.core import enrich_prices, normalize_prices, summarize
from tempus.demo import synthetic_series

def test_five_assets():
    assert set(ASSETS) == {"JNJ", "MSFT", "JPM", "WMT", "XOM"}
    assert all(ASSETS[t].metadata()["ticker"] == t for t in ASSETS)

def test_enrich_has_no_future_leakage():
    data = synthetic_series()
    before = enrich_prices(data.iloc[:90])
    changed = data.copy()
    changed.loc[changed.index[95:], ["Open", "High", "Low", "Close"]] *= 4
    after = enrich_prices(changed).iloc[:90]
    pd.testing.assert_frame_equal(before, after)

def test_reject_duplicate_dates():
    data = synthetic_series()
    duplicate = pd.concat([data, data.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicadas"):
        normalize_prices(duplicate)

def test_reject_negative_price():
    data = synthetic_series()
    data.iloc[0, data.columns.get_loc("Close")] = -1
    with pytest.raises(ValueError, match="positivos"):
        normalize_prices(data)

def test_descriptive_summary():
    result = summarize(synthetic_series())
    assert result["observaciones"] == 130
    assert np.isfinite(result["volatilidad_anual_pct"])

def test_reject_missing_close():
    data = synthetic_series()
    data.loc[data.index[2], "Close"] = np.nan
    with pytest.raises(ValueError, match="ausentes"):
        normalize_prices(data)
