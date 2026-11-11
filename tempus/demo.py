"""Serie sintética para pruebas o demostración claramente etiquetada."""
import numpy as np
import pandas as pd
from tempus.core import normalize_prices

def synthetic_series(n: int = 130, seed: int = 42) -> pd.DataFrame:
    """Genera OHLCV simulado sin representar precios de mercado reales."""
    if not 60 <= n <= 10000:
        raise ValueError("n fuera de rango")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2025-01-02", periods=n)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    opening = prices * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(prices, opening) * 1.008
    low = np.minimum(prices, opening) * 0.992
    return normalize_prices(pd.DataFrame({
        "Open": opening, "High": high, "Low": low, "Close": prices,
        "Volume": rng.integers(1000, 100000, size=n)
    }, index=dates))
