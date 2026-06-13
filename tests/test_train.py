from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import DAYS_PER_YEAR, DEFAULT_HALF_LIFE_YEARS
from src.train import recency_weights


def test_recency_weights_have_no_floor() -> None:
    newest = pd.Timestamp("2026-01-01")
    dates = pd.Series([newest - pd.Timedelta(days=DAYS_PER_YEAR * 8), newest])

    weights = recency_weights(dates, half_life_years=DEFAULT_HALF_LIFE_YEARS)

    assert np.allclose(weights, [0.5, 1.0])
