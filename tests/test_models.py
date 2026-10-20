"""Tests for the forecasting models on simulated data (no database needed)."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR

from src.models import (
    TARGETS,
    ar_forecast,
    bvar_forecast,
    lag_matrix,
    random_walk_forecast,
    rf_forecast,
    var_forecast,
)

COLUMNS = ["gdp_growth", "inflation", "unemployment", "euribor3m"]


@pytest.fixture
def data():
    """80 quarters simulated from a stable VAR(1), with the same columns as the real dataset."""
    rng = np.random.default_rng(0)
    A = np.array([
        [0.3, 0.0, 0.0, -0.3],
        [0.02, 0.85, 0.0, 0.02],
        [-0.03, 0.0, 0.9, 0.02],
        [0.03, 0.05, -0.02, 0.9],
    ])
    values = np.zeros((180, 4))
    for t in range(1, 180):
        values[t] = A @ values[t - 1] + rng.normal(0, [1.5, 0.3, 0.15, 0.2])
    index = pd.period_range("2000Q1", periods=80, freq="Q")
    return pd.DataFrame(values[100:] + [1.5, 2.0, 8.0, 2.0], index=index, columns=COLUMNS)


def test_lag_matrix_aligns_lags_with_targets():
    values = np.arange(40, dtype=float).reshape(10, 4)
    Y, X = lag_matrix(values, lags=2)
    assert Y.shape == (8, 4) and X.shape == (8, 1 + 2 * 4)
    assert np.all(X[:, 0] == 1.0)                 # intercept column
    assert np.array_equal(X[0, 1:5], values[1])   # lag 1 of the first target row (t = 2)
    assert np.array_equal(X[0, 5:9], values[0])   # lag 2


def test_bvar_with_a_loose_prior_matches_the_ols_var(data):
    # With an almost flat prior the posterior mean must coincide with OLS
    bvar = bvar_forecast(data, steps=8, lags=2, tightness=1e4, cross_tightness=1.0)
    ols = VAR(data).fit(2).forecast(data.values[-2:], steps=8)
    ols = pd.DataFrame(ols, index=bvar.index, columns=COLUMNS)[TARGETS]
    assert np.abs(bvar.values - ols.values).max() < 1e-3


@pytest.mark.parametrize(
    "forecast_fn",
    [random_walk_forecast, ar_forecast, var_forecast, bvar_forecast,
     lambda train, steps: rf_forecast(train, steps, n_trees=20)],
)
def test_every_model_returns_the_next_eight_quarters(data, forecast_fn):
    forecasts = forecast_fn(data, 8)
    expected = pd.period_range("2020Q1", periods=8, freq="Q")
    assert list(forecasts.columns) == TARGETS
    assert forecasts.index.equals(expected)
    assert not forecasts.isna().any().any()
