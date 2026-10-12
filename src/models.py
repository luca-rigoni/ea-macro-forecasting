"""Forecasting models: AR benchmark (VAR and BVAR to follow)."""

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import VAR

from src.database import connect, DB_PATH
from src.dataset import build_quarterly_dataset

#Maximum number of lags to consider: 8 quarters = 2 years
MAX_LAGS = 8

#Maximum lags for VAR (quarters)
MAX_LAGS_VAR = 4

TARGETS = ["gdp_growth", "inflation"]

def forecast_ar(series: pd.Series, steps: int=8):
    """Select the lag order by BIC, fit an AR(p) model and forecast `steps` quarters ahead."""

    # 1. Select the optimal lag structure based on BIC criterion
    selection = ar_select_order(series, maxlag=MAX_LAGS, ic="bic")
    chosen_lags = selection.ar_lags 

    # 2. Fit the AR model with the selected lags
    model = AutoReg(series, lags=chosen_lags)
    model_fit = model.fit()

    # 3. Generates forecasts for the "steps" number of periods ahead
    forecast = model_fit.forecast(steps=steps)

    return forecast, chosen_lags

def ar_forecast(train: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Wrapper function to generate AR forecasts for all target variables."""
    return pd.DataFrame({v: forecast_ar(train[v], steps)[0] for v in TARGETS})

def random_walk_forecast(train, steps):
    future = pd.period_range(train.index[-1] + 1, periods=steps, freq="Q")
    return pd.DataFrame({v: [train[v].iloc[-1]] * steps for v in TARGETS}, index=future)

def var_forecast(train: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Estimates a VAR(p) model with lag length selected by BIC and generates out-of sample forecasts for all target variables."""

    #1 Inizialize VAR object using all 4 dataset columns
    model = VAR(train)

    #2 Select the optimal lag structure based on BIC criterion (forcing at least 1 lag)
    selected_order = model.select_order(MAX_LAGS_VAR)
    lags = max(1, selected_order.bic)

   #3 Estimates equations' parameters via OLS
    results = model.fit(lags)

    #4 Generates forecasts for the "steps" number of periods ahead 
    values = results.forecast(train.values[-lags:], steps=steps)

    #5 Builds future index and returns a DataFrame with the forecasts
    future = pd.period_range(train.index[-1] + 1, periods=steps, freq="Q")

    return pd.DataFrame(values, index=future, columns=train.columns)[TARGETS]

if __name__ == "__main__":
    # 1. Connect to the database and build the quarterly dataset
    connection = connect(DB_PATH)
    dataset = build_quarterly_dataset(connection)
    connection.close()

    #2. Define variables to forecast
    target_variables = ["gdp_growth", "inflation"]

    # 3. Forecast each series in the dataset
    for var in target_variables:
        forecast, lags = forecast_ar(dataset.loc[:"2019Q4", "gdp_growth"], steps=8)

        print(f"Forecast for {var}:")
        print(f"Chosen lags (BIC): {lags}")
        print(f"Forecasted values: {forecast}\n")