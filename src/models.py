"""Forecasting models: AR benchmark (VAR and BVAR to follow)."""

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

from src.database import connect, DB_PATH
from src.dataset import build_quarterly_dataset

#Maximum number of lags to consider: 8 quarters = 2 years
MAX_LAGS = 8

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