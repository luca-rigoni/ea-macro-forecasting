"""Build the quarterly dataset used by the forecasting models."""

import numpy as np
import pandas as pd
from src.database import connect, load_series, DB_PATH
from src.download import SERIES

def to_quarterly(data: pd.DataFrame, frequency: str) -> pd.Series:
    """Convert a series to quarterly frequency, averaging the months of complete quarters."""
    series = pd.Series(
        data["value"].values,
        index=pd.PeriodIndex(data["period"], freq=frequency),
    )
    if frequency == "Q":
        return series

    quarterly = series.groupby(series.index.asfreq("Q")).agg(["mean", "count"])
    complete_quarters = quarterly[quarterly["count"] == 3]["mean"]

    return complete_quarters

def build_quarterly_dataset (connection)-> pd.DataFrame:
    gdp_raw =   load_series(connection, "gdp")
    gdp = to_quarterly(gdp_raw, SERIES ["gdp"]["frequency"])

    hicp_raw =  load_series(connection, "hicp")
    hicp = to_quarterly(hicp_raw, SERIES ["hicp"]["frequency"])  

    unemp_raw = load_series(connection, "unemployment")
    unemp = to_quarterly(unemp_raw, SERIES ["unemployment"]["frequency"])

    euribor_raw = load_series(connection, "euribor3m")
    euribor = to_quarterly(euribor_raw, SERIES ["euribor3m"]["frequency"])

    gdp_growth = 400 * np.log(gdp).diff()

    dataset = pd.concat(
    {
        "gdp_growth": gdp_growth,
        "inflation": hicp,
        "unemployment": unemp,
        "euribor3m": euribor,
    },
    axis=1,
    )

    dataset_clean = dataset.loc["2000Q1":].dropna()

    return dataset_clean

if __name__ == "__main__":
    connection = connect(DB_PATH)
    df = build_quarterly_dataset(connection)
    print(df.head())
    print(df.tail())
    print(f"Total Rows: {len(df)}")
    
connection.close()