"""Download euro area series from the ECB Data Portal API."""

import io
import pandas as pd
import requests

BASE_URL = "https://data-api.ecb.europa.eu/service/data"

SERIES = {
    "hicp": {
        "flow": "HICP",
        "key": "M.U2.N.000000.4D0.ANR",
        "description": "HICP inflation, annual rate of change",
        "frequency": "M",
        "unit": "percent",
    },
    "unemployment": {
        "flow": "LFSI",
        "key": "M.I10.S.UNEHRT.TOTAL0.15_74.T",
        "description": "Seasonally adjusted rate, age 15-74, as a percentage of the workforce",
        "frequency": "M",
        "unit": "percent",
    },
    "euribor3m": {
        "flow": "FM",
        "key": "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
        "description": "3-month Euribor, monthly average",
        "frequency": "M",
        "unit": "percent per annum",
    },
    "gdp": {
        "flow": "MNA",
        "key": "Q.Y.I10.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N",
        "description": "Real GDP, chain-linked volumes, seasonally and calendar adjusted",
        "frequency": "Q",
        "unit": "EUR millions",
    },
}

def fetch_series(flow, key):
    """Download one series and return a table with columns period and value."""

    # 1. URL construction
    url = f"{BASE_URL}/{flow}/{key}"

    # 2. HTTP request with parameters and timeout
    response = requests.get(url, params={"format": "csvdata"}, timeout=30)

    # 3. Server errors control (raises an exception if status code is 4xx or 5xx)
    response.raise_for_status()

    # 4. Reads CSV data received via io.StringIO
    df = pd.read_csv(io.StringIO(response.text))

    # 5. Keep and rename the two columns we need; drop periods without a value
    df_filtered = df[["TIME_PERIOD", "OBS_VALUE"]]
    df_renamed = df_filtered.rename(
        columns={"TIME_PERIOD": "period", "OBS_VALUE": "value"}
    )
    df_clean = df_renamed.dropna(subset=["value"]).reset_index(drop=True)

    # 6. Return of the resulting DataFrame
    return df_clean


if __name__ == "__main__":
    for series_id, info in SERIES.items():
        data = fetch_series(info["flow"], info["key"])

        num_obs = len(data)
        first_period = data["period"].iloc[0]
        last_period = data["period"].iloc[-1]
        last_value = data["value"].iloc[-1]

        print(
            f"{series_id} | Observations: {num_obs} | "
            f"Period: {first_period} -> {last_period} | "
            f"Last value: {last_value}"
        )