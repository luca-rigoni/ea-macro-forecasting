"""Download euro area series from the ECB Data Portal API."""

import io
import pandas as pd
import requests

BASE_URL = "https://data-api.ecb.europa.eu/service/data"


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
    hicp = fetch_series("HICP", "M.U2.N.000000.4D0.ANR")
    print(hicp.head()) 
    print(hicp.tail())
    print(len(hicp), hicp["period"].iloc[0], hicp["period"].iloc[-1])