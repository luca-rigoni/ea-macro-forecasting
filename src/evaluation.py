import pandas as pd
import numpy as np

HORIZONS = [1, 4, 8]
FIRST_ORIGIN = "2009Q4"   # first forecasts are made for 2010Q1

COVID = pd.period_range("2020Q1", "2021Q4", freq="Q")


def pseudo_out_of_sample(dataset, forecast_fn, first_origin=FIRST_ORIGIN, horizons=HORIZONS):
    """Recursive forecasts: at each origin, estimate on data up to that quarter only."""
    rows = []
    for origin in dataset.loc[first_origin:].index:
        train = dataset.loc[:origin]                       # only the past: no peeking
        forecasts = forecast_fn(train, max(horizons))
        for h in horizons:
            target = origin + h                            # quarter being forecast
            if target not in dataset.index:                # not observed yet
                continue
            for variable in forecasts.columns:
                rows.append({
                    "variable": variable, "horizon": h,
                    "origin": origin, "target": target,
                    "forecast": forecasts.loc[target, variable],
                    "actual": dataset.loc[target, variable],
                })
    return pd.DataFrame(rows)

def compute_rmse(
    eval_df: pd.DataFrame, exclude: pd.PeriodIndex = None) -> pd.DataFrame:
    """
    Computes Root Mean Squared Error (RMSE) grouped by variable and horizon.
    Optionally filters out specific target quarters (e.g. COVID period).
    """
    df = eval_df.copy()
    df["sq_error"] = (df["forecast"] - df["actual"]) ** 2

    if exclude is not None:
        df = df[~df["target"].isin(exclude) & ~df["origin"].isin(exclude)]

    mean_sq_error = df.groupby(["variable", "horizon"])["sq_error"].mean()
    rmse_table = np.sqrt(mean_sq_error).unstack("horizon")

    return rmse_table