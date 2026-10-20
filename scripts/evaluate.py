"""Pseudo-out-of-sample evaluation of the AR, random walk, VAR and BVAR forecasts."""

from functools import partial

import pandas as pd

from src.database import DB_PATH, connect
from src.dataset import build_quarterly_dataset
from src.evaluation import COVID, compute_rmse, pseudo_out_of_sample
from src.models import ar_forecast, bvar_forecast, random_walk_forecast, rf_forecast, var_forecast

PRE_COVID_END = pd.Period("2019Q4", freq="Q")
TIGHTNESS_GRID = [0.05, 0.1, 0.2, 0.5]
NOTE = "(< 1: the first model outperforms the second)"


def show(title: str, table: pd.DataFrame, note: str | None = None) -> None:
    """Print a titled table of results."""
    print(f"\n=== {title} ===")
    print(table.round(3))
    if note:
        print(note)


def pre_covid(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Keep forecasts made from origins up to 2019Q4, so no Covid quarters enter the estimation."""
    return eval_df[eval_df["origin"] <= PRE_COVID_END]


def main():
    connection = connect(DB_PATH)
    dataset = build_quarterly_dataset(connection)
    connection.close()

    print("Running pseudo-out-of-sample evaluation...")
    evals = {
        "AR": pseudo_out_of_sample(dataset, ar_forecast),
        "RW": pseudo_out_of_sample(dataset, random_walk_forecast),
        "VAR": pseudo_out_of_sample(dataset, var_forecast),
        "BVAR": pseudo_out_of_sample(dataset, bvar_forecast),
        "RF": pseudo_out_of_sample(dataset, rf_forecast),
    }

    # RMSE excluding Covid origins and targets (2020Q1-2021Q4)
    rmse = {name: compute_rmse(ev, exclude=COVID) for name, ev in evals.items()}
    # Same, keeping only origins up to 2019Q4
    rmse_pre = {name: compute_rmse(pre_covid(ev), exclude=COVID) for name, ev in evals.items()}

    show("AR RMSE, Covid included", compute_rmse(evals["AR"]))
    for name in ["AR", "VAR", "BVAR", "RF"]:
        show(f"{name} RMSE, excluding Covid", rmse[name])

    show("AR / RW, excluding Covid", rmse["AR"] / rmse["RW"], NOTE)
    show("VAR / AR, excluding Covid", rmse["VAR"] / rmse["AR"], NOTE)
    show("BVAR / AR, excluding Covid", rmse["BVAR"] / rmse["AR"], NOTE)
    show("BVAR / VAR, excluding Covid", rmse["BVAR"] / rmse["VAR"], NOTE)

    show("VAR / AR, origins up to 2019Q4", rmse_pre["VAR"] / rmse_pre["AR"], NOTE)
    show("BVAR / AR, origins up to 2019Q4", rmse_pre["BVAR"] / rmse_pre["AR"], NOTE)
    show("BVAR / VAR, origins up to 2019Q4", rmse_pre["BVAR"] / rmse_pre["VAR"], NOTE)

    # Sensitivity of the BVAR to the overall tightness lambda
    sensitivity = {}
    for tightness in TIGHTNESS_GRID:
        eval_lambda = pseudo_out_of_sample(dataset, partial(bvar_forecast, tightness=tightness))
        sensitivity[tightness] = compute_rmse(eval_lambda, exclude=COVID) / rmse["AR"]
    show("BVAR / AR by overall tightness (lambda), excluding Covid",
         pd.concat(sensitivity, names=["tightness"]), NOTE)

    show("RF / AR, excluding Covid", rmse["RF"] / rmse["AR"], NOTE)
    show("RF / AR, origins up to 2019Q4", rmse_pre["RF"] / rmse_pre["AR"], NOTE)

if __name__ == "__main__":
    main()