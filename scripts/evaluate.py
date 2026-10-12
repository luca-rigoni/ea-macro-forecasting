import pandas as pd

from src.database import DB_PATH, connect
from src.dataset import build_quarterly_dataset
from src.evaluation import COVID, compute_rmse, pseudo_out_of_sample
from src.models import ar_forecast, random_walk_forecast, var_forecast


def main():
    # 1. Opens the database connection and builds the quarterly dataset
    connection = connect(DB_PATH)
    df = build_quarterly_dataset(connection)
    connection.close()

    print("Running pseudo out-of-sample evaluation...\n")

    # 2. Generate out-of-sample forecasts for both models
    eval_ar = pseudo_out_of_sample(df, ar_forecast)
    eval_rw = pseudo_out_of_sample(df, random_walk_forecast)
    eval_var = pseudo_out_of_sample(df, var_forecast)

    # 3. Calculate the RMSE tables
    rmse_ar_full = compute_rmse(eval_ar)
    rmse_ar_nocovid = compute_rmse(eval_ar, exclude=COVID)
    rmse_rw_nocovid = compute_rmse(eval_rw, exclude=COVID)
    rmse_var_nocovid = compute_rmse(eval_var, exclude=COVID)

    # 4. Calculate the RMSE ratio (Relative RMSE: AR / Random Walk)
    relative_ar_rw = rmse_ar_nocovid / rmse_rw_nocovid
    relative_var_ar = rmse_var_nocovid / rmse_ar_nocovid

    # 5. Print the results
    print("=== AR Model RMSE (Full Sample) ===")
    print(rmse_ar_full.round(3))
    print("\n" )

    print("=== AR Model RMSE (Excluding COVID: 2020Q1 - 2021Q4) ===")
    print(rmse_ar_nocovid.round(3))
    print("\n" )

    print("=== VAR Model RMSE (Excluding COVID: 2020Q1 - 2021Q4) ===")
    print(rmse_var_nocovid.round(3))
    print("\n" )

    print("=== Relative RMSE: AR / Random Walk (Excluding COVID) ===")
    print(relative_ar_rw.round(3))
    print("(Values < 1.000 indicate AR outperforms Random Walk)\n\n")

    print("=== Relative RMSE: VAR / AR (Excluding COVID) ===")
    print(relative_var_ar.round(3))
    print("(Values < 1.000 indicate VAR outperforms AR)\n\n")


#6 . Additional comparison of AR and VAR models before COVID to see if VAR outperforms AR in the pre-pandemic period
    pre_covid = pd.Period("2019Q4", freq="Q")
    ar_pre = compute_rmse(eval_ar[eval_ar["origin"] <= pre_covid], exclude=COVID)
    var_pre = compute_rmse(eval_var[eval_var["origin"] <= pre_covid], exclude=COVID)
    print("=== Relative RMSE: VAR / AR (Pre-COVID) ===")
    print((var_pre / ar_pre).round(3))
    print("(Values > 1.000 indicate AR outperforms VAR, suggesting VAR being less effective due to overfitting)\n\n")

if __name__ == "__main__":
    main()