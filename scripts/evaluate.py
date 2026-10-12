from src.database import connect
from src.dataset import DB_PATH, build_quarterly_dataset
from src.evaluation import COVID, compute_rmse, pseudo_out_of_sample
from src.models import ar_forecast, random_walk_forecast


def main():
    # 1. Opens the database connection and builds the quarterly dataset
    connection = connect(DB_PATH)
    df = build_quarterly_dataset(connection)
    connection.close()

    print("Running pseudo out-of-sample evaluation...\n")

    # 2. Generate out-of-sample forecasts for both models
    eval_ar = pseudo_out_of_sample(df, ar_forecast)
    eval_rw = pseudo_out_of_sample(df, random_walk_forecast)

    # 3. Calculate the RMSE tables
    rmse_ar_full = compute_rmse(eval_ar)
    rmse_ar_nocovid = compute_rmse(eval_ar, exclude=COVID)
    rmse_rw_nocovid = compute_rmse(eval_rw, exclude=COVID)

    # 4. Calculate the RMSE ratio (Relative RMSE: AR / Random Walk)
    relative_rmse_nocovid = rmse_ar_nocovid / rmse_rw_nocovid

    # 5. Print the results
    print("=== AR Model RMSE (Full Sample) ===")
    print(rmse_ar_full.round(3))
    print("\n" )

    print("=== AR Model RMSE (Excluding COVID: 2020Q1 - 2021Q4) ===")
    print(rmse_ar_nocovid.round(3))
    print("\n" )

    print("=== Relative RMSE: AR / Random Walk (Excluding COVID) ===")
    print(relative_rmse_nocovid.round(3))
    print("(Values < 1.000 indicate AR outperforms Random Walk)\n")


if __name__ == "__main__":
    main()