"""
Download every project series from the ECB Data Portal and store it in SQLite.
Run from the project root:  python -m scripts.build_database
"""

from pathlib import Path
from src.database import connect, save_series, save_observations
from src.download import SERIES, fetch_series

# Relative path to the SQLite database file
DB_PATH = Path("data/processed/ea_macro.db")


def main():
    # Ensure that the 'data/processed' folder exists on the disk
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Opens/creates the database connection
    connection = connect(DB_PATH)

    # 2. Iteration over each series present in the SERIES dictionary
    for series_id, info in SERIES.items():
        data = fetch_series(info["flow"], info["key"])

        save_series(connection, series_id, info)
        num_obs = save_observations(connection, series_id, data)

        first_period = data["period"].iloc[0]
        last_period = data["period"].iloc[-1]

        print(
            f"Saved {series_id}: {num_obs} observations "
            f"({first_period} -> {last_period})"
        )

    # 3. Saves the changes to the file and closes the connection
    connection.commit()
    connection.close()
    print(f"\nDatabase '{DB_PATH}' updated successfully.")


if __name__ == "__main__":
    main()