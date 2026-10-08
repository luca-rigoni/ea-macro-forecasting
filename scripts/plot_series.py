"""
Plot the four euro area series stored in the database.
Run from the project root:  python -m scripts.plot_series
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.database import DB_PATH, connect, load_series
from src.download import SERIES

FIGURE_PATH = "figures/ea_macro_series.png"

def main():
    # 1. Opens the database connection
    connection = connect(DB_PATH)

    # 2. Creates a 2x2 grid of subplots for the four series 
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    for ax, (name, info) in zip(axes.flat, SERIES.items()):
        data = load_series(connection, name)
        dates = pd.PeriodIndex(data["period"], freq=info["frequency"]).to_timestamp()

    # GDP is a level: show its year-on-year growth rate instead
        if name == "gdp":
            values = data["value"].pct_change(4) * 100
            title = "Real GDP, year-on-year growth"
            unit = "percent"
        else:
            values = data["value"]
            title = info["description"]
            unit = info["unit"]

        ax.plot(dates, values)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)

    connection.close()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Figure saved to {FIGURE_PATH}")

if __name__ == "__main__":
    main()