"""Impulse responses of the euro area VAR to a monetary policy shock."""

from pathlib import Path

import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

from src.database import DB_PATH, connect
from src.dataset import build_quarterly_dataset

ORDER = ["gdp_growth", "inflation", "unemployment", "euribor3m"]
LABELS = {
    "gdp_growth": "GDP growth (pp, annualised q/q)",
    "inflation": "HICP inflation (pp, y/y)",
    "unemployment": "Unemployment rate (pp)",
    "euribor3m": "3M Euribor (pp)",
}
MAX_LAGS = 4
HORIZON = 16
ESTIMATION_END = "2019Q4"
FIGURE_PATH = Path("figures/var_irf_monetary.png")


def main():
    # 1. Load the dataset and keep the pre-Covid sample, columns in Cholesky order
    connection = connect(DB_PATH)
    dataset = build_quarterly_dataset(connection)
    connection.close()
    data = dataset.loc[:ESTIMATION_END, ORDER]

    # 2. Estimate the VAR, lag order chosen by BIC (at least 1)
    model = VAR(data)
    lags = max(1, model.select_order(maxlags=MAX_LAGS).bic)
    results = model.fit(lags)
    sample = f"{data.index[0]}-{data.index[-1]}"
    print(f"Estimated VAR({lags}) on {sample} ({len(data)} quarters).")

    # 3. Orthogonalised impulse responses and asymptotic standard errors
    irf = results.irf(HORIZON)
    shock = ORDER.index("euribor3m")
    responses = irf.orth_irfs[:, :, shock]
    std_errors = irf.stderr(orth=True)[:, :, shock]
    shock_size = responses[0, shock]
    print(f"Shock size: +{shock_size * 100:.0f} bp on the 3M Euribor (one standard deviation).")

    # 4. Figure: point estimate with +/- 1 standard error bands (about 68%)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    horizons = range(HORIZON + 1)

    for i, (ax, variable) in enumerate(zip(axes.flatten(), ORDER)):
        point = responses[:, i]
        ax.plot(horizons, point, color="#1f77b4", lw=2)
        ax.fill_between(horizons, point - std_errors[:, i], point + std_errors[:, i],
                        color="#1f77b4", alpha=0.25)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(LABELS[variable], fontsize=11, fontweight="bold")
        ax.set_xlabel("Quarters after the shock")
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"Responses to a monetary policy shock (+{shock_size * 100:.0f} bp on 3M Euribor)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01,
             f"VAR({lags}) estimated on {sample}, lag order by BIC. Recursive identification "
             "(GDP growth, inflation, unemployment, Euribor). Bands: +/- 1 asymptotic s.e. (about 68%).",
             ha="center", fontsize=8.5, color="dimgray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {FIGURE_PATH}.")


if __name__ == "__main__":
    main()