"""Forecasting models: random walk, AR, VAR, BVAR with a Minnesota prior and random forest."""

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor

from src.database import connect, DB_PATH
from src.dataset import build_quarterly_dataset

#Maximum number of lags to consider: 8 quarters = 2 years
MAX_LAGS = 8

#Maximum lags for VAR (quarters)
MAX_LAGS_VAR = 4

#BVAR Hyperparameters (prior strength)
BVAR_LAGS = 4
TIGHTNESS = 0.2
CROSS_TIGHTNESS = 0.5

#Prior mean for own lag 1 (delta_i)
OWN_LAG_PRIOR_MEAN= {
    "gdp_growth": 0.0,
    "inflation": 1.0,
    "unemployment": 1.0,
    "euribor3m": 1.0,
}

# Random forest: lags of every variable used as features, number of trees, seed for reproducibility
RF_LAGS = 4
RF_TREES = 300
RF_SEED = 0

TARGETS = ["gdp_growth", "inflation"]

def forecast_ar(series: pd.Series, steps: int=8):
    """Select the lag order by BIC, fit an AR(p) model and forecast `steps` quarters ahead."""

    # 1. Select the optimal lag structure based on BIC criterion
    selection = ar_select_order(series, maxlag=MAX_LAGS, ic="bic")
    chosen_lags = selection.ar_lags 

    # 2. Fit the AR model with the selected lags
    model = AutoReg(series, lags=chosen_lags)
    model_fit = model.fit()

    # 3. Generates forecasts for the "steps" number of periods ahead
    forecast = model_fit.forecast(steps=steps)

    return forecast, chosen_lags

def ar_forecast(train: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Wrapper function to generate AR forecasts for all target variables."""
    return pd.DataFrame({v: forecast_ar(train[v], steps)[0] for v in TARGETS})

def random_walk_forecast(train, steps):
    future = pd.period_range(train.index[-1] + 1, periods=steps, freq="Q")
    return pd.DataFrame({v: [train[v].iloc[-1]] * steps for v in TARGETS}, index=future)

def var_forecast(train: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Estimates a VAR(p) model with lag length selected by BIC and generates out-of sample forecasts for all target variables."""

    #1 Inizialize VAR object using all 4 dataset columns
    model = VAR(train)

    #2 Select the optimal lag structure based on BIC criterion (forcing at least 1 lag)
    selected_order = model.select_order(MAX_LAGS_VAR)
    lags = max(1, selected_order.bic)

   #3 Estimates equations' parameters via OLS
    results = model.fit(lags)

    #4 Generates forecasts for the "steps" number of periods ahead 
    values = results.forecast(train.values[-lags:], steps=steps)

    #5 Builds future index and returns a DataFrame with the forecasts
    future = pd.period_range(train.index[-1] + 1, periods=steps, freq="Q")

    return pd.DataFrame(values, index=future, columns=train.columns)[TARGETS]

def lag_matrix(values:np.ndarray, lags: int)-> tuple[np.ndarray, np.ndarray]:
    """Builds target matrix Y and regressor matrix X"""
    T, k = values.shape

    #Target Matrix Y: from t=lags to T
    Y = values[lags:]

    #Regressor Matrix X starting with a column of ones for the intercept
    cols = [np.ones(T-lags)]

    #Append lag matrices for l=1, .., lags
    for l in range(1, lags+1):
        cols.append(values[lags-l:T-l])

    #Stacks the columns horizontally to form the regressor matrix X
    X = np.column_stack(cols)

    return Y, X

def ar_residual_variance(series: np.ndarray, lags: int) -> float:
    """Estimates an AR(p) model via OLS and returns the residual variance."""
    values=series.reshape(-1, 1)
    Y, X = lag_matrix(values, lags)

    #OLS estimation of AR(p) model
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    #Compute OLS residuals
    residuals = Y - X @ beta

    #Degree of freedom adjustment for unbiased variance estimation: N-num_params
    dof = len(Y) - X.shape[1]
    sigma2 = np.sum(residuals**2) / dof

    return float(sigma2)

def bvar_coefficients(train: pd.DataFrame, lags: int = BVAR_LAGS, tightness: float = TIGHTNESS, cross_tightness: float = CROSS_TIGHTNESS) -> np.ndarray:
    """Estimates the BVAR coefficients using the Minnesota prior."""
    cols = list(train.columns)
    k = len(cols)
    values = train.values

    #1 Build Y and X Matrices
    Y, X = lag_matrix(values, lags)

    #2 Compute univariate AR(p) residual variances for scale adjustments
    sigma2 = np.array([ar_residual_variance(values[:, i], lags) for i in range(k)])

    num_params = 1 + k * lags
    B = np.zeros((num_params, k))

    #3 Estimate equation by equation
    for i in range(k):
        var_name = cols[i]
        delta_i = OWN_LAG_PRIOR_MEAN[var_name]

        prior_mean = np.zeros(num_params)
        prior_var = np.zeros(num_params)

        # Intercept Prior
        prior_mean[0] = 0.0
        prior_var[0] = 1e6

        # Lags Prior
        for l in range(1, lags + 1):
            for j in range(k):
                col_idx = 1 + (l-1) * k + j

                if j == i:
                # Own lag: prior mean delta_i only at lag 1, variance (lambda / l)^2 at every lag
                    if l == 1:
                        prior_mean[col_idx] = delta_i
                    prior_var[col_idx] = (tightness / l) ** 2
                else:
                    # Cross variable lag
                    std_dev = (tightness * cross_tightness / l) * np.sqrt(sigma2[i]/sigma2[j])
                    prior_var[col_idx] = std_dev ** 2

        assert np.all(prior_var > 0), "every coefficient needs a positive prior variance"

        #4 Posterior Mean Calculation
        precision = (X.T @ X) / sigma2[i] + np.diag(1.0 / prior_var)
        rhs = (X.T @ Y[:, i]) / sigma2[i] + prior_mean / prior_var

        # Solve system: precision * B[:, i] = rhs
        B[:, i] = np.linalg.solve(precision, rhs)

    return B

def bvar_forecast(train: pd.DataFrame, steps: int, lags: int = BVAR_LAGS, tightness: float = TIGHTNESS, cross_tightness: float = CROSS_TIGHTNESS) -> pd.DataFrame:
    """Generates dynamic multi-step out-of-sample forecasts using BVAR with Minnesota prior."""
    # 1. Estimate BVAR coefficient matrix B of shape (1 + k*lags, k)
    B = bvar_coefficients(train, lags=lags, tightness=tightness, cross_tightness=cross_tightness)

    # Convert last 'lags' observations into a list for recursive updating
    history = list(train.values[-lags:])
    forecasts = []

    # 2. Iterative forecasting loop
    for _ in range(steps):
        # Build regressor vector x = [1, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
        x = [1.0]
        for l in range(1, lags + 1):
            x.extend(history[-l])

        x_arr = np.array(x)

        # Compute point prediction for next period
        y_next = x_arr @ B

        forecasts.append(y_next)
        history.append(y_next)

    # 3. Create future period index and return target variables DataFrame
    future = pd.period_range(train.index[-1] + 1, periods=steps, freq="Q")
    forecast_df = pd.DataFrame(forecasts, index=future, columns=train.columns)

    return forecast_df[TARGETS]

def rf_features(data: pd.DataFrame, lags: int = RF_LAGS) -> pd.DataFrame:
    """Features known at each date t: the values of every variable at t, t-1, ..., t-(lags-1)."""
    return pd.concat(
        {f"{col}_lag{l}": data[col].shift(l) for col in data.columns for l in range(lags)},
        axis=1,
    )


def rf_forecast(train: pd.DataFrame, steps: int, lags: int = RF_LAGS, n_trees: int = RF_TREES) -> pd.DataFrame:
    """Direct multi-step random forest forecasts: one forest per target variable and horizon."""
    features = rf_features(train, lags)
    latest = features.iloc[[-1]]  # what we know at the forecast origin

    future = pd.period_range(train.index[-1] + 1, periods=steps, freq="Q")
    forecasts = pd.DataFrame(index=future, columns=TARGETS, dtype=float)

    for h in range(1, steps + 1):
        for target in TARGETS:
            # Pair the features at t with the target at t+h, and drop incomplete rows
            sample = features.assign(y=train[target].shift(-h)).dropna()
            model = RandomForestRegressor(n_estimators=n_trees, random_state=RF_SEED, n_jobs=-1)
            model.fit(sample.drop(columns="y"), sample["y"])
            forecasts.loc[future[h - 1], target] = model.predict(latest)[0]

    return forecasts


if __name__ == "__main__":
    # 1. Connect to the database and build the quarterly dataset
    connection = connect(DB_PATH)
    dataset = build_quarterly_dataset(connection)
    connection.close()

    #2. Define variables to forecast
    target_variables = ["gdp_growth", "inflation"]

    # 3. Forecast each series in the dataset
    for var in target_variables:
        forecast, lags = forecast_ar(dataset.loc[:"2019Q4", "gdp_growth"], steps=8)

        print(f"Forecast for {var}:")
        print(f"Chosen lags (BIC): {lags}")
        print(f"Forecasted values: {forecast}\n")


if __name__ == "__main__":
    from statsmodels.tsa.api import VAR
    from src.database import connect
    from src.dataset import DB_PATH, build_quarterly_dataset

    connection = connect(DB_PATH)
    df = build_quarterly_dataset(connection)
    connection.close()

    # Test BVAR convergence to OLS VAR with loose prior on train slice
    train_slice = df.loc[:"2015Q4"]

    bvar_test = bvar_forecast(train_slice, steps=8, lags=2, tightness=1e4, cross_tightness=1.0 )

    ols_var = VAR(train_slice).fit(2)
    ols_forecast_values = ols_var.forecast(train_slice.values[-2:], steps=8)
    ols_test = pd.DataFrame(ols_forecast_values, index=bvar_test.index, columns=df.columns )[TARGETS]

    max_diff = np.abs(bvar_test.values - ols_test.values).max()
    print(f"=== OLS CONVERGENCE TEST ===")
    print(f"Maximum absolute difference BVAR(loose) vs VAR(OLS): {max_diff:.8f}")

    if max_diff < 1e-4:
        print("SUCCESS: BVAR implementation correctly matches OLS when prior is uninformative!\n" )
    else:
        print( "WARNING: Mismatch detected! Check column ordering or lag index alignment.\n" )