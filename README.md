# Euro Area Macro Forecasting

**Status: work in progress**

Forecasting euro area inflation and output with Bayesian VARs, compared against simple benchmarks, plus the effects of monetary policy shocks from a structural VAR and a small New Keynesian DSGE model.

## Goals

- [ ] **Data pipeline:** download macroeconomic series from the ECB Data Portal, Eurostat, OECD and IMF APIs and store them in a SQLite database
- [ ] **Benchmarks:** univariate AR and a classical VAR
- [ ] **Bayesian VAR:** BVAR with a Minnesota prior, implemented from scratch in NumPy
- [ ] **Forecast evaluation:** pseudo-out-of-sample forecasts with a rolling window, RMSE by horizon
- [ ] **Structural identification:** impulse responses to a monetary policy shock (recursive identification)
- [ ] **DSGE:** three-equation New Keynesian model in Dynare, with impulse responses compared to the VAR
- [ ] **Report:** short write-up of methods and results

## Repository structure

```
data/raw/          downloaded series (not tracked, recreated by the scripts)
data/processed/    cleaned datasets
src/               reusable Python code
scripts/           entry-point scripts
dynare/            Dynare .mod files
notebooks/         exploratory notebooks
figures/           charts used in this README and the report
tests/             automated tests (run on every push via GitHub Actions)
```

## Reproducing the results

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_setup.py
pytest
```

The Dynare part requires MATLAB (or GNU Octave) with Dynare installed.

## Data sources

- ECB Data Portal
- Eurostat
- OECD
- IMF

## Author

Luca Rigoni, MSc student in Economics (QEM), Ca' Foscari University of Venice
