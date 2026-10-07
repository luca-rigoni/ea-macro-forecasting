f"""Check that the Python environment is ready and the ECB Data Portal API is reachable.

Run from the project root:  python scripts/check_setup.py
"""

import importlib
import io
import sys

REQUIRED_PACKAGES = [
    "numpy", "pandas", "scipy", "statsmodels",
    "sklearn", "matplotlib", "requests", "openpyxl",
]

# Euro area HICP, overall index, annual rate of change (monthly)
ECB_URL = (
    "https://data-api.ecb.europa.eu/service/data/HICP/M.U2.N.000000.4D0.ANR"
    "?format=csvdata&lastNObservations=6"
)


def check_python():
    version = sys.version_info
    ok = version >= (3, 10)
    status = "[OK]  " if ok else "[FAIL]"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro} (3.10+ required)")
    return ok


def check_packages():
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            print(f"[OK]   {name} {getattr(module, '__version__', '')}")
        except ImportError:
            print(f"[FAIL] {name} is missing: run  pip install -r requirements.txt")
            all_ok = False
    return all_ok


def check_ecb_api():
    try:
        import pandas as pd
        import requests

        response = requests.get(ECB_URL, timeout=30)
        response.raise_for_status()
        data = pd.read_csv(io.StringIO(response.text))
        latest = data[["TIME_PERIOD", "OBS_VALUE"]].tail(3)
        print("[OK]   ECB Data Portal reachable. Latest euro area HICP inflation (%):")
        print(latest.to_string(index=False))
        return True
    except Exception as error:  # report any network or parsing problem
        print(f"[FAIL] ECB Data Portal not reachable: {error}")
        return False


if __name__ == "__main__":
    results = [check_python(), check_packages(), check_ecb_api()]
    print("\nSetup complete." if all(results) else "\nSome checks failed: see above.")
    sys.exit(0 if all(results) else 1)
