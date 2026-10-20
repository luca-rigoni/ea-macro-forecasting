"""Tests for the conversion to quarterly frequency in src/dataset.py."""

import pandas as pd

from src.dataset import to_quarterly


def test_monthly_series_are_averaged_over_complete_quarters_only():
    data = pd.DataFrame({
        "period": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
        "value": [1.0, 2.0, 3.0, 10.0, 20.0],
    })
    quarterly = to_quarterly(data, "M")
    assert quarterly.index.tolist() == [pd.Period("2024Q1", freq="Q")]  # 2024Q2 has only two months
    assert quarterly.iloc[0] == 2.0


def test_quarterly_series_are_left_unchanged():
    data = pd.DataFrame({"period": ["2024Q1", "2024Q2"], "value": [1.0, 2.0]})
    quarterly = to_quarterly(data, "Q")
    assert quarterly.tolist() == [1.0, 2.0]
