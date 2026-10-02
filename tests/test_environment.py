"""Basic checks that the environment used by the project is complete."""

import importlib
import sys

import pytest


def test_python_version():
    assert sys.version_info >= (3, 10)


@pytest.mark.parametrize(
    "package",
    ["numpy", "pandas", "scipy", "statsmodels", "sklearn", "matplotlib", "requests"],
)
def test_package_importable(package):
    importlib.import_module(package)
