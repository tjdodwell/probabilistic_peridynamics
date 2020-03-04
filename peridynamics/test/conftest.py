"""Shared definitions for test modules."""
import pathlib
import pytest


@pytest.fixture(scope="session")
def data_path():
    """Path to the test data directory."""
    path = pathlib.Path(__file__).parent.absolute() / "data"
    return path
