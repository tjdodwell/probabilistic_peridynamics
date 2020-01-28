import pathlib
import pytest


@pytest.fixture(scope="session")
def data_path():
    path = pathlib.Path(__file__).parent.absolute() / "data"
    return path
