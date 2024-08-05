import polars as pl
import pytest
from tests.common import DIR_PYREDESMYN


@pytest.fixture()
def irises() -> pl.DataFrame:
    return pl.read_csv(DIR_PYREDESMYN / "examples/iris/data/iris.csv")
