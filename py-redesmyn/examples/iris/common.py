from functools import lru_cache
from pathlib import Path

import polars as pl


@lru_cache()
def project_dir():
    return Path(__file__).parent


@lru_cache()
def load_irises() -> pl.DataFrame:
    return pl.read_csv(project_dir() / "data/iris.csv")
