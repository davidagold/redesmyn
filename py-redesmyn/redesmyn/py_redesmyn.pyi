from asyncio import Future
from datetime import timedelta
from pathlib import Path
from typing import Callable, Generic, Optional, Self, Tuple, TypeVar

import polars as pl
from pandas._libs.tslibs.timedeltas import Timedelta

from redesmyn.artifacts import CacheConfig, Cron

class PySchema:
    def __new__(cls) -> Self: ...
    @classmethod
    def from_struct_type(cls, struct_type: pl.Struct) -> PySchema: ...
    def as_str(self) -> str: ...

class PyEndpoint:
    def __new__(
        cls,
        signature: Tuple[pl.Struct, pl.Struct],
        path: str,
        handler: Callable,
        batch_max_delay_ms: int = 10,
        batch_max_size: int = 32,
    ) -> "PyEndpoint": ...

class PyServer:
    def __new__(cls) -> Self: ...
    def register(self, endpoint: PyEndpoint, cache_config: CacheConfig) -> Self: ...
    def serve(self) -> Future: ...

class FsClient:
    def __new__(cls, base_path: Path, path_template: str) -> Self: ...

M = TypeVar("M")

class Cache:
    def __new__(
        cls,
        client: FsClient,
        load_model: Callable[..., M],
        max_size: Optional[int] = None,
        schedule: Optional[Cron] = None,
        interval: Optional[timedelta] = None,
        pre_fetch_all: Optional[bool] = None,
    ) -> Self: ...

class LogConfig:
    def __new__(cls, path: Path) -> Self: ...
    def init(self): ...
