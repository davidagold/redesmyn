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
        signature: Tuple[Optional[pl.Struct], Optional[pl.Struct]],
        path: str,
        handler: Callable,
        batch_max_delay_ms: int = 10,
        batch_max_size: int = 64,
        validate_artifact_params: bool = False,
    ) -> "PyEndpoint": ...

class PyServer:
    def __new__(cls) -> Self: ...
    def register(self, endpoint: PyEndpoint, cache_config: Optional[CacheConfig]) -> Self: ...
    def serve(self) -> Future: ...
    def handle(self) -> "ServerHandle": ...

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
    def start(self) -> None: ...

class LogConfig:
    def __new__(cls, path: Path, emf_path: Optional[Path] = None) -> Self: ...
    def init(self): ...

class ServerHandle:
    def stop(self, graceful: Optional[bool]) -> Future: ...
