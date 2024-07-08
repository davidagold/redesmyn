from asyncio import Future
from pathlib import Path
from typing import Callable, Generic, Tuple, Self, TypeVar
import polars as pl


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

    def register(self, endpoint: PyEndpoint) -> Self: ...

    def serve(self) -> Future: ...


class FsClient:
    def __new__(cls, base_path: Path, path_template: str) -> Self: ...


M = TypeVar("M")


class Cache:
    def __new__(cls, client: FsClient, load_model: Callable[..., M]) -> Self: ...
