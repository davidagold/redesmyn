from asyncio import Future
from typing import Callable, Tuple, Self
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
