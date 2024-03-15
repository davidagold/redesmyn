from asyncio import Future
from typing import Tuple, Self
import polars as pl


class PySchema:
    def __new__(cls) -> Self: ...

    @classmethod
    def from_struct_type(cls, struct_type: pl.Struct) -> PySchema: ...

    def as_str(self) -> str: ...


class Endpoint:
    def __new__(
        cls,
        signature: Tuple[pl.Struct, pl.Struct],
        path: str,
        handler: str,
        batch_max_delay_ms: int = 10,
        batch_max_size: int = 50,
    ) -> "Endpoint": ...

class PyServer:
    def __new__(cls) -> Self: ...

    def register(self, endpoint: Endpoint) -> Self: ...

    def serve(self) -> Future: ...
