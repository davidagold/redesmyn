
from asyncio import Future
from redesmyn.py_redesmyn import PyEndpoint, PyServer
from redesmyn.endpoint import Endpoint
from typing import Callable, List, Self, Tuple, Type, cast
import polars as pl
from redesmyn.schema import Schema


SchemaLike = pl.Struct | Type[Schema]
Signature = Tuple[pl.Struct, pl.Struct]


class Server:
    _pyserver: PyServer
    _endpoints: List[Endpoint]

    def __init__(self) -> None:
        self._pyserver = PyServer()
        self._endpoints = []

    def register(self, endpoint: Endpoint) -> Self:
        self._endpoints.append(endpoint)
        self._pyserver.register(endpoint._pyendpoint)
        return self

    def serve(self) -> Future:
        return self._pyserver.serve()

    def __repr__(self) -> str:
        tab = " " * 4
        repr_endpoints = "\n".join(f"{tab}{endpoint}" for endpoint in self._endpoints)
        repr = f"Endpoints\n{repr_endpoints}"
        return repr
