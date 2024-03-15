from asyncio import Future
from redesmyn.py_redesmyn import Endpoint, PyServer
from typing import Callable, List, Self, Tuple
import polars as pl


class Server:
    _pyserver: PyServer
    _endpoints: List[Endpoint]

    def __init__(self) -> None:
        self._pyserver = PyServer()
        self._endpoints = []

    def register(self, endpoint: Endpoint) -> Self:
        self._endpoints.append(endpoint)
        self._pyserver.register(endpoint)
        return self
    
    def serve(self) -> Future:
        return self._pyserver.serve()

    def __repr__(self) -> str:
        tab = " " * 4
        repr_endpoints = "\n".join(f"{tab}{endpoint}" for endpoint in self._endpoints)
        repr = f"Endpoints\n{repr_endpoints}"
        return repr
    
    def endpoint(self, path: str, signature: Tuple[pl.Struct, pl.Struct]) -> Callable[[Callable], Callable]:
        def wrapper(fn):
            endpoint = Endpoint(
                path=path,
                signature=signature,
                handler=f"{fn.__module__}.{fn.__name__}",
            )
            self.register(endpoint=endpoint)
            return fn
        return wrapper