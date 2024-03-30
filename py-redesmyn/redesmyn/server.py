from asyncio import Future
from redesmyn.py_redesmyn import PyServer
from redesmyn.endpoint import Endpoint
from typing import List, Self, Tuple, Type
import polars as pl
from redesmyn.schema import Schema


SchemaLike = pl.Struct | Type[Schema]
Signature = Tuple[pl.Struct, pl.Struct]


class Server:
    """An HTTP server.

    Use :py:func:`Server.register` to register endpoints with a given server.

    Run the server with :py:func:`Server.serve`.
    """

    def __init__(self) -> None:
        self._pyserver: PyServer = PyServer()
        self._endpoints: List[Endpoint] = []

    def register(self, endpoint: Endpoint) -> Self:
        """Register an endpoint with the present `Server`.

        ..  code-block:: python

            import polars as pl
            from redesmyn.endpoint import endpoint
            from redesmyn.server import Server


            @endpoint(path="/predictions/{model_version}")
            def handler(model, records: pl.DataFrame) -> pl.DataFrame:
                return model.predict(records)


            server = Server()
            server.register(handler)
        """
        self._endpoints.append(endpoint)
        self._pyserver.register(endpoint._pyendpoint)
        return self

    def serve(self) -> Future:
        """Run the present `Server`.

        ..  code-block:: python

            import asyncio

            from redesmyn.server import Server


            async def main():
                server = Server()
                await server.serve()


            asyncio.run(main())
        """
        return self._pyserver.serve()

    def __repr__(self) -> str:
        tab = " " * 4
        repr_endpoints = "\n".join(f"{tab}{endpoint}" for endpoint in self._endpoints)
        repr = f"Endpoints\n{repr_endpoints}"
        return repr
