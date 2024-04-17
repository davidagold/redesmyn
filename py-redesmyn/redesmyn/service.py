from asyncio import Future
import inspect
from itertools import islice
from more_itertools import first, one
from redesmyn.artifacts import ModelCache
from redesmyn.py_redesmyn import PyEndpoint, PyServer
from typing import Callable, List, Self, Tuple, Type, get_args
import polars as pl
from redesmyn.schema import Schema


def extract_schema(annotation: Type) -> pl.Struct:
    schema_cls = one(e for e in islice(get_args(annotation), 1, None) if issubclass(e, Schema))
    return schema_cls.to_struct_type()


def get_signature(f: Callable) -> Tuple[pl.Struct, pl.Struct]:
    s = inspect.signature(f)
    param_df = one(
        param
        for param in s.parameters.values()
        if issubclass(first(get_args(param.annotation)), pl.DataFrame)
    )
    return (extract_schema(param_df.annotation), extract_schema(s.return_annotation))


class Endpoint:
    def __init__(
        self,
        handler: Callable[..., pl.DataFrame],
        signature: Tuple[pl.Struct, pl.Struct],
        path: str,
        cache: ModelCache,
        batch_max_delay_ms: int,
        batch_max_size: int,
    ) -> None:
        self._handler = handler
        self._cache = cache
        self._pyendpoint = PyEndpoint(
            signature=signature,
            path=path,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
            handler=handler,
        )

    def __call__(self, *args, **kawrgs) -> pl.DataFrame:
        return self._handler(*args, **kawrgs)


def endpoint(
    path: str,
    cache: ModelCache,
    batch_max_delay_ms: int = 10,
    batch_max_size: int = 32,
) -> Callable[[Callable], Endpoint]:
    def wrapper(handler: Callable) -> Endpoint:
        return Endpoint(
            handler=handler,
            signature=get_signature(handler),
            path=path,
            cache=cache,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
        )

    return wrapper


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
            from redesmyn import service as svc
            from sklearn.linear_model import Lasso


            @svc.endpoint(
                ...
            )
            def handler(model: Lasso, records: pl.DataFrame) -> pl.DataFrame:
                return model.predict(records)


            server = svc.Server()
            server.register(handler)
        """
        self._endpoints.append(endpoint)
        self._pyserver.register(endpoint._pyendpoint)
        return self

    def serve(self) -> Future:
        """Run the present `Server`.

        ..  code-block:: python

            import asyncio

            from redesmyn import service as svc


            async def main():
                server = svc.Server()
                await server.serve()


            asyncio.run(main())
        """
        return self._pyserver.serve()

    def __repr__(self) -> str:
        tab = " " * 4
        repr_endpoints = "\n".join(f"{tab}{endpoint}" for endpoint in self._endpoints)
        repr = f"Endpoints\n{repr_endpoints}"
        return repr
