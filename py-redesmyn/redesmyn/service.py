import inspect
from asyncio import Future
from itertools import islice
from typing import Callable, Generic, List, Self, Tuple, Type, TypeVar, get_args

import polars as pl
from more_itertools import first, one

from redesmyn.artifacts import ArtifactSpec, ModelCache
from redesmyn.py_redesmyn import PyEndpoint, PyServer
from redesmyn.schema import Schema


def extract_schema(annotation: Type) -> pl.Struct:
    schema_cls = one(e for e in islice(get_args(annotation), 1, None) if issubclass(e, Schema))
    return schema_cls.to_struct_type()


def get_signature(f: Callable) -> Tuple[pl.Struct, pl.Struct]:
    s = inspect.signature(f)
    print(f"{list(get_args(t.annotation) for t in s.parameters.values())}")
    param_df = one(
        param
        for param in s.parameters.values()
        if (
            len((type_args := get_args(param.annotation))) > 0
            and issubclass(first(type_args), pl.DataFrame)
        )
    )
    return (extract_schema(param_df.annotation), extract_schema(s.return_annotation))


M = TypeVar("M")


class Endpoint(Generic[M]):
    def __init__(
        self,
        handler: Callable[[M, pl.DataFrame], pl.DataFrame],
        signature: Tuple[pl.Struct, pl.Struct],
        path: str,
        cache: ModelCache[ArtifactSpec[M], M],
        batch_max_delay_ms: int,
        batch_max_size: int,
    ) -> None:
        self._handler = handler
        self._pyendpoint = PyEndpoint(
            signature=signature,
            path=path,
            batch_max_delay_ms=batch_max_delay_ms,
            batch_max_size=batch_max_size,
            handler=handler,
        )
        self._cache = cache

    def __call__(self, records: pl.DataFrame, **kwargs) -> pl.DataFrame:
        model: M = self._cache.get(**kwargs)
        return self._handler(model, records)


def endpoint(
    path: str,
    cache: ModelCache[ArtifactSpec[M], M],
    batch_max_delay_ms: int = 10,
    batch_max_size: int = 32,
) -> Callable[[Callable[[M, pl.DataFrame], pl.DataFrame]], Endpoint]:
    def wrapper(handler: Callable[[M, pl.DataFrame], pl.DataFrame]) -> Endpoint:
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
